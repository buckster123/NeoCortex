"""
Memory Health Engine

Manages memory lifecycle:
- Decay calculations based on layer
- Promotion rules (sensory → working → long_term → cortex)
- Stale memory detection
- Duplicate detection and consolidation
- Access pattern analysis

Design based on ApexAurum Memory Protocol Enhancement.
"""

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    COLLECTION_PRIVATE,
    COLLECTION_VILLAGE,
    COLLECTION_BRIDGES,
    COLLECTION_CRUMBS,
    COLLECTION_SENSORY,
    LAYER_SENSORY,
    LAYER_WORKING,
    LAYER_LONG_TERM,
    LAYER_CORTEX,
    LAYER_CONFIG,
)
from .storage.base import MemoryRecord, StorageBackend

logger = logging.getLogger(__name__)

# Collections that support layers
LAYERED_COLLECTIONS = [
    COLLECTION_PRIVATE,
    COLLECTION_VILLAGE,
    COLLECTION_BRIDGES,
    COLLECTION_SENSORY,
]


class HealthEngine:
    """
    Memory health management for Neo-Cortex.

    Handles:
    - Attention weight decay over time
    - Layer promotion based on access patterns
    - Stale memory detection
    - Duplicate detection and consolidation
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    # =========================================================================
    # Decay Calculations
    # =========================================================================

    def calculate_attention_weight(
        self,
        record: MemoryRecord,
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        Calculate current attention weight based on decay.

        Attention decays exponentially based on layer's half-life.
        Access count provides a boost.

        Formula:
            base_decay = 0.5 ^ (hours_since_access / half_life)
            access_boost = min(access_count * 0.1, 0.5)
            attention = base_decay + access_boost

        Args:
            record: The memory record
            current_time: Current time (defaults to now)

        Returns:
            Attention weight (0.0 to ~1.5)
        """
        now = current_time or datetime.now()
        layer = record.layer or LAYER_WORKING
        config = LAYER_CONFIG.get(layer, LAYER_CONFIG[LAYER_WORKING])

        # Cortex never decays
        if layer == LAYER_CORTEX:
            return 1.0

        # Get half-life
        half_life_hours = config.get("decay_half_life_hours")
        if not half_life_hours:
            return 1.0

        # Calculate time since last access
        last_accessed = record.last_accessed_at or record.created_at or now
        hours_since = (now - last_accessed).total_seconds() / 3600

        # Exponential decay
        base_decay = math.pow(0.5, hours_since / half_life_hours)

        # Access boost (capped)
        access_boost = min(record.access_count * 0.1, 0.5)

        # Combined weight
        attention = base_decay + access_boost

        # Enforce minimum for layer
        min_attention = config.get("min_attention", 0.1)
        return max(attention, min_attention)

    def update_attention_weights(
        self,
        collection: str,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Update attention weights for all records in a collection.

        Should be run periodically (e.g., hourly) to keep weights fresh.

        Args:
            collection: Collection to update
            batch_size: Process in batches

        Returns:
            Dict with update statistics
        """
        try:
            records = self.storage.list_all(collection)
            now = datetime.now()

            updated = 0
            below_threshold = 0
            updates = []

            for record in records:
                old_weight = record.attention_weight
                new_weight = self.calculate_attention_weight(record, now)

                if abs(new_weight - old_weight) > 0.01:  # Only update if changed
                    record.attention_weight = new_weight
                    updates.append(record)
                    updated += 1

                    # Check if below layer minimum
                    layer_config = LAYER_CONFIG.get(record.layer or LAYER_WORKING, {})
                    if new_weight < layer_config.get("min_attention", 0.1):
                        below_threshold += 1

                # Batch update
                if len(updates) >= batch_size:
                    self.storage.update(collection, updates)
                    updates = []

            # Final batch
            if updates:
                self.storage.update(collection, updates)

            return {
                "success": True,
                "collection": collection,
                "total_records": len(records),
                "updated": updated,
                "below_threshold": below_threshold,
            }

        except Exception as e:
            logger.error(f"update_attention_weights failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Layer Promotion
    # =========================================================================

    def check_promotion_eligibility(
        self,
        record: MemoryRecord,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Check if a record is eligible for promotion to the next layer.

        Args:
            record: The memory record
            current_time: Current time

        Returns:
            Tuple of (is_eligible, target_layer, reason)
        """
        now = current_time or datetime.now()
        current_layer = record.layer or LAYER_WORKING
        config = LAYER_CONFIG.get(current_layer, {})

        # Already at cortex - no promotion possible
        if current_layer == LAYER_CORTEX:
            return (False, None, "Already at cortex layer")

        # Determine target layer
        layer_order = [LAYER_SENSORY, LAYER_WORKING, LAYER_LONG_TERM, LAYER_CORTEX]
        try:
            current_idx = layer_order.index(current_layer)
            target_layer = layer_order[current_idx + 1]
        except (ValueError, IndexError):
            return (False, None, "Invalid layer state")

        # Check promotion threshold
        threshold = config.get("promotion_threshold")
        if threshold is None:
            # Layer requires special promotion (e.g., convergence for cortex)
            return (False, target_layer, "Requires special promotion criteria")

        # Check access count
        if record.access_count < threshold:
            return (False, target_layer, f"Access count {record.access_count} < threshold {threshold}")

        # Check minimum age (if applicable)
        min_age_hours = config.get("min_age_hours", 0)
        if min_age_hours > 0 and record.created_at:
            age_hours = (now - record.created_at).total_seconds() / 3600
            if age_hours < min_age_hours:
                return (False, target_layer, f"Age {age_hours:.1f}h < minimum {min_age_hours}h")

        return (True, target_layer, "Eligible for promotion")

    def promote_record(
        self,
        collection: str,
        record_id: str,
        target_layer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Promote a record to a higher layer.

        Args:
            collection: Collection containing the record
            record_id: ID of the record to promote
            target_layer: Target layer (auto-determined if None)

        Returns:
            Dict with promotion result
        """
        try:
            records = self.storage.get(collection, [record_id])
            if not records:
                return {"success": False, "error": "Record not found"}

            record = records[0]
            eligible, auto_target, reason = self.check_promotion_eligibility(record)

            if target_layer is None:
                target_layer = auto_target

            if not target_layer:
                return {"success": False, "error": reason}

            # Update record
            old_layer = record.layer
            record.layer = target_layer
            record.attention_weight = 1.0  # Reset attention on promotion

            self.storage.update(collection, [record])

            logger.info(f"Promoted {record_id}: {old_layer} -> {target_layer}")

            return {
                "success": True,
                "record_id": record_id,
                "old_layer": old_layer,
                "new_layer": target_layer,
                "was_eligible": eligible,
            }

        except Exception as e:
            logger.error(f"promote_record failed: {e}")
            return {"success": False, "error": str(e)}

    def run_promotions(
        self,
        collection: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Check all records and promote eligible ones.

        Args:
            collection: Collection to process
            dry_run: If True, report but don't actually promote

        Returns:
            Dict with promotion statistics
        """
        try:
            records = self.storage.list_all(collection)
            now = datetime.now()

            eligible = []
            promoted = []

            for record in records:
                is_eligible, target, reason = self.check_promotion_eligibility(record, now)
                if is_eligible and target:
                    eligible.append({
                        "id": record.id,
                        "current_layer": record.layer,
                        "target_layer": target,
                        "access_count": record.access_count,
                    })

                    if not dry_run:
                        result = self.promote_record(collection, record.id, target)
                        if result["success"]:
                            promoted.append(record.id)

            return {
                "success": True,
                "collection": collection,
                "total_records": len(records),
                "eligible_count": len(eligible),
                "promoted_count": len(promoted),
                "dry_run": dry_run,
                "eligible": eligible if dry_run else None,
                "promoted": promoted if not dry_run else None,
            }

        except Exception as e:
            logger.error(f"run_promotions failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Stale Detection
    # =========================================================================

    def get_stale_memories(
        self,
        collection: str,
        days_unused: int = 30,
        min_attention: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Find memories not accessed in X days.

        Args:
            collection: Collection to scan
            days_unused: Threshold in days
            min_attention: Only return items BELOW this attention (cleanup targets)
            limit: Maximum results

        Returns:
            Dict with stale memories
        """
        try:
            records = self.storage.list_all(collection)
            cutoff = datetime.now() - timedelta(days=days_unused)

            stale = []
            for record in records:
                last_accessed = record.last_accessed_at or record.created_at
                if not last_accessed:
                    continue

                if last_accessed < cutoff:
                    # Check attention filter
                    if min_attention is not None and record.attention_weight >= min_attention:
                        continue

                    days_since = (datetime.now() - last_accessed).days
                    stale.append({
                        "id": record.id,
                        "content_preview": record.content[:150] + "..." if len(record.content) > 150 else record.content,
                        "layer": record.layer,
                        "last_accessed": last_accessed.isoformat(),
                        "days_since_access": days_since,
                        "access_count": record.access_count,
                        "attention_weight": record.attention_weight,
                    })

                    if limit and len(stale) >= limit:
                        break

            # Sort by days since access (oldest first)
            stale.sort(key=lambda x: x["days_since_access"], reverse=True)

            return {
                "success": True,
                "collection": collection,
                "days_threshold": days_unused,
                "stale_count": len(stale),
                "stale_memories": stale,
            }

        except Exception as e:
            logger.error(f"get_stale_memories failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Duplicate Detection
    # =========================================================================

    def get_duplicate_candidates(
        self,
        collection: str,
        similarity_threshold: float = 0.95,
        sample_size: Optional[int] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Find potential duplicate memories using semantic similarity.

        Args:
            collection: Collection to scan
            similarity_threshold: Minimum similarity for duplicate (0.0-1.0)
            sample_size: Only check this many records (for performance)
            limit: Maximum duplicate pairs to return

        Returns:
            Dict with duplicate pairs
        """
        try:
            records = self.storage.list_all(collection, limit=sample_size)

            if len(records) < 2:
                return {
                    "success": True,
                    "collection": collection,
                    "duplicate_pairs": [],
                    "total_checked": len(records),
                }

            seen_pairs = set()
            duplicates = []

            for record in records:
                # Search for similar records
                similar = self.storage.search(
                    collection=collection,
                    query=record.content,
                    n_results=5,
                )

                for match in similar:
                    # Skip self-match
                    if match.id == record.id:
                        continue

                    # Check threshold
                    if (match.similarity or 0) < similarity_threshold:
                        continue

                    # Create canonical pair ID
                    pair_id = tuple(sorted([record.id, match.id]))
                    if pair_id in seen_pairs:
                        continue

                    seen_pairs.add(pair_id)
                    duplicates.append({
                        "id1": record.id,
                        "id2": match.id,
                        "similarity": match.similarity,
                        "content1_preview": record.content[:100] + "..." if len(record.content) > 100 else record.content,
                        "content2_preview": match.content[:100] + "..." if len(match.content) > 100 else match.content,
                    })

                    if len(duplicates) >= limit:
                        break

                if len(duplicates) >= limit:
                    break

            # Sort by similarity
            duplicates.sort(key=lambda x: x["similarity"], reverse=True)

            return {
                "success": True,
                "collection": collection,
                "threshold": similarity_threshold,
                "total_checked": len(records),
                "duplicate_count": len(duplicates),
                "duplicate_pairs": duplicates,
            }

        except Exception as e:
            logger.error(f"get_duplicate_candidates failed: {e}")
            return {"success": False, "error": str(e)}

    def consolidate_memories(
        self,
        collection: str,
        id1: str,
        id2: str,
        keep: str = "higher_access",
    ) -> Dict[str, Any]:
        """
        Merge two similar memories into one.

        The kept memory gets:
        - Combined access_count
        - Most recent last_accessed_at
        - Higher attention_weight

        Args:
            collection: Collection name
            id1: First memory ID
            id2: Second memory ID
            keep: Strategy - "higher_access", "higher_attention", "id1", "id2"

        Returns:
            Dict with consolidation result
        """
        try:
            records = self.storage.get(collection, [id1, id2])

            if len(records) != 2:
                return {"success": False, "error": f"Could not find both records: {id1}, {id2}"}

            rec1, rec2 = records[0], records[1]
            if rec1.id == id2:
                rec1, rec2 = rec2, rec1  # Ensure order matches input

            # Determine which to keep
            if keep == "higher_access":
                keep_rec = rec1 if rec1.access_count >= rec2.access_count else rec2
                discard_rec = rec2 if keep_rec == rec1 else rec1
            elif keep == "higher_attention":
                keep_rec = rec1 if rec1.attention_weight >= rec2.attention_weight else rec2
                discard_rec = rec2 if keep_rec == rec1 else rec1
            elif keep == "id1":
                keep_rec, discard_rec = rec1, rec2
            elif keep == "id2":
                keep_rec, discard_rec = rec2, rec1
            else:
                return {"success": False, "error": f"Invalid keep strategy: {keep}"}

            # Merge metadata
            keep_rec.access_count += discard_rec.access_count

            if discard_rec.last_accessed_at and keep_rec.last_accessed_at:
                if discard_rec.last_accessed_at > keep_rec.last_accessed_at:
                    keep_rec.last_accessed_at = discard_rec.last_accessed_at

            keep_rec.attention_weight = max(
                keep_rec.attention_weight,
                discard_rec.attention_weight
            )

            # Merge tags
            all_tags = set(keep_rec.tags) | set(discard_rec.tags)
            keep_rec.tags = list(all_tags)

            # Update kept record
            self.storage.update(collection, [keep_rec])

            # Delete discarded record
            self.storage.delete(collection, [discard_rec.id])

            logger.info(f"Consolidated {id1} + {id2} -> kept {keep_rec.id}")

            return {
                "success": True,
                "kept_id": keep_rec.id,
                "discarded_id": discard_rec.id,
                "new_access_count": keep_rec.access_count,
                "new_attention_weight": keep_rec.attention_weight,
            }

        except Exception as e:
            logger.error(f"consolidate_memories failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Health Report
    # =========================================================================

    def health_report(
        self,
        collections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive health report for the memory system.

        Args:
            collections: Which collections to analyze (defaults to layered collections)

        Returns:
            Dict with health metrics and recommendations
        """
        try:
            target_collections = collections or LAYERED_COLLECTIONS
            now = datetime.now()

            report = {
                "success": True,
                "generated_at": now.isoformat(),
                "collections": {},
                "summary": {
                    "total_memories": 0,
                    "total_stale": 0,
                    "total_duplicates": 0,
                    "promotion_candidates": 0,
                },
                "recommendations": [],
            }

            for coll in target_collections:
                try:
                    records = self.storage.list_all(coll)
                    coll_report = {
                        "total": len(records),
                        "by_layer": defaultdict(int),
                        "avg_attention": 0,
                        "stale_count": 0,
                        "promotion_candidates": 0,
                    }

                    total_attention = 0
                    stale_cutoff = now - timedelta(days=30)

                    for record in records:
                        # Layer distribution
                        layer = record.layer or LAYER_WORKING
                        coll_report["by_layer"][layer] += 1

                        # Attention stats
                        total_attention += record.attention_weight

                        # Stale check
                        last_access = record.last_accessed_at or record.created_at
                        if last_access and last_access < stale_cutoff:
                            coll_report["stale_count"] += 1

                        # Promotion check
                        eligible, _, _ = self.check_promotion_eligibility(record, now)
                        if eligible:
                            coll_report["promotion_candidates"] += 1

                    if records:
                        coll_report["avg_attention"] = round(total_attention / len(records), 3)

                    coll_report["by_layer"] = dict(coll_report["by_layer"])
                    report["collections"][coll] = coll_report

                    # Update summary
                    report["summary"]["total_memories"] += len(records)
                    report["summary"]["total_stale"] += coll_report["stale_count"]
                    report["summary"]["promotion_candidates"] += coll_report["promotion_candidates"]

                except Exception as e:
                    report["collections"][coll] = {"error": str(e)}

            # Generate recommendations
            if report["summary"]["total_stale"] > 10:
                report["recommendations"].append(
                    f"Consider reviewing {report['summary']['total_stale']} stale memories (unused for 30+ days)"
                )

            if report["summary"]["promotion_candidates"] > 5:
                report["recommendations"].append(
                    f"{report['summary']['promotion_candidates']} memories are eligible for layer promotion"
                )

            return report

        except Exception as e:
            logger.error(f"health_report failed: {e}")
            return {"success": False, "error": str(e)}


# =============================================================================
# Tool Schemas
# =============================================================================

HEALTH_TOOL_SCHEMAS = {
    "memory_health_report": {
        "name": "memory_health_report",
        "description": (
            "Generate a comprehensive health report for the memory system. "
            "Shows layer distribution, stale memories, promotion candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Collections to analyze (omit for all)"
                }
            },
            "required": []
        }
    },
    "memory_get_stale": {
        "name": "memory_get_stale",
        "description": (
            "Find memories not accessed in X days. "
            "Use this to identify knowledge that may need review or cleanup."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection to scan"
                },
                "days_unused": {
                    "type": "integer",
                    "description": "Threshold in days (default: 30)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results"
                }
            },
            "required": ["collection"]
        }
    },
    "memory_get_duplicates": {
        "name": "memory_get_duplicates",
        "description": (
            "Find potential duplicate memories with high similarity. "
            "Returns pairs of similar memories that could be consolidated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection to scan"
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Similarity cutoff 0.0-1.0 (default: 0.95)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum duplicate pairs"
                }
            },
            "required": ["collection"]
        }
    },
    "memory_consolidate": {
        "name": "memory_consolidate",
        "description": (
            "Merge two similar memories into one. "
            "The kept memory gets combined access counts and metadata."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name"
                },
                "id1": {
                    "type": "string",
                    "description": "First memory ID"
                },
                "id2": {
                    "type": "string",
                    "description": "Second memory ID"
                },
                "keep": {
                    "type": "string",
                    "enum": ["higher_access", "higher_attention", "id1", "id2"],
                    "description": "Which memory to keep (default: higher_access)"
                }
            },
            "required": ["collection", "id1", "id2"]
        }
    },
    "memory_run_promotions": {
        "name": "memory_run_promotions",
        "description": (
            "Check all memories and promote eligible ones to higher layers. "
            "Use dry_run=true to preview without making changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection to process"
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview only, don't actually promote"
                }
            },
            "required": ["collection"]
        }
    },
}
