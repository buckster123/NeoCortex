"""
Neo-Cortex Engine: Unified Memory Coordinator

The main interface for all memory operations across:
- Knowledge (curated docs)
- Village (multi-agent shared memory)
- Forward Crumbs (session continuity)
- Memory Health (access tracking, decay)

Supports both local (ChromaDB) and cloud (pgvector) backends.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import math

from .config import (
    CHROMA_PATH,
    COLLECTION_KNOWLEDGE,
    COLLECTION_PRIVATE,
    COLLECTION_VILLAGE,
    COLLECTION_BRIDGES,
    COLLECTION_CRUMBS,
    COLLECTION_SENSORY,
    ALL_COLLECTIONS,
    LAYER_SENSORY,
    LAYER_WORKING,
    LAYER_LONG_TERM,
    LAYER_CORTEX,
    LAYER_CONFIG,
    MESSAGE_TYPES,
    AGENT_PROFILES,
    DEFAULT_BACKEND,
    DEFAULT_SEARCH_RESULTS,
    DEFAULT_SIMILARITY_THRESHOLD,
    CONVERGENCE_HARMONY,
    CONVERGENCE_CONSENSUS,
)
from .embeddings import get_embedding_function

logger = logging.getLogger(__name__)


class CortexEngine:
    """
    Unified memory engine for Neo-Cortex.

    Coordinates all memory operations across collections and layers.
    """

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        chroma_path: Optional[Path] = None,
        db_url: Optional[str] = None,  # For pgvector
    ):
        self.backend = backend
        self._client = None
        self._collections: Dict[str, Any] = {}
        self._embedding_fn = None
        self._current_agent_id = "CLAUDE"

        if backend == "chroma":
            self._chroma_path = chroma_path or CHROMA_PATH
            self._chroma_path.mkdir(parents=True, exist_ok=True)
        elif backend == "pgvector":
            self._db_url = db_url
            # TODO: Initialize pgvector connection
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # =========================================================================
    # Initialization
    # =========================================================================

    def _get_client(self):
        """Get or create the ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(self._chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"ChromaDB client initialized at {self._chroma_path}")
        return self._client

    def _get_embedding_fn(self):
        """Get or create the embedding function."""
        if self._embedding_fn is None:
            self._embedding_fn = get_embedding_function("auto")
        return self._embedding_fn

    def _get_collection(self, name: str):
        """Get or create a collection."""
        if name not in self._collections:
            client = self._get_client()
            embedding_fn = self._get_embedding_fn()
            self._collections[name] = client.get_or_create_collection(
                name=name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[name]

    # =========================================================================
    # Agent Management
    # =========================================================================

    def set_current_agent(self, agent_id: str):
        """Set the current active agent for operations."""
        self._current_agent_id = agent_id.upper()
        logger.info(f"Current agent set to: {self._current_agent_id}")

    def get_current_agent(self) -> str:
        """Get the current active agent ID."""
        return self._current_agent_id

    def get_agent_profile(self, agent_id: str) -> Optional[Dict]:
        """Get profile for an agent."""
        return AGENT_PROFILES.get(agent_id.upper())

    def list_agents(self) -> List[Dict]:
        """List all registered agents."""
        return [
            {"id": agent_id, **profile}
            for agent_id, profile in AGENT_PROFILES.items()
        ]

    # =========================================================================
    # Core Memory Operations
    # =========================================================================

    def remember(
        self,
        content: str,
        layer: str = LAYER_SENSORY,
        visibility: str = "private",
        agent_id: Optional[str] = None,
        message_type: str = "observation",
        tags: Optional[List[str]] = None,
        responding_to: Optional[List[str]] = None,
        conversation_thread: Optional[str] = None,
        related_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Store a memory in the cortex.

        Args:
            content: The memory content
            layer: Memory layer (sensory/working/long_term/cortex)
            visibility: Visibility realm (private/village/bridge)
            agent_id: Agent storing the memory (uses current if None)
            message_type: Type of memory (fact/dialogue/observation/etc.)
            tags: Optional tags
            responding_to: IDs of memories this responds to
            conversation_thread: Thread identifier
            related_agents: Other agents involved

        Returns:
            Dict with success status and memory ID
        """
        try:
            # Determine collection based on visibility
            if visibility == "private":
                collection_name = COLLECTION_PRIVATE
            elif visibility == "bridge":
                collection_name = COLLECTION_BRIDGES
            else:
                collection_name = COLLECTION_VILLAGE

            collection = self._get_collection(collection_name)

            # Get agent info
            agent = agent_id or self._current_agent_id
            profile = self.get_agent_profile(agent)

            # Generate memory ID
            timestamp = datetime.now()
            memory_id = f"cortex_{agent}_{timestamp.timestamp()}"

            # Build metadata
            metadata = {
                "agent_id": agent,
                "agent_display": profile["display_name"] if profile else agent,
                "visibility": visibility,
                "message_type": message_type,
                "layer": layer,
                "responding_to": json.dumps(responding_to or []),
                "conversation_thread": conversation_thread or "",
                "related_agents": json.dumps(related_agents or []),
                "tags": json.dumps(tags or []),
                "posted_at": timestamp.isoformat(),
                "access_count": 0,
                "last_accessed_ts": timestamp.timestamp(),
                "attention_weight": 1.0,  # Starts at full attention
            }

            # Add to collection
            collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[memory_id]
            )

            logger.info(f"Memory stored: {agent} -> {visibility}/{layer} ({message_type})")

            return {
                "success": True,
                "id": memory_id,
                "agent_id": agent,
                "visibility": visibility,
                "layer": layer,
                "collection": collection_name,
            }

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return {"success": False, "error": str(e)}

    def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        visibility: str = "all",
        agent_filter: Optional[str] = None,
        min_attention: float = 0.0,
        n_results: int = DEFAULT_SEARCH_RESULTS,
        track_access: bool = True,
    ) -> Dict[str, Any]:
        """
        Search across memory collections.

        Args:
            query: Search query
            collections: Which collections to search (defaults to all)
            layers: Filter by memory layers
            visibility: "private", "village", "all"
            agent_filter: Filter by agent ID
            min_attention: Minimum attention weight threshold
            n_results: Maximum results
            track_access: Whether to track this access

        Returns:
            Dict with search results
        """
        try:
            # Determine which collections to search
            if collections:
                search_collections = collections
            elif visibility == "private":
                search_collections = [COLLECTION_PRIVATE]
            elif visibility == "village":
                search_collections = [COLLECTION_VILLAGE, COLLECTION_BRIDGES]
            else:
                search_collections = [
                    COLLECTION_PRIVATE,
                    COLLECTION_VILLAGE,
                    COLLECTION_BRIDGES,
                    COLLECTION_KNOWLEDGE,
                ]

            all_results = []

            for coll_name in search_collections:
                try:
                    collection = self._get_collection(coll_name)

                    # Build filter
                    where_filter = {}
                    if agent_filter:
                        where_filter["agent_id"] = agent_filter.upper()
                    if min_attention > 0:
                        where_filter["attention_weight"] = {"$gte": min_attention}

                    # Query
                    results = collection.query(
                        query_texts=[query],
                        n_results=n_results,
                        where=where_filter if where_filter else None,
                        include=["documents", "metadatas", "distances"]
                    )

                    # Process results
                    if results["ids"] and results["ids"][0]:
                        for i, doc_id in enumerate(results["ids"][0]):
                            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                            distance = results["distances"][0][i] if results["distances"] else 1.0

                            # Filter by layer if specified
                            if layers and metadata.get("layer") not in layers:
                                continue

                            result = {
                                "id": doc_id,
                                "content": results["documents"][0][i],
                                "similarity": round(1 - distance, 4),
                                "collection": coll_name,
                                **self._parse_metadata(metadata)
                            }
                            all_results.append(result)

                except Exception as e:
                    logger.warning(f"Error searching {coll_name}: {e}")

            # Sort by similarity
            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            all_results = all_results[:n_results]

            # Track access
            if track_access and all_results:
                self._track_access([r["id"] for r in all_results], search_collections)

            return {
                "success": True,
                "query": query,
                "count": len(all_results),
                "results": all_results,
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    def _parse_metadata(self, metadata: Dict) -> Dict:
        """Parse JSON fields in metadata."""
        result = dict(metadata)
        for field in ["responding_to", "related_agents", "tags"]:
            if field in result and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except:
                    result[field] = []
        return result

    def _track_access(self, memory_ids: List[str], collections: List[str]):
        """Track access to memories (non-blocking)."""
        try:
            current_time = datetime.now().timestamp()

            for coll_name in collections:
                try:
                    collection = self._get_collection(coll_name)

                    # Get current metadata
                    for memory_id in memory_ids:
                        try:
                            results = collection.get(ids=[memory_id], include=["metadatas"])
                            if results["ids"] and results["metadatas"]:
                                metadata = results["metadatas"][0]
                                metadata["access_count"] = metadata.get("access_count", 0) + 1
                                metadata["last_accessed_ts"] = current_time
                                # Recalculate attention (simple: access count boost)
                                metadata["attention_weight"] = min(
                                    1.0 + (metadata["access_count"] * 0.1),
                                    2.0
                                )
                                collection.update(ids=[memory_id], metadatas=[metadata])
                        except:
                            pass  # Non-blocking
                except:
                    pass  # Non-blocking

        except Exception as e:
            logger.debug(f"Access tracking failed (non-blocking): {e}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Get statistics about the cortex."""
        stats = {
            "current_agent": self._current_agent_id,
            "registered_agents": len(AGENT_PROFILES),
            "backend": self.backend,
            "collections": {},
            "total_memories": 0,
        }

        for coll_name in ALL_COLLECTIONS:
            try:
                collection = self._get_collection(coll_name)
                count = collection.count()
                stats["collections"][coll_name] = count
                stats["total_memories"] += count
            except:
                stats["collections"][coll_name] = 0

        return stats

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def quick_save(self, content: str, tags: Optional[List[str]] = None) -> Dict:
        """Quick save to working memory."""
        return self.remember(
            content=content,
            layer=LAYER_WORKING,
            visibility="private",
            message_type="observation",
            tags=tags,
        )

    def village_post(self, content: str, message_type: str = "dialogue", **kwargs) -> Dict:
        """Post to the village square."""
        return self.remember(
            content=content,
            layer=LAYER_WORKING,
            visibility="village",
            message_type=message_type,
            **kwargs
        )

    def bridge_to(
        self,
        content: str,
        target_agents: List[str],
        **kwargs
    ) -> Dict:
        """Create a bridge message to specific agents."""
        return self.remember(
            content=content,
            layer=LAYER_WORKING,
            visibility="bridge",
            related_agents=target_agents,
            **kwargs
        )


# =============================================================================
# Module-level convenience functions
# =============================================================================

_engine: Optional[CortexEngine] = None


def get_engine() -> CortexEngine:
    """Get or create the global cortex engine."""
    global _engine
    if _engine is None:
        _engine = CortexEngine()
    return _engine


def remember(content: str, **kwargs) -> Dict:
    """Store a memory."""
    return get_engine().remember(content, **kwargs)


def search(query: str, **kwargs) -> Dict:
    """Search memories."""
    return get_engine().search(query, **kwargs)


def stats() -> Dict:
    """Get cortex statistics."""
    return get_engine().stats()
