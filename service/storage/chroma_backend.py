"""
ChromaDB Storage Backend

Local vector storage using ChromaDB with sentence-transformers embeddings.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import StorageBackend, MemoryRecord
from ..config import CHROMA_PATH, EMBEDDING_DIM, ALL_COLLECTIONS
from ..embeddings import get_embedding_function

logger = logging.getLogger(__name__)


class ChromaBackend(StorageBackend):
    """
    ChromaDB implementation of StorageBackend.

    Uses sentence-transformers for embeddings (384 dimensions).
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self._persist_path = persist_path or CHROMA_PATH
        self._client = None
        self._embedding_fn = None
        self._collections: Dict[str, Any] = {}

    @property
    def backend_name(self) -> str:
        return "chroma"

    @property
    def embedding_dimension(self) -> int:
        return EMBEDDING_DIM

    def _get_client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self._persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._persist_path),
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"ChromaDB initialized at {self._persist_path}")
        return self._client

    def _get_embedding_fn(self):
        """Lazy-load embedding function."""
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

    def initialize(self) -> None:
        """Initialize all collections."""
        for coll_name in ALL_COLLECTIONS:
            self._get_collection(coll_name)
        logger.info(f"Initialized {len(ALL_COLLECTIONS)} collections")

    def _record_to_metadata(self, record: MemoryRecord) -> Dict[str, Any]:
        """Convert MemoryRecord to ChromaDB metadata dict."""
        return {
            "agent_id": record.agent_id,
            "visibility": record.visibility,
            "layer": record.layer,
            "message_type": record.message_type,
            "responding_to": json.dumps(record.responding_to),
            "conversation_thread": record.conversation_thread or "",
            "related_agents": json.dumps(record.related_agents),
            "tags": json.dumps(record.tags),
            "created_at": record.created_at.isoformat() if record.created_at else datetime.now().isoformat(),
            "access_count": record.access_count,
            "last_accessed_ts": record.last_accessed_at.timestamp() if record.last_accessed_at else datetime.now().timestamp(),
            "attention_weight": record.attention_weight,
        }

    def _metadata_to_record(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        distance: Optional[float] = None,
        collection: Optional[str] = None,
    ) -> MemoryRecord:
        """Convert ChromaDB result to MemoryRecord."""
        # Parse JSON fields
        responding_to = []
        related_agents = []
        tags = []
        try:
            responding_to = json.loads(metadata.get("responding_to", "[]"))
        except:
            pass
        try:
            related_agents = json.loads(metadata.get("related_agents", "[]"))
        except:
            pass
        try:
            tags = json.loads(metadata.get("tags", "[]"))
        except:
            pass

        # Parse timestamps
        created_at = None
        if metadata.get("created_at"):
            try:
                created_at = datetime.fromisoformat(metadata["created_at"])
            except:
                pass

        last_accessed_at = None
        if metadata.get("last_accessed_ts"):
            try:
                last_accessed_at = datetime.fromtimestamp(metadata["last_accessed_ts"])
            except:
                pass

        return MemoryRecord(
            id=doc_id,
            content=content,
            agent_id=metadata.get("agent_id", "CLAUDE"),
            visibility=metadata.get("visibility", "private"),
            layer=metadata.get("layer", "working"),
            message_type=metadata.get("message_type", "observation"),
            responding_to=responding_to,
            conversation_thread=metadata.get("conversation_thread") or None,
            related_agents=related_agents,
            tags=tags,
            created_at=created_at,
            access_count=metadata.get("access_count", 0),
            last_accessed_at=last_accessed_at,
            attention_weight=metadata.get("attention_weight", 1.0),
            similarity=round(1 - distance, 4) if distance is not None else None,
            collection=collection,
        )

    def add(
        self,
        collection: str,
        records: List[MemoryRecord],
    ) -> List[str]:
        """Add records to collection."""
        if not records:
            return []

        coll = self._get_collection(collection)

        ids = []
        documents = []
        metadatas = []

        for record in records:
            # Generate ID if not provided
            if not record.id:
                record.id = f"cortex_{record.agent_id}_{datetime.now().timestamp()}"

            # Set created_at if not set
            if not record.created_at:
                record.created_at = datetime.now()

            ids.append(record.id)
            documents.append(record.content)
            metadatas.append(self._record_to_metadata(record))

        coll.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug(f"Added {len(records)} records to {collection}")
        return ids

    def search(
        self,
        collection: str,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """Semantic search in collection."""
        coll = self._get_collection(collection)

        results = coll.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        records = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0

                record = self._metadata_to_record(
                    doc_id=doc_id,
                    content=content,
                    metadata=metadata,
                    distance=distance,
                    collection=collection,
                )
                records.append(record)

        return records

    def get(
        self,
        collection: str,
        ids: List[str],
    ) -> List[MemoryRecord]:
        """Get records by ID."""
        if not ids:
            return []

        coll = self._get_collection(collection)
        results = coll.get(ids=ids, include=["documents", "metadatas"])

        records = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                content = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}

                record = self._metadata_to_record(
                    doc_id=doc_id,
                    content=content,
                    metadata=metadata,
                    collection=collection,
                )
                records.append(record)

        return records

    def update(
        self,
        collection: str,
        records: List[MemoryRecord],
    ) -> bool:
        """Update existing records."""
        if not records:
            return True

        coll = self._get_collection(collection)

        try:
            ids = [r.id for r in records]
            documents = [r.content for r in records]
            metadatas = [self._record_to_metadata(r) for r in records]

            coll.update(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    def delete(
        self,
        collection: str,
        ids: List[str],
    ) -> bool:
        """Delete records by ID."""
        if not ids:
            return True

        coll = self._get_collection(collection)
        try:
            coll.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def count(self, collection: str) -> int:
        """Count records in collection."""
        coll = self._get_collection(collection)
        return coll.count()

    def list_all(
        self,
        collection: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """List all records (for export)."""
        coll = self._get_collection(collection)

        # ChromaDB doesn't have great pagination, get all and slice
        results = coll.get(include=["documents", "metadatas"])

        records = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                content = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}

                record = self._metadata_to_record(
                    doc_id=doc_id,
                    content=content,
                    metadata=metadata,
                    collection=collection,
                )
                records.append(record)

        # Apply offset and limit
        if offset:
            records = records[offset:]
        if limit:
            records = records[:limit]

        return records


# =============================================================================
# Export/Import Functions
# =============================================================================

def export_to_memory_core(
    backend: ChromaBackend,
    agent_id: Optional[str] = None,
    collections: Optional[List[str]] = None,
) -> "MemoryCore":
    """
    Export memories to portable MemoryCore format.

    Args:
        backend: ChromaBackend instance
        agent_id: Filter by agent (None for all)
        collections: Which collections to export (None for all)

    Returns:
        MemoryCore object ready for JSON serialization
    """
    from .base import MemoryCore

    export_collections = collections or ALL_COLLECTIONS
    core = MemoryCore(
        agent_id=agent_id or "ALL",
        exported_at=datetime.now(),
        metadata={
            "source_backend": backend.backend_name,
            "embedding_dimension": backend.embedding_dimension,
            "total_memories": 0,
        }
    )

    for coll_name in export_collections:
        try:
            records = backend.list_all(coll_name)

            # Filter by agent if specified
            if agent_id:
                records = [r for r in records if r.agent_id == agent_id]

            core.collections[coll_name] = [r.to_dict() for r in records]
            core.metadata["total_memories"] += len(records)

        except Exception as e:
            logger.warning(f"Failed to export {coll_name}: {e}")

    return core


def import_from_memory_core(
    backend: ChromaBackend,
    core: "MemoryCore",
    re_embed: bool = True,
) -> Dict[str, int]:
    """
    Import memories from MemoryCore format.

    Args:
        backend: ChromaBackend instance
        core: MemoryCore object to import
        re_embed: Whether to regenerate embeddings (recommended for cross-backend)

    Returns:
        Dict with import stats per collection
    """
    stats = {}

    for coll_name, records_data in core.collections.items():
        try:
            records = [MemoryRecord.from_dict(r) for r in records_data]

            # Clear embeddings if re-embedding (backend will regenerate)
            if re_embed:
                for r in records:
                    r.embedding = None

            ids = backend.add(coll_name, records)
            stats[coll_name] = len(ids)

        except Exception as e:
            logger.error(f"Failed to import {coll_name}: {e}")
            stats[coll_name] = 0

    return stats
