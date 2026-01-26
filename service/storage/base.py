"""
Storage Backend Protocol and Base Types

Defines the interface that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Union
import json


@dataclass
class MemoryRecord:
    """
    Universal memory record format.

    Used for:
    - Internal operations
    - Export/Import between backends
    - API responses
    """
    id: str
    content: str
    embedding: Optional[List[float]] = None

    # Core metadata
    agent_id: str = "CLAUDE"
    visibility: str = "private"  # private/village/bridge
    layer: str = "working"       # sensory/working/long_term/cortex
    message_type: str = "observation"

    # Relationships
    responding_to: List[str] = field(default_factory=list)
    conversation_thread: Optional[str] = None
    related_agents: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: Optional[datetime] = None

    # Attention/Health
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    attention_weight: float = 1.0

    # Search result fields (populated on query)
    similarity: Optional[float] = None
    collection: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "agent_id": self.agent_id,
            "visibility": self.visibility,
            "layer": self.layer,
            "message_type": self.message_type,
            "responding_to": self.responding_to,
            "conversation_thread": self.conversation_thread,
            "related_agents": self.related_agents,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "attention_weight": self.attention_weight,
            "similarity": self.similarity,
            "collection": self.collection,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        """Create from dictionary."""
        # Parse datetime fields
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except:
                pass

        last_accessed_at = None
        if data.get("last_accessed_at"):
            try:
                last_accessed_at = datetime.fromisoformat(data["last_accessed_at"])
            except:
                pass

        return cls(
            id=data["id"],
            content=data["content"],
            embedding=data.get("embedding"),
            agent_id=data.get("agent_id", "CLAUDE"),
            visibility=data.get("visibility", "private"),
            layer=data.get("layer", "working"),
            message_type=data.get("message_type", "observation"),
            responding_to=data.get("responding_to", []),
            conversation_thread=data.get("conversation_thread"),
            related_agents=data.get("related_agents", []),
            tags=data.get("tags", []),
            created_at=created_at,
            access_count=data.get("access_count", 0),
            last_accessed_at=last_accessed_at,
            attention_weight=data.get("attention_weight", 1.0),
            similarity=data.get("similarity"),
            collection=data.get("collection"),
        )


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    All backends (ChromaDB, pgvector) must implement this interface.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return backend identifier (e.g., 'chroma', 'pgvector')."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding dimension used by this backend."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (create collections/tables if needed)."""
        pass

    @abstractmethod
    def add(
        self,
        collection: str,
        records: List[MemoryRecord],
    ) -> List[str]:
        """
        Add records to a collection.

        Args:
            collection: Collection name
            records: List of MemoryRecord objects (embeddings will be generated if missing)

        Returns:
            List of record IDs
        """
        pass

    @abstractmethod
    def search(
        self,
        collection: str,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """
        Semantic search in a collection.

        Args:
            collection: Collection name
            query: Search query text
            n_results: Maximum results
            where: Filter conditions

        Returns:
            List of MemoryRecord objects with similarity scores
        """
        pass

    @abstractmethod
    def get(
        self,
        collection: str,
        ids: List[str],
    ) -> List[MemoryRecord]:
        """
        Get records by ID.

        Args:
            collection: Collection name
            ids: List of record IDs

        Returns:
            List of MemoryRecord objects
        """
        pass

    @abstractmethod
    def update(
        self,
        collection: str,
        records: List[MemoryRecord],
    ) -> bool:
        """
        Update existing records.

        Args:
            collection: Collection name
            records: Records with updated fields (must have id)

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def delete(
        self,
        collection: str,
        ids: List[str],
    ) -> bool:
        """
        Delete records by ID.

        Args:
            collection: Collection name
            ids: List of record IDs

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def count(self, collection: str) -> int:
        """
        Count records in a collection.

        Args:
            collection: Collection name

        Returns:
            Record count
        """
        pass

    @abstractmethod
    def list_all(
        self,
        collection: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """
        List all records in a collection (for export).

        Args:
            collection: Collection name
            limit: Maximum records (None for all)
            offset: Skip first N records

        Returns:
            List of MemoryRecord objects
        """
        pass


# =============================================================================
# Export/Import Format
# =============================================================================

@dataclass
class MemoryCore:
    """
    Portable memory export format.

    Used for transferring memories between backends.
    """
    format_version: str = "1.0"
    agent_id: str = "CLAUDE"
    exported_at: Optional[datetime] = None
    collections: Dict[str, List[Dict]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "format_version": self.format_version,
            "agent_id": self.agent_id,
            "exported_at": self.exported_at.isoformat() if self.exported_at else datetime.now().isoformat(),
            "collections": self.collections,
            "metadata": self.metadata,
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryCore":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        exported_at = None
        if data.get("exported_at"):
            try:
                exported_at = datetime.fromisoformat(data["exported_at"])
            except:
                pass

        return cls(
            format_version=data.get("format_version", "1.0"),
            agent_id=data.get("agent_id", "CLAUDE"),
            exported_at=exported_at,
            collections=data.get("collections", {}),
            metadata=data.get("metadata", {}),
        )
