"""
PostgreSQL + pgvector Storage Backend

Cloud vector storage using PostgreSQL with pgvector extension.

This implementation integrates with ApexAurum Cloud's async SQLAlchemy setup
and OpenAI embedding service.

Usage (in cloud backend):
    from neo_cortex.service.storage.pgvector_backend import PgVectorBackend

    backend = PgVectorBackend(
        db_url=settings.database_url,
        embedding_api_key=settings.openai_api_key,
    )
    await backend.initialize()

    # Now use same interface as ChromaDB
    ids = await backend.add("cortex_village", records)
    results = await backend.search("cortex_village", "query text")
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from .base import StorageBackend, MemoryRecord

logger = logging.getLogger(__name__)

# Cloud uses OpenAI embeddings (1536 dims)
PGVECTOR_EMBEDDING_DIM = 1536


class PgVectorBackend(StorageBackend):
    """
    PostgreSQL + pgvector implementation of StorageBackend.

    Uses OpenAI text-embedding-3-small for embeddings (1536 dimensions).
    Designed for async operation with SQLAlchemy AsyncSession.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        user_id: Optional[UUID] = None,
    ):
        """
        Initialize pgvector backend.

        Args:
            db_url: PostgreSQL connection URL
            embedding_api_key: OpenAI API key for embeddings
            embedding_model: Embedding model to use
            user_id: Default user ID for operations (can be overridden per-call)
        """
        self._db_url = db_url
        self._embedding_api_key = embedding_api_key
        self._embedding_model = embedding_model
        self._user_id = user_id
        self._engine = None
        self._session_factory = None
        self._initialized = False

    @property
    def backend_name(self) -> str:
        return "pgvector"

    @property
    def embedding_dimension(self) -> int:
        return PGVECTOR_EMBEDDING_DIM

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self) -> None:
        """
        Initialize database connection.

        Creates async SQLAlchemy engine and session factory.
        """
        if self._initialized:
            return

        try:
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker

            # Convert standard postgres URL to async
            db_url = self._db_url
            if db_url and db_url.startswith("postgresql://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

            self._engine = create_async_engine(db_url, echo=False)
            self._session_factory = sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Test connection
            async with self._session_factory() as session:
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                result.fetchone()

            self._initialized = True
            logger.info("PgVectorBackend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize pgvector backend: {e}")
            raise

    def initialize(self) -> None:
        """Sync wrapper for initialize - runs the async version."""
        asyncio.get_event_loop().run_until_complete(self._async_initialize())

    async def _async_initialize(self) -> None:
        """Async initialization."""
        await self.initialize()

    async def _get_session(self):
        """Get a database session."""
        if not self._initialized:
            await self.initialize()
        return self._session_factory()

    # =========================================================================
    # Embedding Generation
    # =========================================================================

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API."""
        if not self._embedding_api_key:
            logger.warning("No embedding API key configured")
            return None

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._embedding_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._embedding_model,
                        "input": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        if not self._embedding_api_key:
            return [None] * len(texts)

        try:
            import httpx

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._embedding_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._embedding_model,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                return [e["embedding"] for e in embeddings]

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [None] * len(texts)

    # =========================================================================
    # Collection Mapping
    # =========================================================================

    def _collection_to_filters(self, collection: str) -> Dict[str, Any]:
        """
        Map collection names to visibility/layer filters.

        Collection names:
        - cortex_knowledge -> visibility='knowledge', layer='cortex'
        - cortex_private   -> visibility='private'
        - cortex_village   -> visibility='village'
        - cortex_bridges   -> visibility='bridge'
        - cortex_crumbs    -> visibility='private', message_type='forward_crumb'
        - cortex_sensory   -> visibility='private', layer='sensory'
        """
        mapping = {
            "cortex_knowledge": {"visibility": "knowledge", "layer": "cortex"},
            "cortex_private": {"visibility": "private"},
            "cortex_village": {"visibility": "village"},
            "cortex_bridges": {"visibility": "bridge"},
            "cortex_crumbs": {"visibility": "private", "message_type": "forward_crumb"},
            "cortex_sensory": {"visibility": "private", "layer": "sensory"},
        }
        return mapping.get(collection, {"collection": collection})

    def _record_to_db_values(self, record: MemoryRecord, user_id: UUID) -> Dict[str, Any]:
        """Convert MemoryRecord to database column values."""
        return {
            "id": UUID(record.id) if isinstance(record.id, str) else record.id or uuid4(),
            "user_id": user_id,
            "collection": record.collection or "default",
            "content": record.content,
            "metadata": {},  # Original metadata field (keep for backwards compat)
            "layer": record.layer,
            "visibility": record.visibility,
            "agent_id": record.agent_id,
            "message_type": record.message_type,
            "attention_weight": record.attention_weight,
            "access_count": record.access_count,
            "last_accessed_at": record.last_accessed_at,
            "responding_to": record.responding_to or [],
            "conversation_thread": record.conversation_thread,
            "related_agents": record.related_agents or [],
            "tags": record.tags or [],
            "created_at": record.created_at or datetime.utcnow(),
        }

    def _db_row_to_record(
        self,
        row: Any,
        similarity: Optional[float] = None,
        collection: Optional[str] = None,
    ) -> MemoryRecord:
        """Convert database row to MemoryRecord."""
        return MemoryRecord(
            id=str(row.id),
            content=row.content,
            agent_id=row.agent_id or "CLAUDE",
            visibility=row.visibility or "private",
            layer=row.layer or "working",
            message_type=row.message_type or "observation",
            responding_to=row.responding_to or [],
            conversation_thread=row.conversation_thread,
            related_agents=row.related_agents or [],
            tags=row.tags or [],
            created_at=row.created_at,
            access_count=row.access_count or 0,
            last_accessed_at=row.last_accessed_at,
            attention_weight=row.attention_weight or 1.0,
            similarity=round(similarity, 4) if similarity else None,
            collection=collection or row.collection,
        )

    # =========================================================================
    # Core Operations (Async)
    # =========================================================================

    async def add_async(
        self,
        collection: str,
        records: List[MemoryRecord],
        user_id: Optional[UUID] = None,
    ) -> List[str]:
        """
        Add records to PostgreSQL (async).

        Args:
            collection: Collection name
            records: List of MemoryRecord objects
            user_id: User ID (uses default if not provided)

        Returns:
            List of record IDs
        """
        if not records:
            return []

        user_id = user_id or self._user_id
        if not user_id:
            raise ValueError("user_id is required for pgvector operations")

        # Generate embeddings for all records
        texts = [r.content for r in records]
        embeddings = await self._generate_embeddings_batch(texts)

        ids = []
        async with await self._get_session() as session:
            from sqlalchemy import text

            for i, record in enumerate(records):
                embedding = embeddings[i]
                if embedding is None:
                    logger.warning(f"Skipping record without embedding: {record.id}")
                    continue

                # Set collection from mapping
                filters = self._collection_to_filters(collection)
                if "visibility" in filters:
                    record.visibility = filters["visibility"]
                if "layer" in filters:
                    record.layer = filters["layer"]
                if "message_type" in filters:
                    record.message_type = filters["message_type"]
                record.collection = collection

                # Generate ID if not set
                if not record.id:
                    record.id = str(uuid4())
                if not record.created_at:
                    record.created_at = datetime.utcnow()

                values = self._record_to_db_values(record, user_id)
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"

                await session.execute(
                    text("""
                        INSERT INTO user_vectors (
                            id, user_id, collection, content, metadata,
                            layer, visibility, agent_id, message_type,
                            attention_weight, access_count, last_accessed_at,
                            responding_to, conversation_thread, related_agents, tags,
                            created_at, embedding
                        ) VALUES (
                            :id, :user_id, :collection, :content, :metadata,
                            :layer, :visibility, :agent_id, :message_type,
                            :attention_weight, :access_count, :last_accessed_at,
                            :responding_to, :conversation_thread, :related_agents, :tags,
                            :created_at, :embedding
                        )
                    """),
                    {
                        **values,
                        "metadata": json.dumps(values["metadata"]),
                        "responding_to": json.dumps(values["responding_to"]),
                        "related_agents": json.dumps(values["related_agents"]),
                        "tags": json.dumps(values["tags"]),
                        "embedding": embedding_str,
                    }
                )
                ids.append(record.id)

            await session.commit()

        logger.debug(f"Added {len(ids)} records to {collection}")
        return ids

    async def search_async(
        self,
        collection: str,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        user_id: Optional[UUID] = None,
    ) -> List[MemoryRecord]:
        """
        Semantic search using pgvector (async).

        Args:
            collection: Collection name
            query: Search query text
            n_results: Maximum results
            where: Additional filter conditions
            user_id: User ID (uses default if not provided)

        Returns:
            List of MemoryRecord objects with similarity scores
        """
        user_id = user_id or self._user_id
        if not user_id:
            raise ValueError("user_id is required for pgvector operations")

        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []

        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        # Build WHERE clause
        filters = self._collection_to_filters(collection)
        where_clauses = ["user_id = :user_id"]
        params = {"user_id": user_id, "embedding": embedding_str, "limit": n_results}

        if "visibility" in filters:
            where_clauses.append("visibility = :visibility")
            params["visibility"] = filters["visibility"]
        if "layer" in filters:
            where_clauses.append("layer = :layer")
            params["layer"] = filters["layer"]
        if "message_type" in filters:
            where_clauses.append("message_type = :message_type")
            params["message_type"] = filters["message_type"]
        if "collection" in filters:
            where_clauses.append("collection = :collection")
            params["collection"] = filters["collection"]

        # Add custom where conditions
        if where:
            for key, value in where.items():
                where_clauses.append(f"{key} = :{key}")
                params[key] = value

        where_sql = " AND ".join(where_clauses)

        async with await self._get_session() as session:
            from sqlalchemy import text

            result = await session.execute(
                text(f"""
                    SELECT
                        id, user_id, collection, content, metadata,
                        layer, visibility, agent_id, message_type,
                        attention_weight, access_count, last_accessed_at,
                        responding_to, conversation_thread, related_agents, tags,
                        created_at,
                        1 - (embedding <=> :embedding) as similarity
                    FROM user_vectors
                    WHERE {where_sql}
                    ORDER BY embedding <=> :embedding
                    LIMIT :limit
                """),
                params
            )
            rows = result.fetchall()

        records = []
        for row in rows:
            record = self._db_row_to_record(row, similarity=row.similarity, collection=collection)
            records.append(record)

        return records

    async def get_async(
        self,
        collection: str,
        ids: List[str],
        user_id: Optional[UUID] = None,
    ) -> List[MemoryRecord]:
        """Get records by ID (async)."""
        if not ids:
            return []

        user_id = user_id or self._user_id
        if not user_id:
            raise ValueError("user_id is required")

        async with await self._get_session() as session:
            from sqlalchemy import text

            # Convert string IDs to UUIDs for query
            uuid_ids = [UUID(id) if isinstance(id, str) else id for id in ids]

            result = await session.execute(
                text("""
                    SELECT
                        id, user_id, collection, content, metadata,
                        layer, visibility, agent_id, message_type,
                        attention_weight, access_count, last_accessed_at,
                        responding_to, conversation_thread, related_agents, tags,
                        created_at
                    FROM user_vectors
                    WHERE user_id = :user_id AND id = ANY(:ids)
                """),
                {"user_id": user_id, "ids": uuid_ids}
            )
            rows = result.fetchall()

        return [self._db_row_to_record(row, collection=collection) for row in rows]

    async def update_async(
        self,
        collection: str,
        records: List[MemoryRecord],
        user_id: Optional[UUID] = None,
    ) -> bool:
        """Update existing records (async)."""
        if not records:
            return True

        user_id = user_id or self._user_id
        if not user_id:
            raise ValueError("user_id is required")

        try:
            async with await self._get_session() as session:
                from sqlalchemy import text

                for record in records:
                    await session.execute(
                        text("""
                            UPDATE user_vectors SET
                                content = :content,
                                layer = :layer,
                                visibility = :visibility,
                                agent_id = :agent_id,
                                message_type = :message_type,
                                attention_weight = :attention_weight,
                                access_count = :access_count,
                                last_accessed_at = :last_accessed_at,
                                responding_to = :responding_to,
                                conversation_thread = :conversation_thread,
                                related_agents = :related_agents,
                                tags = :tags
                            WHERE id = :id AND user_id = :user_id
                        """),
                        {
                            "id": UUID(record.id) if isinstance(record.id, str) else record.id,
                            "user_id": user_id,
                            "content": record.content,
                            "layer": record.layer,
                            "visibility": record.visibility,
                            "agent_id": record.agent_id,
                            "message_type": record.message_type,
                            "attention_weight": record.attention_weight,
                            "access_count": record.access_count,
                            "last_accessed_at": record.last_accessed_at,
                            "responding_to": json.dumps(record.responding_to or []),
                            "conversation_thread": record.conversation_thread,
                            "related_agents": json.dumps(record.related_agents or []),
                            "tags": json.dumps(record.tags or []),
                        }
                    )

                await session.commit()
            return True

        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    async def delete_async(
        self,
        collection: str,
        ids: List[str],
        user_id: Optional[UUID] = None,
    ) -> bool:
        """Delete records by ID (async)."""
        if not ids:
            return True

        user_id = user_id or self._user_id
        if not user_id:
            raise ValueError("user_id is required")

        try:
            async with await self._get_session() as session:
                from sqlalchemy import text

                uuid_ids = [UUID(id) if isinstance(id, str) else id for id in ids]

                await session.execute(
                    text("""
                        DELETE FROM user_vectors
                        WHERE user_id = :user_id AND id = ANY(:ids)
                    """),
                    {"user_id": user_id, "ids": uuid_ids}
                )
                await session.commit()
            return True

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def count_async(
        self,
        collection: str,
        user_id: Optional[UUID] = None,
    ) -> int:
        """Count records in collection (async)."""
        user_id = user_id or self._user_id
        if not user_id:
            raise ValueError("user_id is required")

        filters = self._collection_to_filters(collection)
        where_clauses = ["user_id = :user_id"]
        params = {"user_id": user_id}

        if "visibility" in filters:
            where_clauses.append("visibility = :visibility")
            params["visibility"] = filters["visibility"]
        if "layer" in filters:
            where_clauses.append("layer = :layer")
            params["layer"] = filters["layer"]
        if "collection" in filters:
            where_clauses.append("collection = :collection")
            params["collection"] = filters["collection"]

        where_sql = " AND ".join(where_clauses)

        async with await self._get_session() as session:
            from sqlalchemy import text

            result = await session.execute(
                text(f"SELECT COUNT(*) FROM user_vectors WHERE {where_sql}"),
                params
            )
            row = result.fetchone()
            return row[0] if row else 0

    async def list_all_async(
        self,
        collection: str,
        limit: Optional[int] = None,
        offset: int = 0,
        user_id: Optional[UUID] = None,
    ) -> List[MemoryRecord]:
        """List all records for export (async)."""
        user_id = user_id or self._user_id
        if not user_id:
            raise ValueError("user_id is required")

        filters = self._collection_to_filters(collection)
        where_clauses = ["user_id = :user_id"]
        params = {"user_id": user_id, "offset": offset}

        if "visibility" in filters:
            where_clauses.append("visibility = :visibility")
            params["visibility"] = filters["visibility"]
        if "layer" in filters:
            where_clauses.append("layer = :layer")
            params["layer"] = filters["layer"]
        if "collection" in filters:
            where_clauses.append("collection = :collection")
            params["collection"] = filters["collection"]

        where_sql = " AND ".join(where_clauses)
        limit_sql = f"LIMIT {limit}" if limit else ""

        async with await self._get_session() as session:
            from sqlalchemy import text

            result = await session.execute(
                text(f"""
                    SELECT
                        id, user_id, collection, content, metadata,
                        layer, visibility, agent_id, message_type,
                        attention_weight, access_count, last_accessed_at,
                        responding_to, conversation_thread, related_agents, tags,
                        created_at
                    FROM user_vectors
                    WHERE {where_sql}
                    ORDER BY created_at DESC
                    OFFSET :offset
                    {limit_sql}
                """),
                params
            )
            rows = result.fetchall()

        return [self._db_row_to_record(row, collection=collection) for row in rows]

    # =========================================================================
    # Sync Wrappers (for StorageBackend interface compatibility)
    # =========================================================================

    def add(
        self,
        collection: str,
        records: List[MemoryRecord],
    ) -> List[str]:
        """Sync wrapper for add_async."""
        return asyncio.get_event_loop().run_until_complete(
            self.add_async(collection, records)
        )

    def search(
        self,
        collection: str,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """Sync wrapper for search_async."""
        return asyncio.get_event_loop().run_until_complete(
            self.search_async(collection, query, n_results, where)
        )

    def get(
        self,
        collection: str,
        ids: List[str],
    ) -> List[MemoryRecord]:
        """Sync wrapper for get_async."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_async(collection, ids)
        )

    def update(
        self,
        collection: str,
        records: List[MemoryRecord],
    ) -> bool:
        """Sync wrapper for update_async."""
        return asyncio.get_event_loop().run_until_complete(
            self.update_async(collection, records)
        )

    def delete(
        self,
        collection: str,
        ids: List[str],
    ) -> bool:
        """Sync wrapper for delete_async."""
        return asyncio.get_event_loop().run_until_complete(
            self.delete_async(collection, ids)
        )

    def count(self, collection: str) -> int:
        """Sync wrapper for count_async."""
        return asyncio.get_event_loop().run_until_complete(
            self.count_async(collection)
        )

    def list_all(
        self,
        collection: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """Sync wrapper for list_all_async."""
        return asyncio.get_event_loop().run_until_complete(
            self.list_all_async(collection, limit, offset)
        )


# =============================================================================
# Export/Import Functions (Async)
# =============================================================================

async def export_to_memory_core_async(
    backend: PgVectorBackend,
    agent_id: Optional[str] = None,
    collections: Optional[List[str]] = None,
    user_id: Optional[UUID] = None,
) -> "MemoryCore":
    """
    Export memories to portable MemoryCore format (async).

    Args:
        backend: PgVectorBackend instance
        agent_id: Filter by agent (None for all)
        collections: Which collections to export (None for all)
        user_id: User ID for export

    Returns:
        MemoryCore object ready for JSON serialization
    """
    from .base import MemoryCore
    from ..config import ALL_COLLECTIONS

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
            records = await backend.list_all_async(coll_name, user_id=user_id)

            # Filter by agent if specified
            if agent_id:
                records = [r for r in records if r.agent_id == agent_id]

            # Don't include embeddings in export (will be regenerated on import)
            core.collections[coll_name] = [r.to_dict() for r in records]
            core.metadata["total_memories"] += len(records)

        except Exception as e:
            logger.warning(f"Failed to export {coll_name}: {e}")

    return core


async def import_from_memory_core_async(
    backend: PgVectorBackend,
    core: "MemoryCore",
    user_id: UUID,
    re_embed: bool = True,
) -> Dict[str, int]:
    """
    Import memories from MemoryCore format (async).

    Args:
        backend: PgVectorBackend instance
        core: MemoryCore object to import
        user_id: User ID for import
        re_embed: Whether to regenerate embeddings (always True for pgvector)

    Returns:
        Dict with import stats per collection
    """
    from .base import MemoryRecord

    stats = {}

    for coll_name, records_data in core.collections.items():
        try:
            records = [MemoryRecord.from_dict(r) for r in records_data]

            # Clear embeddings - will be regenerated
            for r in records:
                r.embedding = None

            ids = await backend.add_async(coll_name, records, user_id=user_id)
            stats[coll_name] = len(ids)

        except Exception as e:
            logger.error(f"Failed to import {coll_name}: {e}")
            stats[coll_name] = 0

    return stats
