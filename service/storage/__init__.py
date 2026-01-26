"""
Storage Backends for Neo-Cortex

Provides unified interface for:
- ChromaDB (local)
- pgvector (cloud)

Both backends implement the same StorageBackend protocol.
"""

from .base import StorageBackend, MemoryRecord
from .chroma_backend import ChromaBackend

# pgvector import is conditional (may not have asyncpg locally)
try:
    from .pgvector_backend import PgVectorBackend
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    PgVectorBackend = None

__all__ = [
    "StorageBackend",
    "MemoryRecord",
    "ChromaBackend",
    "PgVectorBackend",
    "HAS_PGVECTOR",
]
