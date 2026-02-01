"""
Neo-Cortex Engine: Unified Memory Coordinator

The main interface for all memory operations across:
- Knowledge (curated docs)
- Shared Memory (multi-agent memory)
- Sessions (session continuity)
- Memory Health (access tracking, decay)

Integrates all subsystems:
- storage/ (ChromaDB/pgvector backends)
- shared_engine.py (multi-agent memory)
- session_engine.py (session continuity)

Supports both local (ChromaDB) and cloud (pgvector) backends.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    CHROMA_PATH,
    ALL_COLLECTIONS,
    COLLECTION_KNOWLEDGE,
    DEFAULT_BACKEND,
    DEFAULT_SEARCH_RESULTS,
    AGENT_PROFILES,
)
from .storage import ChromaBackend, StorageBackend, MemoryRecord, MemoryCore
from .storage.chroma_backend import export_to_memory_core, import_from_memory_core
from .shared_engine import SharedMemoryEngine, SHARED_TOOL_SCHEMAS
from .session_engine import SessionEngine, SESSION_TOOL_SCHEMAS
from .health_engine import HealthEngine, HEALTH_TOOL_SCHEMAS

logger = logging.getLogger(__name__)


class CortexEngine:
    """
    Unified memory engine for Neo-Cortex.

    Coordinates all memory subsystems:
    - Shared Memory (multi-agent memory)
    - Sessions (session continuity)
    - Knowledge Base (curated docs)
    - Memory Health (access tracking, decay)
    """

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        chroma_path: Optional[Path] = None,
        db_url: Optional[str] = None,
    ):
        self.backend_name = backend

        if backend == "chroma":
            self.storage: StorageBackend = ChromaBackend(
                persist_path=chroma_path or CHROMA_PATH
            )
        elif backend == "pgvector":
            from .storage import PgVectorBackend, HAS_PGVECTOR
            if not HAS_PGVECTOR:
                raise RuntimeError("pgvector backend not available (missing dependencies)")
            self.storage = PgVectorBackend(db_url=db_url)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Initialize subsystems
        self.shared = SharedMemoryEngine(self.storage)
        self.sessions = SessionEngine(self.storage)
        self.health = HealthEngine(self.storage)

        self._current_agent_id = "CLAUDE"

        logger.info(f"CortexEngine initialized with {backend} backend")

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self):
        """Initialize all collections/tables."""
        self.storage.initialize()
        logger.info("Cortex initialized")

    # =========================================================================
    # Agent Management (delegated to shared)
    # =========================================================================

    def set_current_agent(self, agent_id: str):
        """Set the current active agent for all subsystems."""
        self._current_agent_id = agent_id.upper()
        self.shared.set_current_agent(agent_id)
        self.sessions.set_current_agent(agent_id)
        logger.info(f"Current agent set to: {self._current_agent_id}")

    def get_current_agent(self) -> str:
        return self._current_agent_id

    def get_agent_profile(self, agent_id: str) -> Optional[Dict]:
        return self.shared.get_agent_profile(agent_id)

    def list_agents(self) -> Dict[str, Any]:
        return self.shared.list_agents()

    # =========================================================================
    # Shared Memory (delegated)
    # =========================================================================

    def memory_store(self, content: str, **kwargs) -> Dict[str, Any]:
        """Store a memory (see shared_engine.post)."""
        return self.shared.post(content, **kwargs)

    def memory_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search shared memory (see shared_engine.search)."""
        return self.shared.search(query, **kwargs)

    def memory_convergence(self, query: str, **kwargs) -> Dict[str, Any]:
        """Detect convergence (see shared_engine.detect_convergence)."""
        return self.shared.detect_convergence(query, **kwargs)

    def register_agent(self, **kwargs) -> Dict[str, Any]:
        """Register an agent (see shared_engine.register_agent)."""
        return self.shared.register_agent(**kwargs)

    def agent_greeting(self, **kwargs) -> Dict[str, Any]:
        """Agent greeting (see shared_engine.agent_greeting)."""
        return self.shared.agent_greeting(**kwargs)

    def memory_get_thread(self, thread_id: str, **kwargs) -> Dict[str, Any]:
        """Get thread (see shared_engine.get_thread)."""
        return self.shared.get_thread(thread_id, **kwargs)

    # =========================================================================
    # Sessions (delegated)
    # =========================================================================

    def session_save(self, session_summary: str, **kwargs) -> Dict[str, Any]:
        """Save a session note (see session_engine.save_session)."""
        return self.sessions.save_session(session_summary, **kwargs)

    def session_recall(self, **kwargs) -> Dict[str, Any]:
        """Recall session notes (see session_engine.recall_sessions)."""
        return self.sessions.recall_sessions(**kwargs)

    def quick_session_note(self, summary: str, **kwargs) -> Dict[str, Any]:
        """Quick session note shortcut."""
        return self.sessions.quick_session_note(summary, **kwargs)

    def get_unfinished_tasks(self) -> List[str]:
        """Get unfinished tasks from recent sessions."""
        return self.sessions.get_unfinished_tasks()

    # =========================================================================
    # Memory Health (delegated)
    # =========================================================================

    def health_report(self, **kwargs) -> Dict[str, Any]:
        return self.health.health_report(**kwargs)

    def get_stale_memories(self, collection: str, **kwargs) -> Dict[str, Any]:
        return self.health.get_stale_memories(collection, **kwargs)

    def get_duplicate_candidates(self, collection: str, **kwargs) -> Dict[str, Any]:
        return self.health.get_duplicate_candidates(collection, **kwargs)

    def consolidate_memories(self, collection: str, id1: str, id2: str, **kwargs) -> Dict[str, Any]:
        return self.health.consolidate_memories(collection, id1, id2, **kwargs)

    def run_promotions(self, collection: str, **kwargs) -> Dict[str, Any]:
        return self.health.run_promotions(collection, **kwargs)

    def update_attention_weights(self, collection: str, **kwargs) -> Dict[str, Any]:
        return self.health.update_attention_weights(collection, **kwargs)

    # =========================================================================
    # Direct Storage Operations (for knowledge and advanced use)
    # =========================================================================

    def remember(
        self,
        content: str,
        collection: str = COLLECTION_KNOWLEDGE,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Store a memory directly in a collection.

        For shared/private/thread, use memory_store() instead.
        This is for knowledge base entries and direct storage.
        """
        try:
            record = MemoryRecord(
                id=f"cortex_{self._current_agent_id}_{datetime.now().timestamp()}",
                content=content,
                agent_id=kwargs.get("agent_id", self._current_agent_id),
                visibility=kwargs.get("visibility", "private"),
                layer=kwargs.get("layer", "working"),
                message_type=kwargs.get("message_type", "fact"),
                tags=kwargs.get("tags", []),
                created_at=datetime.now(),
            )

            ids = self.storage.add(collection, [record])

            return {
                "success": True,
                "id": ids[0] if ids else None,
                "collection": collection,
            }

        except Exception as e:
            logger.error(f"remember failed: {e}")
            return {"success": False, "error": str(e)}

    def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        n_results: int = DEFAULT_SEARCH_RESULTS,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search across specified collections.

        For shared memory search, use memory_search() instead.
        This is for cross-collection and knowledge base search.
        """
        try:
            search_collections = collections or [COLLECTION_KNOWLEDGE]
            all_results = []

            for coll in search_collections:
                results = self.storage.search(
                    collection=coll,
                    query=query,
                    n_results=n_results,
                    where=kwargs.get("where"),
                )
                for r in results:
                    r.collection = coll
                all_results.extend(results)

            all_results.sort(key=lambda x: x.similarity or 0, reverse=True)
            all_results = all_results[:n_results]

            return {
                "success": True,
                "query": query,
                "count": len(all_results),
                "results": [r.to_dict() for r in all_results],
            }

        except Exception as e:
            logger.error(f"search failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    # =========================================================================
    # Export/Import
    # =========================================================================

    def export_memory_core(
        self,
        agent_id: Optional[str] = None,
        collections: Optional[List[str]] = None,
    ) -> MemoryCore:
        if isinstance(self.storage, ChromaBackend):
            return export_to_memory_core(
                self.storage,
                agent_id=agent_id,
                collections=collections,
            )
        else:
            raise NotImplementedError("Export not yet implemented for this backend")

    def import_memory_core(
        self,
        core: MemoryCore,
        re_embed: bool = True,
    ) -> Dict[str, int]:
        if isinstance(self.storage, ChromaBackend):
            return import_from_memory_core(
                self.storage,
                core=core,
                re_embed=re_embed,
            )
        else:
            raise NotImplementedError("Import not yet implemented for this backend")

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        try:
            stats = {
                "success": True,
                "backend": self.backend_name,
                "embedding_dimension": self.storage.embedding_dimension,
                "current_agent": self._current_agent_id,
                "registered_agents": len(AGENT_PROFILES),
                "collections": {},
                "total_memories": 0,
                "shared": None,
                "sessions": None,
            }

            for coll_name in ALL_COLLECTIONS:
                try:
                    count = self.storage.count(coll_name)
                    stats["collections"][coll_name] = count
                    stats["total_memories"] += count
                except:
                    stats["collections"][coll_name] = 0

            stats["shared"] = self.shared.stats()
            stats["sessions"] = self.sessions.stats()

            return stats

        except Exception as e:
            logger.error(f"stats failed: {e}")
            return {"success": False, "error": str(e)}


# =============================================================================
# Module-level convenience functions
# =============================================================================

_engine: Optional[CortexEngine] = None


def get_engine(backend: str = DEFAULT_BACKEND) -> CortexEngine:
    """Get or create the global cortex engine."""
    global _engine
    if _engine is None:
        _engine = CortexEngine(backend=backend)
    return _engine


def set_engine(engine: CortexEngine):
    """Set the global cortex engine (for testing)."""
    global _engine
    _engine = engine


# Convenience functions
def memory_store(content: str, **kwargs) -> Dict:
    return get_engine().memory_store(content, **kwargs)


def memory_search(query: str, **kwargs) -> Dict:
    return get_engine().memory_search(query, **kwargs)


def session_save(session_summary: str, **kwargs) -> Dict:
    return get_engine().session_save(session_summary, **kwargs)


def session_recall(**kwargs) -> Dict:
    return get_engine().session_recall(**kwargs)


def stats() -> Dict:
    return get_engine().stats()


# =============================================================================
# Combined Tool Schemas
# =============================================================================

CORTEX_TOOL_SCHEMAS = {
    **SHARED_TOOL_SCHEMAS,
    **SESSION_TOOL_SCHEMAS,
    **HEALTH_TOOL_SCHEMAS,
    "knowledge_search": {
        "name": "knowledge_search",
        "description": (
            "Search the knowledge base for documentation and reference material. "
            "Use this to find information about frameworks, tools, APIs, and guides "
            "that have been ingested into the cortex knowledge collection."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 20)",
                    "default": 5
                },
            },
            "required": ["query"]
        }
    },
    "cortex_stats": {
        "name": "cortex_stats",
        "description": "Get comprehensive statistics about the neo-cortex memory system.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    "cortex_export": {
        "name": "cortex_export",
        "description": (
            "Export memories to portable JSON format. "
            "Use this to backup or transfer memories between local and cloud."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Filter by agent (omit for all)"
                },
                "collections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Which collections to export (omit for all)"
                }
            },
            "required": []
        }
    },
}
