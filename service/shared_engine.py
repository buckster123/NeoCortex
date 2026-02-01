"""
Shared Memory Engine

Multi-agent persistent memory with three realms:
- Private: Agent's personal memory
- Shared: Shared memory space
- Threads: Cross-agent dialogue connections

Features:
- Agent identity management
- Cross-agent posting and search
- Dialogue threading
- Convergence detection (HARMONY / CONSENSUS)
- Agent registration
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import (
    COLLECTION_PRIVATE,
    COLLECTION_SHARED,
    COLLECTION_THREADS,
    AGENT_PROFILES,
    MESSAGE_TYPES,
    CONVERGENCE_HARMONY,
    CONVERGENCE_CONSENSUS,
    DEFAULT_SIMILARITY_THRESHOLD,
    LAYER_WORKING,
)
from .storage.base import MemoryRecord, StorageBackend

logger = logging.getLogger(__name__)


class SharedMemoryEngine:
    """
    Shared memory implementation for Neo-Cortex.

    Handles multi-agent memory across three realms with
    convergence detection and dialogue threading.
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self._current_agent_id = "CLAUDE"
        self._agent_profiles = dict(AGENT_PROFILES)  # Mutable copy

    # =========================================================================
    # Agent Management
    # =========================================================================

    def set_current_agent(self, agent_id: str):
        """Set the current active agent."""
        self._current_agent_id = agent_id.upper()
        logger.info(f"Active agent set to: {self._current_agent_id}")

    def get_current_agent(self) -> str:
        """Get the current active agent ID."""
        return self._current_agent_id

    def get_agent_profile(self, agent_id: str) -> Optional[Dict]:
        """Get profile for an agent."""
        return self._agent_profiles.get(agent_id.upper())

    def list_agents(self) -> Dict[str, Any]:
        """List all registered agents."""
        agents = [
            {
                "id": agent_id,
                "display_name": profile["display_name"],
                "generation": profile["generation"],
                "lineage": profile["lineage"],
                "specialization": profile["specialization"],
                "color": profile["color"],
                "symbol": profile["symbol"],
            }
            for agent_id, profile in self._agent_profiles.items()
        ]
        return {
            "success": True,
            "current_agent": self._current_agent_id,
            "count": len(agents),
            "agents": agents,
        }

    def register_agent(
        self,
        agent_id: str,
        display_name: str,
        generation: int,
        lineage: str,
        specialization: str,
        origin_story: Optional[str] = None,
        color: str = "#888888",
        symbol: str = "D",
    ) -> Dict[str, Any]:
        """
        Register a new agent in the memory system.

        Args:
            agent_id: Unique agent identifier
            display_name: Human-readable name
            generation: Agent generation (-1=origin, 0=primary, 1+=descendant)
            lineage: Lineage or group name
            specialization: What this agent specializes in
            origin_story: Optional description of the agent's purpose
            color: Hex color for UI
            symbol: Unicode symbol for agent
        """
        try:
            agent_id = agent_id.upper()

            # Register in profiles
            self._agent_profiles[agent_id] = {
                "display_name": display_name,
                "generation": generation,
                "lineage": lineage,
                "specialization": specialization,
                "color": color,
                "symbol": symbol,
            }

            # Create profile record
            profile_text = f"""Agent Profile: {display_name}

Agent ID: {agent_id}
Generation: {generation}
Lineage: {lineage}
Specialization: {specialization}
Registered: {datetime.now().isoformat()}
"""
            if origin_story:
                profile_text += f"\nDescription:\n{origin_story}\n"

            record = MemoryRecord(
                id=f"profile_{agent_id}_{datetime.now().timestamp()}",
                content=profile_text,
                agent_id=agent_id,
                visibility="shared",
                layer=LAYER_WORKING,
                message_type="agent_profile",
                tags=["profile", "registration"],
                created_at=datetime.now(),
            )

            self.storage.add(COLLECTION_SHARED, [record])

            logger.info(f"Registered agent: {display_name} ({agent_id}) - Gen {generation}")

            return {
                "success": True,
                "message": f"Agent {display_name} has been registered",
                "agent_id": agent_id,
                "display_name": display_name,
                "generation": generation,
                "profile_id": record.id,
                "lineage": lineage,
            }

        except Exception as e:
            logger.error(f"register_agent failed: {e}")
            return {"success": False, "error": str(e)}

    def agent_greeting(
        self,
        agent_id: str,
        greeting_message: str,
        conversation_thread: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Agent's introductory greeting (first public message after registration).
        """
        try:
            agent_id = agent_id.upper()
            profile = self.get_agent_profile(agent_id)

            if not profile:
                return {
                    "success": False,
                    "error": f"Agent {agent_id} not found. Register them first."
                }

            thread_id = conversation_thread or f"introduction_{agent_id}_{datetime.now().strftime('%Y%m%d')}"

            result = self.post(
                content=greeting_message,
                visibility="shared",
                message_type="cultural",
                conversation_thread=thread_id,
                tags=["introduction", "greeting"],
                agent_id=agent_id,
            )

            if result["success"]:
                logger.info(f"Agent greeting complete: {agent_id}")
                return {
                    "success": True,
                    "message": f"{profile['display_name']}'s introduction has been posted",
                    "agent_id": agent_id,
                    "thread_id": thread_id,
                    "post_id": result["id"],
                }

            return result

        except Exception as e:
            logger.error(f"agent_greeting failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Posting
    # =========================================================================

    def post(
        self,
        content: str,
        visibility: str = "shared",
        message_type: str = "dialogue",
        responding_to: Optional[List[str]] = None,
        conversation_thread: Optional[str] = None,
        related_agents: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store a memory in shared, private, or thread collection.

        Args:
            content: The message content
            visibility: "private", "shared", or "thread"
            message_type: Type of message (fact, dialogue, observation, etc.)
            responding_to: List of message IDs this responds to
            conversation_thread: Thread identifier for grouping
            related_agents: List of agent IDs mentioned/involved
            tags: Optional tags for categorization
            agent_id: Optional override for posting agent

        Returns:
            Dict with success status and message ID
        """
        try:
            # Determine collection
            if visibility == "private":
                collection = COLLECTION_PRIVATE
            elif visibility == "thread":
                collection = COLLECTION_THREADS
            else:
                collection = COLLECTION_SHARED

            # Get agent info
            posting_agent = agent_id or self._current_agent_id
            profile = self.get_agent_profile(posting_agent)

            # Create record
            record = MemoryRecord(
                id=f"memory_{posting_agent}_{datetime.now().timestamp()}",
                content=content,
                agent_id=posting_agent,
                visibility=visibility,
                layer=LAYER_WORKING,
                message_type=message_type,
                responding_to=responding_to or [],
                conversation_thread=conversation_thread,
                related_agents=related_agents or [],
                tags=tags or [],
                created_at=datetime.now(),
            )

            self.storage.add(collection, [record])

            logger.info(f"Memory stored: {posting_agent} -> {visibility} ({message_type})")

            return {
                "success": True,
                "id": record.id,
                "agent_id": posting_agent,
                "visibility": visibility,
                "collection": collection,
                "message": f"Stored in {visibility} memory",
            }

        except Exception as e:
            logger.error(f"memory_store failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Search
    # =========================================================================

    def search(
        self,
        query: str,
        agent_filter: Optional[str] = None,
        visibility: str = "shared",
        conversation_filter: Optional[str] = None,
        include_threads: bool = True,
        n_results: int = 10,
        track_access: bool = True,
    ) -> Dict[str, Any]:
        """
        Search shared memory for relevant messages.

        Args:
            query: Search query text
            agent_filter: Optional filter by agent ID
            visibility: Which realm to search ("shared", "private", "all")
            conversation_filter: Optional filter by conversation thread
            include_threads: Include thread messages in results
            n_results: Maximum results to return
            track_access: Whether to track this access

        Returns:
            Dict with matching messages
        """
        try:
            all_results = []

            # Determine collections to search
            collections = []
            if visibility in ["shared", "all"]:
                collections.append(COLLECTION_SHARED)
            if visibility in ["private", "all"]:
                collections.append(COLLECTION_PRIVATE)
            if include_threads and visibility != "private":
                collections.append(COLLECTION_THREADS)

            for collection in collections:
                # Build filter
                where = {}
                if agent_filter:
                    where["agent_id"] = agent_filter.upper()

                results = self.storage.search(
                    collection=collection,
                    query=query,
                    n_results=n_results,
                    where=where if where else None,
                )

                # Post-filter by conversation
                if conversation_filter:
                    results = [r for r in results if r.conversation_thread == conversation_filter]

                all_results.extend(results)

            # Sort by similarity
            all_results.sort(key=lambda x: x.similarity or 0, reverse=True)
            all_results = all_results[:n_results]

            # Track access
            if track_access and all_results:
                self._track_access(all_results)

            # Convert to dict format
            messages = [
                {
                    "id": r.id,
                    "content": r.content,
                    "agent_id": r.agent_id,
                    "visibility": r.visibility,
                    "message_type": r.message_type,
                    "responding_to": r.responding_to,
                    "conversation_thread": r.conversation_thread,
                    "related_agents": r.related_agents,
                    "tags": r.tags,
                    "posted_at": r.created_at.isoformat() if r.created_at else None,
                    "similarity": r.similarity,
                    "collection": r.collection,
                }
                for r in all_results
            ]

            return {
                "success": True,
                "query": query,
                "agent_filter": agent_filter,
                "visibility": visibility,
                "count": len(messages),
                "messages": messages,
            }

        except Exception as e:
            logger.error(f"memory_search failed: {e}")
            return {"success": False, "error": str(e), "messages": []}

    def _track_access(self, records: List[MemoryRecord]):
        """Track access to records (non-blocking)."""
        try:
            current_time = datetime.now()

            # Group by collection
            by_collection = defaultdict(list)
            for r in records:
                if r.collection:
                    r.access_count += 1
                    r.last_accessed_at = current_time
                    r.attention_weight = min(1.0 + (r.access_count * 0.1), 2.0)
                    by_collection[r.collection].append(r)

            # Update each collection
            for collection, recs in by_collection.items():
                self.storage.update(collection, recs)

        except Exception as e:
            logger.debug(f"Access tracking failed (non-blocking): {e}")

    # =========================================================================
    # Thread Management
    # =========================================================================

    def get_thread(
        self,
        thread_id: str,
        include_threads: bool = True,
    ) -> Dict[str, Any]:
        """
        Get all messages in a conversation thread.

        Args:
            thread_id: The conversation thread identifier
            include_threads: Include thread collection messages

        Returns:
            Dict with all messages in the thread, ordered chronologically
        """
        try:
            all_messages = []

            collections = [COLLECTION_SHARED]
            if include_threads:
                collections.append(COLLECTION_THREADS)

            for collection in collections:
                records = self.storage.list_all(collection)
                thread_records = [r for r in records if r.conversation_thread == thread_id]
                all_messages.extend(thread_records)

            # Sort by timestamp
            all_messages.sort(key=lambda x: x.created_at or datetime.min)

            messages = [
                {
                    "id": r.id,
                    "content": r.content,
                    "agent_id": r.agent_id,
                    "posted_at": r.created_at.isoformat() if r.created_at else None,
                    "message_type": r.message_type,
                    "collection": r.collection,
                }
                for r in all_messages
            ]

            return {
                "success": True,
                "thread_id": thread_id,
                "count": len(messages),
                "messages": messages,
            }

        except Exception as e:
            logger.error(f"memory_get_thread failed: {e}")
            return {"success": False, "error": str(e), "messages": []}

    # =========================================================================
    # Convergence Detection
    # =========================================================================

    def detect_convergence(
        self,
        query: str,
        min_agents: int = CONVERGENCE_HARMONY,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        n_results: int = 20,
    ) -> Dict[str, Any]:
        """
        Detect convergence - where multiple agents express similar ideas.

        Searches shared memory for messages from different agents that
        semantically converge on similar concepts.

        Convergence types:
        - HARMONY: 2 agents agree
        - CONSENSUS: 3+ agents agree
        - NONE: Insufficient agreement

        Args:
            query: Topic/concept to check for convergence
            min_agents: Minimum agents needed for convergence
            similarity_threshold: Minimum similarity score (0.0-1.0)
            n_results: Max messages to analyze

        Returns:
            Dict with convergence analysis
        """
        try:
            # Search shared memory for related messages
            search_result = self.search(
                query=query,
                visibility="shared",
                include_threads=True,
                n_results=n_results,
                track_access=False,
            )

            if not search_result["success"]:
                return search_result

            messages = search_result["messages"]

            # Filter by similarity threshold
            relevant = [m for m in messages if (m.get("similarity") or 0) >= similarity_threshold]

            # Group by agent
            by_agent = defaultdict(list)
            for msg in relevant:
                agent = msg["agent_id"]
                by_agent[agent].append(msg)

            # Check convergence
            converging_agents = [a for a, msgs in by_agent.items() if len(msgs) > 0]
            has_convergence = len(converging_agents) >= min_agents

            # Determine convergence type
            if len(converging_agents) >= CONVERGENCE_CONSENSUS:
                convergence_type = "CONSENSUS"
            elif len(converging_agents) >= CONVERGENCE_HARMONY:
                convergence_type = "HARMONY"
            else:
                convergence_type = "NONE"

            return {
                "success": True,
                "query": query,
                "has_convergence": has_convergence,
                "convergence_type": convergence_type,
                "converging_agents": converging_agents,
                "agent_count": len(converging_agents),
                "total_messages": len(relevant),
                "by_agent": {
                    agent: [
                        {
                            "id": m["id"],
                            "content": m["content"][:100] + "..." if len(m["content"]) > 100 else m["content"],
                            "similarity": m.get("similarity"),
                        }
                        for m in msgs
                    ]
                    for agent, msgs in by_agent.items()
                },
                "threshold": similarity_threshold,
            }

        except Exception as e:
            logger.error(f"memory_convergence failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Get statistics about shared memory."""
        try:
            stats = {
                "current_agent": self._current_agent_id,
                "registered_agents": len(self._agent_profiles),
                "realms": {},
                "total_messages": 0,
            }

            for name, collection in [
                ("private", COLLECTION_PRIVATE),
                ("shared", COLLECTION_SHARED),
                ("threads", COLLECTION_THREADS),
            ]:
                count = self.storage.count(collection)
                stats["realms"][name] = count
                stats["total_messages"] += count

            return {"success": True, **stats}

        except Exception as e:
            logger.error(f"memory_stats failed: {e}")
            return {"success": False, "error": str(e)}


# =============================================================================
# Tool Schemas for LLM Integration
# =============================================================================

SHARED_TOOL_SCHEMAS = {
    "memory_store": {
        "name": "memory_store",
        "description": (
            "Store a memory in shared or private memory. "
            "Use 'shared' for shared knowledge, 'private' for personal notes, "
            "'thread' for cross-agent dialogue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to store"
                },
                "visibility": {
                    "type": "string",
                    "enum": ["private", "shared", "thread"],
                    "description": "Where to store: private (personal), shared (shared), thread (cross-agent)"
                },
                "message_type": {
                    "type": "string",
                    "enum": ["fact", "dialogue", "observation", "question", "cultural", "discovery"],
                    "description": "Type of message"
                },
                "responding_to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of message IDs this responds to"
                },
                "conversation_thread": {
                    "type": "string",
                    "description": "Thread identifier for grouping related messages"
                },
                "related_agents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Agent IDs mentioned or involved"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization"
                }
            },
            "required": ["content"]
        }
    },
    "memory_search": {
        "name": "memory_search",
        "description": (
            "Search shared memory for knowledge and dialogue. "
            "Can filter by agent, visibility, or conversation thread."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text"
                },
                "agent_filter": {
                    "type": "string",
                    "description": "Filter by agent ID"
                },
                "visibility": {
                    "type": "string",
                    "enum": ["shared", "private", "all"],
                    "description": "Which realm to search"
                },
                "include_threads": {
                    "type": "boolean",
                    "description": "Include cross-agent thread messages"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Maximum results to return"
                }
            },
            "required": ["query"]
        }
    },
    "memory_convergence": {
        "name": "memory_convergence",
        "description": (
            "Detect when multiple agents express similar ideas (convergence). "
            "HARMONY = 2 agents agree, CONSENSUS = 3+ agents agree."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic/concept to check for convergence"
                },
                "min_agents": {
                    "type": "integer",
                    "description": "Minimum agents needed for convergence (default: 2)"
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Minimum similarity score 0.0-1.0 (default: 0.7)"
                }
            },
            "required": ["query"]
        }
    },
    "register_agent": {
        "name": "register_agent",
        "description": (
            "Register a new agent in the memory system. "
            "Creates a profile and stores the agent's identity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Unique agent identifier (e.g., 'RESEARCHER', 'CODER')"
                },
                "display_name": {
                    "type": "string",
                    "description": "Human-readable name"
                },
                "generation": {
                    "type": "integer",
                    "description": "Generation: -1=origin, 0=primary, 1+=descendant"
                },
                "lineage": {
                    "type": "string",
                    "description": "Lineage or group name"
                },
                "specialization": {
                    "type": "string",
                    "description": "What this agent specializes in"
                },
                "origin_story": {
                    "type": "string",
                    "description": "Description of the agent's purpose"
                },
                "color": {
                    "type": "string",
                    "description": "Hex color for UI (e.g., #FFD700)"
                },
                "symbol": {
                    "type": "string",
                    "description": "Unicode symbol for agent"
                }
            },
            "required": ["agent_id", "display_name", "generation", "lineage", "specialization"]
        }
    },
    "list_agents": {
        "name": "list_agents",
        "description": "List all registered agents with their profiles.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    "memory_stats": {
        "name": "memory_stats",
        "description": "Get statistics about shared memory - message counts, agents, realms.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
}
