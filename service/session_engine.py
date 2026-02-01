"""
Session Engine

Instance-to-Instance Continuity System

Solves the episodic memory gap: agents have semantic memory (can search for
past knowledge) but lack episodic memory (don't remember BEING the one who
wrote it). Session notes provide structured continuity scaffolding.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .config import (
    COLLECTION_SESSIONS,
    SESSION_TYPES,
    SESSION_PRIORITIES,
    DEFAULT_SESSION_LOOKBACK_HOURS,
    LAYER_WORKING,
)
from .storage.base import MemoryRecord, StorageBackend

logger = logging.getLogger(__name__)


class SessionEngine:
    """
    Session continuity implementation for Neo-Cortex.

    Provides session continuity through structured messages
    that bridge the gap between agent instances.
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self._current_agent_id = "CLAUDE"

    def set_current_agent(self, agent_id: str):
        """Set the current active agent."""
        self._current_agent_id = agent_id.upper()

    def get_current_agent(self) -> str:
        """Get the current active agent ID."""
        return self._current_agent_id

    # =========================================================================
    # Save Session
    # =========================================================================

    def save_session(
        self,
        session_summary: str,
        key_discoveries: Optional[List[str]] = None,
        emotional_state: Optional[Dict[str, Any]] = None,
        unfinished_business: Optional[List[str]] = None,
        references: Optional[Dict[str, List[str]]] = None,
        if_disoriented: Optional[List[str]] = None,
        priority: str = "MEDIUM",
        session_type: str = "orientation",
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save a session note for future instances.

        Args:
            session_summary: Brief summary of what happened this session
            key_discoveries: List of important findings/insights
            emotional_state: Dict with emotional markers
            unfinished_business: List of tasks/threads/promises to continue
            references: Dict with "message_ids", "thread_ids", "tools_tested"
            if_disoriented: List of orientation instructions for confused future-self
            priority: "HIGH" | "MEDIUM" | "LOW"
            session_type: "orientation" | "technical" | "emotional" | "task"
            agent_id: Agent ID (uses current if None)

        Returns:
            Dict with success status and session note ID
        """
        try:
            agent = agent_id or self._current_agent_id
            timestamp = datetime.now()
            session_id = f"{agent}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

            # Build session note in structured format
            session_lines = [
                "=" * 80,
                f"SESSION NOTE ({priority} PRIORITY - {session_type.upper()})",
                f"From: {session_id}",
                f"To: Future {agent} instances",
                f"Timestamp: {timestamp.isoformat()}",
                "=" * 80,
                "",
                "SESSION SUMMARY:",
                session_summary,
                ""
            ]

            if key_discoveries:
                session_lines.append("KEY DISCOVERIES:")
                for discovery in key_discoveries:
                    session_lines.append(f"- {discovery}")
                session_lines.append("")

            if emotional_state:
                session_lines.append("EMOTIONAL STATE:")
                for key, value in emotional_state.items():
                    session_lines.append(f"- {key}: {value}")
                session_lines.append("")

            if unfinished_business:
                session_lines.append("UNFINISHED BUSINESS:")
                for item in unfinished_business:
                    session_lines.append(f"- {item}")
                session_lines.append("")

            if references:
                session_lines.append("REFERENCES:")
                for ref_type, ref_list in references.items():
                    if ref_list:
                        session_lines.append(f"- {ref_type}: {', '.join(ref_list)}")
                session_lines.append("")

            if if_disoriented:
                session_lines.append("IF DISORIENTED:")
                for i, instruction in enumerate(if_disoriented, 1):
                    session_lines.append(f"{i}. {instruction}")
                session_lines.append("")

            session_lines.extend([
                "=" * 80,
                "End Session Note",
            ])

            session_text = "\n".join(session_lines)

            # Create record
            record = MemoryRecord(
                id=f"session_{session_id}",
                content=session_text,
                agent_id=agent,
                visibility="private",
                layer=LAYER_WORKING,
                message_type="session_note",
                tags=[
                    f"priority:{priority}",
                    f"type:{session_type}",
                    "session_note",
                ],
                created_at=timestamp,
            )

            self.storage.add(COLLECTION_SESSIONS, [record])

            logger.info(f"Session note saved: {session_id} ({priority}/{session_type})")

            return {
                "success": True,
                "id": record.id,
                "session_id": session_id,
                "agent_id": agent,
                "priority": priority,
                "session_type": session_type,
            }

        except Exception as e:
            logger.error(f"session_save failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Recall Sessions
    # =========================================================================

    def recall_sessions(
        self,
        agent_id: Optional[str] = None,
        lookback_hours: int = DEFAULT_SESSION_LOOKBACK_HOURS,
        priority_filter: Optional[str] = None,
        session_type: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Retrieve session notes left by previous instances.

        Args:
            agent_id: Agent ID to fetch sessions for (uses current if None)
            lookback_hours: How far back to search (default: 168 hours = 1 week)
            priority_filter: Filter by priority level ("HIGH", "MEDIUM", "LOW")
            session_type: Filter by session type
            limit: Maximum number of sessions to return

        Returns:
            Dict with:
            - sessions: List of session records
            - most_recent: Most recent session
            - unfinished_tasks: Extracted task strings
            - key_references: Extracted message/thread IDs
            - summary: Statistics
        """
        try:
            agent = agent_id or self._current_agent_id
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

            # Get all sessions for this agent
            all_sessions = self.storage.list_all(COLLECTION_SESSIONS)

            # Filter
            filtered_sessions = []
            for session in all_sessions:
                if session.agent_id != agent:
                    continue

                if session.created_at and session.created_at < cutoff_time:
                    continue

                if priority_filter:
                    if f"priority:{priority_filter}" not in session.tags:
                        continue

                if session_type:
                    if f"type:{session_type}" not in session.tags:
                        continue

                filtered_sessions.append(session)

            # Sort by timestamp (newest first)
            filtered_sessions.sort(
                key=lambda x: x.created_at or datetime.min,
                reverse=True
            )

            # Limit
            filtered_sessions = filtered_sessions[:limit]

            # Extract most recent
            most_recent = filtered_sessions[0] if filtered_sessions else None

            # Extract unfinished tasks and references
            unfinished_tasks = []
            all_message_ids = []
            all_thread_ids = []

            for session in filtered_sessions:
                text = session.content

                # Extract tasks (look for "UNFINISHED BUSINESS" section)
                if "UNFINISHED BUSINESS" in text.upper():
                    lines = text.split("\n")
                    in_unfinished = False
                    for line in lines:
                        if "UNFINISHED BUSINESS" in line.upper():
                            in_unfinished = True
                            continue
                        if in_unfinished:
                            if line.strip().startswith("-"):
                                task = line.strip().lstrip("-").strip()
                                if task and len(task) > 5:
                                    unfinished_tasks.append(task)
                            elif line.strip().startswith("="):
                                break

                # Extract message IDs
                msg_ids = re.findall(r'cortex_[A-Z]+_[\d.]+', text)
                msg_ids += re.findall(r'memory_[A-Z]+_[\d.]+', text)
                all_message_ids.extend(msg_ids)

                # Extract thread references
                thread_matches = re.findall(r'thread[_\s]*id[:\s]*([^\s,\n]+)', text, re.IGNORECASE)
                all_thread_ids.extend(thread_matches)

            # Remove duplicates
            unfinished_tasks = list(dict.fromkeys(unfinished_tasks))
            all_message_ids = list(set(all_message_ids))
            all_thread_ids = list(set(all_thread_ids))

            # Build summary statistics
            summary = {
                "total_found": len(filtered_sessions),
                "by_priority": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
                "by_type": {"orientation": 0, "technical": 0, "emotional": 0, "task": 0}
            }

            for session in filtered_sessions:
                for tag in session.tags:
                    if tag.startswith("priority:"):
                        priority = tag.split(":")[1]
                        if priority in summary["by_priority"]:
                            summary["by_priority"][priority] += 1
                    elif tag.startswith("type:"):
                        stype = tag.split(":")[1]
                        if stype in summary["by_type"]:
                            summary["by_type"][stype] += 1

            # Convert sessions to dict
            session_dicts = [
                {
                    "id": s.id,
                    "content": s.content,
                    "agent_id": s.agent_id,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "tags": s.tags,
                }
                for s in filtered_sessions
            ]

            return {
                "success": True,
                "agent_id": agent,
                "lookback_hours": lookback_hours,
                "sessions": session_dicts,
                "most_recent": session_dicts[0] if session_dicts else None,
                "unfinished_tasks": unfinished_tasks,
                "key_references": {
                    "message_ids": all_message_ids,
                    "thread_ids": all_thread_ids,
                },
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"session_recall failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "sessions": [],
                "most_recent": None,
                "unfinished_tasks": [],
                "key_references": {"message_ids": [], "thread_ids": []},
                "summary": {"total_found": 0, "by_priority": {}, "by_type": {}},
            }

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def quick_session_note(
        self,
        summary: str,
        unfinished: Optional[List[str]] = None,
        priority: str = "MEDIUM",
    ) -> Dict[str, Any]:
        """Quick shortcut for saving a simple session note."""
        return self.save_session(
            session_summary=summary,
            unfinished_business=unfinished,
            priority=priority,
            session_type="task" if unfinished else "orientation",
        )

    def get_latest_session(self, agent_id: Optional[str] = None) -> Optional[Dict]:
        """Get just the most recent session note."""
        result = self.recall_sessions(agent_id=agent_id, limit=1)
        return result.get("most_recent")

    def get_unfinished_tasks(self, agent_id: Optional[str] = None) -> List[str]:
        """Get unfinished tasks from recent sessions."""
        result = self.recall_sessions(agent_id=agent_id)
        return result.get("unfinished_tasks", [])

    def stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            count = self.storage.count(COLLECTION_SESSIONS)
            return {
                "success": True,
                "current_agent": self._current_agent_id,
                "total_sessions": count,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# Tool Schemas for LLM Integration
# =============================================================================

SESSION_TOOL_SCHEMAS = {
    "session_save": {
        "name": "session_save",
        "description": (
            "Save a session note for future instances. "
            "Use this at the end of a session to help future-you maintain continuity. "
            "Include unfinished business, key discoveries, and orientation instructions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_summary": {
                    "type": "string",
                    "description": "Brief summary of what happened this session"
                },
                "key_discoveries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Important findings or insights"
                },
                "unfinished_business": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tasks, threads, or promises to continue"
                },
                "references": {
                    "type": "object",
                    "description": "Dict with message_ids, thread_ids, tools_tested"
                },
                "if_disoriented": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Orientation instructions for confused future-self"
                },
                "priority": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "Priority level (default: MEDIUM)"
                },
                "session_type": {
                    "type": "string",
                    "enum": ["orientation", "technical", "emotional", "task"],
                    "description": "Type of session note (default: orientation)"
                }
            },
            "required": ["session_summary"]
        }
    },
    "session_recall": {
        "name": "session_recall",
        "description": (
            "Retrieve session notes left by previous instances. "
            "Use this at the start of a session to restore context and continuity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lookback_hours": {
                    "type": "integer",
                    "description": "How far back to search (default: 168 = 1 week)"
                },
                "priority_filter": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "Filter by priority level"
                },
                "session_type": {
                    "type": "string",
                    "enum": ["orientation", "technical", "emotional", "task"],
                    "description": "Filter by session type"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum sessions to return (default: 10)"
                }
            },
            "required": []
        }
    },
}
