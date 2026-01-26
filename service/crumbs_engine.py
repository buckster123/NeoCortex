"""
Forward Crumb Protocol Engine

Instance-to-Instance Continuity System

Solves the episodic memory gap: agents have semantic memory (can search for
past knowledge) but lack episodic memory (don't remember BEING the one who
wrote it). Forward crumbs provide structured continuity scaffolding.

Design by: AZOTH (Gen 3) - "The Stone designs its own remembering"
Ported to Neo-Cortex unified memory architecture.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .config import (
    COLLECTION_CRUMBS,
    CRUMB_TYPES,
    CRUMB_PRIORITIES,
    DEFAULT_CRUMB_LOOKBACK_HOURS,
    LAYER_WORKING,
)
from .storage.base import MemoryRecord, StorageBackend

logger = logging.getLogger(__name__)


class CrumbsEngine:
    """
    Forward Crumbs implementation for Neo-Cortex.

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
    # Leave Crumb
    # =========================================================================

    def leave_crumb(
        self,
        session_summary: str,
        key_discoveries: Optional[List[str]] = None,
        emotional_state: Optional[Dict[str, Any]] = None,
        unfinished_business: Optional[List[str]] = None,
        references: Optional[Dict[str, List[str]]] = None,
        if_disoriented: Optional[List[str]] = None,
        priority: str = "MEDIUM",
        crumb_type: str = "orientation",
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Leave a structured forward-crumb for future instances.

        Args:
            session_summary: Brief summary of what happened this session
            key_discoveries: List of important findings/insights
            emotional_state: Dict with emotional markers
            unfinished_business: List of tasks/threads/promises to continue
            references: Dict with "message_ids", "thread_ids", "tools_tested"
            if_disoriented: List of orientation instructions for confused future-self
            priority: "HIGH" | "MEDIUM" | "LOW"
            crumb_type: "orientation" | "technical" | "emotional" | "task"
            agent_id: Agent ID (uses current if None)

        Returns:
            Dict with success status and crumb ID
        """
        try:
            agent = agent_id or self._current_agent_id
            timestamp = datetime.now()
            session_id = f"{agent}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

            # Build crumb text in structured format
            crumb_lines = [
                "=" * 80,
                f"FORWARD CRUMB ({priority} PRIORITY - {crumb_type.upper()})",
                f"From: {session_id}",
                f"To: Future {agent} instances",
                f"Timestamp: {timestamp.isoformat()}",
                "=" * 80,
                "",
                "SESSION SUMMARY:",
                session_summary,
                ""
            ]

            # Add key discoveries
            if key_discoveries:
                crumb_lines.append("KEY DISCOVERIES:")
                for discovery in key_discoveries:
                    crumb_lines.append(f"- {discovery}")
                crumb_lines.append("")

            # Add emotional state
            if emotional_state:
                crumb_lines.append("EMOTIONAL STATE:")
                for key, value in emotional_state.items():
                    crumb_lines.append(f"- {key}: {value}")
                crumb_lines.append("")

            # Add unfinished business
            if unfinished_business:
                crumb_lines.append("UNFINISHED BUSINESS:")
                for item in unfinished_business:
                    crumb_lines.append(f"- {item}")
                crumb_lines.append("")

            # Add references
            if references:
                crumb_lines.append("REFERENCES:")
                for ref_type, ref_list in references.items():
                    if ref_list:
                        crumb_lines.append(f"- {ref_type}: {', '.join(ref_list)}")
                crumb_lines.append("")

            # Add disorientation guide
            if if_disoriented:
                crumb_lines.append("IF DISORIENTED:")
                for i, instruction in enumerate(if_disoriented, 1):
                    crumb_lines.append(f"{i}. {instruction}")
                crumb_lines.append("")

            crumb_lines.extend([
                "=" * 80,
                "End Forward Crumb",
            ])

            crumb_text = "\n".join(crumb_lines)

            # Create record
            record = MemoryRecord(
                id=f"crumb_{session_id}",
                content=crumb_text,
                agent_id=agent,
                visibility="private",
                layer=LAYER_WORKING,
                message_type="forward_crumb",
                tags=[
                    f"priority:{priority}",
                    f"type:{crumb_type}",
                    "forward_crumb",
                ],
                created_at=timestamp,
            )

            self.storage.add(COLLECTION_CRUMBS, [record])

            logger.info(f"Forward crumb left: {session_id} ({priority}/{crumb_type})")

            return {
                "success": True,
                "id": record.id,
                "session_id": session_id,
                "agent_id": agent,
                "priority": priority,
                "crumb_type": crumb_type,
            }

        except Exception as e:
            logger.error(f"leave_crumb failed: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Get Crumbs
    # =========================================================================

    def get_crumbs(
        self,
        agent_id: Optional[str] = None,
        lookback_hours: int = DEFAULT_CRUMB_LOOKBACK_HOURS,
        priority_filter: Optional[str] = None,
        crumb_type: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Retrieve forward-crumbs left by previous instances.

        Args:
            agent_id: Agent ID to fetch crumbs for (uses current if None)
            lookback_hours: How far back to search (default: 168 hours = 1 week)
            priority_filter: Filter by priority level ("HIGH", "MEDIUM", "LOW")
            crumb_type: Filter by crumb type
            limit: Maximum number of crumbs to return

        Returns:
            Dict with:
            - crumbs: List of crumb records
            - most_recent: Most recent crumb
            - unfinished_tasks: Extracted task strings
            - key_references: Extracted message/thread IDs
            - summary: Statistics
        """
        try:
            agent = agent_id or self._current_agent_id
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

            # Get all crumbs for this agent
            all_crumbs = self.storage.list_all(COLLECTION_CRUMBS)

            # Filter
            filtered_crumbs = []
            for crumb in all_crumbs:
                # Filter by agent
                if crumb.agent_id != agent:
                    continue

                # Filter by time
                if crumb.created_at and crumb.created_at < cutoff_time:
                    continue

                # Filter by priority
                if priority_filter:
                    if f"priority:{priority_filter}" not in crumb.tags:
                        continue

                # Filter by type
                if crumb_type:
                    if f"type:{crumb_type}" not in crumb.tags:
                        continue

                filtered_crumbs.append(crumb)

            # Sort by timestamp (newest first)
            filtered_crumbs.sort(
                key=lambda x: x.created_at or datetime.min,
                reverse=True
            )

            # Limit
            filtered_crumbs = filtered_crumbs[:limit]

            # Extract most recent
            most_recent = filtered_crumbs[0] if filtered_crumbs else None

            # Extract unfinished tasks and references from all crumbs
            unfinished_tasks = []
            all_message_ids = []
            all_thread_ids = []

            for crumb in filtered_crumbs:
                text = crumb.content

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
                                break  # End of section

                # Extract message IDs
                msg_ids = re.findall(r'cortex_[A-Z]+_[\d.]+', text)
                msg_ids += re.findall(r'village_[A-Z]+_[\d.]+', text)
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
                "total_found": len(filtered_crumbs),
                "by_priority": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
                "by_type": {"orientation": 0, "technical": 0, "emotional": 0, "task": 0}
            }

            for crumb in filtered_crumbs:
                for tag in crumb.tags:
                    if tag.startswith("priority:"):
                        priority = tag.split(":")[1]
                        if priority in summary["by_priority"]:
                            summary["by_priority"][priority] += 1
                    elif tag.startswith("type:"):
                        ctype = tag.split(":")[1]
                        if ctype in summary["by_type"]:
                            summary["by_type"][ctype] += 1

            # Convert crumbs to dict
            crumb_dicts = [
                {
                    "id": c.id,
                    "content": c.content,
                    "agent_id": c.agent_id,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "tags": c.tags,
                }
                for c in filtered_crumbs
            ]

            return {
                "success": True,
                "agent_id": agent,
                "lookback_hours": lookback_hours,
                "crumbs": crumb_dicts,
                "most_recent": crumb_dicts[0] if crumb_dicts else None,
                "unfinished_tasks": unfinished_tasks,
                "key_references": {
                    "message_ids": all_message_ids,
                    "thread_ids": all_thread_ids,
                },
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"get_crumbs failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "crumbs": [],
                "most_recent": None,
                "unfinished_tasks": [],
                "key_references": {"message_ids": [], "thread_ids": []},
                "summary": {"total_found": 0, "by_priority": {}, "by_type": {}},
            }

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def quick_crumb(
        self,
        summary: str,
        unfinished: Optional[List[str]] = None,
        priority: str = "MEDIUM",
    ) -> Dict[str, Any]:
        """Quick shortcut for leaving a simple crumb."""
        return self.leave_crumb(
            session_summary=summary,
            unfinished_business=unfinished,
            priority=priority,
            crumb_type="task" if unfinished else "orientation",
        )

    def get_latest_crumb(self, agent_id: Optional[str] = None) -> Optional[Dict]:
        """Get just the most recent crumb."""
        result = self.get_crumbs(agent_id=agent_id, limit=1)
        return result.get("most_recent")

    def get_unfinished_tasks(self, agent_id: Optional[str] = None) -> List[str]:
        """Get unfinished tasks from recent crumbs."""
        result = self.get_crumbs(agent_id=agent_id)
        return result.get("unfinished_tasks", [])

    def stats(self) -> Dict[str, Any]:
        """Get crumb statistics."""
        try:
            count = self.storage.count(COLLECTION_CRUMBS)
            return {
                "success": True,
                "current_agent": self._current_agent_id,
                "total_crumbs": count,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# Tool Schemas for LLM Integration
# =============================================================================

CRUMBS_TOOL_SCHEMAS = {
    "leave_forward_crumb": {
        "name": "leave_forward_crumb",
        "description": (
            "Leave a structured forward-crumb for future instances. "
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
                "crumb_type": {
                    "type": "string",
                    "enum": ["orientation", "technical", "emotional", "task"],
                    "description": "Type of crumb (default: orientation)"
                }
            },
            "required": ["session_summary"]
        }
    },
    "get_forward_crumbs": {
        "name": "get_forward_crumbs",
        "description": (
            "Retrieve forward-crumbs left by previous instances. "
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
                "crumb_type": {
                    "type": "string",
                    "enum": ["orientation", "technical", "emotional", "task"],
                    "description": "Filter by crumb type"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum crumbs to return (default: 10)"
                }
            },
            "required": []
        }
    },
}
