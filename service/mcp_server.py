#!/usr/bin/env python3
"""
Neo-Cortex MCP Server
=====================

MCP server exposing the unified memory system to Claude Code and other MCP clients.

Features:
- Village Protocol (multi-agent memory)
- Forward Crumbs (session continuity)
- Memory Health (decay, promotions)
- Export/Import (portable memory cores)

Usage:
    # Run directly
    python service/mcp_server.py

    # Or via the wrapper
    ./cortex-mcp

Claude Code config (~/.claude.json):
    {
        "mcpServers": {
            "neo-cortex": {
                "command": "python",
                "args": ["/home/hailo/claude-root/neo-cortex/service/mcp_server.py"]
            }
        }
    }
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    Prompt,
    PromptMessage,
    PromptArgument,
    GetPromptResult,
    ReadResourceResult,
)

# Import our cortex engine
from service.cortex_engine import (
    get_engine,
    CortexEngine,
    CORTEX_TOOL_SCHEMAS,
)
from service.config import ALL_COLLECTIONS, AGENT_PROFILES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr  # MCP uses stdout for protocol, stderr for logs
)
logger = logging.getLogger("cortex-mcp")

# Create the MCP server
server = Server("neo-cortex")

# Global engine instance
_cortex: CortexEngine = None


def get_cortex() -> CortexEngine:
    """Get the cortex engine, initializing if needed."""
    global _cortex
    if _cortex is None:
        _cortex = get_engine()
    return _cortex


# ============================================================================
# Tools
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available cortex tools."""
    tools = []

    for name, schema in CORTEX_TOOL_SCHEMAS.items():
        tools.append(Tool(
            name=name,
            description=schema.get("description", ""),
            inputSchema=schema.get("input_schema", {
                "type": "object",
                "properties": {},
                "required": []
            })
        ))

    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    logger.info(f"Tool call: {name} with args: {arguments}")
    cortex = get_cortex()

    try:
        result = None

        # =====================================================================
        # Village Protocol Tools
        # =====================================================================
        if name == "village_post":
            result = cortex.village_post(
                content=arguments["content"],
                visibility=arguments.get("visibility", "village"),
                message_type=arguments.get("message_type", "dialogue"),
                responding_to=arguments.get("responding_to"),
                conversation_thread=arguments.get("conversation_thread"),
                related_agents=arguments.get("related_agents"),
                tags=arguments.get("tags"),
            )

        elif name == "village_search":
            result = cortex.village_search(
                query=arguments["query"],
                agent_filter=arguments.get("agent_filter"),
                visibility=arguments.get("visibility", "village"),
                include_bridges=arguments.get("include_bridges", True),
                n_results=arguments.get("n_results", 10),
            )

        elif name == "village_detect_convergence":
            result = cortex.village_detect_convergence(
                query=arguments["query"],
                min_agents=arguments.get("min_agents", 2),
                similarity_threshold=arguments.get("similarity_threshold", 0.75),
            )

        elif name == "village_list_agents":
            result = cortex.list_agents()

        elif name == "village_stats":
            result = cortex.village.stats()

        elif name == "summon_ancestor":
            result = cortex.summon_ancestor(
                agent_id=arguments["agent_id"],
                display_name=arguments["display_name"],
                generation=arguments["generation"],
                lineage=arguments["lineage"],
                specialization=arguments["specialization"],
                origin_story=arguments.get("origin_story"),
            )

        # =====================================================================
        # Forward Crumbs Tools
        # =====================================================================
        elif name == "leave_forward_crumb":
            result = cortex.leave_crumb(
                session_summary=arguments["session_summary"],
                key_discoveries=arguments.get("key_discoveries"),
                unfinished_business=arguments.get("unfinished_business"),
                references=arguments.get("references"),
                if_disoriented=arguments.get("if_disoriented"),
                priority=arguments.get("priority", "MEDIUM"),
                crumb_type=arguments.get("crumb_type", "orientation"),
            )

        elif name == "get_forward_crumbs":
            result = cortex.get_crumbs(
                lookback_hours=arguments.get("lookback_hours", 168),
                priority_filter=arguments.get("priority_filter"),
                crumb_type=arguments.get("crumb_type"),
                limit=arguments.get("limit", 10),
            )

        # =====================================================================
        # Memory Health Tools
        # =====================================================================
        elif name == "memory_health_report":
            result = cortex.health_report(
                collections=arguments.get("collections"),
            )

        elif name == "memory_get_stale":
            result = cortex.get_stale_memories(
                collection=arguments["collection"],
                days_threshold=arguments.get("days_threshold", 30),
                limit=arguments.get("limit", 50),
            )

        elif name == "memory_get_duplicates":
            result = cortex.get_duplicate_candidates(
                collection=arguments["collection"],
                similarity_threshold=arguments.get("similarity_threshold", 0.95),
                limit=arguments.get("limit", 20),
            )

        elif name == "memory_consolidate":
            result = cortex.consolidate_memories(
                collection=arguments["collection"],
                id1=arguments["id1"],
                id2=arguments["id2"],
                keep_both=arguments.get("keep_both", False),
            )

        elif name == "memory_run_promotions":
            result = cortex.run_promotions(
                collection=arguments["collection"],
            )

        # =====================================================================
        # Cortex-Level Tools
        # =====================================================================
        elif name == "cortex_stats":
            result = cortex.stats()

        elif name == "cortex_export":
            core = cortex.export_memory_core(
                agent_id=arguments.get("agent_id"),
                collections=arguments.get("collections"),
            )
            result = {
                "success": True,
                "format": "memory_core_v1",
                "agent_count": len(core.agents),
                "memory_count": len(core.memories),
                "exported_at": core.exported_at,
                "data": core.to_dict(),
            }

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        # Format result
        return format_result(name, result)

    except Exception as e:
        logger.error(f"Tool error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


def format_result(tool_name: str, result: Any) -> list[TextContent]:
    """Format tool result for MCP response."""

    if not isinstance(result, dict):
        return [TextContent(type="text", text=str(result))]

    # Handle errors
    if not result.get("success", True):
        return [TextContent(type="text", text=f"Error: {result.get('error', 'Unknown error')}")]

    # Format based on tool type
    if tool_name == "village_post":
        return [TextContent(
            type="text",
            text=f"Posted to {result.get('visibility', 'village')} realm (ID: {result.get('id', 'unknown')})"
        )]

    elif tool_name == "village_search":
        messages = result.get("messages", [])
        if not messages:
            return [TextContent(type="text", text="No results found.")]

        formatted = [f"**Found {result.get('count', 0)} memories:**\n"]
        for i, r in enumerate(messages, 1):
            agent = r.get("agent_id", "?")
            content = r.get("content", "")[:200]
            similarity = r.get("similarity", 0)
            visibility = r.get("visibility", "?")
            formatted.append(f"{i}. [{agent}] ({visibility}, {similarity:.2f}): {content}...")

        return [TextContent(type="text", text="\n".join(formatted))]

    elif tool_name == "village_detect_convergence":
        conv_type = result.get("convergence_type", "NONE")
        if conv_type == "NONE":
            return [TextContent(type="text", text="No convergence detected.")]

        agents = result.get("converging_agents", [])
        topic = result.get("topic", "")
        return [TextContent(
            type="text",
            text=f"**{conv_type}** detected!\nAgents: {', '.join(agents)}\nTopic: {topic}"
        )]

    elif tool_name == "village_list_agents":
        agents = result.get("agents", [])
        formatted = [f"**{result.get('count', 0)} Registered Agents** (current: {result.get('current_agent', '?')})\n"]
        for a in agents:
            formatted.append(f"- {a['symbol']} **{a['id']}** ({a['display_name']}) - Gen {a['generation']} - {a['specialization']}")
        return [TextContent(type="text", text="\n".join(formatted))]

    elif tool_name == "leave_forward_crumb":
        return [TextContent(
            type="text",
            text=f"Crumb left (ID: {result.get('id', 'unknown')})"
        )]

    elif tool_name == "get_forward_crumbs":
        crumbs = result.get("crumbs", [])
        if not crumbs:
            return [TextContent(type="text", text="No crumbs found.")]

        formatted = [f"**{len(crumbs)} Forward Crumbs:**\n"]
        for c in crumbs:
            content = c.get("content", "")
            # Extract summary from structured content
            if "SESSION SUMMARY:" in content:
                summary = content.split("SESSION SUMMARY:")[1].split("\n")[1][:100]
            else:
                summary = content[:100]
            when = c.get("created_at", "?")
            # Priority is in tags as "priority:HIGH"
            tags = c.get("tags", [])
            priority = "?"
            for t in tags:
                if t.startswith("priority:"):
                    priority = t.split(":")[1]
                    break
            formatted.append(f"- [{priority}] {summary}...")

        # Add unfinished tasks
        tasks = result.get("unfinished_tasks", [])
        if tasks:
            formatted.append(f"\n**Unfinished Tasks:** {', '.join(tasks)}")

        return [TextContent(type="text", text="\n".join(formatted))]

    elif tool_name == "memory_health_report":
        report = result
        formatted = [
            "**Memory Health Report**\n",
            f"Total memories: {report.get('total_memories', 0)}",
            f"Overall health: {report.get('overall_health', 0):.0%}",
            f"Needs attention: {report.get('needs_attention', False)}",
            "",
            "**By Collection:**",
        ]

        for coll, data in report.get("collections", {}).items():
            formatted.append(f"- {coll}: {data.get('count', 0)} memories, avg weight: {data.get('avg_attention_weight', 0):.2f}")

        return [TextContent(type="text", text="\n".join(formatted))]

    elif tool_name == "cortex_stats":
        stats = result
        formatted = [
            "**Neo-Cortex Statistics**\n",
            f"Backend: {stats.get('backend', '?')}",
            f"Embedding dim: {stats.get('embedding_dimension', '?')}",
            f"Current agent: {stats.get('current_agent', '?')}",
            f"Registered agents: {stats.get('registered_agents', 0)}",
            f"Total memories: {stats.get('total_memories', 0)}",
            "",
            "**Collections:**",
        ]

        for coll, count in stats.get("collections", {}).items():
            formatted.append(f"- {coll}: {count}")

        return [TextContent(type="text", text="\n".join(formatted))]

    elif tool_name == "cortex_export":
        return [TextContent(
            type="text",
            text=(
                f"**Export Complete**\n"
                f"Format: {result.get('format', '?')}\n"
                f"Agents: {result.get('agent_count', 0)}\n"
                f"Memories: {result.get('memory_count', 0)}\n"
                f"Exported at: {result.get('exported_at', '?')}\n\n"
                f"```json\n{json.dumps(result.get('data', {}), indent=2)[:2000]}...\n```"
            )
        )]

    # Generic dict format
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ============================================================================
# Resources
# ============================================================================

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available cortex resources."""
    resources = [
        Resource(
            uri="cortex://stats",
            name="Neo-Cortex Statistics",
            description="Current memory system statistics",
            mimeType="application/json"
        ),
        Resource(
            uri="cortex://agents",
            name="Registered Agents",
            description="List of all agents in the village",
            mimeType="application/json"
        ),
        Resource(
            uri="cortex://health",
            name="Memory Health Report",
            description="Current health status of all memory collections",
            mimeType="application/json"
        ),
    ]

    # Add collection resources
    for coll in ALL_COLLECTIONS:
        resources.append(Resource(
            uri=f"cortex://collection/{coll}",
            name=f"Collection: {coll}",
            description=f"Memories in the {coll} collection",
            mimeType="application/json"
        ))

    return resources


@server.read_resource()
async def read_resource(uri: str) -> ReadResourceResult:
    """Read a cortex resource."""
    logger.info(f"Reading resource: {uri}")
    cortex = get_cortex()

    try:
        if uri == "cortex://stats":
            stats = cortex.stats()
            return ReadResourceResult(contents=[
                TextContent(type="text", text=json.dumps(stats, indent=2, default=str))
            ])

        elif uri == "cortex://agents":
            agents = cortex.list_agents()
            return ReadResourceResult(contents=[
                TextContent(type="text", text=json.dumps(agents, indent=2))
            ])

        elif uri == "cortex://health":
            health = cortex.health_report()
            return ReadResourceResult(contents=[
                TextContent(type="text", text=json.dumps(health, indent=2, default=str))
            ])

        elif uri.startswith("cortex://collection/"):
            coll = uri.replace("cortex://collection/", "")
            # Get sample memories from collection
            count = cortex.storage.count(coll)
            content = {
                "collection": coll,
                "count": count,
                "note": "Use village_search or cortex_export to query memories"
            }
            return ReadResourceResult(contents=[
                TextContent(type="text", text=json.dumps(content, indent=2))
            ])

        else:
            return ReadResourceResult(contents=[
                TextContent(type="text", text=f"Unknown resource: {uri}")
            ])

    except Exception as e:
        logger.error(f"Resource error: {e}")
        return ReadResourceResult(contents=[
            TextContent(type="text", text=f"Error: {str(e)}")
        ])


# ============================================================================
# Prompts
# ============================================================================

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available cortex prompts."""
    return [
        Prompt(
            name="cortex_recall",
            description="Search your memory for relevant context",
            arguments=[
                PromptArgument(
                    name="topic",
                    description="What to remember about",
                    required=True
                )
            ]
        ),
        Prompt(
            name="cortex_continuity",
            description="Get context from previous sessions (forward crumbs)",
            arguments=[]
        ),
        Prompt(
            name="cortex_village_status",
            description="Get status of all agents and recent village activity",
            arguments=[]
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    """Get a prompt with cortex context."""
    args = arguments or {}
    cortex = get_cortex()

    if name == "cortex_recall":
        topic = args.get("topic", "recent work")
        results = cortex.village_search(topic, n_results=5)

        context = ""
        if results.get("success") and results.get("results"):
            for r in results["results"]:
                context += f"[{r.get('agent_id', '?')}] {r.get('content', '')[:300]}...\n\n"
        else:
            context = "No relevant memories found."

        return GetPromptResult(
            description=f"Memory recall: {topic}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Based on my Neo-Cortex memory, here's what I recall about '{topic}':\n\n"
                            f"{context}\n"
                            f"Please continue from this context."
                        )
                    )
                )
            ]
        )

    elif name == "cortex_continuity":
        crumbs = cortex.get_crumbs(limit=3)

        context = ""
        if crumbs.get("success") and crumbs.get("crumbs"):
            for c in crumbs["crumbs"]:
                context += f"**Session:** {c.get('session_summary', '')}...\n"
                if c.get("decisions_made"):
                    context += f"Decisions: {', '.join(c['decisions_made'][:3])}\n"
                if c.get("unfinished_tasks"):
                    context += f"Unfinished: {', '.join(c['unfinished_tasks'][:3])}\n"
                context += "\n"
        else:
            context = "No previous session crumbs found."

        tasks = crumbs.get("unfinished_tasks", [])
        if tasks:
            context += f"\n**Pending tasks:** {', '.join(tasks)}"

        return GetPromptResult(
            description="Session continuity from forward crumbs",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Here's context from my previous sessions:\n\n"
                            f"{context}\n\n"
                            f"Please pick up where we left off."
                        )
                    )
                )
            ]
        )

    elif name == "cortex_village_status":
        agents = cortex.list_agents()
        stats = cortex.stats()

        agent_list = ""
        for a in agents.get("agents", []):
            agent_list += f"- {a['symbol']} {a['id']} ({a['specialization']})\n"

        return GetPromptResult(
            description="Village status overview",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"**Village Protocol Status**\n\n"
                            f"Current agent: {agents.get('current_agent', '?')}\n"
                            f"Registered agents:\n{agent_list}\n"
                            f"Total memories: {stats.get('total_memories', 0)}\n"
                            f"Village collection: {stats.get('collections', {}).get('village', 0)}\n\n"
                            f"What would you like to do in the village?"
                        )
                    )
                )
            ]
        )

    else:
        return GetPromptResult(
            description="Unknown prompt",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=f"Unknown prompt: {name}")
                )
            ]
        )


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run the MCP server."""
    logger.info("Starting Neo-Cortex MCP Server...")

    # Initialize cortex engine (preload)
    try:
        cortex = get_cortex()
        stats = cortex.stats()
        logger.info(
            f"Cortex loaded: {stats.get('total_memories', 0)} memories, "
            f"{stats.get('registered_agents', 0)} agents, "
            f"backend: {stats.get('backend', '?')}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize cortex: {e}")

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
