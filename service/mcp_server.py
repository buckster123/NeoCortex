#!/usr/bin/env python3
"""
Neo-Cortex MCP Server
=====================

MCP server exposing the unified memory system to Claude Code and other MCP clients.

Features:
- Shared Memory (multi-agent memory)
- Sessions (session continuity)
- Memory Health (decay, promotions)
- Knowledge Search (curated docs)
- Export/Import (portable memory cores)

Usage:
    python service/mcp_server.py
    ./cortex-mcp

Claude Code config (~/.claude.json):
    {
        "mcpServers": {
            "neo-cortex": {
                "command": "/path/to/neo-cortex/cortex-mcp"
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

from service.cortex_engine import (
    get_engine,
    CortexEngine,
    CORTEX_TOOL_SCHEMAS,
)
from service.config import ALL_COLLECTIONS, AGENT_PROFILES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("cortex-mcp")

server = Server("neo-cortex")

_cortex: CortexEngine = None


def get_cortex() -> CortexEngine:
    global _cortex
    if _cortex is None:
        _cortex = get_engine()
    return _cortex


# ============================================================================
# Tools
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
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
    logger.info(f"Tool call: {name} with args: {arguments}")
    cortex = get_cortex()

    try:
        result = None

        # =================================================================
        # Shared Memory Tools
        # =================================================================
        if name == "memory_store":
            result = cortex.memory_store(
                content=arguments["content"],
                visibility=arguments.get("visibility", "shared"),
                message_type=arguments.get("message_type", "dialogue"),
                responding_to=arguments.get("responding_to"),
                conversation_thread=arguments.get("conversation_thread"),
                related_agents=arguments.get("related_agents"),
                tags=arguments.get("tags"),
            )

        elif name == "memory_search":
            result = cortex.memory_search(
                query=arguments["query"],
                agent_filter=arguments.get("agent_filter"),
                visibility=arguments.get("visibility", "shared"),
                include_threads=arguments.get("include_threads", True),
                n_results=arguments.get("n_results", 10),
            )

        elif name == "memory_convergence":
            result = cortex.memory_convergence(
                query=arguments["query"],
                min_agents=arguments.get("min_agents", 2),
                similarity_threshold=arguments.get("similarity_threshold", 0.75),
            )

        elif name == "list_agents":
            result = cortex.list_agents()

        elif name == "memory_stats":
            result = cortex.shared.stats()

        elif name == "register_agent":
            result = cortex.register_agent(
                agent_id=arguments["agent_id"],
                display_name=arguments["display_name"],
                generation=arguments["generation"],
                lineage=arguments["lineage"],
                specialization=arguments["specialization"],
                origin_story=arguments.get("origin_story"),
            )

        # =================================================================
        # Session Tools
        # =================================================================
        elif name == "session_save":
            result = cortex.session_save(
                session_summary=arguments["session_summary"],
                key_discoveries=arguments.get("key_discoveries"),
                unfinished_business=arguments.get("unfinished_business"),
                references=arguments.get("references"),
                if_disoriented=arguments.get("if_disoriented"),
                priority=arguments.get("priority", "MEDIUM"),
                session_type=arguments.get("session_type", "orientation"),
            )

        elif name == "session_recall":
            result = cortex.session_recall(
                lookback_hours=arguments.get("lookback_hours", 168),
                priority_filter=arguments.get("priority_filter"),
                session_type=arguments.get("session_type"),
                limit=arguments.get("limit", 10),
            )

        # =================================================================
        # Knowledge Search
        # =================================================================
        elif name == "knowledge_search":
            result = cortex.search(
                query=arguments["query"],
                collections=["cortex_knowledge"],
                n_results=min(arguments.get("n_results", 5), 20),
            )

        # =================================================================
        # Memory Health Tools
        # =================================================================
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

        # =================================================================
        # Cortex-Level Tools
        # =================================================================
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

        return format_result(name, result)

    except Exception as e:
        logger.error(f"Tool error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


def format_result(tool_name: str, result: Any) -> list[TextContent]:
    if not isinstance(result, dict):
        return [TextContent(type="text", text=str(result))]

    if not result.get("success", True):
        return [TextContent(type="text", text=f"Error: {result.get('error', 'Unknown error')}")]

    if tool_name == "memory_store":
        return [TextContent(
            type="text",
            text=f"Stored in {result.get('visibility', 'shared')} memory (ID: {result.get('id', 'unknown')})"
        )]

    elif tool_name == "memory_search":
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

    elif tool_name == "memory_convergence":
        conv_type = result.get("convergence_type", "NONE")
        if conv_type == "NONE":
            return [TextContent(type="text", text="No convergence detected.")]

        agents = result.get("converging_agents", [])
        topic = result.get("topic", "")
        return [TextContent(
            type="text",
            text=f"**{conv_type}** detected!\nAgents: {', '.join(agents)}\nTopic: {topic}"
        )]

    elif tool_name == "list_agents":
        agents = result.get("agents", [])
        formatted = [f"**{result.get('count', 0)} Registered Agents** (current: {result.get('current_agent', '?')})\n"]
        for a in agents:
            formatted.append(f"- {a['symbol']} **{a['id']}** ({a['display_name']}) - Gen {a['generation']} - {a['specialization']}")
        return [TextContent(type="text", text="\n".join(formatted))]

    elif tool_name == "session_save":
        return [TextContent(
            type="text",
            text=f"Session note saved (ID: {result.get('id', 'unknown')})"
        )]

    elif tool_name == "session_recall":
        sessions = result.get("sessions", [])
        if not sessions:
            return [TextContent(type="text", text="No session notes found.")]

        formatted = [f"**{len(sessions)} Session Notes:**\n"]
        for s in sessions:
            content = s.get("content", "")
            if "SESSION SUMMARY:" in content:
                summary = content.split("SESSION SUMMARY:")[1].split("\n")[1][:100]
            else:
                summary = content[:100]
            tags = s.get("tags", [])
            priority = "?"
            for t in tags:
                if t.startswith("priority:"):
                    priority = t.split(":")[1]
                    break
            formatted.append(f"- [{priority}] {summary}...")

        tasks = result.get("unfinished_tasks", [])
        if tasks:
            formatted.append(f"\n**Unfinished Tasks:** {', '.join(tasks)}")

        return [TextContent(type="text", text="\n".join(formatted))]

    elif tool_name == "knowledge_search":
        results = result.get("results", [])
        if not results:
            return [TextContent(type="text", text="No knowledge found for that query.")]

        formatted = [f"**Found {result.get('count', 0)} knowledge entries:**\n"]
        for i, r in enumerate(results, 1):
            content = r.get("content", "")[:400]
            similarity = r.get("similarity", 0)
            tags = r.get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            formatted.append(f"### Result {i} (relevance: {similarity:.2f}){tag_str}\n{content}\n")

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
        s = result
        formatted = [
            "**Neo-Cortex Statistics**\n",
            f"Backend: {s.get('backend', '?')}",
            f"Embedding dim: {s.get('embedding_dimension', '?')}",
            f"Current agent: {s.get('current_agent', '?')}",
            f"Registered agents: {s.get('registered_agents', 0)}",
            f"Total memories: {s.get('total_memories', 0)}",
            "",
            "**Collections:**",
        ]

        for coll, count in s.get("collections", {}).items():
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

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ============================================================================
# Resources
# ============================================================================

@server.list_resources()
async def list_resources() -> list[Resource]:
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
            description="List of all registered agents",
            mimeType="application/json"
        ),
        Resource(
            uri="cortex://health",
            name="Memory Health Report",
            description="Current health status of all memory collections",
            mimeType="application/json"
        ),
    ]

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
    logger.info(f"Reading resource: {uri}")
    cortex = get_cortex()

    try:
        if uri == "cortex://stats":
            data = cortex.stats()
            return ReadResourceResult(contents=[
                TextContent(type="text", text=json.dumps(data, indent=2, default=str))
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
            count = cortex.storage.count(coll)
            content = {
                "collection": coll,
                "count": count,
                "note": "Use memory_search or knowledge_search to query memories"
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
            description="Get context from previous sessions",
            arguments=[]
        ),
        Prompt(
            name="cortex_memory_status",
            description="Get status of all agents and recent memory activity",
            arguments=[]
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    args = arguments or {}
    cortex = get_cortex()

    if name == "cortex_recall":
        topic = args.get("topic", "recent work")
        results = cortex.memory_search(topic, n_results=5)

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
        sessions = cortex.session_recall(limit=3)

        context = ""
        if sessions.get("success") and sessions.get("sessions"):
            for s in sessions["sessions"]:
                context += f"**Session:** {s.get('session_summary', '')}...\n"
                if s.get("decisions_made"):
                    context += f"Decisions: {', '.join(s['decisions_made'][:3])}\n"
                if s.get("unfinished_tasks"):
                    context += f"Unfinished: {', '.join(s['unfinished_tasks'][:3])}\n"
                context += "\n"
        else:
            context = "No previous session notes found."

        tasks = sessions.get("unfinished_tasks", [])
        if tasks:
            context += f"\n**Pending tasks:** {', '.join(tasks)}"

        return GetPromptResult(
            description="Session continuity",
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

    elif name == "cortex_memory_status":
        agents = cortex.list_agents()
        s = cortex.stats()

        agent_list = ""
        for a in agents.get("agents", []):
            agent_list += f"- {a['symbol']} {a['id']} ({a['specialization']})\n"

        return GetPromptResult(
            description="Memory status overview",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"**Neo-Cortex Memory Status**\n\n"
                            f"Current agent: {agents.get('current_agent', '?')}\n"
                            f"Registered agents:\n{agent_list}\n"
                            f"Total memories: {s.get('total_memories', 0)}\n"
                            f"Shared memory: {s.get('collections', {}).get('cortex_shared', 0)}\n\n"
                            f"What would you like to do?"
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
    logger.info("Starting Neo-Cortex MCP Server...")

    try:
        cortex = get_cortex()
        s = cortex.stats()
        logger.info(
            f"Cortex loaded: {s.get('total_memories', 0)} memories, "
            f"{s.get('registered_agents', 0)} agents, "
            f"backend: {s.get('backend', '?')}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize cortex: {e}")

    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
