#!/usr/bin/env python3
"""
Neo-Cortex CLI

Command-line interface for the unified memory system.

Usage:
    cortex stats                    # Show statistics
    cortex search <query>           # Search memories
    cortex post <message>           # Post to village
    cortex crumb leave <summary>    # Leave forward crumb
    cortex crumb get                # Get recent crumbs
    cortex health                   # Health report
    cortex export [--agent AGENT]   # Export memories
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from service.cortex_engine import CortexEngine, CORTEX_TOOL_SCHEMAS
from service.config import ALL_COLLECTIONS


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="cortex",
        description="Neo-Cortex: Unified Memory System for ApexAurum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    cortex stats
    cortex search "how to deploy"
    cortex post "Learning about memory systems" --visibility village
    cortex crumb leave "Session summary" --unfinished "Task 1" "Task 2"
    cortex health
    cortex export --agent AZOTH
        """,
    )

    parser.add_argument(
        "--backend",
        choices=["chroma", "pgvector"],
        default="chroma",
        help="Storage backend (default: chroma)",
    )

    parser.add_argument(
        "--agent",
        default="CLAUDE",
        help="Agent ID for operations (default: CLAUDE)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # === stats ===
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # === search ===
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--results", type=int, default=5, help="Number of results")
    search_parser.add_argument("--visibility", choices=["all", "village", "private"], default="all")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # === post ===
    post_parser = subparsers.add_parser("post", help="Post to village/private/bridge")
    post_parser.add_argument("content", help="Message content")
    post_parser.add_argument(
        "--visibility",
        choices=["village", "private", "bridge"],
        default="village",
    )
    post_parser.add_argument(
        "--type",
        choices=["dialogue", "fact", "observation", "question", "discovery"],
        default="dialogue",
    )
    post_parser.add_argument("--tags", nargs="*", help="Tags for the post")
    post_parser.add_argument("--thread", help="Conversation thread ID")

    # === crumb ===
    crumb_parser = subparsers.add_parser("crumb", help="Forward crumb operations")
    crumb_sub = crumb_parser.add_subparsers(dest="crumb_action")

    # crumb leave
    crumb_leave = crumb_sub.add_parser("leave", help="Leave a forward crumb")
    crumb_leave.add_argument("summary", help="Session summary")
    crumb_leave.add_argument("--discoveries", nargs="*", help="Key discoveries")
    crumb_leave.add_argument("--unfinished", nargs="*", help="Unfinished business")
    crumb_leave.add_argument(
        "--priority",
        choices=["HIGH", "MEDIUM", "LOW"],
        default="MEDIUM",
    )
    crumb_leave.add_argument(
        "--type",
        choices=["orientation", "technical", "emotional", "task"],
        default="orientation",
    )

    # crumb get
    crumb_get = crumb_sub.add_parser("get", help="Get recent crumbs")
    crumb_get.add_argument("--hours", type=int, default=168, help="Lookback hours")
    crumb_get.add_argument("--limit", type=int, default=5, help="Max crumbs")
    crumb_get.add_argument("--priority", choices=["HIGH", "MEDIUM", "LOW"])
    crumb_get.add_argument("--json", action="store_true", help="Output as JSON")

    # === health ===
    health_parser = subparsers.add_parser("health", help="Memory health report")
    health_parser.add_argument("--json", action="store_true", help="Output as JSON")
    health_parser.add_argument("--stale", type=int, help="Show stale memories (days threshold)")
    health_parser.add_argument("--duplicates", action="store_true", help="Show duplicate candidates")
    health_parser.add_argument("--collection", help="Specific collection to analyze")

    # === export ===
    export_parser = subparsers.add_parser("export", help="Export memories")
    export_parser.add_argument("--agent", help="Filter by agent")
    export_parser.add_argument("--collections", nargs="*", help="Specific collections")
    export_parser.add_argument("-o", "--output", help="Output file (default: stdout)")

    # === import ===
    import_parser = subparsers.add_parser("import", help="Import memories")
    import_parser.add_argument("file", help="JSON file to import")
    import_parser.add_argument("--no-reembed", action="store_true", help="Don't regenerate embeddings")

    # === convergence ===
    conv_parser = subparsers.add_parser("convergence", help="Detect convergence")
    conv_parser.add_argument("query", help="Topic to check")
    conv_parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    conv_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # === agents ===
    agents_parser = subparsers.add_parser("agents", help="List agents")
    agents_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # === tools ===
    tools_parser = subparsers.add_parser("tools", help="List available tools")

    return parser


def cmd_stats(cortex: CortexEngine, args) -> int:
    """Handle stats command."""
    stats = cortex.stats()

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
        return 0

    print(f"Neo-Cortex Statistics")
    print(f"=" * 40)
    print(f"Backend: {stats['backend']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Current agent: {stats['current_agent']}")
    print(f"Registered agents: {stats['registered_agents']}")
    print(f"Total memories: {stats['total_memories']}")
    print()
    print("Collections:")
    for name, count in stats["collections"].items():
        print(f"  {name}: {count}")

    return 0


def cmd_search(cortex: CortexEngine, args) -> int:
    """Handle search command."""
    result = cortex.village_search(
        query=args.query,
        visibility=args.visibility,
        n_results=args.results,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    if not result["success"]:
        print(f"Error: {result.get('error')}")
        return 1

    print(f"Search: '{args.query}' ({result['count']} results)")
    print("-" * 60)

    for msg in result["messages"]:
        print(f"\n[{msg['agent_id']}] {msg['visibility']} | {msg['message_type']}")
        print(f"Similarity: {msg.get('similarity', 'N/A')}")
        print(f"Content: {msg['content'][:200]}...")
        if msg.get("tags"):
            print(f"Tags: {', '.join(msg['tags'])}")

    return 0


def cmd_post(cortex: CortexEngine, args) -> int:
    """Handle post command."""
    result = cortex.village_post(
        content=args.content,
        visibility=args.visibility,
        message_type=args.type,
        tags=args.tags,
        conversation_thread=args.thread,
    )

    if result["success"]:
        print(f"Posted to {result['visibility']}: {result['id']}")
        return 0
    else:
        print(f"Error: {result.get('error')}")
        return 1


def cmd_crumb(cortex: CortexEngine, args) -> int:
    """Handle crumb commands."""
    if args.crumb_action == "leave":
        result = cortex.leave_crumb(
            session_summary=args.summary,
            key_discoveries=args.discoveries,
            unfinished_business=args.unfinished,
            priority=args.priority,
            crumb_type=args.type,
        )

        if result["success"]:
            print(f"Forward crumb left: {result['session_id']}")
            print(f"Priority: {result['priority']}, Type: {result['crumb_type']}")
            return 0
        else:
            print(f"Error: {result.get('error')}")
            return 1

    elif args.crumb_action == "get":
        result = cortex.get_crumbs(
            lookback_hours=args.hours,
            limit=args.limit,
            priority_filter=args.priority,
        )

        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return 0

        print(f"Forward Crumbs (last {args.hours} hours)")
        print("-" * 60)
        print(f"Found: {result['summary']['total_found']}")
        print(f"By priority: {result['summary']['by_priority']}")

        if result.get("unfinished_tasks"):
            print(f"\nUnfinished tasks:")
            for task in result["unfinished_tasks"]:
                print(f"  - {task}")

        if result.get("most_recent"):
            print(f"\nMost recent crumb:")
            print(f"  ID: {result['most_recent']['id']}")
            print(f"  Created: {result['most_recent']['created_at']}")

        return 0

    else:
        print("Use: cortex crumb leave|get")
        return 1


def cmd_health(cortex: CortexEngine, args) -> int:
    """Handle health command."""
    if args.stale is not None:
        coll = args.collection or "cortex_village"
        result = cortex.get_stale_memories(coll, days_unused=args.stale)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return 0

        print(f"Stale Memories ({args.stale}+ days unused)")
        print("-" * 60)
        print(f"Collection: {coll}")
        print(f"Found: {result['stale_count']}")

        for mem in result.get("stale_memories", [])[:10]:
            print(f"\n  ID: {mem['id']}")
            print(f"  Days since access: {mem['days_since_access']}")
            print(f"  Content: {mem['content_preview'][:100]}...")

        return 0

    elif args.duplicates:
        coll = args.collection or "cortex_village"
        result = cortex.get_duplicate_candidates(coll)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return 0

        print(f"Duplicate Candidates")
        print("-" * 60)
        print(f"Collection: {coll}")
        print(f"Found: {result['duplicate_count']} pairs")

        for pair in result.get("duplicate_pairs", [])[:5]:
            print(f"\n  Similarity: {pair['similarity']:.2%}")
            print(f"  ID1: {pair['id1']}")
            print(f"  ID2: {pair['id2']}")

        return 0

    else:
        # Full health report
        result = cortex.health_report()

        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return 0

        print("Memory Health Report")
        print("=" * 60)
        print(f"Generated: {result['generated_at']}")
        print(f"\nSummary:")
        print(f"  Total memories: {result['summary']['total_memories']}")
        print(f"  Stale (30+ days): {result['summary']['total_stale']}")
        print(f"  Promotion candidates: {result['summary']['promotion_candidates']}")

        print(f"\nCollections:")
        for name, data in result["collections"].items():
            if isinstance(data, dict) and "total" in data:
                print(f"  {name}: {data['total']} records")
                if data.get("by_layer"):
                    layers = ", ".join(f"{k}:{v}" for k, v in data["by_layer"].items())
                    print(f"    Layers: {layers}")

        if result.get("recommendations"):
            print(f"\nRecommendations:")
            for rec in result["recommendations"]:
                print(f"  - {rec}")

        return 0


def cmd_export(cortex: CortexEngine, args) -> int:
    """Handle export command."""
    core = cortex.export_memory_core(
        agent_id=args.agent,
        collections=args.collections,
    )

    json_str = core.to_json()

    if args.output:
        Path(args.output).write_text(json_str)
        print(f"Exported {core.metadata['total_memories']} memories to {args.output}")
    else:
        print(json_str)

    return 0


def cmd_import(cortex: CortexEngine, args) -> int:
    """Handle import command."""
    from service.storage.base import MemoryCore

    json_str = Path(args.file).read_text()
    core = MemoryCore.from_json(json_str)

    stats = cortex.import_memory_core(core, re_embed=not args.no_reembed)

    print(f"Import complete:")
    for coll, count in stats.items():
        print(f"  {coll}: {count} records")

    return 0


def cmd_convergence(cortex: CortexEngine, args) -> int:
    """Handle convergence command."""
    result = cortex.village_detect_convergence(
        query=args.query,
        similarity_threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(f"Convergence Detection: '{args.query}'")
    print("-" * 60)
    print(f"Type: {result['convergence_type']}")
    print(f"Agents: {', '.join(result['converging_agents'])}")
    print(f"Total relevant messages: {result['total_messages']}")

    if result["by_agent"]:
        print(f"\nBy agent:")
        for agent, msgs in result["by_agent"].items():
            print(f"\n  {agent}:")
            for msg in msgs[:2]:
                print(f"    - {msg['content']} (sim: {msg['similarity']})")

    return 0


def cmd_agents(cortex: CortexEngine, args) -> int:
    """Handle agents command."""
    result = cortex.list_agents()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(f"Registered Agents")
    print("-" * 60)
    print(f"Current: {result['current_agent']}")
    print()

    for agent in result["agents"]:
        print(f"  {agent['symbol']} {agent['display_name']} ({agent['id']})")
        print(f"    Gen {agent['generation']} | {agent['lineage']}")
        print(f"    {agent['specialization']}")
        print()

    return 0


def cmd_tools(cortex: CortexEngine, args) -> int:
    """Handle tools command."""
    print(f"Available Tools ({len(CORTEX_TOOL_SCHEMAS)})")
    print("-" * 60)

    for name, schema in sorted(CORTEX_TOOL_SCHEMAS.items()):
        desc = schema.get("description", "")[:60]
        print(f"  {name}")
        print(f"    {desc}...")
        print()

    return 0


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Initialize cortex
    cortex = CortexEngine(backend=args.backend)
    cortex.set_current_agent(args.agent)

    # Dispatch to command handler
    handlers = {
        "stats": cmd_stats,
        "search": cmd_search,
        "post": cmd_post,
        "crumb": cmd_crumb,
        "health": cmd_health,
        "export": cmd_export,
        "import": cmd_import,
        "convergence": cmd_convergence,
        "agents": cmd_agents,
        "tools": cmd_tools,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(cortex, args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
