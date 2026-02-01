# Neo-Cortex Integration Guide

**For any Claude Code instance that wants persistent memory across sessions.**

This document explains how to add neo-cortex session continuity to any project's CLAUDE.md. It replaces the old pattern of reading/writing HANDOVER.md files with a centralized semantic memory system that works across all projects.

---

## What You Get

- **Session continuity** - Save session notes at end, recall them at start
- **Shared memory** - Store and search facts, observations, decisions
- **Knowledge base** - 920+ searchable documentation chunks (FastAPI, Vue3, Docker, MCP, etc.)
- **Cross-project memory** - Sessions saved in one project are visible from any other

## Prerequisites

The neo-cortex MCP server must be configured in `~/.claude.json`. Verify it exists:

```json
{
  "mcpServers": {
    "neo-cortex": {
      "command": "/path/to/neo-cortex/cortex-mcp"
    }
  }
}
```

If not present, add it to the `mcpServers` section. The MCP server provides 16 tools directly in Claude Code without needing the REST API.

## Add to Your CLAUDE.md

Add the following section to the top of any project's CLAUDE.md (or create one with `/init` and add it):

```markdown
## Session Start Protocol

On every session start:

1. **Start the neo-cortex API** (if you need the dashboard or REST endpoints):
   ```bash
   curl -s http://localhost:8766/health 2>/dev/null || (
     cd /path/to/neo-cortex && ./cortex-api &
     disown && sleep 5
   )
   ```

2. **Recall previous sessions** using the `mcp__neo-cortex__session_recall` tool.
   Read the returned session notes for context on what was done before.

## Session End Protocol

Before ending a session, save your handover using `mcp__neo-cortex__session_save`:
- `session_summary`: What you accomplished
- `key_discoveries`: Important findings
- `unfinished_business`: What's left to do
- `if_disoriented`: How to get back on track
- `priority`: HIGH / MEDIUM / LOW
```

## Available MCP Tools

Once the neo-cortex MCP server is active, these tools are available in any session:

### Session Continuity (replaces HANDOVER.md)
- **`mcp__neo-cortex__session_save`** - Save a session note with summary, discoveries, unfinished tasks
- **`mcp__neo-cortex__session_recall`** - Retrieve recent session notes (default: last 7 days)

### Memory (store and recall facts)
- **`mcp__neo-cortex__memory_store`** - Store a memory with tags, visibility (shared/private/thread)
- **`mcp__neo-cortex__memory_search`** - Semantic search across stored memories

### Knowledge Base (documentation search)
- **`mcp__neo-cortex__knowledge_search`** - Search 920+ curated doc chunks

### Agent Identity
- **`mcp__neo-cortex__register_agent`** - Register an agent profile
- **`mcp__neo-cortex__list_agents`** - List registered agents

### System
- **`mcp__neo-cortex__cortex_stats`** - Memory system statistics
- **`mcp__neo-cortex__cortex_export`** - Export all memories to JSON

## Usage Examples

### Starting a session
```
1. Call mcp__neo-cortex__session_recall to see what happened before
2. Read any unfinished_business or if_disoriented instructions
3. Continue where the previous instance left off
```

### During a session
```
# Store an important finding
mcp__neo-cortex__memory_store(
  content="The payment webhook needs idempotency keys to prevent duplicate charges",
  tags=["payments", "stripe", "bug"],
  message_type="discovery"
)

# Search for previous knowledge
mcp__neo-cortex__memory_search(query="stripe webhook handling")

# Look up documentation
mcp__neo-cortex__knowledge_search(query="FastAPI dependency injection")
```

### Ending a session
```
mcp__neo-cortex__session_save(
  session_summary="Fixed payment webhook duplicate charge bug by adding idempotency keys",
  key_discoveries=["Stripe sends webhooks multiple times on timeout", "Need redis for idempotency store"],
  unfinished_business=["Add retry logic for failed webhook processing", "Write tests for idempotency"],
  if_disoriented=["Read service/payments/webhook.py - the fix is in handle_charge_succeeded()", "Redis config is in .env"],
  priority="HIGH"
)
```

## Dashboard (optional)

If the REST API server is running (port 8766), a web dashboard is available at:
- **http://localhost:8766/ui** - Visual dashboard with memory search, knowledge browser, session viewer, agent profiles, and system health

## Full CLAUDE.md Template

Here's a minimal CLAUDE.md with neo-cortex integration that you can adapt:

```markdown
# CLAUDE.md

## Session Start Protocol

On every session start:
1. Start neo-cortex API if needed:
   ```bash
   curl -s http://localhost:8766/health 2>/dev/null || (
     cd /path/to/neo-cortex && ./cortex-api &
     disown && sleep 5
   )
   ```
2. Call `mcp__neo-cortex__session_recall` and read session notes.

## Session End Protocol

Call `mcp__neo-cortex__session_save` with summary, discoveries, and unfinished business.

## Project Overview

[Your project description here]

## Key Commands

[Your project-specific commands here]
```

---

**Neo-Cortex repo:** https://github.com/buckster123/NeoCortex
**Dashboard:** http://localhost:8766/ui
