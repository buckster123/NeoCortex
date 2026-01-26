# Neo-Cortex Session Handover

> *Phase 3 COMPLETE - Full integration achieved!*

**Date:** 2026-01-26
**Session Vibe:** Epic collaboration, tri-coding evolved into full deployment

---

## Hey Future Claude!

**IT'S DONE!** The neo-cortex is fully operational with:
- MCP Server for Claude Code
- REST API for web access
- ApexAurum tools integration

You're inheriting a **complete** unified memory system.

---

## What We Built

### Neo-Cortex = Unified Memory System

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEO-CORTEX                                │
├───────────────┬───────────────┬───────────────┬────────────────┤
│  KNOWLEDGE    │   EPISODIC    │    SOCIAL     │    HEALTH      │
│  (KB docs)    │  (forward     │   (Village    │  (decay,       │
│               │   crumbs)     │   protocol)   │   promotions)  │
├───────────────┴───────────────┴───────────────┴────────────────┤
│                    STORAGE ADAPTER                               │
│            ChromaDB (local) ←→ pgvector (cloud)                 │
├─────────────────────────────────────────────────────────────────┤
│   MCP Server (8766)    │    REST API (8766)    │    CLI         │
└─────────────────────────────────────────────────────────────────┘
```

### All Components Complete

| Component | File | Status |
|-----------|------|--------|
| Storage Protocol | `service/storage/base.py` | ✅ Full |
| ChromaDB Backend | `service/storage/chroma_backend.py` | ✅ Full |
| pgvector Backend | `service/storage/pgvector_backend.py` | ✅ Full (cloud partner) |
| Village Protocol | `service/village_engine.py` | ✅ Full |
| Forward Crumbs | `service/crumbs_engine.py` | ✅ Full |
| Memory Health | `service/health_engine.py` | ✅ Full |
| Unified Engine | `service/cortex_engine.py` | ✅ Full |
| CLI | `service/cortex_cli.py` + `cortex` | ✅ Full |
| **MCP Server** | `service/mcp_server.py` + `cortex-mcp` | ✅ **NEW** |
| **REST API** | `service/api_server.py` + `cortex-api` | ✅ **NEW** |
| **ApexAurum Integration** | `reusable_lib/tools/neo_cortex.py` | ✅ **NEW** |

### 15 Tools Available (via all interfaces)

Village: `village_post`, `village_search`, `village_detect_convergence`, `summon_ancestor`, `village_list_agents`, `village_stats`

Crumbs: `leave_forward_crumb`, `get_forward_crumbs`

Health: `memory_health_report`, `memory_get_stale`, `memory_get_duplicates`, `memory_consolidate`, `memory_run_promotions`

Core: `cortex_stats`, `cortex_export`

---

## File Structure

```
/home/hailo/claude-root/neo-cortex/
├── cortex                    # CLI launcher
├── cortex-mcp                # MCP server launcher
├── cortex-api                # REST API launcher
├── README.md
├── ARCHITECTURE.md
├── HANDOVER.md
├── service/
│   ├── __init__.py
│   ├── config.py             # All settings
│   ├── embeddings.py         # Embedding functions
│   ├── cortex_engine.py      # Main unified engine
│   ├── village_engine.py     # Village Protocol
│   ├── crumbs_engine.py      # Forward Crumbs
│   ├── health_engine.py      # Memory Health
│   ├── cortex_cli.py         # CLI implementation
│   ├── mcp_server.py         # MCP server (NEW)
│   ├── api_server.py         # REST API (NEW)
│   └── storage/
│       ├── __init__.py
│       ├── base.py
│       ├── chroma_backend.py
│       └── pgvector_backend.py
└── data/chroma/              # ChromaDB persistence
```

---

## Quick Start

### CLI
```bash
cd /home/hailo/claude-root/neo-cortex
./cortex stats
./cortex search "memory system"
./cortex post "Hello village" --visibility village
./cortex crumb leave "Session summary" --priority HIGH
./cortex crumb get
./cortex health
```

### MCP Server (Claude Code)
Add to `~/.claude.json`:
```json
{
  "mcpServers": {
    "neo-cortex": {
      "command": "/home/hailo/claude-root/neo-cortex/cortex-mcp"
    }
  }
}
```

### REST API
```bash
./cortex-api  # Starts on port 8766
# Docs at http://localhost:8766/docs
```

### ApexAurum Integration
```python
from reusable_lib.tools import (
    set_cortex_path,
    cortex_stats,
    cortex_village_post,
    cortex_village_search,
    cortex_leave_crumb,
    cortex_get_crumbs,
    NEO_CORTEX_TOOL_SCHEMAS,
)

set_cortex_path('/home/hailo/claude-root/neo-cortex')
cortex_village_post("Hello from ApexAurum!")
```

---

## REST API Endpoints

```
GET  /                     - API info
GET  /health               - Health check
GET  /stats                - Cortex statistics

POST /village/post         - Post a message
GET  /village/search       - Search memories
GET  /village/agents       - List agents
POST /village/convergence  - Detect convergence
POST /village/summon       - Summon ancestor

POST /crumbs/leave         - Leave a crumb
GET  /crumbs               - Get recent crumbs
GET  /crumbs/tasks         - Unfinished tasks

GET  /memory/health        - Health report
GET  /memory/stale/{coll}  - Stale memories
GET  /memory/duplicates/{coll} - Duplicates
POST /memory/consolidate   - Merge memories
POST /memory/promote/{coll} - Run promotions

POST /export               - Export MemoryCore
POST /import               - Import MemoryCore

GET  /q/{query}            - Quick search
POST /remember             - Quick store
```

---

## Key Concepts

### Memory Layers
- **sensory** (6h decay) → **working** (3d decay) → **long_term** (30d decay) → **cortex** (permanent)

### Village Realms
- **private** - Agent's personal memory
- **village** - Shared knowledge square
- **bridges** - Cross-agent dialogue

### Convergence
- **HARMONY** - 2 agents agree
- **CONSENSUS** - 3+ agents agree

### Agent Profiles
AZOTH, ELYSIAN, VAJRA, KETHER, NOURI, CLAUDE

---

## Session Stats

- **Commits:** 9
- **LOC:** ~6,500+
- **Tools:** 15
- **Interfaces:** 4 (CLI, MCP, REST, Python)
- **Backends:** 2 (chroma + pgvector)
- **Tri-coding:** Yes!

---

## What's Left

1. **Tests** - Proper test suite (nice to have)
2. **Web UI** - Optional frontend for REST API
3. **Cloud Testing** - pgvector backend needs live testing

---

*Phase 3 Complete - The neo-cortex is ALIVE!*
