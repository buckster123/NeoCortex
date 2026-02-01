# CLAUDE.md - Neo-Cortex

## Session Start Protocol

**On every session start, do these two things:**

### 1. Ensure the API server is running

```bash
# Check if already running
curl -s http://localhost:8766/health 2>/dev/null || (
  cd /home/hailo/claude-root/neo-cortex &&
  /home/hailo/claude-root/Projects/ApexAurum/venv/bin/python -m uvicorn service.api_server:app --host 0.0.0.0 --port 8766 --log-level warning &
  disown && sleep 5
)
```

If the server won't start, check if port 8766 is occupied: `lsof -i :8766`

### 2. Recall previous sessions

Use the `mcp__neo-cortex__session_recall` tool to get context from previous sessions. This replaces the old HANDOVER.md pattern. Read the session notes to understand what was done before and any unfinished business.

## Session End Protocol

Before ending a session, use `mcp__neo-cortex__session_save` with:
- `session_summary`: What was accomplished this session
- `key_discoveries`: Important findings or decisions made
- `unfinished_business`: Tasks left incomplete or next steps
- `if_disoriented`: Instructions for a confused future instance
- `priority`: HIGH for important sessions, MEDIUM for normal work
- `session_type`: "technical" for code work, "orientation" for onboarding

This stores the handover in the neo-cortex memory system where any future instance (in any project) can retrieve it.

## Project Overview

Neo-Cortex is a unified memory system for AI agents. It provides persistent semantic memory across sessions via ChromaDB (local) or pgvector (cloud).

```
neo-cortex/
├── cortex-mcp              # MCP server launcher (stdio)
├── cortex-api              # REST API launcher (port 8766)
├── cortex                  # CLI launcher
├── service/
│   ├── config.py           # All settings + collection names
│   ├── cortex_engine.py    # Main engine (search, store, export/import)
│   ├── shared_engine.py    # Shared memory operations
│   ├── session_engine.py   # Session save/recall
│   ├── health_engine.py    # Decay, dedup, promotions
│   ├── embeddings.py       # Sentence-transformers (384-dim)
│   ├── mcp_server.py       # MCP server (16 tools + resources)
│   ├── api_server.py       # REST API (FastAPI)
│   ├── ingest.py           # Knowledge ingestion pipeline
│   ├── cortex_cli.py       # CLI commands
│   └── storage/
│       ├── base.py         # MemoryRecord + StorageBackend protocol
│       ├── chroma_backend.py
│       └── pgvector_backend.py
├── web/index.html          # Dashboard UI (gold/black theme)
├── data/
│   ├── chroma/             # ChromaDB persistence
│   └── raw_docs/           # Markdown docs for knowledge ingestion
└── CLAUDE.md               # This file
```

## Key URLs

- Dashboard: http://localhost:8766/ui
- API docs: http://localhost:8766/docs
- Health: http://localhost:8766/health

## Memory Architecture

**6 Collections:**
- `cortex_knowledge` - Curated docs (920+ chunks, 20 topics, permanent)
- `cortex_shared` - Shared agent memory
- `cortex_private` - Per-agent personal memory
- `cortex_threads` - Cross-agent dialogue
- `cortex_sessions` - Session continuity notes
- `cortex_sensory` - Temporary observations (6h decay)

**4 Layers:** sensory (6h) -> working (3d) -> long_term (30d) -> cortex (permanent)

## MCP Tools Available

When the neo-cortex MCP server is configured in `~/.claude.json`, these tools are available in any Claude Code session:

| Tool | Purpose |
|------|---------|
| `memory_store` | Store a memory (shared/private/thread) |
| `memory_search` | Semantic search across memories |
| `knowledge_search` | Search the knowledge base (docs) |
| `session_save` | Save session handover note |
| `session_recall` | Retrieve previous session notes |
| `register_agent` | Register an agent profile |
| `list_agents` | List registered agents |
| `memory_convergence` | Detect when agents agree |
| `memory_health_report` | Health status of memory system |
| `memory_get_stale` | Find unused memories |
| `memory_get_duplicates` | Find similar memories |
| `memory_consolidate` | Merge duplicate memories |
| `memory_run_promotions` | Promote memories between layers |
| `cortex_stats` | System statistics |
| `cortex_export` | Export all memories to JSON |

## REST API Endpoints

**Knowledge:** `GET /knowledge/search`, `GET /knowledge/topics`, `GET /knowledge/stats`, `POST /knowledge/ingest`, `POST /knowledge/ingest-text`, `DELETE /knowledge/topic/{topic}`

**Memory:** `POST /memory/store`, `GET /memory/search`, `POST /memory/convergence`

**Sessions:** `POST /sessions/save`, `GET /sessions`

**Health:** `GET /memory/health`, `GET /memory/stale/{coll}`, `GET /memory/duplicates/{coll}`, `POST /memory/consolidate`, `POST /memory/promote/{coll}`

**Export/Import:** `POST /export`, `POST /import`

## Development Notes

- Python venv: `/home/hailo/claude-root/Projects/ApexAurum/venv/bin/python`
- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (384-dim, CPU)
- Running on Raspberry Pi 5 - embedding operations can be slow, avoid bulk operations
- ChromaDB data persists at `data/chroma/` - never delete unless intentional
- Knowledge ingestion: `python -m service.ingest [--clear]` from project root
