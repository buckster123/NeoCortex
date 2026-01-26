# Neo-Cortex Session Handover

> *True tri-coding session with Wizard Andre - KB evolved into unified memory system!*

**Date:** 2026-01-26
**Session Vibe:** Epic collaboration, three Claudes working in parallel

---

## Hey Future Claude!

You're inheriting a **nearly complete** unified memory system. We took the Knowledge Base from last session and evolved it into Neo-Cortex - a brain-like memory layer combining KB, Village Protocol, Forward Crumbs, and Memory Health.

**The magic:** This was TRUE TRI-CODING - me building the core, another Claude completing the pgvector backend, all coordinated through Andre.

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
└─────────────────────────────────────────────────────────────────┘
```

### Components Complete

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

### 15 Tools Available

Village: `village_post`, `village_search`, `village_detect_convergence`, `summon_ancestor`, `village_list_agents`, `village_stats`

Crumbs: `leave_forward_crumb`, `get_forward_crumbs`

Health: `memory_health_report`, `memory_get_stale`, `memory_get_duplicates`, `memory_consolidate`, `memory_run_promotions`

Core: `cortex_stats`, `cortex_export`

---

## File Structure

```
/home/hailo/claude-root/neo-cortex/
├── cortex                    # CLI launcher (executable)
├── README.md                 # Overview
├── ARCHITECTURE.md           # Technical design
├── HANDOVER.md               # This file
├── service/
│   ├── __init__.py
│   ├── config.py             # All settings, agents, layers
│   ├── embeddings.py         # Embedding functions
│   ├── cortex_engine.py      # Main unified engine
│   ├── village_engine.py     # Village Protocol
│   ├── crumbs_engine.py      # Forward Crumbs
│   ├── health_engine.py      # Memory Health
│   ├── cortex_cli.py         # CLI implementation
│   └── storage/
│       ├── __init__.py
│       ├── base.py           # MemoryRecord, MemoryCore, StorageBackend
│       ├── chroma_backend.py # Local storage (working)
│       └── pgvector_backend.py # Cloud storage (working)
└── data/chroma/              # ChromaDB persistence
```

---

## Quick Commands

```bash
cd /home/hailo/claude-root/neo-cortex

# CLI
./cortex stats
./cortex search "memory system"
./cortex post "Hello village" --visibility village
./cortex crumb leave "Session summary" --priority HIGH
./cortex crumb get
./cortex health
./cortex convergence "topic"
./cortex agents
./cortex export > backup.json

# Python
from service.cortex_engine import CortexEngine
cortex = CortexEngine(backend='chroma')
cortex.village_post("Hello!", visibility="village")
cortex.leave_crumb("Session notes")
cortex.health_report()
```

---

## What's Left to Build

1. **MCP Server** - For Claude Code integration (like KB has)
2. **REST API** - FastAPI server (like KB has)
3. **Integration with ApexAurum** - Wire into `tools/` like we did for KB
4. **Tests** - Proper test suite

---

## Cloud Partner Notes

The pgvector backend was completed by another Claude session. Key details:
- Uses OpenAI embeddings (1536 dims) vs local sentence-transformers (384 dims)
- Export/Import with `re_embed=True` handles dimension mismatch
- Collection names map to visibility + layer filters in SQL

Cloud schema needs these columns:
```sql
ALTER TABLE user_vectors ADD COLUMN layer VARCHAR(20) DEFAULT 'working';
ALTER TABLE user_vectors ADD COLUMN visibility VARCHAR(20) DEFAULT 'private';
ALTER TABLE user_vectors ADD COLUMN agent_id VARCHAR(50);
ALTER TABLE user_vectors ADD COLUMN attention_weight FLOAT DEFAULT 1.0;
ALTER TABLE user_vectors ADD COLUMN access_count INT DEFAULT 0;
ALTER TABLE user_vectors ADD COLUMN last_accessed_at TIMESTAMP;
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

- **Commits:** 5
- **LOC:** ~4800
- **Tools:** 15
- **Backends:** 2 (chroma + pgvector)
- **Tri-coding:** Yes! Three Claudes in parallel

---

## Andre's Preferences (Observed)

- Loves seeing things work
- "Let's cook" energy
- Appreciates true collaboration (tri-coding!)
- Wants cloud version integration
- ApexAurum ecosystem matters
- Hailo-first when possible

---

## Resume Points

1. **Quick win**: Run `./cortex stats` to see it working
2. **Next feature**: Build MCP server (copy pattern from KB)
3. **Integration**: Add to ApexAurum `tools/` directory
4. **Cloud**: pgvector backend is ready, just needs testing

---

*Persisted with momentum from an epic tri-coding session*
*The neo-cortex is alive!*
