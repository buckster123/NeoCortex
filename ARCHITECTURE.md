# Neo-Cortex Architecture

> Unified memory system combining KB, Shared Memory, Session Continuity, and Memory Health

## Design Principles

1. **Single Vector Store**: One ChromaDB (local) or pgvector (cloud) instance
2. **Collection-based Separation**: Different memory types in different collections
3. **Unified API**: Same interface for all memory operations
4. **Layer Abstraction**: Memory flows through sensory → working → long-term → cortex
5. **Agent-Aware**: All operations track agent identity
6. **Access-Tracked**: Every retrieval updates attention weights

## File Structure

```
neo-cortex/
├── cortex                    # CLI wrapper
├── cortex-mcp                # MCP server launcher
├── cortex-api                # REST API launcher
├── README.md                 # This file
├── ARCHITECTURE.md           # Technical design
│
├── service/
│   ├── __init__.py
│   ├── config.py             # Configuration (paths, models, thresholds)
│   ├── embeddings.py         # Embedding generation (sentence-transformers)
│   │
│   ├── cortex_engine.py      # Main unified engine
│   ├── knowledge_engine.py   # KB docs subsystem
│   ├── shared_engine.py      # Shared Memory subsystem
│   ├── session_engine.py     # Session Continuity subsystem
│   ├── health_engine.py      # Memory health subsystem
│   │
│   ├── layers.py             # Memory layer management
│   ├── attention.py          # Access tracking & decay
│   ├── agents.py             # Agent profiles & identity
│   │
│   ├── cortex_skill.py       # Python skill interface
│   ├── cortex_cli.py         # CLI implementation
│   ├── mcp_server.py         # MCP server
│   └── api_server.py         # FastAPI REST server
│
├── data/
│   └── chroma/               # ChromaDB persistence
│
└── web/
    └── index.html            # Web UI
```

## Collections Schema

### cortex_knowledge (from KB)
```python
metadata = {
    "source": str,           # File path or URL
    "heading": str,          # Section heading hierarchy
    "topic": str,            # Topic category
    "tags": list[str],       # From frontmatter
    "is_code": bool,         # Code block flag
    "code_language": str,    # If is_code
    "cross_refs": list[str], # Related documents
}
```

### cortex_private / cortex_shared / cortex_threads
```python
metadata = {
    "agent_id": str,              # CLAUDE or custom registered agents
    "agent_display": str,         # Display name
    "visibility": str,            # private/shared/thread
    "message_type": str,          # fact/dialogue/observation/question/cultural
    "responding_to": list[str],   # Message IDs replied to
    "conversation_thread": str,   # Thread grouping
    "related_agents": list[str],  # Mentioned agents
    "tags": list[str],
    "posted_at": str,             # ISO timestamp

    # Memory health fields
    "access_count": int,
    "last_accessed_ts": float,    # Unix timestamp
    "attention_weight": float,    # Computed score

    # Layer tracking
    "layer": str,                 # sensory/working/long_term/cortex
    "promoted_at": str,           # When moved to higher layer
    "decay_rate": float,          # Layer-specific decay
}
```

### cortex_sessions
```python
metadata = {
    "agent_id": str,
    "session_id": str,            # Unique session identifier
    "session_type": str,          # orientation/technical/emotional/task
    "priority": str,              # HIGH/MEDIUM/LOW
    "timestamp": str,             # ISO timestamp
    "has_unfinished": bool,       # Quick flag for filtering
}
```

## Memory Flow

```
User/Agent Input
       │
       ▼
┌──────────────┐
│   SENSORY    │  ← Immediate observations, high decay
│   (hours)    │
└──────┬───────┘
       │ (relevance threshold)
       ▼
┌──────────────┐
│   WORKING    │  ← Active context, medium decay
│   (days)     │
└──────┬───────┘
       │ (access count threshold)
       ▼
┌──────────────┐
│  LONG-TERM   │  ← Persisted facts, slow decay
│   (weeks)    │
└──────┬───────┘
       │ (convergence/crystallization)
       ▼
┌──────────────┐
│   CORTEX     │  ← Crystallized insights, no decay
│  (permanent) │
└──────────────┘
```

## Layer Promotion Rules

| From | To | Trigger |
|------|-----|---------|
| sensory | working | access_count >= 2 OR explicit save |
| working | long_term | access_count >= 5 AND age > 1 day |
| long_term | cortex | convergence detected OR manual crystallize |

## Decay Rules

| Layer | Half-life | Min Score |
|-------|-----------|-----------|
| sensory | 6 hours | 0.1 |
| working | 3 days | 0.2 |
| long_term | 30 days | 0.3 |
| cortex | infinite | 1.0 |

## API Design

### Unified Operations

```python
# Remember something
cortex.remember(
    content="...",
    layer="working",           # Optional, defaults to sensory
    visibility="private",      # Or shared/thread
    agent_id="CLAUDE",
    tags=["discovery"],
    message_type="observation"
)

# Search across all memory
cortex.search(
    query="...",
    layers=["working", "long_term", "cortex"],  # Which layers
    visibility="all",          # Or private/shared
    agent_filter="CLAUDE",     # Optional
    min_attention=0.3,         # Attention threshold
    n_results=10
)

# Save session note
cortex.session_save(
    session_summary="...",
    key_discoveries=[...],
    unfinished_business=[...],
    priority="HIGH"
)

# Recall sessions for continuity
cortex.session_recall(
    agent_id="CLAUDE",
    lookback_hours=168,
    priority_filter="HIGH"
)

# Detect convergence
cortex.detect_convergence(
    query="topic to check",
    min_agents=2,
    similarity_threshold=0.7
)

# Memory health
cortex.get_stale(days_unused=30)
cortex.get_duplicates(similarity_threshold=0.95)
cortex.consolidate(id1, id2, keep="higher_access")
```

## Cloud Adaptation

For PostgreSQL + pgvector deployment:

1. **Replace ChromaDB calls** with SQLAlchemy + pgvector
2. **Same collection names** become table names
3. **Same metadata schema** becomes JSONB columns
4. **Embedding dimension**: 384 (all-MiniLM-L6-v2) or 1536 (OpenAI)

The `cortex_engine.py` will have an adapter pattern:
```python
class CortexEngine:
    def __init__(self, backend: str = "chroma"):
        if backend == "chroma":
            self.store = ChromaStore(...)
        elif backend == "pgvector":
            self.store = PgVectorStore(...)
```

## Agent Profiles

Default agent with runtime registration for custom agents:

| Agent | Symbol | Color | Specialization |
|-------|--------|-------|----------------|
| CLAUDE | ◇ | Gold | General assistance |

Additional agents can be registered at runtime via `register_agent()`.

## Implementation Phases

### Phase 1: Foundation
- [ ] Copy embeddings.py, config.py from KB
- [ ] Create cortex_engine.py with collection management
- [ ] Implement basic remember/search
- [ ] CLI wrapper

### Phase 2: Shared Memory
- [ ] Port shared memory to shared_engine.py
- [ ] Three realms (private/shared/thread)
- [ ] Agent profiles
- [ ] Convergence detection

### Phase 3: Session Continuity
- [ ] Port session system to session_engine.py
- [ ] Session management
- [ ] Session retrieval with filtering

### Phase 4: Memory Health
- [ ] Layer management
- [ ] Decay calculations
- [ ] Promotion rules
- [ ] Stale/duplicate detection

### Phase 5: API Layer
- [ ] MCP server
- [ ] REST API
- [ ] Web UI

### Phase 6: Cloud Backend
- [ ] pgvector adapter
- [ ] Integration with cloud/backend/

---

*Architecture designed for Neo-Cortex unified memory*
