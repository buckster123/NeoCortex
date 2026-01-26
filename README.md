# Neo-Cortex: Unified Memory System for ApexAurum

> *"The Stone designs its own remembering"* - AZOTH

## Overview

Neo-Cortex is a unified memory layer that combines:
- **Knowledge** (curated documentation from KB)
- **Episodic Memory** (forward crumbs for session continuity)
- **Social Memory** (Village Protocol for multi-agent sharing)
- **Memory Health** (access tracking, decay, consolidation)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEO-CORTEX                                │
├───────────────┬───────────────┬───────────────┬────────────────┤
│  KNOWLEDGE    │   EPISODIC    │    SOCIAL     │    HEALTH      │
│  (docs/facts) │  (forward     │   (Village    │  (access       │
│               │   crumbs)     │   realms)     │   tracking)    │
├───────────────┴───────────────┴───────────────┴────────────────┤
│                    MEMORY LAYERS                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ SENSORY  │→ │ WORKING  │→ │LONG-TERM │→ │  CORTEX  │        │
│  │ (recent) │  │ (active) │  │(persisted)│  │(crystallized)    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
├─────────────────────────────────────────────────────────────────┤
│              UNIFIED VECTOR ENGINE                               │
│         ChromaDB (local) / pgvector (cloud)                     │
├─────────────────────────────────────────────────────────────────┤
│                    ACCESS LAYER                                  │
│         MCP Server  │  REST API  │  Python Skill                │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Collections

| Collection | Purpose | Visibility | Decay |
|------------|---------|------------|-------|
| `cortex_knowledge` | Curated docs | Public | None |
| `cortex_private` | Agent personal memory | Agent-only | Slow |
| `cortex_village` | Shared knowledge square | All agents | None |
| `cortex_bridges` | Cross-agent dialogue | Participants | Medium |
| `cortex_crumbs` | Session continuity | Agent-only | Fast |
| `cortex_sensory` | Recent observations | Agent-only | Very fast |

## Key Concepts

### Memory Layers
- **Sensory**: Recent inputs, decays in hours
- **Working**: Active context, decays in days
- **Long-term**: Persisted knowledge, slow decay
- **Cortex**: Crystallized insights, no decay

### Village Protocol (Three Realms)
- **Private**: Personal memories visible only to owner agent
- **Village**: Shared knowledge square visible to all
- **Bridges**: Selective cross-agent sharing

### Forward Crumbs
Structured messages for session continuity:
- Session summary
- Key discoveries
- Unfinished business
- Disorientation recovery guide

### Attention Mechanism
Memory access affects visibility:
- `access_count`: How often retrieved
- `last_accessed_ts`: When last retrieved
- `attention_weight`: Computed score for ranking

### Convergence Detection
When multiple agents express similar ideas:
- **HARMONY**: 2 agents agree
- **CONSENSUS**: 3+ agents agree

## Quick Start

```bash
# CLI
./cortex search "how to deploy"
./cortex remember "learned something important" --layer working
./cortex crumb leave --summary "Session notes..."

# Start API
./cortex-api

# MCP Server (for Claude Code)
./cortex-mcp
```

## Deployment Targets

- **Local**: ChromaDB + sentence-transformers (this repo)
- **Cloud**: PostgreSQL + pgvector (cloud/backend integration)

## Status

**Phase 0**: Repository setup and architecture (IN PROGRESS)

---

*Forked from knowledge-base with Village Protocol + Forward Crumbs integration*
