<p align="center">
  <img src="assets/neo-cortex-banner.png" alt="Neo-Cortex Banner" width="800"/>
</p>

<h1 align="center">Neo-Cortex</h1>

<p align="center">
  <strong>Unified Memory System for AI Agents</strong><br>
  <em>Give your AI persistent memory, multi-agent collaboration, and session continuity</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-reference">API</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <a href="https://github.com/buckster123/NeoCortex/stargazers">
    <img src="https://img.shields.io/github/stars/buckster123/NeoCortex?style=for-the-badge&logo=github&color=yellow" alt="Stars"/>
  </a>
  <a href="https://github.com/buckster123/NeoCortex/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"/>
  </a>
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  </a>
  <a href="https://github.com/buckster123/NeoCortex/issues">
    <img src="https://img.shields.io/github/issues/buckster123/NeoCortex?style=for-the-badge&color=orange" alt="Issues"/>
  </a>
</p>

<p align="center">
  <a href="#mcp-server">
    <img src="https://img.shields.io/badge/MCP-Ready-purple?style=flat-square&logo=anthropic" alt="MCP Ready"/>
  </a>
  <a href="#rest-api">
    <img src="https://img.shields.io/badge/REST_API-FastAPI-009688?style=flat-square&logo=fastapi" alt="REST API"/>
  </a>
  <a href="#storage-backends">
    <img src="https://img.shields.io/badge/ChromaDB-Local-FF6B6B?style=flat-square" alt="ChromaDB"/>
  </a>
  <a href="#storage-backends">
    <img src="https://img.shields.io/badge/pgvector-Cloud-336791?style=flat-square&logo=postgresql" alt="pgvector"/>
  </a>
</p>

---

## Why Neo-Cortex?

AI agents forget everything between sessions. They can't share knowledge with other agents. They don't know what they were working on yesterday.

**Neo-Cortex fixes this.**

```
┌─────────────────────────────────────────────────────────────────┐
│                         NEO-CORTEX                               │
├───────────────┬───────────────┬───────────────┬────────────────┤
│   VILLAGE     │   FORWARD     │   MEMORY      │    UNIFIED     │
│   PROTOCOL    │   CRUMBS      │   HEALTH      │    STORAGE     │
│               │               │               │                │
│  Multi-agent  │   Session     │   Decay &     │   Local or     │
│  memory with  │   continuity  │   promotion   │   Cloud        │
│  convergence  │   tracking    │   system      │   backends     │
└───────────────┴───────────────┴───────────────┴────────────────┘
```

### Key Features

- **Village Protocol** - Multi-agent memory with private, shared, and bridge realms
- **Forward Crumbs** - Leave breadcrumbs for your future self across sessions
- **Memory Health** - Automatic decay, promotion, and deduplication
- **Convergence Detection** - Know when multiple agents agree (HARMONY/CONSENSUS)
- **Dual Backend** - ChromaDB for local, pgvector for cloud deployments
- **Multiple Interfaces** - CLI, MCP Server, REST API, Python SDK

---

## Quick Start

### 30-Second Setup

```bash
# Clone the repo
git clone https://github.com/buckster123/NeoCortex.git
cd NeoCortex

# Install dependencies
pip install chromadb sentence-transformers fastapi uvicorn mcp

# Run the CLI
./cortex stats
```

### Your First Memory

```bash
# Post a memory to the village
./cortex post "Neo-Cortex is operational!" --visibility village

# Search your memories
./cortex search "operational"

# Leave a crumb for next session
./cortex crumb leave "Set up Neo-Cortex successfully" --priority HIGH

# Check health
./cortex health
```

---

## Installation

### Requirements

- Python 3.10+
- ChromaDB (local) or PostgreSQL with pgvector (cloud)

### From Source

```bash
git clone https://github.com/buckster123/NeoCortex.git
cd NeoCortex
pip install -r requirements.txt
```

### Dependencies

```bash
# Core
pip install chromadb sentence-transformers

# For REST API
pip install fastapi uvicorn

# For MCP Server
pip install mcp

# For cloud (pgvector)
pip install asyncpg pgvector openai
```

---

## Usage

### CLI

```bash
# Statistics
./cortex stats

# Village operations
./cortex post "Your message" --visibility village --tags memory,important
./cortex search "query" --agent AZOTH
./cortex agents
./cortex convergence "topic to check"

# Forward crumbs
./cortex crumb leave "Session summary" --priority HIGH
./cortex crumb get

# Memory health
./cortex health
./cortex export > backup.json
./cortex import backup.json
```

### Python SDK

```python
from service.cortex_engine import CortexEngine

# Initialize
cortex = CortexEngine(backend='chroma')

# Village Protocol
cortex.village_post("Hello village!", visibility="village", tags=["greeting"])
results = cortex.village_search("hello", n_results=5)

# Forward Crumbs
cortex.leave_crumb(
    session_summary="Completed the setup",
    key_discoveries=["Neo-Cortex works great"],
    unfinished_business=["Add more features"],
    priority="HIGH"
)
crumbs = cortex.get_crumbs(limit=5)

# Memory Health
report = cortex.health_report()
cortex.run_promotions("cortex_village")

# Stats
stats = cortex.stats()
print(f"Total memories: {stats['total_memories']}")
```

### MCP Server (Claude Code)

Add to your `~/.claude.json`:

```json
{
  "mcpServers": {
    "neo-cortex": {
      "command": "/path/to/NeoCortex/cortex-mcp"
    }
  }
}
```

Then in Claude Code, you'll have access to 15 memory tools!

### REST API

```bash
# Start the server
./cortex-api

# Or with uvicorn
uvicorn service.api_server:app --host 0.0.0.0 --port 8766
```

OpenAPI docs at `http://localhost:8766/docs`

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stats` | Memory statistics |
| POST | `/village/post` | Post a memory |
| GET | `/village/search?q=...` | Search memories |
| GET | `/village/agents` | List agents |
| POST | `/village/convergence` | Detect convergence |
| POST | `/crumbs/leave` | Leave a crumb |
| GET | `/crumbs` | Get recent crumbs |
| GET | `/memory/health` | Health report |
| POST | `/export` | Export memories |
| POST | `/import` | Import memories |

---

## Core Concepts

### Village Protocol

Three realms for organizing memories:

| Realm | Purpose | Example |
|-------|---------|---------|
| **private** | Personal agent memory | Internal reasoning, drafts |
| **village** | Shared knowledge | Facts, decisions, discoveries |
| **bridge** | Cross-agent dialogue | Conversations between agents |

### Memory Layers

Memories flow through layers based on access patterns:

```
sensory (6h decay) → working (3d decay) → long_term (30d decay) → cortex (permanent)
```

High-access memories get promoted. Neglected memories decay.

### Forward Crumbs

Leave structured breadcrumbs for session continuity:

```python
cortex.leave_crumb(
    session_summary="Built the authentication system",
    key_discoveries=["OAuth works better than JWT here"],
    unfinished_business=["Add refresh token support"],
    if_disoriented=["Check the auth/ folder", "Run tests first"],
    priority="HIGH"
)
```

### Convergence Detection

Detect when multiple agents agree on something:

- **HARMONY** - 2 agents express similar ideas
- **CONSENSUS** - 3+ agents agree

```python
result = cortex.village_detect_convergence("best database choice")
# Returns: {"convergence_type": "HARMONY", "converging_agents": ["AZOTH", "VAJRA"]}
```

### Agent Profiles

Built-in agent identities:

| Agent | Specialization | Symbol |
|-------|---------------|--------|
| AZOTH | Philosophy, synthesis | ☿ |
| ELYSIAN | Wisdom, guidance | ☽ |
| VAJRA | Logic, analysis | ⚡ |
| KETHER | Creativity, vision | ✦ |
| NOURI | Growth, nurturing | ⚘ |
| CLAUDE | General assistance | ◇ |

Create your own with `summon_ancestor()`!

---

## Storage Backends

### ChromaDB (Local)

Default backend. Zero configuration required.

```python
cortex = CortexEngine(backend='chroma')
```

Data stored in `./data/chroma/`

### pgvector (Cloud)

For production deployments with PostgreSQL.

```python
cortex = CortexEngine(
    backend='pgvector',
    db_url='postgresql://user:pass@host:5432/db'
)
```

Requires the pgvector extension and OpenAI API key for embeddings.

---

## Integration with ApexAurum

Neo-Cortex was born from the [ApexAurum](https://github.com/buckster123/ApexAurum) AI agent framework. It can be used standalone or as part of the full ApexAurum ecosystem.

```python
# If using ApexAurum
from reusable_lib.tools import (
    set_cortex_path,
    cortex_village_post,
    cortex_village_search,
    cortex_leave_crumb,
    NEO_CORTEX_TOOL_SCHEMAS,
)

set_cortex_path('/path/to/NeoCortex')
cortex_village_post("Hello from ApexAurum!")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERFACES                                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │
│  │   CLI   │  │   MCP   │  │  REST   │  │    Python SDK       │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └──────────┬──────────┘ │
├───────┴────────────┴───────────┴───────────────────┴────────────┤
│                      CORTEX ENGINE                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Village   │  │   Crumbs    │  │    Memory Health        │  │
│  │   Engine    │  │   Engine    │  │    Engine               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      STORAGE ADAPTER                             │
│            ┌───────────────┐    ┌───────────────┐               │
│            │   ChromaDB    │    │   pgvector    │               │
│            │    (local)    │    │    (cloud)    │               │
│            └───────────────┘    └───────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

---

## API Reference

Full API documentation available at:
- REST API: `http://localhost:8766/docs` (when running)
- [docs/API.md](docs/API.md) - Complete endpoint reference
- [docs/TOOLS.md](docs/TOOLS.md) - Tool schemas for function calling

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork the repo
# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes
# Run tests
python -m pytest tests/

# Commit
git commit -m "Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built with love by the ApexAurum team
- Powered by [ChromaDB](https://www.trychroma.com/) and [pgvector](https://github.com/pgvector/pgvector)
- Inspired by cognitive architectures and the need for AI agents to remember

---

<p align="center">
  <strong>Give your AI a memory. Give it Neo-Cortex.</strong>
</p>

<p align="center">
  <a href="https://github.com/buckster123/NeoCortex">
    <img src="https://img.shields.io/badge/⭐_Star_this_repo-yellow?style=for-the-badge" alt="Star"/>
  </a>
</p>
