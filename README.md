<p align="center">
  <img src="assets/neo-cortex-banner.jpg" alt="Neo-Cortex Banner" width="800"/>
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
│   SHARED      │   SESSION     │   MEMORY      │    UNIFIED     │
│   MEMORY      │   CONTINUITY  │   HEALTH      │    STORAGE     │
│               │               │               │                │
│  Multi-agent  │   Session     │   Decay &     │   Local or     │
│  memory with  │   continuity  │   promotion   │   Cloud        │
│  convergence  │   tracking    │   system      │   backends     │
└───────────────┴───────────────┴───────────────┴────────────────┘
```

### Key Features

- **Knowledge Base** - Ingest markdown docs, search 920+ chunks across 20+ topics
- **Shared Memory** - Multi-agent memory with private, shared, and thread realms
- **Session Continuity** - Leave session notes for your future self across sessions
- **Memory Health** - Automatic decay, promotion, and deduplication
- **Convergence Detection** - Know when multiple agents agree (HARMONY/CONSENSUS)
- **Web Dashboard** - Visual UI for browsing knowledge, memory, sessions, agents, and health
- **Dual Backend** - ChromaDB for local, pgvector for cloud deployments
- **Multiple Interfaces** - CLI, MCP Server, REST API, Python SDK, Web Dashboard

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
# Post a memory to shared space
./cortex post "Neo-Cortex is operational!" --visibility shared

# Search your memories
./cortex search "operational"

# Save a session note for next time
./cortex session leave "Set up Neo-Cortex successfully" --priority HIGH

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

# Memory operations
./cortex post "Your message" --visibility shared --tags memory,important
./cortex search "query" --agent CLAUDE
./cortex agents
./cortex convergence "topic to check"

# Session notes
./cortex session leave "Session summary" --priority HIGH
./cortex session get

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

# Shared Memory
cortex.memory_store("Hello world!", visibility="shared", tags=["greeting"])
results = cortex.memory_search("hello", n_results=5)

# Session Continuity
cortex.session_save(
    session_summary="Completed the setup",
    key_discoveries=["Neo-Cortex works great"],
    unfinished_business=["Add more features"],
    priority="HIGH"
)
sessions = cortex.session_recall(limit=5)

# Memory Health
report = cortex.health_report()
cortex.run_promotions("cortex_shared")

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

Then in Claude Code, you'll have access to 16 memory tools.

See [INTEGRATE.md](INTEGRATE.md) for a complete guide on adding neo-cortex session continuity to any project.

### REST API

```bash
# Start the server
./cortex-api

# Or with uvicorn
uvicorn service.api_server:app --host 0.0.0.0 --port 8766
```

- OpenAPI docs at `http://localhost:8766/docs`
- Web dashboard at `http://localhost:8766/ui`

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stats` | Memory statistics |
| POST | `/memory/store` | Store a memory |
| GET | `/memory/search?q=...` | Search memories |
| POST | `/memory/convergence` | Detect convergence |
| GET | `/knowledge/search?q=...` | Search knowledge base |
| GET | `/knowledge/topics` | List topics with counts |
| GET | `/knowledge/stats` | Knowledge base stats |
| POST | `/knowledge/ingest` | Upload & ingest markdown file |
| POST | `/knowledge/ingest-text` | Ingest raw text content |
| DELETE | `/knowledge/topic/{topic}` | Delete a topic |
| GET | `/agents` | List agents |
| POST | `/agents/register` | Register new agent |
| POST | `/sessions/save` | Save session note |
| GET | `/sessions` | Get recent sessions |
| GET | `/memory/health` | Health report |
| POST | `/export` | Export memories |
| POST | `/import` | Import memories |

---

## Core Concepts

### Memory Realms

Three realms for organizing memories:

| Realm | Purpose | Example |
|-------|---------|---------|
| **private** | Personal agent memory | Internal reasoning, drafts |
| **shared** | Shared knowledge | Facts, decisions, discoveries |
| **thread** | Cross-agent dialogue | Conversations between agents |

### Memory Layers

Memories flow through layers based on access patterns:

```
sensory (6h decay) → working (3d decay) → long_term (30d decay) → cortex (permanent)
```

High-access memories get promoted. Neglected memories decay.

### Session Notes

Leave structured notes for session continuity:

```python
cortex.session_save(
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
result = cortex.memory_convergence("best database choice")
# Returns: {"convergence_type": "HARMONY", "converging_agents": ["agent1", "agent2"]}
```

### Agent Profiles

One built-in agent, with runtime registration for custom agents:

| Agent | Specialization | Symbol |
|-------|---------------|--------|
| CLAUDE | General assistance | ◇ |

Register your own with `register_agent()`!

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

## Integration

Neo-Cortex can be used standalone or integrated into any AI agent framework via its Python SDK, MCP Server, or REST API.

```python
from service.cortex_engine import CortexEngine

cortex = CortexEngine(backend='chroma')
cortex.memory_store("Hello from my agent!", visibility="shared")
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
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────────┐  │
│  │ Knowledge │ │  Shared   │ │  Session  │ │  Memory Health  │  │
│  │  Engine   │ │  Memory   │ │  Engine   │ │     Engine      │  │
│  └───────────┘ └───────────┘ └───────────┘ └─────────────────┘  │
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

- Built with love by the Neo-Cortex contributors
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
