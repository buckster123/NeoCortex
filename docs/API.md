# Neo-Cortex REST API Reference

Base URL: `http://localhost:8766`

OpenAPI docs available at `/docs` when server is running.

---

## Core Endpoints

### GET /
API information and available endpoints.

**Response:**
```json
{
  "name": "Neo-Cortex Memory API",
  "version": "1.0.0",
  "docs": "/docs",
  "endpoints": {...}
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "backend": "chroma",
  "total_memories": 42,
  "registered_agents": 1
}
```

### GET /stats
Comprehensive cortex statistics.

**Response:**
```json
{
  "success": true,
  "backend": "chroma",
  "embedding_dimension": 384,
  "current_agent": "CLAUDE",
  "registered_agents": 1,
  "total_memories": 42,
  "collections": {
    "cortex_shared": 30,
    "cortex_private": 10,
    "cortex_sessions": 2
  }
}
```

---

## Memory Endpoints

### POST /memory/store
Store a memory.

**Request Body:**
```json
{
  "content": "string (required)",
  "visibility": "private|shared|thread",
  "message_type": "fact|dialogue|observation|question|cultural|discovery",
  "responding_to": ["message_id_1"],
  "conversation_thread": "thread_id",
  "related_agents": ["CLAUDE"],
  "tags": ["important", "architecture"]
}
```

**Response:**
```json
{
  "success": true,
  "id": "memory_CLAUDE_1234567890.123",
  "agent_id": "CLAUDE",
  "visibility": "shared",
  "collection": "cortex_shared"
}
```

### GET /memory/search
Search for memories.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| q | string | required | Search query |
| agent | string | null | Filter by agent ID |
| visibility | string | "shared" | Realm to search |
| n | int | 10 | Max results |

**Example:**
```
GET /memory/search?q=memory%20system&agent=CLAUDE&n=5
```

**Response:**
```json
{
  "success": true,
  "query": "memory system",
  "count": 5,
  "messages": [
    {
      "id": "memory_CLAUDE_123...",
      "content": "The memory system should...",
      "agent_id": "CLAUDE",
      "visibility": "shared",
      "similarity": 0.89,
      "tags": ["memory"]
    }
  ]
}
```

### POST /memory/search
Advanced search with full options.

**Request Body:**
```json
{
  "query": "string (required)",
  "agent_filter": "CLAUDE",
  "visibility": "shared",
  "include_threads": true,
  "n_results": 10
}
```

### POST /memory/convergence
Detect convergence on a topic.

**Request Body:**
```json
{
  "query": "best database choice",
  "min_agents": 2,
  "similarity_threshold": 0.75
}
```

**Response:**
```json
{
  "success": true,
  "convergence_type": "HARMONY",
  "converging_agents": ["agent1", "agent2"],
  "topic": "best database choice",
  "evidence": [...]
}
```

### GET /memory/stats
Get memory-specific statistics.

---

## Agent Endpoints

### GET /agents
List all registered agents.

**Response:**
```json
{
  "success": true,
  "current_agent": "CLAUDE",
  "count": 1,
  "agents": [
    {
      "id": "CLAUDE",
      "display_name": "Claude",
      "generation": 0,
      "lineage": "Assistant",
      "specialization": "General assistance",
      "symbol": "◇"
    }
  ]
}
```

### GET /agents/{agent_id}
Get a specific agent's profile.

**Response:**
```json
{
  "agent_id": "CLAUDE",
  "profile": {
    "display_name": "Claude",
    "generation": 0,
    "lineage": "Assistant",
    "specialization": "General assistance",
    "color": "#FFD700",
    "symbol": "◇"
  }
}
```

### POST /agents/register
Register a new agent.

**Request Body:**
```json
{
  "agent_id": "RESEARCHER",
  "display_name": "Researcher",
  "generation": 0,
  "lineage": "Custom",
  "specialization": "Research and analysis",
  "origin_story": "Created for research tasks..."
}
```

---

## Session Endpoints

### POST /sessions/save
Save a session note for future instances.

**Request Body:**
```json
{
  "session_summary": "string (required)",
  "key_discoveries": ["Discovery 1", "Discovery 2"],
  "unfinished_business": ["Task 1", "Task 2"],
  "references": {
    "message_ids": ["id1", "id2"],
    "thread_ids": ["thread1"]
  },
  "if_disoriented": ["Check the auth/ folder", "Run tests first"],
  "priority": "HIGH|MEDIUM|LOW",
  "session_type": "orientation|technical|emotional|task"
}
```

**Response:**
```json
{
  "success": true,
  "id": "session_CLAUDE_20260126_123456"
}
```

### GET /sessions
Get recent session notes.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 10 | Max sessions |
| hours | int | 168 | Lookback period |
| priority | string | null | Filter by priority |
| session_type | string | null | Filter by type |

**Response:**
```json
{
  "success": true,
  "sessions": [...],
  "unfinished_tasks": ["Task 1", "Task 2"],
  "summary": {
    "total_found": 5,
    "by_priority": {"HIGH": 2, "MEDIUM": 3, "LOW": 0}
  }
}
```

### GET /sessions/tasks
Get unfinished tasks from sessions.

**Response:**
```json
{
  "tasks": ["Task 1", "Task 2", "Task 3"],
  "count": 3
}
```

---

## Knowledge Endpoints

### GET /knowledge/search
Search the knowledge base with optional topic filter.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| q | string | required | Search query |
| topic | string | null | Filter by topic name |
| n | int | 10 | Max results (1-50) |

**Example:**
```
GET /knowledge/search?q=dependency%20injection&topic=fastapi&n=5
```

**Response:**
```json
{
  "success": true,
  "query": "dependency injection",
  "topic": "fastapi",
  "count": 5,
  "results": [
    {
      "id": "kb_3ad0c4a274a3d6ec",
      "content": "[fastapi/FastAPI Complete Guide] Basic Dependency...",
      "tags": ["fastapi", "fastapi-complete-guide"],
      "similarity": 0.59,
      "collection": "cortex_knowledge"
    }
  ]
}
```

### GET /knowledge/topics
List all knowledge topics with chunk counts.

**Response:**
```json
{
  "success": true,
  "topics": [
    {"name": "anthropic", "count": 377},
    {"name": "dev-tools", "count": 141},
    {"name": "fastapi", "count": 35}
  ],
  "total_topics": 20
}
```

### GET /knowledge/stats
Get knowledge base statistics.

**Response:**
```json
{
  "success": true,
  "total_chunks": 920,
  "total_topics": 20,
  "avg_chunk_size": 427,
  "total_chars": 392878,
  "top_topics": [["anthropic", 377], ["dev-tools", 141]]
}
```

### POST /knowledge/ingest
Upload a markdown file to ingest into the knowledge base.

**Content-Type:** `multipart/form-data`

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | file | yes | Markdown file (.md, .txt) |
| topic | string | yes | Topic name for categorization |
| title | string | no | Document title (auto-detected from H1 if omitted) |

**Response:**
```json
{
  "success": true,
  "chunks": 12,
  "topic": "my-topic",
  "title": "My Document",
  "source": "my-document.md"
}
```

### POST /knowledge/ingest-text
Ingest raw text/markdown content.

**Request Body:**
```json
{
  "content": "# My Document\n\n## Section 1\n\nContent here...",
  "topic": "my-topic",
  "title": "My Document"
}
```

**Response:**
```json
{
  "success": true,
  "chunks": 3,
  "topic": "my-topic",
  "title": "My Document"
}
```

### DELETE /knowledge/topic/{topic}
Delete all knowledge chunks for a topic.

**Response:**
```json
{
  "success": true,
  "topic": "my-topic",
  "deleted": 12
}
```

---

## Memory Health Endpoints

### GET /memory/health
Get memory health report.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| collections | string | Comma-separated collection names |

**Response:**
```json
{
  "success": true,
  "total_memories": 42,
  "overall_health": 0.85,
  "needs_attention": false,
  "collections": {
    "cortex_shared": {
      "count": 30,
      "avg_attention_weight": 0.75,
      "stale_count": 2
    }
  }
}
```

### GET /memory/stale/{collection}
Get stale memories in a collection.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| days | int | 30 | Days threshold |
| limit | int | 50 | Max results |

### GET /memory/duplicates/{collection}
Get duplicate candidates.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| threshold | float | 0.95 | Similarity threshold |
| limit | int | 20 | Max pairs |

### POST /memory/consolidate
Consolidate duplicate memories.

**Request Body:**
```json
{
  "collection": "cortex_shared",
  "id1": "memory_id_1",
  "id2": "memory_id_2",
  "keep_both": false
}
```

### POST /memory/promote/{collection}
Run layer promotions.

---

## Import/Export Endpoints

### POST /export
Export memories to portable format.

**Request Body:**
```json
{
  "agent_id": "CLAUDE",
  "collections": ["cortex_shared", "cortex_sessions"]
}
```

**Response:**
```json
{
  "success": true,
  "format": "memory_core_v1",
  "agent_count": 1,
  "memory_count": 25,
  "exported_at": "2026-01-26T12:00:00",
  "data": {...}
}
```

### POST /import
Import memories from portable format.

**Request Body:**
```json
{
  "data": {...},
  "re_embed": true
}
```

---

## Convenience Endpoints

### GET /q/{query}
Ultra-quick search.

**Example:**
```
GET /q/memory%20system
```

**Response:**
```json
{
  "query": "memory system",
  "results": [
    {"agent": "CLAUDE", "content": "...", "similarity": 0.89}
  ]
}
```

### POST /remember
Quick way to store a memory.

**Request Body:**
```json
{
  "content": "Something to remember",
  "tags": ["quick", "note"]
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common status codes:
- `400` - Bad request (invalid parameters)
- `404` - Not found
- `500` - Server error
- `503` - Service unavailable

---

## Rate Limiting

Currently no rate limiting. For production, consider adding nginx or API gateway rate limiting.

---

## Authentication

Currently no authentication. For production, add API key or OAuth.
