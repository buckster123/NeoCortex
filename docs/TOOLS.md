# Neo-Cortex Tool Schemas

Tool schemas for function calling with Claude, OpenAI, or other LLMs.

---

## Memory Tools

### memory_store

Store a memory in shared or private memory.

```json
{
  "name": "memory_store",
  "description": "Store a memory in shared or private memory. Use 'shared' for shared knowledge, 'private' for personal notes, 'thread' for cross-agent dialogue.",
  "input_schema": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "The message content to store"
      },
      "visibility": {
        "type": "string",
        "enum": ["private", "shared", "thread"],
        "description": "Where to store: private (personal), shared (shared), thread (cross-agent)"
      },
      "message_type": {
        "type": "string",
        "enum": ["fact", "dialogue", "observation", "question", "cultural", "discovery"],
        "description": "Type of message"
      },
      "tags": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Tags for categorization"
      }
    },
    "required": ["content"]
  }
}
```

### memory_search

Search shared memory for knowledge and dialogue.

```json
{
  "name": "memory_search",
  "description": "Search shared memory for knowledge and dialogue. Can filter by agent, visibility, or conversation thread.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query text"
      },
      "agent_filter": {
        "type": "string",
        "description": "Filter by agent ID"
      },
      "visibility": {
        "type": "string",
        "enum": ["shared", "private", "all"],
        "description": "Which realm to search"
      },
      "n_results": {
        "type": "integer",
        "description": "Maximum results to return"
      }
    },
    "required": ["query"]
  }
}
```

### memory_convergence

Detect when multiple agents express similar ideas.

```json
{
  "name": "memory_convergence",
  "description": "Detect when multiple agents express similar ideas (convergence). HARMONY = 2 agents agree, CONSENSUS = 3+ agents agree.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Topic/concept to check for convergence"
      },
      "min_agents": {
        "type": "integer",
        "description": "Minimum agents needed for convergence (default: 2)"
      },
      "similarity_threshold": {
        "type": "number",
        "description": "How similar memories must be (default: 0.75)"
      }
    },
    "required": ["query"]
  }
}
```

### memory_stats

Get statistics about the memory system.

```json
{
  "name": "memory_stats",
  "description": "Get statistics about shared memory - message counts, agents, realms.",
  "input_schema": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```

---

## Agent Tools

### register_agent

Register a new agent in the memory system.

```json
{
  "name": "register_agent",
  "description": "Register a new agent in the memory system. Creates a permanent agent profile.",
  "input_schema": {
    "type": "object",
    "properties": {
      "agent_id": {
        "type": "string",
        "description": "Unique identifier (uppercase, e.g., RESEARCHER)"
      },
      "display_name": {
        "type": "string",
        "description": "Display name for the agent"
      },
      "generation": {
        "type": "integer",
        "description": "Generation number (-1=origin, 0=primary, 1+=descendant)"
      },
      "lineage": {
        "type": "string",
        "description": "Lineage or group name"
      },
      "specialization": {
        "type": "string",
        "description": "What this agent specializes in"
      },
      "origin_story": {
        "type": "string",
        "description": "Description of the agent's purpose (optional)"
      }
    },
    "required": ["agent_id", "display_name", "generation", "lineage", "specialization"]
  }
}
```

### list_agents

List all registered agents.

```json
{
  "name": "list_agents",
  "description": "List all registered agents with their profiles.",
  "input_schema": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```

---

## Session Tools

### session_save

Save a session note for future instances.

```json
{
  "name": "session_save",
  "description": "Save a session note for future instances. Use this at the end of a session to help future-you maintain continuity.",
  "input_schema": {
    "type": "object",
    "properties": {
      "session_summary": {
        "type": "string",
        "description": "Brief summary of what happened this session"
      },
      "key_discoveries": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Important findings or insights"
      },
      "unfinished_business": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Tasks, threads, or promises to continue"
      },
      "if_disoriented": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Orientation instructions for confused future-self"
      },
      "priority": {
        "type": "string",
        "enum": ["HIGH", "MEDIUM", "LOW"],
        "description": "Priority level (default: MEDIUM)"
      },
      "session_type": {
        "type": "string",
        "enum": ["orientation", "technical", "emotional", "task"],
        "description": "Type of session note (default: orientation)"
      }
    },
    "required": ["session_summary"]
  }
}
```

### session_recall

Retrieve session notes from previous instances.

```json
{
  "name": "session_recall",
  "description": "Retrieve session notes left by previous instances. Use at the start of a session to restore context and continuity.",
  "input_schema": {
    "type": "object",
    "properties": {
      "lookback_hours": {
        "type": "integer",
        "description": "How far back to search (default: 168 = 1 week)"
      },
      "priority_filter": {
        "type": "string",
        "enum": ["HIGH", "MEDIUM", "LOW"],
        "description": "Filter by priority level"
      },
      "session_type": {
        "type": "string",
        "enum": ["orientation", "technical", "emotional", "task"],
        "description": "Filter by session type"
      },
      "limit": {
        "type": "integer",
        "description": "Maximum sessions to return (default: 10)"
      }
    },
    "required": []
  }
}
```

---

## Memory Health Tools

### memory_health_report

Generate a memory health report.

```json
{
  "name": "memory_health_report",
  "description": "Generate a memory health report showing decay status, access patterns, and maintenance needs.",
  "input_schema": {
    "type": "object",
    "properties": {
      "collections": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Specific collections to report on (omit for all)"
      }
    },
    "required": []
  }
}
```

### memory_get_stale

Get stale memories that need attention.

```json
{
  "name": "memory_get_stale",
  "description": "Get memories that haven't been accessed recently and may need review or archival.",
  "input_schema": {
    "type": "object",
    "properties": {
      "collection": {
        "type": "string",
        "description": "Collection to check"
      },
      "days_threshold": {
        "type": "integer",
        "description": "Days since last access to consider stale (default: 30)"
      },
      "limit": {
        "type": "integer",
        "description": "Maximum results (default: 50)"
      }
    },
    "required": ["collection"]
  }
}
```

### memory_get_duplicates

Find duplicate or near-duplicate memories.

```json
{
  "name": "memory_get_duplicates",
  "description": "Find memories that are very similar and might be duplicates for consolidation.",
  "input_schema": {
    "type": "object",
    "properties": {
      "collection": {
        "type": "string",
        "description": "Collection to check"
      },
      "similarity_threshold": {
        "type": "number",
        "description": "Minimum similarity to consider duplicate (default: 0.95)"
      },
      "limit": {
        "type": "integer",
        "description": "Maximum pairs to return (default: 20)"
      }
    },
    "required": ["collection"]
  }
}
```

### memory_consolidate

Consolidate two similar memories into one.

```json
{
  "name": "memory_consolidate",
  "description": "Merge two similar memories into one, optionally keeping both.",
  "input_schema": {
    "type": "object",
    "properties": {
      "collection": {
        "type": "string",
        "description": "Collection name"
      },
      "id1": {
        "type": "string",
        "description": "First memory ID"
      },
      "id2": {
        "type": "string",
        "description": "Second memory ID"
      },
      "keep_both": {
        "type": "boolean",
        "description": "Keep both after consolidation (default: false)"
      }
    },
    "required": ["collection", "id1", "id2"]
  }
}
```

### memory_run_promotions

Run layer promotions for frequently accessed memories.

```json
{
  "name": "memory_run_promotions",
  "description": "Promote memories with high access counts to higher layers (sensory -> working -> long_term -> cortex).",
  "input_schema": {
    "type": "object",
    "properties": {
      "collection": {
        "type": "string",
        "description": "Collection to run promotions on"
      }
    },
    "required": ["collection"]
  }
}
```

---

## Core Tools

### knowledge_search

Search the knowledge base for documentation.

```json
{
  "name": "knowledge_search",
  "description": "Search the knowledge base for documentation and reference material.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query"
      },
      "n_results": {
        "type": "integer",
        "description": "Number of results to return (default: 5)"
      }
    },
    "required": ["query"]
  }
}
```

### cortex_stats

Get comprehensive cortex statistics.

```json
{
  "name": "cortex_stats",
  "description": "Get comprehensive statistics about the neo-cortex memory system.",
  "input_schema": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```

### cortex_export

Export memories to portable format.

```json
{
  "name": "cortex_export",
  "description": "Export memories to portable JSON format for backup or transfer between local and cloud.",
  "input_schema": {
    "type": "object",
    "properties": {
      "agent_id": {
        "type": "string",
        "description": "Filter by agent (omit for all)"
      },
      "collections": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Which collections to export (omit for all)"
      }
    },
    "required": []
  }
}
```

---

## Using with Claude (Anthropic API)

```python
import anthropic

client = anthropic.Anthropic()

# Add tools to your request
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[
        {
            "name": "memory_search",
            "description": "Search shared memory for knowledge",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ],
    messages=[{"role": "user", "content": "Search for memories about databases"}]
)
```

---

## Using with OpenAI

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Search for memories about databases"}],
    functions=[
        {
            "name": "memory_search",
            "description": "Search shared memory for knowledge",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ]
)
```

---

## Full Schema Export

To get all tool schemas programmatically:

```python
from service.cortex_engine import CORTEX_TOOL_SCHEMAS

# All 16 tools
for name, schema in CORTEX_TOOL_SCHEMAS.items():
    print(f"{name}: {schema['description'][:50]}...")
```
