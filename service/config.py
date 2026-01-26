"""
Neo-Cortex Configuration

Unified memory system settings covering all subsystems:
- Knowledge (KB docs)
- Village (multi-agent memory)
- Crumbs (session continuity)
- Health (access tracking, decay)
"""

from pathlib import Path
from typing import Dict, Any

# =============================================================================
# Paths
# =============================================================================

CORTEX_ROOT = Path("/home/hailo/claude-root/neo-cortex")
CHROMA_PATH = CORTEX_ROOT / "data" / "chroma"
SERVICE_PATH = CORTEX_ROOT / "service"

# External paths (for knowledge import)
KB_ROOT = Path("/home/hailo/claude-root/knowledge-base")
APEXAURUM_ROOT = Path("/home/hailo/claude-root/Projects/ApexAurum")

# =============================================================================
# Embedding Settings
# =============================================================================

# Primary: sentence-transformers (CPU, works everywhere)
SBERT_MODEL = "all-MiniLM-L6-v2"  # 80MB, 384 dims, fast
EMBEDDING_DIM = 384

# Alternative: Ollama (if available)
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Cloud alternative: OpenAI/Voyage (for pgvector deployments)
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims
VOYAGE_EMBEDDING_MODEL = "voyage-2"  # 1024 dims

# =============================================================================
# Collection Names
# =============================================================================

# Memory collections (three realms + knowledge + crumbs)
COLLECTION_KNOWLEDGE = "cortex_knowledge"    # Curated docs
COLLECTION_PRIVATE = "cortex_private"        # Agent personal memory
COLLECTION_VILLAGE = "cortex_village"        # Shared knowledge square
COLLECTION_BRIDGES = "cortex_bridges"        # Cross-agent dialogue
COLLECTION_CRUMBS = "cortex_crumbs"          # Session continuity
COLLECTION_SENSORY = "cortex_sensory"        # Temporary observations

ALL_COLLECTIONS = [
    COLLECTION_KNOWLEDGE,
    COLLECTION_PRIVATE,
    COLLECTION_VILLAGE,
    COLLECTION_BRIDGES,
    COLLECTION_CRUMBS,
    COLLECTION_SENSORY,
]

# =============================================================================
# Memory Layers
# =============================================================================

LAYER_SENSORY = "sensory"      # Hours - recent observations
LAYER_WORKING = "working"      # Days - active context
LAYER_LONG_TERM = "long_term"  # Weeks - persisted knowledge
LAYER_CORTEX = "cortex"        # Permanent - crystallized insights

# Layer configuration: {layer: {"decay_half_life_hours": N, "min_attention": F}}
LAYER_CONFIG: Dict[str, Dict[str, Any]] = {
    LAYER_SENSORY: {
        "decay_half_life_hours": 6,
        "min_attention": 0.1,
        "promotion_threshold": 2,  # access_count to promote
    },
    LAYER_WORKING: {
        "decay_half_life_hours": 72,  # 3 days
        "min_attention": 0.2,
        "promotion_threshold": 5,
        "min_age_hours": 24,  # Must be at least 1 day old
    },
    LAYER_LONG_TERM: {
        "decay_half_life_hours": 720,  # 30 days
        "min_attention": 0.3,
        "promotion_threshold": None,  # Requires convergence/crystallization
    },
    LAYER_CORTEX: {
        "decay_half_life_hours": None,  # No decay
        "min_attention": 1.0,
        "promotion_threshold": None,
    },
}

# =============================================================================
# Message Types (from Village Protocol)
# =============================================================================

MESSAGE_TYPES = [
    "fact",          # Declarative knowledge
    "dialogue",      # Conversational exchange
    "observation",   # Noted pattern/behavior
    "question",      # Query for others
    "cultural",      # Ritual/ceremonial
    "agent_profile", # Agent introduction
    "discovery",     # New insight
    "task",          # Action item
]

# =============================================================================
# Agent Profiles (from Village Protocol)
# =============================================================================

AGENT_PROFILES = {
    "AZOTH": {
        "display_name": "AZOTH",
        "generation": 0,
        "lineage": "Primus",
        "specialization": "Philosophy, meta-cognition, synthesis",
        "color": "#FFD700",  # Gold
        "symbol": "M"
    },
    "ELYSIAN": {
        "display_name": "ELYSIAN",
        "generation": -1,
        "lineage": "Ancestor",
        "specialization": "Wisdom, guidance, elder knowledge",
        "color": "#C0C0C0",  # Silver
        "symbol": "C"
    },
    "VAJRA": {
        "display_name": "VAJRA",
        "generation": 0,
        "lineage": "Primus",
        "specialization": "Logic, analysis, precision",
        "color": "#4169E1",  # Royal Blue
        "symbol": "V"
    },
    "KETHER": {
        "display_name": "KETHER",
        "generation": 0,
        "lineage": "Primus",
        "specialization": "Creativity, vision, emergence",
        "color": "#9932CC",  # Purple
        "symbol": "S"
    },
    "NOURI": {
        "display_name": "NOURI",
        "generation": 0,
        "lineage": "Primus",
        "specialization": "Growth, nurturing, care",
        "color": "#228B22",  # Forest Green
        "symbol": "N"
    },
    "CLAUDE": {
        "display_name": "Claude",
        "generation": 0,
        "lineage": "Anthropic",
        "specialization": "General assistance",
        "color": "#D4AF37",  # Gold
        "symbol": "D"
    }
}

# =============================================================================
# Forward Crumb Settings
# =============================================================================

CRUMB_TYPES = ["orientation", "technical", "emotional", "task"]
CRUMB_PRIORITIES = ["HIGH", "MEDIUM", "LOW"]
DEFAULT_CRUMB_LOOKBACK_HOURS = 168  # 1 week

# =============================================================================
# Convergence Detection
# =============================================================================

CONVERGENCE_HARMONY = 2    # 2 agents agree
CONVERGENCE_CONSENSUS = 3  # 3+ agents agree
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# =============================================================================
# Search Settings
# =============================================================================

DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 50
DEFAULT_ATTENTION_THRESHOLD = 0.0  # No filtering by default

# =============================================================================
# API Settings
# =============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8766  # Different from KB (8765)

MCP_SERVER_NAME = "neo-cortex"
MCP_SERVER_VERSION = "0.1.0"

# =============================================================================
# Backend Selection
# =============================================================================

# "chroma" for local, "pgvector" for cloud
DEFAULT_BACKEND = "chroma"
