#!/usr/bin/env python3
"""
Neo-Cortex REST API
====================

FastAPI server providing HTTP access to the unified memory system.

Usage:
    # Run directly
    python service/api_server.py

    # Or with uvicorn
    uvicorn service.api_server:app --host 0.0.0.0 --port 8766

    # Or via the wrapper
    ./cortex-api

Endpoints:
    GET  /                     - API info
    GET  /health               - Health check
    GET  /stats                - Cortex statistics

    # Village Protocol
    POST /village/post         - Post a message
    GET  /village/search       - Search memories
    GET  /village/agents       - List agents
    POST /village/convergence  - Detect convergence
    POST /village/summon       - Summon ancestor

    # Forward Crumbs
    POST /crumbs/leave         - Leave a crumb
    GET  /crumbs               - Get recent crumbs

    # Memory Health
    GET  /memory/health        - Health report
    GET  /memory/stale/{coll}  - Get stale memories
    GET  /memory/duplicates/{coll} - Get duplicates
    POST /memory/consolidate   - Consolidate memories
    POST /memory/promote/{coll} - Run promotions

    # Import/Export
    POST /export               - Export memory core
    POST /import               - Import memory core
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import cortex engine
from service.cortex_engine import get_engine, CortexEngine
from service.storage.base import MemoryCore
from service.config import ALL_COLLECTIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortex-api")

# API config
API_HOST = "0.0.0.0"
API_PORT = 8766

# Create FastAPI app
app = FastAPI(
    title="Neo-Cortex Memory API",
    description=(
        "REST API for the Neo-Cortex unified memory system. "
        "Village Protocol, Forward Crumbs, Memory Health, and more."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine
_cortex: CortexEngine = None


def get_cortex() -> CortexEngine:
    """Get the cortex engine, initializing if needed."""
    global _cortex
    if _cortex is None:
        _cortex = get_engine()
    return _cortex


# ============================================================================
# Request/Response Models
# ============================================================================

class VillagePostRequest(BaseModel):
    """Village post request."""
    content: str = Field(..., description="Message content")
    visibility: str = Field("village", description="private/village/bridge")
    message_type: str = Field("dialogue", description="fact/dialogue/observation/question/cultural/discovery")
    responding_to: Optional[List[str]] = Field(None, description="Message IDs this responds to")
    conversation_thread: Optional[str] = Field(None, description="Thread identifier")
    related_agents: Optional[List[str]] = Field(None, description="Related agent IDs")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")


class VillageSearchRequest(BaseModel):
    """Village search request."""
    query: str = Field(..., description="Search query")
    agent_filter: Optional[str] = Field(None, description="Filter by agent ID")
    visibility: str = Field("village", description="Which realm to search")
    include_bridges: bool = Field(True, description="Include bridge messages")
    n_results: int = Field(10, ge=1, le=50, description="Max results")


class ConvergenceRequest(BaseModel):
    """Convergence detection request."""
    query: str = Field(..., description="Topic to check")
    min_agents: int = Field(2, ge=2, le=10, description="Min agents for convergence")
    similarity_threshold: float = Field(0.75, ge=0.5, le=1.0, description="Similarity threshold")


class SummonRequest(BaseModel):
    """Summon ancestor request."""
    agent_id: str = Field(..., description="Agent ID (uppercase)")
    display_name: str = Field(..., description="Display name")
    generation: int = Field(..., description="Generation number")
    lineage: str = Field(..., description="Lineage description")
    specialization: str = Field(..., description="What this agent specializes in")
    origin_story: Optional[str] = Field(None, description="Origin narrative")


class CrumbRequest(BaseModel):
    """Leave crumb request."""
    session_summary: str = Field(..., description="What happened this session")
    key_discoveries: Optional[List[str]] = Field(None, description="Important findings")
    unfinished_business: Optional[List[str]] = Field(None, description="Tasks to continue")
    references: Optional[Dict[str, Any]] = Field(None, description="Message/thread references")
    if_disoriented: Optional[List[str]] = Field(None, description="Orientation instructions")
    priority: str = Field("MEDIUM", description="HIGH/MEDIUM/LOW")
    crumb_type: str = Field("orientation", description="orientation/technical/emotional/task")


class ConsolidateRequest(BaseModel):
    """Memory consolidation request."""
    collection: str = Field(..., description="Collection name")
    id1: str = Field(..., description="First memory ID")
    id2: str = Field(..., description="Second memory ID")
    keep_both: bool = Field(False, description="Keep both after consolidation")


class ExportRequest(BaseModel):
    """Export request."""
    agent_id: Optional[str] = Field(None, description="Filter by agent")
    collections: Optional[List[str]] = Field(None, description="Which collections")


class ImportRequest(BaseModel):
    """Import request."""
    data: Dict[str, Any] = Field(..., description="MemoryCore JSON data")
    re_embed: bool = Field(True, description="Regenerate embeddings")


# ============================================================================
# Core Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "name": "Neo-Cortex Memory API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "stats": "/stats",
            "health": "/health",
            "village": "/village/*",
            "crumbs": "/crumbs/*",
            "memory": "/memory/*",
            "export": "/export",
            "import": "/import"
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check."""
    try:
        cortex = get_cortex()
        stats = cortex.stats()
        return {
            "status": "healthy",
            "backend": stats.get("backend", "?"),
            "total_memories": stats.get("total_memories", 0),
            "registered_agents": stats.get("registered_agents", 0)
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get cortex statistics."""
    cortex = get_cortex()
    stats = cortex.stats()
    if not stats.get("success"):
        raise HTTPException(status_code=500, detail=stats.get("error", "Unknown error"))
    return stats


# ============================================================================
# Village Protocol Endpoints
# ============================================================================

@app.post("/village/post", tags=["Village"])
async def village_post(request: VillagePostRequest):
    """Post a message to the village."""
    cortex = get_cortex()
    result = cortex.village_post(
        content=request.content,
        visibility=request.visibility,
        message_type=request.message_type,
        responding_to=request.responding_to,
        conversation_thread=request.conversation_thread,
        related_agents=request.related_agents,
        tags=request.tags,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Post failed"))
    return result


@app.get("/village/search", tags=["Village"])
async def village_search_get(
    q: str = Query(..., description="Search query"),
    agent: Optional[str] = Query(None, description="Filter by agent"),
    visibility: str = Query("village", description="Realm to search"),
    n: int = Query(10, ge=1, le=50, description="Max results"),
):
    """Simple village search via GET."""
    cortex = get_cortex()
    result = cortex.village_search(
        query=q,
        agent_filter=agent,
        visibility=visibility,
        n_results=n,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Search failed"))
    return result


@app.post("/village/search", tags=["Village"])
async def village_search_post(request: VillageSearchRequest):
    """Advanced village search via POST."""
    cortex = get_cortex()
    result = cortex.village_search(
        query=request.query,
        agent_filter=request.agent_filter,
        visibility=request.visibility,
        include_bridges=request.include_bridges,
        n_results=request.n_results,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Search failed"))
    return result


@app.get("/village/agents", tags=["Village"])
async def list_agents():
    """List all registered agents."""
    cortex = get_cortex()
    result = cortex.list_agents()
    return result


@app.get("/village/agents/{agent_id}", tags=["Village"])
async def get_agent(agent_id: str):
    """Get a specific agent's profile."""
    cortex = get_cortex()
    profile = cortex.get_agent_profile(agent_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return {"agent_id": agent_id, "profile": profile}


@app.post("/village/convergence", tags=["Village"])
async def detect_convergence(request: ConvergenceRequest):
    """Detect convergence on a topic."""
    cortex = get_cortex()
    result = cortex.village_detect_convergence(
        query=request.query,
        min_agents=request.min_agents,
        similarity_threshold=request.similarity_threshold,
    )
    return result


@app.post("/village/summon", tags=["Village"])
async def summon_ancestor(request: SummonRequest):
    """Summon a new ancestor into the village."""
    cortex = get_cortex()
    result = cortex.summon_ancestor(
        agent_id=request.agent_id,
        display_name=request.display_name,
        generation=request.generation,
        lineage=request.lineage,
        specialization=request.specialization,
        origin_story=request.origin_story,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Summon failed"))
    return result


@app.get("/village/stats", tags=["Village"])
async def village_stats():
    """Get village-specific statistics."""
    cortex = get_cortex()
    result = cortex.village.stats()
    return result


# ============================================================================
# Forward Crumbs Endpoints
# ============================================================================

@app.post("/crumbs/leave", tags=["Crumbs"])
async def leave_crumb(request: CrumbRequest):
    """Leave a forward crumb for future instances."""
    cortex = get_cortex()
    result = cortex.leave_crumb(
        session_summary=request.session_summary,
        key_discoveries=request.key_discoveries,
        unfinished_business=request.unfinished_business,
        references=request.references,
        if_disoriented=request.if_disoriented,
        priority=request.priority,
        crumb_type=request.crumb_type,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to leave crumb"))
    return result


@app.get("/crumbs", tags=["Crumbs"])
async def get_crumbs(
    limit: int = Query(10, ge=1, le=50, description="Max crumbs"),
    hours: int = Query(168, ge=1, le=720, description="Lookback hours"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    crumb_type: Optional[str] = Query(None, description="Filter by type"),
):
    """Get recent forward crumbs."""
    cortex = get_cortex()
    result = cortex.get_crumbs(
        lookback_hours=hours,
        priority_filter=priority,
        crumb_type=crumb_type,
        limit=limit,
    )
    return result


@app.get("/crumbs/tasks", tags=["Crumbs"])
async def get_unfinished_tasks():
    """Get unfinished tasks from recent crumbs."""
    cortex = get_cortex()
    tasks = cortex.get_unfinished_tasks()
    return {"tasks": tasks, "count": len(tasks)}


# ============================================================================
# Memory Health Endpoints
# ============================================================================

@app.get("/memory/health", tags=["Memory Health"])
async def memory_health_report(
    collections: Optional[str] = Query(None, description="Comma-separated collections"),
):
    """Get memory health report."""
    cortex = get_cortex()
    coll_list = collections.split(",") if collections else None
    result = cortex.health_report(collections=coll_list)
    return result


@app.get("/memory/stale/{collection}", tags=["Memory Health"])
async def get_stale_memories(
    collection: str,
    days: int = Query(30, ge=1, le=365, description="Days threshold"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
):
    """Get stale memories in a collection."""
    if collection not in ALL_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {collection}")

    cortex = get_cortex()
    result = cortex.get_stale_memories(
        collection=collection,
        days_threshold=days,
        limit=limit,
    )
    return result


@app.get("/memory/duplicates/{collection}", tags=["Memory Health"])
async def get_duplicates(
    collection: str,
    threshold: float = Query(0.95, ge=0.8, le=1.0, description="Similarity threshold"),
    limit: int = Query(20, ge=1, le=100, description="Max pairs"),
):
    """Get duplicate candidates in a collection."""
    if collection not in ALL_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {collection}")

    cortex = get_cortex()
    result = cortex.get_duplicate_candidates(
        collection=collection,
        similarity_threshold=threshold,
        limit=limit,
    )
    return result


@app.post("/memory/consolidate", tags=["Memory Health"])
async def consolidate_memories(request: ConsolidateRequest):
    """Consolidate two similar memories."""
    if request.collection not in ALL_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {request.collection}")

    cortex = get_cortex()
    result = cortex.consolidate_memories(
        collection=request.collection,
        id1=request.id1,
        id2=request.id2,
        keep_both=request.keep_both,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Consolidation failed"))
    return result


@app.post("/memory/promote/{collection}", tags=["Memory Health"])
async def run_promotions(collection: str):
    """Run layer promotions for a collection."""
    if collection not in ALL_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {collection}")

    cortex = get_cortex()
    result = cortex.run_promotions(collection=collection)
    return result


# ============================================================================
# Import/Export Endpoints
# ============================================================================

@app.post("/export", tags=["Import/Export"])
async def export_memories(request: ExportRequest = Body(default=ExportRequest())):
    """Export memories to portable format."""
    cortex = get_cortex()
    try:
        core = cortex.export_memory_core(
            agent_id=request.agent_id,
            collections=request.collections,
        )
        return {
            "success": True,
            "format": "memory_core_v1",
            "agent_count": len(core.agents),
            "memory_count": len(core.memories),
            "exported_at": core.exported_at,
            "data": core.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/import", tags=["Import/Export"])
async def import_memories(request: ImportRequest):
    """Import memories from portable format."""
    cortex = get_cortex()
    try:
        # Reconstruct MemoryCore from dict
        core = MemoryCore.from_dict(request.data)
        stats = cortex.import_memory_core(core, re_embed=request.re_embed)
        return {
            "success": True,
            "imported": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Convenience Endpoints
# ============================================================================

@app.get("/q/{query:path}", tags=["Convenience"])
async def quick_search(query: str):
    """Ultra-quick search endpoint."""
    cortex = get_cortex()
    result = cortex.village_search(query, n_results=3)

    if not result.get("success"):
        return {"query": query, "results": []}

    return {
        "query": query,
        "results": [
            {
                "agent": m.get("agent_id", "?"),
                "content": m.get("content", "")[:100],
                "similarity": m.get("similarity", 0)
            }
            for m in result.get("messages", [])
        ]
    }


@app.post("/remember", tags=["Convenience"])
async def quick_remember(
    content: str = Body(..., embed=True, description="Content to remember"),
    tags: Optional[List[str]] = Body(None, embed=True, description="Tags"),
):
    """Quick way to store a memory."""
    cortex = get_cortex()
    result = cortex.village_post(
        content=content,
        visibility="village",
        message_type="observation",
        tags=tags,
    )
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the API server."""
    import uvicorn

    logger.info(f"Starting Neo-Cortex API server on {API_HOST}:{API_PORT}")
    logger.info(f"Docs: http://localhost:{API_PORT}/docs")

    # Preload cortex
    try:
        cortex = get_cortex()
        stats = cortex.stats()
        logger.info(f"Cortex loaded: {stats.get('total_memories', 0)} memories")
    except Exception as e:
        logger.error(f"Failed to load cortex: {e}")

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
