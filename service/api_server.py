#!/usr/bin/env python3
"""
Neo-Cortex REST API
====================

FastAPI server providing HTTP access to the unified memory system.

Usage:
    python service/api_server.py
    uvicorn service.api_server:app --host 0.0.0.0 --port 8766
    ./cortex-api

Endpoints:
    GET  /                         - API info
    GET  /health                   - Health check
    GET  /stats                    - Cortex statistics

    # Shared Memory
    POST /memory/store             - Store a memory
    GET  /memory/search            - Search memories
    POST /memory/search            - Advanced search
    POST /memory/convergence       - Detect convergence
    GET  /memory/stats             - Memory statistics

    # Agents
    GET  /agents                   - List agents
    GET  /agents/{id}              - Get agent profile
    POST /agents/register          - Register new agent

    # Sessions
    POST /sessions/save            - Save session note
    GET  /sessions                 - Get recent sessions
    GET  /sessions/tasks           - Get unfinished tasks

    # Memory Health
    GET  /memory/health            - Health report
    GET  /memory/stale/{coll}      - Get stale memories
    GET  /memory/duplicates/{coll} - Get duplicates
    POST /memory/consolidate       - Consolidate memories
    POST /memory/promote/{coll}    - Run promotions

    # Import/Export
    POST /export                   - Export memory core
    POST /import                   - Import memory core
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from service.cortex_engine import get_engine, CortexEngine
from service.storage.base import MemoryCore
from service.config import ALL_COLLECTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortex-api")

API_HOST = "0.0.0.0"
API_PORT = 8766

app = FastAPI(
    title="Neo-Cortex Memory API",
    description=(
        "REST API for the Neo-Cortex unified memory system. "
        "Shared Memory, Sessions, Memory Health, and more."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cortex: CortexEngine = None


def get_cortex() -> CortexEngine:
    global _cortex
    if _cortex is None:
        _cortex = get_engine()
    return _cortex


# ============================================================================
# Request/Response Models
# ============================================================================

class MemoryStoreRequest(BaseModel):
    content: str = Field(..., description="Message content")
    visibility: str = Field("shared", description="private/shared/thread")
    message_type: str = Field("dialogue", description="fact/dialogue/observation/question/cultural/discovery")
    responding_to: Optional[List[str]] = Field(None, description="Message IDs this responds to")
    conversation_thread: Optional[str] = Field(None, description="Thread identifier")
    related_agents: Optional[List[str]] = Field(None, description="Related agent IDs")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    agent_filter: Optional[str] = Field(None, description="Filter by agent ID")
    visibility: str = Field("shared", description="Which realm to search")
    include_threads: bool = Field(True, description="Include thread messages")
    n_results: int = Field(10, ge=1, le=50, description="Max results")


class ConvergenceRequest(BaseModel):
    query: str = Field(..., description="Topic to check")
    min_agents: int = Field(2, ge=2, le=10, description="Min agents for convergence")
    similarity_threshold: float = Field(0.75, ge=0.5, le=1.0, description="Similarity threshold")


class RegisterAgentRequest(BaseModel):
    agent_id: str = Field(..., description="Agent ID (uppercase)")
    display_name: str = Field(..., description="Display name")
    generation: int = Field(..., description="Generation number")
    lineage: str = Field(..., description="Lineage description")
    specialization: str = Field(..., description="What this agent specializes in")
    origin_story: Optional[str] = Field(None, description="Agent description")


class SessionSaveRequest(BaseModel):
    session_summary: str = Field(..., description="What happened this session")
    key_discoveries: Optional[List[str]] = Field(None, description="Important findings")
    unfinished_business: Optional[List[str]] = Field(None, description="Tasks to continue")
    references: Optional[Dict[str, Any]] = Field(None, description="Message/thread references")
    if_disoriented: Optional[List[str]] = Field(None, description="Orientation instructions")
    priority: str = Field("MEDIUM", description="HIGH/MEDIUM/LOW")
    session_type: str = Field("orientation", description="orientation/technical/emotional/task")


class ConsolidateRequest(BaseModel):
    collection: str = Field(..., description="Collection name")
    id1: str = Field(..., description="First memory ID")
    id2: str = Field(..., description="Second memory ID")
    keep_both: bool = Field(False, description="Keep both after consolidation")


class ExportRequest(BaseModel):
    agent_id: Optional[str] = Field(None, description="Filter by agent")
    collections: Optional[List[str]] = Field(None, description="Which collections")


class ImportRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="MemoryCore JSON data")
    re_embed: bool = Field(True, description="Regenerate embeddings")


# ============================================================================
# Core Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Neo-Cortex Memory API",
        "version": "1.0.0",
        "docs": "/docs",
        "ui": "/ui",
        "endpoints": {
            "stats": "/stats",
            "health": "/health",
            "memory": "/memory/*",
            "agents": "/agents",
            "sessions": "/sessions/*",
            "export": "/export",
            "import": "/import"
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    try:
        cortex = get_cortex()
        s = cortex.stats()
        return {
            "status": "healthy",
            "backend": s.get("backend", "?"),
            "total_memories": s.get("total_memories", 0),
            "registered_agents": s.get("registered_agents", 0)
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


@app.get("/stats", tags=["Info"])
async def get_stats():
    cortex = get_cortex()
    s = cortex.stats()
    if not s.get("success"):
        raise HTTPException(status_code=500, detail=s.get("error", "Unknown error"))
    return s


# ============================================================================
# Shared Memory Endpoints
# ============================================================================

@app.post("/memory/store", tags=["Memory"])
async def memory_store(request: MemoryStoreRequest):
    """Store a memory."""
    cortex = get_cortex()
    result = cortex.memory_store(
        content=request.content,
        visibility=request.visibility,
        message_type=request.message_type,
        responding_to=request.responding_to,
        conversation_thread=request.conversation_thread,
        related_agents=request.related_agents,
        tags=request.tags,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Store failed"))
    return result


@app.get("/memory/search", tags=["Memory"])
async def memory_search_get(
    q: str = Query(..., description="Search query"),
    agent: Optional[str] = Query(None, description="Filter by agent"),
    visibility: str = Query("shared", description="Realm to search"),
    n: int = Query(10, ge=1, le=50, description="Max results"),
):
    """Search shared memory via GET."""
    cortex = get_cortex()
    result = cortex.memory_search(
        query=q,
        agent_filter=agent,
        visibility=visibility,
        n_results=n,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Search failed"))
    return result


@app.post("/memory/search", tags=["Memory"])
async def memory_search_post(request: MemorySearchRequest):
    """Advanced memory search via POST."""
    cortex = get_cortex()
    result = cortex.memory_search(
        query=request.query,
        agent_filter=request.agent_filter,
        visibility=request.visibility,
        include_threads=request.include_threads,
        n_results=request.n_results,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Search failed"))
    return result


@app.post("/memory/convergence", tags=["Memory"])
async def detect_convergence(request: ConvergenceRequest):
    """Detect convergence on a topic."""
    cortex = get_cortex()
    result = cortex.memory_convergence(
        query=request.query,
        min_agents=request.min_agents,
        similarity_threshold=request.similarity_threshold,
    )
    return result


@app.get("/memory/stats", tags=["Memory"])
async def memory_stats():
    """Get shared memory statistics."""
    cortex = get_cortex()
    result = cortex.shared.stats()
    return result


# ============================================================================
# Agent Endpoints
# ============================================================================

@app.get("/agents", tags=["Agents"])
async def list_agents():
    """List all registered agents."""
    cortex = get_cortex()
    return cortex.list_agents()


@app.get("/agents/{agent_id}", tags=["Agents"])
async def get_agent(agent_id: str):
    """Get a specific agent's profile."""
    cortex = get_cortex()
    profile = cortex.get_agent_profile(agent_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return {"agent_id": agent_id, "profile": profile}


@app.post("/agents/register", tags=["Agents"])
async def register_agent(request: RegisterAgentRequest):
    """Register a new agent."""
    cortex = get_cortex()
    result = cortex.register_agent(
        agent_id=request.agent_id,
        display_name=request.display_name,
        generation=request.generation,
        lineage=request.lineage,
        specialization=request.specialization,
        origin_story=request.origin_story,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Registration failed"))
    return result


# ============================================================================
# Session Endpoints
# ============================================================================

@app.post("/sessions/save", tags=["Sessions"])
async def session_save(request: SessionSaveRequest):
    """Save a session note for future instances."""
    cortex = get_cortex()
    result = cortex.session_save(
        session_summary=request.session_summary,
        key_discoveries=request.key_discoveries,
        unfinished_business=request.unfinished_business,
        references=request.references,
        if_disoriented=request.if_disoriented,
        priority=request.priority,
        session_type=request.session_type,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to save session"))
    return result


@app.get("/sessions", tags=["Sessions"])
async def session_recall(
    limit: int = Query(10, ge=1, le=50, description="Max sessions"),
    hours: int = Query(168, ge=1, le=720, description="Lookback hours"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    session_type: Optional[str] = Query(None, description="Filter by type"),
):
    """Get recent session notes."""
    cortex = get_cortex()
    result = cortex.session_recall(
        lookback_hours=hours,
        priority_filter=priority,
        session_type=session_type,
        limit=limit,
    )
    return result


@app.get("/sessions/tasks", tags=["Sessions"])
async def get_unfinished_tasks():
    """Get unfinished tasks from recent sessions."""
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
    cortex = get_cortex()
    coll_list = collections.split(",") if collections else None
    result = cortex.health_report(collections=coll_list)
    return result


@app.get("/memory/stale/{collection}", tags=["Memory Health"])
async def get_stale_memories(
    collection: str,
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(50, ge=1, le=200),
):
    if collection not in ALL_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {collection}")
    cortex = get_cortex()
    return cortex.get_stale_memories(collection=collection, days_threshold=days, limit=limit)


@app.get("/memory/duplicates/{collection}", tags=["Memory Health"])
async def get_duplicates(
    collection: str,
    threshold: float = Query(0.95, ge=0.8, le=1.0),
    limit: int = Query(20, ge=1, le=100),
):
    if collection not in ALL_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {collection}")
    cortex = get_cortex()
    return cortex.get_duplicate_candidates(collection=collection, similarity_threshold=threshold, limit=limit)


@app.post("/memory/consolidate", tags=["Memory Health"])
async def consolidate_memories(request: ConsolidateRequest):
    if request.collection not in ALL_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {request.collection}")
    cortex = get_cortex()
    result = cortex.consolidate_memories(collection=request.collection, id1=request.id1, id2=request.id2, keep_both=request.keep_both)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Consolidation failed"))
    return result


@app.post("/memory/promote/{collection}", tags=["Memory Health"])
async def run_promotions(collection: str):
    if collection not in ALL_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid collection: {collection}")
    cortex = get_cortex()
    return cortex.run_promotions(collection=collection)


# ============================================================================
# Import/Export Endpoints
# ============================================================================

@app.post("/export", tags=["Import/Export"])
async def export_memories(request: ExportRequest = Body(default=ExportRequest())):
    cortex = get_cortex()
    try:
        core = cortex.export_memory_core(agent_id=request.agent_id, collections=request.collections)
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
    cortex = get_cortex()
    try:
        core = MemoryCore.from_dict(request.data)
        stats = cortex.import_memory_core(core, re_embed=request.re_embed)
        return {"success": True, "imported": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Convenience Endpoints
# ============================================================================

@app.get("/q/{query:path}", tags=["Convenience"])
async def quick_search(query: str):
    """Quick search endpoint."""
    cortex = get_cortex()
    result = cortex.memory_search(query, n_results=3)
    if not result.get("success"):
        return {"query": query, "results": []}
    return {
        "query": query,
        "results": [
            {"agent": m.get("agent_id", "?"), "content": m.get("content", "")[:100], "similarity": m.get("similarity", 0)}
            for m in result.get("messages", [])
        ]
    }


@app.post("/remember", tags=["Convenience"])
async def quick_remember(
    content: str = Body(..., embed=True),
    tags: Optional[List[str]] = Body(None, embed=True),
):
    """Quick way to store a memory."""
    cortex = get_cortex()
    return cortex.memory_store(content=content, visibility="shared", message_type="observation", tags=tags)


# ============================================================================
# Web UI
# ============================================================================

WEB_DIR = Path(__file__).parent.parent / "web"


@app.get("/ui", include_in_schema=False)
async def serve_ui():
    """Serve the web dashboard."""
    return FileResponse(WEB_DIR / "index.html")


# Mount static files after all API routes
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


# ============================================================================
# Main
# ============================================================================

def main():
    import uvicorn
    logger.info(f"Starting Neo-Cortex API server on {API_HOST}:{API_PORT}")
    logger.info(f"Dashboard: http://localhost:{API_PORT}/ui")
    logger.info(f"API Docs:  http://localhost:{API_PORT}/docs")
    try:
        cortex = get_cortex()
        s = cortex.stats()
        logger.info(f"Cortex loaded: {s.get('total_memories', 0)} memories")
    except Exception as e:
        logger.error(f"Failed to load cortex: {e}")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")


if __name__ == "__main__":
    main()
