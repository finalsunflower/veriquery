"""
FastAPI application entry point for VeriQuery.

Creates and configures the FastAPI application instance with:
  - Lifespan management (startup/shutdown)
  - CORS middleware for cross-origin requests
  - Request tracing middleware (per-request ID + timing)
  - Six business routers under /api/v1
  - Global exception handlers
  - Health check (/health) and root (/) endpoints
"""

import os
import asyncio
import time
import uuid

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

import logging
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import get_settings, create_cleanup_manager
from api.error_handlers import setup_error_handlers
from api.routers import documents, chat, pinout, compare, circuit, erc

settings = get_settings()
logger = logging.getLogger(__name__)

_start_time = time.time()
_storage_writable_cache = {"checked": False, "result": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown.

    Startup:
      1. Ensure data directories exist
      2. Validate configuration
      3. Check and repair SQLite databases
      4. Launch background orphan-data cleanup task
      5. Preload critical services (embedding model, knowledge graph)

    Shutdown:
      1. Cancel background cleanup task if still running
      2. Release GPU resources via ServiceContainer.cleanup()

    Args:
        app: FastAPI application instance (injected by framework).
    """
    startup_start = time.time()

    try:
        settings.ensure_directories()
        settings.validate_config()

        try:
            from core.sqlite_utils import check_database_health, repair_corrupted_database, safe_delete_database

            db_paths_to_check = [
                getattr(settings, 'table_db_path', './data/tables.db'),
                './data/knowledge_graph.db',
            ]

            for db_path in db_paths_to_check:
                if not check_database_health(db_path):
                    logger.warning(f"Database possibly corrupted: {db_path}")
                    if repair_corrupted_database(db_path):
                        logger.info(f"Database auto-repaired: {db_path}")
                    else:
                        logger.error(f"Database unrepairable, deleted: {db_path}")
                        safe_delete_database(db_path)
            logger.info("Database health checks completed")
        except Exception as e:
            logger.warning(f"Database health check skipped: {e}")

        llm_info = f"HuggingFace/{settings.LLM_MODEL}" if settings.USE_HUGGINGFACE else "not configured"
        logger.info(f"VeriQuery API starting {settings.API_HOST}:{settings.API_PORT} | LLM:{llm_info} | Vector:ChromaDB")

        cleanup_task = None
        try:
            mgr = create_cleanup_manager(settings)
            cleanup_task = asyncio.create_task(_background_cleanup(mgr))
        except Exception as e:
            logger.warning(f"Cleanup task creation failed: {e}")

        try:
            from api.dependencies import get_service_container
            container = get_service_container()
            await container.preload_critical_services()
        except Exception as e:
            logger.warning(f"Critical service preload failed (will load on first use): {e}")

        logger.info(f"Startup completed in {time.time() - startup_start:.2f}s")

        yield

        if cleanup_task and not cleanup_task.done():
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

    except asyncio.CancelledError:
        logger.info("Shutdown signal received, closing gracefully...")
        raise
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    finally:
        try:
            from api.dependencies import get_service_container
            container = get_service_container()
            await container.cleanup()
            logger.info("VeriQuery API shutdown complete")
        except Exception as e:
            logger.warning(f"Resource cleanup error: {e}")


async def _background_cleanup(cleanup_manager):
    """Background task to clean orphan data from previous runs.

    Orphan data can accumulate when document deletion fails partway through
    (e.g. server crash), leaving vectors/BM25/table/image entries without
    a corresponding document record.

    Args:
        cleanup_manager: CleanupManager instance for coordinating storage backends.
    """
    try:
        stats = await cleanup_manager.cleanup_orphan_data()
        if stats.orphan_documents > 0:
            logger.info(f"Cleanup: {stats.orphan_documents} orphan docs, "
                       f"vectors={stats.cleaned_vectors} BM25={stats.cleaned_bm25} "
                       f"tables={stats.cleaned_tables} images={stats.cleaned_images} ({stats.duration_seconds:.1f}s)")
        else:
            logger.info("Cleanup: no orphan data found")

        cleanup_manager.cleanup_stale_backups(max_backups=5)
        await cleanup_manager.scan_orphan_uploads()

    except asyncio.TimeoutError:
        logger.warning("Background cleanup timed out, will retry on next startup")
    except Exception as e:
        logger.warning(f"Background cleanup failed: {e}")


app = FastAPI(
    title="VeriQuery API",
    description="Electronic component datasheet intelligent Q&A system",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

_cors_origins = list(settings.CORS_ORIGINS)

if settings.DEBUG:
    _cors_origins.extend([
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)


@app.middleware("http")
async def request_tracing(request: Request, call_next):
    """HTTP middleware that assigns a unique request ID and logs timing.

    Generates a short UUID (8 chars) per request, stores it in
    request.state.request_id, adds X-Request-ID to the response header,
    and logs method, path, status code, and elapsed time.

    Args:
        request: Current HTTP request (injected by FastAPI).
        call_next: Callable to pass the request to the next handler.
    """
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.time()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        elapsed = time.time() - start
        logger.info(f"[{request_id}] {request.method} {request.url.path} {response.status_code} ({elapsed:.2f}s)")
        return response
    except Exception as e:
        logger.error(f"[{request_id}] {request.method} {request.url.path} ERROR: {e}")
        raise


app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(pinout.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api/v1")
app.include_router(circuit.router, prefix="/api/v1")
app.include_router(erc.router, prefix="/api/v1")

setup_error_handlers(app)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring, container probes, and load balancers.

    Checks:
      - api: API service status (always "online" if responding)
      - config: Configuration loaded status
      - storage: Upload directory writability (cached after first check)
      - vector_store: ChromaDB data directory status
      - llm: LLM configuration status

    Returns:
        dict with status/version/timestamp/uptime/services on success,
        JSONResponse with 503 on failure.
    """
    try:
        health_details = {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "timestamp": datetime.now().isoformat(),
            "uptime": f"{time.time() - _start_time:.0f}s",
            "services": {
                "api": "online",
                "config": "loaded"
            }
        }

        global _storage_writable_cache
        if not _storage_writable_cache["checked"]:
            try:
                test_file = settings.UPLOAD_DIR / ".health_check"
                test_file.write_text("test")
                test_file.unlink()
                _storage_writable_cache["result"] = "writable"
            except Exception as e:
                _storage_writable_cache["result"] = f"error: {str(e)}"
            _storage_writable_cache["checked"] = True
        health_details["services"]["storage"] = _storage_writable_cache["result"]

        try:
            chroma_dir = Path(settings.CHROMA_PERSIST_DIR)
            if chroma_dir.exists() and any(chroma_dir.iterdir()):
                health_details["services"]["vector_store"] = "available"
            else:
                health_details["services"]["vector_store"] = "not_initialized"
        except Exception as e:
            health_details["services"]["vector_store"] = f"error: {str(e)}"

        if settings.USE_HUGGINGFACE:
            health_details["services"]["llm"] = "configured"
        else:
            health_details["services"]["llm"] = "disabled"

        return health_details

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/")
async def root():
    """Root endpoint returning system info and available features."""
    return {
        "name": "VeriQuery API",
        "description": "Electronic component datasheet intelligent Q&A system",
        "version": settings.APP_VERSION,
        "features": [
            "Document management (upload/delete/list)",
            "Intelligent Q&A (RAG + citation)",
            "Pinout visualization (SVG rendering)",
            "Parameter comparison (multi-chip)",
            "Circuit retrieval (multimodal)",
            "ERC check (electrical rules)"
        ],
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn

    is_development = os.getenv("ENVIRONMENT", "development").lower() == "development"

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=is_development
    )
