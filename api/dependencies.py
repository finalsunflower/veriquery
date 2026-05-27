"""
Service container and dependency injection module.

Manages all backend service instances (LLM, embedding, vector store, retriever,
etc.) with lazy initialization and singleton caching. Services are created on
first access via @property descriptors and cached for subsequent use.

Provides the FastAPI dependency injection entry point get_service_container()
for use with Depends() in route handlers.

Managed services:
  - llm_client: LLM inference client (Qwen3.5-0.8B)
  - embedding_manager: Text embedding manager (Qwen3-Embedding-0.6B)
  - vector_store: ChromaDB vector database for dense retrieval
  - bm25_store: BM25 sparse retrieval index
  - table_store: Structured table storage (SQLite)
  - visual_indexer: Visual indexer (TrueColPali/Qwen3.5-2B) with graceful degradation
  - ingestion_pipeline: Document ingestion pipeline (PDF → chunks → index)
  - retriever: HybridRetriever (dense + sparse + table, RRF fusion)
  - erc_engine: Four-layer ERC compatibility check engine
  - veriquery_graph: LangGraph Agent workflow
  - svg_renderer: SVG pinout diagram renderer
"""

import logging
from typing import Optional

from core import get_settings

logger = logging.getLogger(__name__)


class ServiceContainer:
    """IoC container for all backend services with lazy initialization.

    Each service is exposed as a @property backed by a private variable
    (prefixed with _). On first access, the property getter creates the
    service instance and caches it; subsequent accesses return the cached
    instance directly (singleton per container).

    Attributes:
        settings: Global Settings instance from core.config.
    """

    def __init__(self):
        self.settings = get_settings()
        self._ingestion_pipeline = None
        self._retriever = None
        self._embedding_manager = None
        self._vector_store = None
        self._bm25_store = None
        self._table_store = None
        self._visual_indexer = None
        self._erc_engine = None
        self._veriquery_graph = None
        self._llm_client = None
        self._svg_renderer = None

    @property
    def llm_client(self):
        """LLM inference client (lazy-loaded).

        Provides a unified interface for text generation, Q&A, and reasoning.
        The underlying model (Qwen3.5-0.8B) is loaded to GPU on first access.

        Raises:
            RuntimeError: If the LLM client cannot be created.
        """
        if self._llm_client is None:
            try:
                from core.llm_client import get_llm_client
                logger.info("Loading LLM model (Qwen3.5-0.8B)...")
                self._llm_client = get_llm_client(self.settings)
                logger.info("LLM client initialized")
            except Exception as e:
                raise RuntimeError(f"LLM客户端创建失败: {e}") from e
        return self._llm_client

    @property
    def embedding_manager(self):
        """Text embedding manager (lazy-loaded) with GPU memory awareness.

        Uses Qwen3-Embedding-0.6B to produce 1024-dim vectors. Checks
        available GPU memory before loading to prevent CUDA OOM errors.

        Raises:
            RuntimeError: If GPU memory is insufficient or loading fails.
        """
        if self._embedding_manager is None:
            from core.memory_manager import get_memory_manager
            from retrieval import EmbeddingManager

            memory_manager = get_memory_manager()
            if not memory_manager.check_and_cleanup(required_memory_gb=2.0, cleanup_threshold_gb=1.5):
                raise RuntimeError("显存不足，无法加载嵌入模型")

            memory_manager.log_memory_usage("Embedding加载前")
            logger.info("Loading Embedding model (Qwen3-Embedding-0.6B)...")
            self._embedding_manager = EmbeddingManager(self.settings)
            logger.info("Embedding model loaded")
            memory_manager.log_memory_usage("Embedding加载后")

        return self._embedding_manager

    @property
    def vector_store(self):
        """ChromaDB vector store (lazy-loaded).

        Synchronizes the embedding dimension from the embedding manager to
        settings before creating the store, ensuring collection dimension
        matches the model output.

        If embedding manager initialization fails, the store is still created
        using the default dimension from settings.
        """
        if self._vector_store is None:
            try:
                embedding_manager = self.embedding_manager
                if embedding_manager and getattr(embedding_manager, "dimension", None):
                    self.settings.EMBEDDING_DIMENSION = embedding_manager.dimension
            except Exception as e:
                logger.warning(f"嵌入模型初始化或维度同步失败: {e}")

            from retrieval import create_vector_store
            self._vector_store = create_vector_store(settings=self.settings)
        return self._vector_store

    @property
    def bm25_store(self):
        """BM25 sparse retrieval store (lazy-loaded).

        Persists the BM25 index to disk (data/bm25_index.pkl) so it can be
        reloaded on restart without rebuilding from scratch.
        """
        if self._bm25_store is None:
            from retrieval import BM25Store
            persist_path = str(self.settings.DATA_DIR / "bm25_index.pkl")
            self._bm25_store = BM25Store(
                settings=self.settings,
                persist_path=persist_path
            )
        return self._bm25_store

    @property
    def table_store(self):
        """Structured table store (lazy-loaded, SQLite-backed).

        Stores parameter tables extracted from PDF datasheets, supporting
        SQL-style queries for precise parameter lookup.
        """
        if self._table_store is None:
            from retrieval import TableStore
            self._table_store = TableStore(settings=self.settings)
        return self._table_store

    @property
    def visual_indexer(self):
        """Visual indexer (lazy-loaded) with graceful degradation.

        Level 1 (full mode): TrueColPaliIndexer with VLM model + CLIP prefilter
            → supports deep visual analysis and MaxSim image-text matching
        Level 2 (degraded mode): TrueColPaliIndexer without VLM/CLIP
            → metadata-only search (filename, chip model)

        Raises:
            RuntimeError: If both full and degraded initialization fail.
        """
        if self._visual_indexer is None:
            try:
                from ingestion.image_indexer import create_visual_indexer
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

                logger.info("Loading visual indexer (VLM+CLIP)...")
                self._visual_indexer = create_visual_indexer(
                    model_name=self.settings.VLM_MODEL,
                    device=device,
                    quantize=self.settings.VLM_QUANTIZE,
                    quantization_bits=self.settings.VLM_QUANTIZATION_BITS
                )
                logger.info("Visual indexer loaded (VLM+CLIP)")

            except ImportError as e:
                logger.error(f"Cannot import TrueColPaliIndexer: {e}")
                raise RuntimeError(f"视觉索引器导入失败: {e}\n请确保ingestion模块正确安装") from e
            except Exception as e:
                logger.warning(f"Visual indexer full init failed: {e}")
                logger.info("Falling back to metadata-only search mode...")
                try:
                    from ingestion.image_indexer import TrueColPaliIndexer
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._visual_indexer = TrueColPaliIndexer(
                        model_name=None,
                        device=device,
                        quantize=False,
                        enable_clip_prefilter=False
                    )
                    logger.info("Visual indexer started in metadata-only mode")
                except Exception as fallback_err:
                    logger.error(f"Visual indexer degraded mode also failed: {fallback_err}")
                    raise RuntimeError(f"视觉索引器完全无法初始化: {fallback_err}") from fallback_err

        return self._visual_indexer

    @property
    def ingestion_pipeline(self):
        """Document ingestion pipeline (lazy-loaded).

        Handles the full PDF processing flow: parse → chunk → embed → index
        (vector, BM25, table, visual).
        """
        if self._ingestion_pipeline is None:
            from ingestion.document_processor import IngestionPipeline
            self._ingestion_pipeline = IngestionPipeline(self.settings)
        return self._ingestion_pipeline

    @property
    def retriever(self):
        """Hybrid retriever (lazy-loaded).

        Fuses dense (vector), sparse (BM25), and table retrieval channels
        using Reciprocal Rank Fusion (RRF). Accessing this property triggers
        lazy initialization of all dependent sub-retrievers.
        """
        if self._retriever is None:
            from retrieval import HybridRetriever

            base_retriever = HybridRetriever(
                embedding_manager=self.embedding_manager,
                vector_store=self.vector_store,
                bm25_store=self.bm25_store,
                table_store=self.table_store,
                settings=self.settings
            )
            self._retriever = base_retriever
        return self._retriever

    @property
    def erc_engine(self):
        """Four-layer ERC compatibility check engine (lazy-loaded).

        Performs electrical rule checks across four layers:
        voltage levels, current drive capability, timing, and special rules.
        """
        if self._erc_engine is None:
            from reasoning.erc_engine import FourLayerERCEngine
            self._erc_engine = FourLayerERCEngine(self.settings)
        return self._erc_engine

    @property
    def veriquery_graph(self):
        """LangGraph Agent workflow (lazy-loaded).

        The core multi-step RAG workflow that orchestrates intent routing,
        retrieval, reasoning, and response generation. Unlike visual_indexer,
        this service has no degradation fallback — initialization failure is
        a fatal error.

        Raises:
            RuntimeError: If the workflow graph cannot be created.
        """
        if self._veriquery_graph is None:
            try:
                from agents.workflow_graph import create_workflow
                self._veriquery_graph = create_workflow(self.settings)
            except Exception as e:
                logger.error(f"Agent workflow init failed: {e}")
                raise RuntimeError(f"Agent工作流图必须正常工作: {e}")
        return self._veriquery_graph

    @property
    def svg_renderer(self):
        """SVG pinout diagram renderer (lazy-loaded).

        Renders chip pin data as SVG diagrams with color-coded pin types
        (power=red, input=blue, output=green, ground=black, etc.).
        """
        if self._svg_renderer is None:
            from core.svg_renderer import PinoutSVGRenderer, SVGConfig
            config = SVGConfig.from_settings(self.settings)
            self._svg_renderer = PinoutSVGRenderer(config=config)
        return self._svg_renderer

    def _check_and_fix_chroma_db(self):
        """Check and fix ChromaDB schema compatibility issues.

        ChromaDB versions may have incompatible database schemas (e.g. the
        'topic' column was removed in 0.4.24). Detects incompatible schemas
        and removes the old database so it will be recreated.
        """
        chroma_dir = self.settings.DATA_DIR / "chroma"
        chroma_sqlite = chroma_dir / "chroma.sqlite3"

        if not chroma_sqlite.exists():
            logger.info(f"ChromaDB database not found, will be created on first use: {chroma_dir}")
            return

        try:
            import sqlite3
            conn = sqlite3.connect(str(chroma_sqlite))
            cursor = conn.cursor()

            cursor.execute("PRAGMA table_info(collections)")
            columns = [col[1] for col in cursor.fetchall()]
            conn.close()

            required_columns_0_4_24 = {'id', 'name', 'dimension'}
            has_topic = 'topic' in columns
            missing_required = required_columns_0_4_24 - set(columns)

            if missing_required:
                logger.warning(f"ChromaDB schema incompatible, missing columns: {missing_required}")
                logger.warning("Removing old database for recreation...")
                import shutil
                if chroma_dir.exists():
                    shutil.rmtree(chroma_dir)
                    logger.info(f"Removed old ChromaDB database: {chroma_dir}")
            elif has_topic:
                logger.warning("Detected old ChromaDB schema (has 'topic' column), rebuilding for 0.4.24+...")
                import shutil
                if chroma_dir.exists():
                    shutil.rmtree(chroma_dir)
                    logger.info(f"Removed old ChromaDB database: {chroma_dir}")
            else:
                logger.info(f"ChromaDB schema compatible: {chroma_sqlite}")

        except Exception as e:
            logger.warning(f"ChromaDB schema check failed: {e}")

    async def preload_critical_services(self):
        """Preload critical services at application startup.

        Initializes the knowledge graph and checks ChromaDB compatibility.
        Failure does not prevent application startup; affected services will
        be retried on first request.
        """
        try:
            self._check_and_fix_chroma_db()
            from knowledge import ensure_knowledge_graph_initialized
            ensure_knowledge_graph_initialized(str(self.settings.DATA_DIR / "knowledge_graph.db"))
        except Exception as e:
            logger.warning(f"Critical service preload failed (will retry on first use): {e}")

    async def preload_vlm_model(self):
        """Preload VLM model in a background thread without blocking the event loop."""
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            def _load_vlm():
                try:
                    from ingestion.image_indexer import get_visual_indexer
                    logger.info("Preloading VLM model...")
                    visual_indexer = get_visual_indexer()
                    visual_indexer._load_model(model_type="qwen")
                    logger.info("VLM model preloaded")
                except Exception as e:
                    logger.warning(f"VLM model preload failed: {e}")

            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm_preload")
            await loop.run_in_executor(executor, _load_vlm)
            executor.shutdown(wait=False)

        except Exception as e:
            logger.warning(f"VLM preload task failed: {e}")

    async def cleanup(self):
        """Release all GPU resources at application shutdown.

        Unloads all registered models via model_manager.unload_all(),
        which deletes model objects and calls torch.cuda.empty_cache().
        """
        try:
            from core.model_manager import model_manager
            model_manager.unload_all()
        except Exception as e:
            logger.warning(f"Model cleanup failed: {e}")


_container: Optional[ServiceContainer] = None


def get_service_container() -> ServiceContainer:
    """Get the global ServiceContainer singleton.

    Creates the container on first call and caches it for all subsequent
    calls. Used as a FastAPI dependency via Depends(get_service_container).

    Returns:
        ServiceContainer: The global singleton instance.
    """
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container
