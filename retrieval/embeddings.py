"""
Text Embedding Service
======================

Converts natural language text into dense vector embeddings for semantic
retrieval. Manages the embedding model lifecycle (loading, device placement,
memory optimization), provides LRU caching to avoid redundant encoding, and
automatically adds instruction prompts for asymmetric encoding.

Data Flow:
    User query / document chunk (text)
        → embed_query / embed_batch (vectorization)
            → vector_store.py (vector storage / retrieval)
                → hybrid_retriever.py (multi-path retrieval + RRF fusion)
                    → agents/workflow_nodes.py (Agent workflow)

Module Call Chain:
    embeddings.py ← vector_store.py ← hybrid_retriever.py ← agents/workflow_nodes.py

Key Concepts:
    - Dense Embedding: Maps discrete text to continuous vector space where
      semantically similar texts are closer. Complements sparse retrieval
      (BM25/TF-IDF) which captures lexical exact matching.
    - SentenceTransformer: Siamese/Bi-encoder framework (Reimers & Gurevych, 2019)
      that encodes query and document separately, then measures similarity
      via cosine distance.
    - Instruction-aware Embedding: Models like BGE and Qwen-Embedding support
      task-specific instruction prefixes. This module adds retrieval instructions
      to queries (asymmetric encoding) while encoding documents without prefixes.
    - LRU Cache: Least Recently Used eviction via OrderedDict; O(1) lookup and
      insertion. Cache key is MD5 hash of input text.
    - Singleton Pattern: Thread-safe via Double-Checked Locking (DCL) to ensure
      only one model instance is loaded globally.

Supported Models:
    - Qwen3-Embedding-0.6B (Chinese/English, 1024-dim, recommended)
    - bge-small-zh-v1.5 (Chinese, 512-dim)
    - Any sentence-transformers compatible model

Configuration (from core/config.py Settings):
    EMBEDDING_MODEL: Model path, default "BAAI/bge-large-zh-v1.5"
    EMBEDDING_DEVICE: Runtime device, default "cuda"
    EMBEDDING_DIMENSION: Embedding dimension, default 1024
    BATCH_SIZE_EMBEDDING: Batch size for encoding
"""

import os

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import logging
import hashlib
import threading
import time
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from core import get_settings, ConfigurationError

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Singleton embedding manager for text vectorization.

    Ensures only one embedding model instance is loaded globally (models
    typically occupy hundreds of MB to several GB of memory/VRAM).

    Supports:
        - Qwen3-Embedding-0.6B (Chinese/English, 1024-dim, recommended)
        - bge-small-zh-v1.5 (Chinese, 512-dim)
        - Any sentence-transformers compatible model

    Example:
        manager = EmbeddingManager()
        embeddings = manager.embed_batch(["text1", "text2"])
        query_vec = manager.embed_query("NE5532 supply voltage")
    """

    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton creation with Double-Checked Locking (DCL).

        First check (no lock): fast path when instance already exists.
        Lock: ensures only one thread creates the instance.
        Second check (under lock): prevents duplicate creation when
        multiple threads pass the first check simultaneously.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, settings=None, model_name: str = None):
        """Initialize embedding manager.

        Args:
            settings: Configuration object. Defaults to get_settings() singleton.
                      Allows injection for testing.
            model_name: Embedding model name/path. Defaults to settings.EMBEDDING_MODEL.
                        Can override config for testing or model switching.
        """
        if EmbeddingManager._initialized:
            return

        EmbeddingManager._initialized = True

        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL
        self.device = self.settings.EMBEDDING_DEVICE
        self.dimension = self.settings.EMBEDDING_DIMENSION

        self.model = None
        self._cache = OrderedDict()
        self._cache_max_size = 2000

        self._load_model()

    def _cleanup_memory_if_needed(self, threshold_percent: int = 50) -> bool:
        """Check memory usage and perform cleanup if above threshold.

        Args:
            threshold_percent: Memory utilization threshold percentage.
                - 0: Force cleanup regardless of usage.
                - >0: Only cleanup when utilization exceeds this value.
                Defaults to 50.

        Returns:
            True if cleanup was performed, False otherwise.
        """
        if threshold_percent <= 0:
            try:
                from core.memory_manager import get_memory_manager
                memory_manager = get_memory_manager()
                if memory_manager:
                    memory_manager.cleanup(aggressive=True)
                    return True
            except Exception as e:
                logger.debug(f"Memory cleanup failed: {e}")
            return False

        try:
            from core.memory_manager import get_memory_manager
            memory_manager = get_memory_manager()

            if memory_manager:
                mem_stats = memory_manager.get_memory_stats()
                if mem_stats.utilization_percent > threshold_percent:
                    logger.debug(f"Memory utilization {mem_stats.utilization_percent:.1f}% > {threshold_percent}%, cleaning up")
                    memory_manager.cleanup(aggressive=True)
                    return True
        except Exception as e:
            logger.debug(f"Memory cleanup failed: {e}")
        return False

    def _handle_embedding_error(self, error: Exception, context: str = "") -> None:
        """Unified embedding error handler.

        Converts all errors to RuntimeError with context. OOM errors receive
        specific guidance (reduce batch_size or increase VRAM).

        Args:
            error: The original exception.
            context: Error context description for logging, e.g. "embed_batch(cache mode)".

        Raises:
            RuntimeError: Always, with descriptive message and chained original error.
        """
        if isinstance(error, RuntimeError):
            error_msg = str(error).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                logger.error(f"GPU OOM error in {context}: {error}")
                raise RuntimeError(f"GPU out of memory, reduce batch_size or increase VRAM: {error}") from error
            else:
                logger.error(f"Runtime error in {context}: {error}")
                raise RuntimeError(f"Embedding operation failed: {error}") from error
        else:
            logger.error(f"Unexpected error in {context}: {error}")
            raise RuntimeError(f"Embedding operation failed: {error}") from error

    def _cleanup_before_load(self):
        """Clean up GPU memory before loading the embedding model.

        Releases PyTorch-cached unused GPU memory blocks and waits for
        asynchronous CUDA operations to complete before cleanup.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self._cleanup_memory_if_needed(threshold_percent=0)
                logger.info("GPU memory cleaned before loading embedding model")
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")

    def _get_optimal_device(self) -> str:
        """Detect the optimal runtime device.

        Prefers GPU (CUDA) for 10-100x faster inference; falls back to
        CPU when CUDA is unavailable. Overrides self.device config if
        the configured device is not actually available.

        Returns:
            "cuda" or "cpu".
        """
        try:
            import torch
            if not torch.cuda.is_available():
                logger.info("CUDA unavailable, using CPU")
                return "cpu"
            return self.device
        except Exception as e:
            logger.warning(f"Device detection failed: {e}, using CPU")
            return "cpu"

    def _load_model(self):
        """Load SentenceTransformer embedding model.

        Skips if model is already loaded. On load:
            1. Clean up GPU memory.
            2. Detect optimal device.
            3. Load model via SentenceTransformer.
            4. Auto-detect actual embedding dimension from model.

        Raises:
            ConfigurationError: If sentence-transformers is not installed,
                                or if model loading fails.
        """
        if self.model is not None:
            logger.debug("Embedding model already loaded, skipping")
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            start_time = time.time()
            logger.info(f"Loading embedding model: {self.model_name}")

            self._cleanup_before_load()
            actual_device = self._get_optimal_device()

            self.model = SentenceTransformer(
                self.model_name,
                device=actual_device
            )

            self.device = actual_device
            self.dimension = self.model.get_sentence_embedding_dimension()

            load_time = time.time() - start_time
            logger.info(f"Embedding model loaded, dim={self.dimension}, device={actual_device}, time={load_time:.2f}s")

        except ImportError:
            raise ConfigurationError(
                "sentence-transformers not installed",
                details={"install": "pip install sentence-transformers"}
            )
        except Exception as e:
            raise ConfigurationError(f"Embedding model loading failed: {e}")

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a user query into an embedding vector with instruction prompt.

        Automatically adds model-specific instruction prefixes for
        instruction-aware models (BGE, Qwen) to improve retrieval quality.
        Models without instruction support are encoded as-is.

        Args:
            query: User query text, e.g. "NE5532的供电电压范围是多少？".

        Returns:
            Embedding vector of shape (dimension,), e.g. (1024,) float32.
        """
        if "bge" in self.model_name.lower():
            query = f"为这个句子生成表示以用于检索相关段落：{query}"
        elif "qwen" in self.model_name.lower():
            query = f"Instruct: 检索与查询相关的电子元器件文档\nQuery: {query}"

        result = self.embed_batch([query], use_cache=False)
        return result[0] if len(result) > 0 else np.array([])

    def embed_batch(self, texts: List[str],
                    use_cache: bool = True) -> np.ndarray:
        """Batch encode texts into embedding vectors.

        Core encoding method with LRU caching and dynamic batch size adjustment:
            - Available memory < 1GB → batch_size ≤ 4
            - Available memory < 2GB → batch_size ≤ 8
            - Average text length > 500 chars → batch_size halved
            - Text count > 100 → batch_size ≤ 4
        All adjustments stack (minimum wins).

        Args:
            texts: Text list to encode, e.g. ["text1", "text2", "text3"].
            use_cache: Whether to use LRU cache. Defaults to True.
                - True: Check cache first, only encode cache misses (suitable
                  for document encoding where texts may repeat).
                - False: Encode directly without cache (suitable for queries
                  which are typically unique).

        Returns:
            Embedding matrix of shape (len(texts), dimension), e.g. (3, 1024).
        """
        if not texts:
            return np.array([])

        if self.model is None:
            self._load_model()
        if self.model is None:
            raise ConfigurationError("Embedding model not loaded, cannot encode texts")

        start_time = time.time()
        total_texts = len(texts)
        avg_text_length = sum(len(t) for t in texts) / total_texts

        batch_size = self.settings.BATCH_SIZE_EMBEDDING

        try:
            from core.memory_manager import get_memory_manager
            memory_manager = get_memory_manager()

            if memory_manager:
                mem_stats = memory_manager.get_memory_stats()

                if mem_stats.free_memory < 1.0:
                    batch_size = min(batch_size, 4)
                    logger.debug(f"Low memory ({mem_stats.free_memory:.2f}GB free), batch_size adjusted to {batch_size}")
                elif mem_stats.free_memory < 2.0:
                    batch_size = min(batch_size, 8)
                    logger.debug(f"Moderate memory ({mem_stats.free_memory:.2f}GB free), batch_size adjusted to {batch_size}")
        except Exception as e:
            logger.debug(f"Memory optimization check failed: {e}")

        if avg_text_length > 500:
            batch_size = max(4, batch_size // 2)
            logger.debug(f"Long texts detected (avg {avg_text_length:.0f} chars), batch_size adjusted to {batch_size}")

        if len(texts) > 100:
            batch_size = min(batch_size, 4)
            logger.debug(f"Large text count ({len(texts)}), batch_size adjusted to {batch_size}")

        self._cleanup_memory_if_needed(threshold_percent=50)

        if use_cache:
            cached_results = {}
            texts_to_encode = []
            text_indices = []

            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached_embedding = self._get_from_cache(cache_key)
                if cached_embedding is not None:
                    cached_results[i] = cached_embedding
                else:
                    texts_to_encode.append(text)
                    text_indices.append(i)

            cache_hit_count = len(cached_results)

            if texts_to_encode:
                try:
                    new_embeddings = self.model.encode(
                        texts_to_encode,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                except Exception as e:
                    self._handle_embedding_error(e, context="embed_batch(cache mode)")

                for j, idx in enumerate(text_indices):
                    cache_key = self._get_cache_key(texts[idx])
                    self._add_to_cache(cache_key, new_embeddings[j])
                    cached_results[idx] = new_embeddings[j]

            result = np.array([cached_results[i] for i in range(len(texts))])

            elapsed = (time.time() - start_time) * 1000
            cache_hit_rate = cache_hit_count / total_texts * 100
            logger.debug(f"Batch embedding done: {total_texts} texts, cache hit {cache_hit_count} ({cache_hit_rate:.1f}%), elapsed {elapsed:.2f}ms, avg length {avg_text_length:.0f}")

            return result

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        except Exception as e:
            self._handle_embedding_error(e, context="embed_batch(no cache)")

        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Batch embedding done (no cache): {total_texts} texts, elapsed {elapsed:.2f}ms, avg length {avg_text_length:.0f}")

        return embeddings

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text via MD5 hash.

        MD5 maps arbitrary-length text to a fixed 32-char hex string.
        Collision probability is negligible for caching purposes.

        Args:
            text: Original text string.

        Returns:
            32-character MD5 hex digest string.
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _add_to_cache(self, key: str, embedding: np.ndarray):
        """Add an entry to the LRU cache.

        If the key already exists, it is re-inserted at the end (most recent).
        If the cache is full, the least recently used entry (head of OrderedDict)
        is evicted.

        Args:
            key: Cache key (MD5 hex digest, 32 chars).
            embedding: Embedding vector (shape=(dimension,), dtype=float32).
        """
        if key in self._cache:
            del self._cache[key]
        elif len(self._cache) >= self._cache_max_size:
            self._cache.popitem(last=False)
            logger.debug(f"Cache full, LRU eviction, current size: {len(self._cache)}")

        self._cache[key] = embedding

    def _get_from_cache(self, key: str) -> Optional[np.ndarray]:
        """Retrieve an embedding from the LRU cache.

        On cache hit, the entry is moved to the end of the OrderedDict
        (marked as most recently used) by pop + re-insert.

        Args:
            key: Cache key (MD5 hex digest).

        Returns:
            Cached embedding vector if hit, None if miss.
        """
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        return None
