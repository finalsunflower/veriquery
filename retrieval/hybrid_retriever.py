"""
Hybrid Retriever — Multi-path Retrieval with RRF Fusion
========================================================

Core orchestrator for VeriQuery's retrieval pipeline. Combines three
heterogeneous retrieval paths (Dense, Sparse, Structured) via Reciprocal
Rank Fusion (RRF) and provides a unified async retrieve() interface.

Architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                     HybridRetriever (RRF Fusion)                     │
    │                                                                      │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
    │  │  VectorStore  │  │  BM25Store   │  │  TableStore   │               │
    │  │  (Dense)      │  │  (Sparse)    │  │  (Structured) │               │
    │  │  ChromaDB     │  │  rank_bm25   │  │  SQLite+FTS5  │               │
    │  │  Semantic     │  │  Keyword     │  │  Table data   │               │
    │  └──────────────┘  └──────────────┘  └──────────────┘               │
    │        w=0.5            w=0.35           w=0.15                      │
    │                                                                      │
    │  Results → Normalize → RRF Fusion → Dedup → Per-doc limit → Final   │
    └──────────────────────────────────────────────────────────────────────┘

Module Call Chain:
    embeddings.py ← vector_store.py ← hybrid_retriever.py ← agents/workflow_nodes.py
    bm25_store.py  ─────────────────→ hybrid_retriever.py
    table_store.py ─────────────────→ hybrid_retriever.py

Key Concepts:
    - Hybrid Retrieval: Combines Dense (semantic), Sparse (keyword), and
      Structured (table) retrieval for complementary coverage. Lin et al.
      (2021) demonstrated that hybrid retrieval significantly outperforms
      any single retrieval method.
    - RRF (Reciprocal Rank Fusion): Score-based fusion that relies only on
      rank positions, avoiding the problem of incomparable score scales
      across different retrieval paths. Cormack et al. (2009) proved RRF
      outperforms CombSUM and Condorcet fusion.
        Formula: score(d) = Σ_s w_s × 1/(k + rank_s(d) + 1)
        k=60 is the recommended smoothing constant from the SIGIR paper.
    - Async Concurrency: asyncio.gather with return_exceptions=True
      executes all three retrieval paths concurrently. Total latency is
      max(T1, T2, T3) instead of T1+T2+T3. Individual failures are
      tolerated — other paths still return results.

Data Flow:
    User query "NE5532 supply voltage range?"
        │
        ├──→ _vector_retrieve(query, top_k*2)  → List[RetrievedChunk] (source="vector")
        ├──→ _bm25_retrieve(query, top_k*2)    → List[RetrievedChunk] (source="bm25")
        └──→ _table_retrieve(query, top_k)     → List[RetrievedChunk] (source="table")
        │
        ▼ (asyncio.gather, return_exceptions=True)
        │
        _rrf_fusion(all_results, k=60)
        │   ├── Group by source → Sort within groups → Compute RRF scores
        │   ├── Multi-source hits accumulate RRF scores (consensus bonus)
        │   ├── Deduplicate by document_id + text[:100] hash
        │   ├── Per-document limit (5 normal, 8 cross-source)
        │   └── Normalize final scores to [0, 1]
        │
        ▼
        List[RetrievedChunk] (source="hybrid")

Configuration (from core/config.py Settings):
    VECTOR_WEIGHT: Vector retrieval weight, default 0.5
    BM25_WEIGHT: BM25 retrieval weight, default 0.35
    STRUCTURED_WEIGHT: Table retrieval weight, default 0.15
    MAX_RESULTS_PER_DOC: Max chunks per document, default 5
    MAX_RESULTS_PER_DOC_CROSS_SOURCE: Max chunks per document (cross-source), default 8
    EMBEDDING_DIMENSION: Embedding dimension, default 1024
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

from core.config import get_settings
from core.schema import RetrievedChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever orchestrating Dense + Sparse + Structured retrieval.

    Facade pattern: encapsulates three heterogeneous retrieval subsystems
    behind a unified async retrieve() interface. Callers (agents/workflow_nodes.py)
    only need to call retrieve() without worrying about multi-path dispatch.

    Features:
        - Dependency injection: all retrieval components are injectable for
          testing; auto-created with lazy init when not provided.
        - Fault tolerance: each component initializes independently; single
          component failure does not affect others. Only raises RuntimeError
          when ALL components fail.
        - RRF fusion with configurable per-source weights.
        - Deduplication and per-document result limiting.

    Example:
        retriever = HybridRetriever()
        results = await retriever.retrieve("NE5532 supply voltage", top_k=10)
    """

    def __init__(
        self,
        settings=None,
        embedding_manager=None,
        vector_store=None,
        bm25_store=None,
        table_store=None,
    ):
        """Initialize hybrid retriever with optional dependency injection.

        All parameters are optional. When not provided, components are
        auto-created. Each component has independent error handling —
        single component failure is logged but does not prevent others
        from initializing.

        Args:
            settings: Configuration object. Defaults to get_settings() singleton.
            embedding_manager: EmbeddingManager instance for query vectorization.
                               Auto-created (singleton) if not provided.
            vector_store: ChromaDBVectorStore for dense retrieval.
            bm25_store: BM25Store for sparse retrieval.
            table_store: TableStore for structured retrieval.
        """
        self.settings = settings or get_settings()
        self.embedding_manager = embedding_manager

        try:
            from retrieval.vector_store import create_vector_store
            from retrieval.bm25_store import BM25Store
            from retrieval.table_store import TableStore

            if self.embedding_manager is None:
                try:
                    from retrieval.embeddings import EmbeddingManager
                    self.embedding_manager = EmbeddingManager(settings=self.settings)
                except Exception as e:
                    logger.warning(f"EmbeddingManager init failed: {e}")
                    self.embedding_manager = None

            if self.embedding_manager and hasattr(self.embedding_manager, 'dimension'):
                self.settings.EMBEDDING_DIMENSION = self.embedding_manager.dimension

            try:
                self.vector_store = vector_store or create_vector_store(settings=self.settings)
            except Exception as e:
                logger.error(f"Vector store init failed: {e}", exc_info=True)
                self.vector_store = None

            try:
                self.bm25_store = bm25_store or BM25Store(settings=self.settings)
            except Exception as e:
                logger.error(f"BM25 store init failed: {e}", exc_info=True)
                self.bm25_store = None

            try:
                self.table_store = table_store or TableStore(settings=self.settings)
            except Exception as e:
                logger.error(f"Table store init failed: {e}", exc_info=True)
                self.table_store = None

            self.source_weights = {
                "vector": getattr(self.settings, "VECTOR_WEIGHT", 0.5),
                "bm25": getattr(self.settings, "BM25_WEIGHT", 0.35),
                "table": getattr(self.settings, "STRUCTURED_WEIGHT", 0.15),
            }

            if not any([self.vector_store, self.bm25_store, self.table_store]):
                raise RuntimeError("All retrievers failed to initialize")

        except ImportError as e:
            raise RuntimeError(f"Retrieval component import failed: {e}") from e
        except Exception:
            raise

    async def retrieve(
        self,
        query: str,
        top_k: int = 50,
        include_tables: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Unified hybrid retrieval interface.

        Two-phase architecture:
            Phase 1: Concurrently execute all available retrieval paths.
            Phase 2: RRF fusion + dedup + per-doc limiting + normalization.

        Args:
            query: Query text, e.g. "NE5532的供电电压范围是多少？".
            top_k: Maximum number of results to return. Defaults to 50.
                   Vector and BM25 paths retrieve top_k*2 candidates
                   (RRF fusion reduces count via dedup/limiting).
            include_tables: Whether to include table retrieval. Defaults to True.
            filters: Optional filter dict (document_id, source, etc.).
            document_ids: Optional document ID list to scope retrieval.
                          Merged into filters internally.

        Returns:
            Fused and ranked List[RetrievedChunk] with source="hybrid".
            Returns empty list if all paths return no results.
        """
        try:
            merged_filters = dict(filters or {})
            if document_ids:
                merged_filters["document_ids"] = [str(d) for d in document_ids]

            tasks = []

            if self.vector_store:
                tasks.append(self._vector_retrieve(query, top_k * 2, merged_filters))

            if self.bm25_store:
                tasks.append(self._bm25_retrieve(query, top_k * 2, merged_filters))

            if include_tables and self.table_store:
                tasks.append(self._table_retrieve(query, top_k, merged_filters))

            if not tasks:
                raise RuntimeError("No retrieval components available")

            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_results = []
            retrieval_errors = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"Retrieval task {i+1} failed: {result}"
                    logger.warning(error_msg)
                    retrieval_errors.append(error_msg)
                elif result:
                    all_results.extend(result)

            if retrieval_errors:
                logger.warning(f"{len(retrieval_errors)} retrieval task(s) failed")

            if not all_results:
                logger.info("All retrieval paths returned no results")
                return []

            fused_results = self._rrf_fusion(all_results)

            final_results = fused_results[:top_k]

            return final_results

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            raise

    async def _vector_retrieve(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Dense retrieval via semantic similarity (VectorStore + ChromaDB).

        Pipeline:
            1. Encode query into embedding vector via EmbeddingManager.
            2. Filter conditions to ChromaDB-supported keys only.
            3. Search ChromaDB (HNSW approximate nearest neighbor).
            4. Convert VectorSearchResult to RetrievedChunk.

        Args:
            query: Query text.
            top_k: Number of candidates to retrieve (typically top_k*2).
            filters: Filter dict supporting document_id and document_ids.

        Returns:
            List[RetrievedChunk] with source="vector". Scores are cosine
            similarity (1 - distance), already in [0, 1].
        """
        try:
            query_embedding = self._get_query_embedding(query)
            if not query_embedding:
                raise RuntimeError("Query embedding is empty, cannot perform vector retrieval")

            filter_dict = None
            if filters:
                supported_filters = {}
                for key in self.vector_store.SUPPORTED_FILTER_KEYS:
                    if key in filters:
                        supported_filters[key] = filters[key]
                if supported_filters:
                    filter_dict = supported_filters

            vector_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict,
            )

            results = []
            for result in vector_results:
                metadata = result.metadata or {}

                results.append(
                    RetrievedChunk(
                        chunk_id=str(result.id),
                        text=result.text or "",
                        score=float(result.score or 0.0),
                        document_id=str(metadata.get("document_id", "")),
                        filename=str(metadata.get("filename", "")),
                        page=int(metadata.get("page", 1) or 1),
                        char_start=int(metadata.get("char_start", 0) or 0),
                        char_end=int(metadata.get("char_end", 0) or 0),
                        source="vector",
                        metadata=metadata,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            raise

    async def _bm25_retrieve(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Sparse retrieval via BM25 keyword matching.

        Pipeline:
            1. Construct filter function from filters dict.
            2. Search BM25Store (tokenize → synonym expand → BM25 score).
            3. Normalize raw BM25 scores to [0, 1] via max normalization.
            4. Convert BM25SearchResult to RetrievedChunk.

        Args:
            query: Query text.
            top_k: Number of candidates to retrieve (typically top_k*2).
            filters: Filter dict supporting document_ids and source.

        Returns:
            List[RetrievedChunk] with source="bm25". Scores normalized to [0, 1].
        """
        try:
            filter_func = None
            document_ids = None
            source_filter = None

            if filters:
                if "document_ids" in filters and isinstance(filters["document_ids"], list):
                    document_ids = set(str(d) for d in filters["document_ids"])
                elif "document_id" in filters:
                    document_ids = {str(filters["document_id"])}

                if "source" in filters:
                    source_filter = filters["source"]

                if document_ids or source_filter:
                    def _filter_fn(doc_id, metadata):
                        if document_ids:
                            meta_doc_id = metadata.get("document_id", "") if metadata else ""
                            if meta_doc_id not in document_ids and doc_id not in document_ids:
                                return False
                        if source_filter:
                            if metadata.get("source", "") != source_filter:
                                return False
                        return True

                    filter_func = _filter_fn

            bm25_results = self.bm25_store.search(query, top_k=top_k, filter_func=filter_func)

            if not bm25_results:
                return []

            max_score = max(r.score for r in bm25_results)
            if max_score <= 0:
                max_score = 1.0

            results = []
            for result in bm25_results:
                normalized_score = result.score / max_score

                metadata = result.metadata or {}

                results.append(
                    RetrievedChunk(
                        chunk_id=result.doc_id,
                        text=result.text or "",
                        score=float(normalized_score),
                        document_id=str(metadata.get("document_id", "")),
                        filename=str(metadata.get("filename", "")),
                        page=int(metadata.get("page", 1) or 1),
                        char_start=int(metadata.get("char_start", 0) or 0),
                        char_end=int(metadata.get("char_end", 0) or 0),
                        source="bm25",
                        metadata=metadata,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            raise

    async def _table_retrieve(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Structured retrieval via TableStore (SQLite + FTS5).

        Table retrieval preserves row/column structure that is lost when
        tables are flattened into text for vector/BM25 retrieval.

        Pipeline:
            1. Search TableStore (FTS5 full-text or LIKE fallback).
            2. Format table data into readable text.
            3. Convert to RetrievedChunk.

        Args:
            query: Query text.
            top_k: Number of candidates to retrieve (typically top_k, not *2).
            filters: Filter dict supporting document_ids.

        Returns:
            List[RetrievedChunk] with source="table".
        """
        try:
            table_results = await self.table_store.search(query=query, top_k=top_k, filters=filters)

            results = []
            for result in table_results:
                table_text = self._format_table_text(result.get("table_data", []))

                results.append(
                    RetrievedChunk(
                        chunk_id=result.get("id", ""),
                        text=table_text,
                        score=float(result.get("score", 0.0)),
                        document_id=result.get("document_id", ""),
                        filename=result.get("filename", ""),
                        page=int(result.get("page", 1)),
                        source="table",
                        metadata={
                            **result.get("metadata", {}),
                            "table_data": result.get("table_data", []),
                            "table_title": result.get("title", ""),
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Table retrieval failed: {e}")
            raise

    def _format_table_text(self, table_data: List[List[str]]) -> str:
        """Format 2D table data into pipe-delimited text for LLM consumption.

        Each row is joined with " | " separators (simplified Markdown table).
        Empty cells and empty rows are filtered out to reduce noise.

        Args:
            table_data: 2D string list, e.g. [["Param", "Min", "Typ", "Max"],
                                              ["VOH", "2.4V", "3.0V", ""]].

        Returns:
            Formatted text, e.g. "Param | Min | Typ | Max\\nVOH | 2.4V | 3.0V".
        """
        if not table_data:
            return ""

        text_lines = []
        for row in table_data:
            if row:
                line = " | ".join(str(cell) for cell in row if cell)
                if line.strip():
                    text_lines.append(line)

        return "\n".join(text_lines)

    def _rrf_fusion(self, results: List[RetrievedChunk], k: int = 60) -> List[RetrievedChunk]:
        """Reciprocal Rank Fusion of multi-source retrieval results.

        RRF formula (Cormack et al., SIGIR 2009):
            score(d) = Σ_s w_s × 1/(k + rank_s(d) + 1)

        where s is the retrieval source, rank_s(d) is the 0-based rank of
        document d in source s, and k=60 is the smoothing constant.

        Pipeline:
            1. Group results by source.
            2. Sort within each group by score (descending) to determine ranks.
            3. Compute RRF scores: weight × 1/(k + rank + 1).
            4. Accumulate scores for chunks hit by multiple sources.
            5. Deduplicate by document_id + text[:100] content hash.
            6. Per-document result limiting (5 normal, 8 cross-source).
            7. Normalize final scores to [0, 1].

        Multi-source consensus: when the same chunk is returned by multiple
        retrieval paths, its RRF scores accumulate, giving it a significant
        boost. This reflects the principle that multi-source agreement
        indicates higher relevance.

        Args:
            results: All retrieval results from vector/bm25/table paths.
            k: RRF smoothing constant. Defaults to 60 (paper recommendation).

        Returns:
            Fused and ranked List[RetrievedChunk] with source="hybrid".
        """
        if not results:
            return []

        results_by_source = {}
        for result in results:
            source = result.source
            if source not in results_by_source:
                results_by_source[source] = []
            results_by_source[source].append(result)

        for source in results_by_source:
            results_by_source[source].sort(key=lambda x: x.score, reverse=True)

        rrf_scores = {}

        for source, source_results in results_by_source.items():
            weight = self.source_weights.get(source, 0.1)

            for rank, result in enumerate(source_results):
                result_id = f"{result.document_id}_{result.chunk_id}"

                rrf_score = weight * (1 / (k + rank + 1))

                if result_id not in rrf_scores:
                    rrf_scores[result_id] = {
                        "result": result,
                        "score": rrf_score,
                        "sources": [source],
                        "rank": rank,
                    }
                else:
                    rrf_scores[result_id]["score"] += rrf_score
                    rrf_scores[result_id]["sources"].append(source)
                    rrf_scores[result_id]["rank"] = min(rrf_scores[result_id]["rank"], rank)

        seen_hashes = {}
        to_remove = set()

        for result_id, score_data in rrf_scores.items():
            result = score_data["result"]
            content_hash = f"{result.document_id}_{result.text[:100]}"

            if content_hash not in seen_hashes:
                seen_hashes[content_hash] = result_id
            else:
                existing_id = seen_hashes[content_hash]
                if score_data["score"] > rrf_scores[existing_id]["score"]:
                    to_remove.add(existing_id)
                    seen_hashes[content_hash] = result_id
                else:
                    to_remove.add(result_id)

        for rid in to_remove:
            del rrf_scores[rid]

        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

        max_per_doc = getattr(self.settings, "MAX_RESULTS_PER_DOC", 5)
        max_per_doc_cross = getattr(self.settings, "MAX_RESULTS_PER_DOC_CROSS_SOURCE", 8)

        final_items = []
        seen_documents = {}

        for item in sorted_results:
            result = item["result"]
            doc_id = result.document_id
            sources = item["sources"]
            doc_count = seen_documents.get(doc_id, 0)

            if doc_count < max_per_doc:
                final_items.append(item)
                seen_documents[doc_id] = doc_count + 1
            elif len(sources) > 1 and doc_count < max_per_doc_cross:
                final_items.append(item)
                seen_documents[doc_id] = doc_count + 1

        for item in final_items:
            result = item["result"]
            result.source = "hybrid"
            result.metadata["fusion_sources"] = item["sources"]
            result.metadata["rrf_score"] = item["score"]
            result.metadata["fusion_rank"] = item["rank"]

        if final_items:
            max_score = final_items[0]["score"]
            if max_score > 0:
                for item in final_items:
                    item["result"].score = item["score"] / max_score

        return [item["result"] for item in final_items]

    def _get_query_embedding(self, query: str):
        """Generate query embedding vector via EmbeddingManager.

        Uses asymmetric encoding: embed_query adds instruction prompts
        (model-specific) to the query, while documents are encoded
        without instructions via embed_batch.

        Args:
            query: Raw query text.

        Returns:
            Embedding as List[float] for ChromaDB compatibility.

        Raises:
            RuntimeError: If EmbeddingManager is not initialized.
            TypeError: If embedding type cannot be converted to list.
        """
        if self.embedding_manager is None:
            error_msg = (
                "EmbeddingManager not initialized. "
                "Pass embedding_manager to HybridRetriever or check vector store init."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            embedding = self.embedding_manager.embed_query(query)

            if isinstance(embedding, (list, tuple)):
                return list(embedding)
            elif hasattr(embedding, "tolist"):
                return embedding.tolist()
            else:
                raise TypeError(f"Cannot convert embedding type: {type(embedding)}")

        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            raise
