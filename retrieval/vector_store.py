"""
Vector Store — Dense Retrieval via ChromaDB.

Persists text embedding vectors in a ChromaDB collection and provides
cosine-similarity-based nearest-neighbor search, forming the dense
retrieval pathway of the hybrid retrieval architecture.

Architecture Position
---------------------
Upstream (writes):
    api/routers/documents.py  →  add_documents()
    api/routers/circuit.py    →  add_documents()

Downstream (reads):
    retrieval/hybrid_retriever.py  →  search()

Management:
    core/cleanup_manager.py  →  delete_by_document()
    api/dependencies.py      →  create_vector_store()

Retrieval Pipeline:
    embeddings.py  →  ★vector_store.py★  →  hybrid_retriever.py  →  agents/workflow_nodes.py

Hybrid Retrieval Architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                     HybridRetriever (RRF Fusion)                    │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
    │  │★VectorStore★ │  │  BM25Store   │  │  TableStore   │              │
    │  │  (Dense)     │  │  (Sparse)    │  │  (Structured) │              │
    │  │  ChromaDB    │  │  rank_bm25   │  │  SQLite+FTS5  │              │
    │  └──────────────┘  └──────────────┘  └──────────────┘              │
    │       w=0.5            w=0.35           w=0.15                      │
    └──────────────────────────────────────────────────────────────────────┘

ChromaDB Collection Schema:
    Each record contains:
        id:         Unique identifier (e.g. "doc1_chunk0")
        embedding:  Dense vector (e.g. 1024-dim float array)
        document:   Original text chunk
        metadata:   {"document_id": str, "filename": str, "page": int, ...}

Key Design Decisions:
    1. PersistentClient — embedded mode with SQLite+Parquet persistence,
       no external server required, suitable for single-node RAG deployment.
    2. HNSW index with cosine distance — O(log N) approximate nearest
       neighbor search; cosine distance suits text embeddings better than
       L2 since it measures directional similarity regardless of magnitude.
    3. Dimension validation on startup — detects embedding model changes
       (e.g. 512-dim BGE → 1024-dim Qwen) and auto-recreates the
       collection to prevent dimension-mismatch errors.
    4. Score conversion: score = 1.0 - cosine_distance, clamped to [0, 1].
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings

try:
    from chromadb.telemetry.product.posthog import Posthog as _Posthog
    _Posthog._direct_capture = lambda self, event: None
except Exception:
    pass

from core import get_settings, RetrievalError

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Single result from a vector similarity search.

    Data flow: ChromaDBVectorStore.search() → List[VectorSearchResult]
    → HybridRetriever._vector_retrieve() → converted to RetrievedChunk.

    Attributes:
        id: Unique record identifier (e.g. "{document_id}_chunk_{index}").
        score: Similarity score in [0.0, 1.0]; 1.0 = identical,
            0.0 = completely unrelated.  Computed as 1.0 - cosine_distance.
        text: Original document text stored in ChromaDB's ``document`` field.
        metadata: Record metadata including document_id, filename, page,
            char_start, char_end, source, etc.
    """
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class ChromaDBVectorStore:
    """ChromaDB-backed vector store for dense retrieval.

    Manages the full lifecycle of a ChromaDB collection: connection,
    dimension validation, document insertion, similarity search, and
    document-level deletion.

    Supported metadata filter keys are restricted via
    ``SUPPORTED_FILTER_KEYS`` to prevent arbitrary field queries.

    Attributes:
        SUPPORTED_FILTER_KEYS: Whitelist of metadata fields allowed in
            filter expressions.  Currently ``["document_id", "document_ids"]``.
    """

    SUPPORTED_FILTER_KEYS = ["document_id", "document_ids"]

    def __init__(self, settings=None):
        """Initialize the vector store and connect to ChromaDB.

        Connection is established immediately (Fail-Fast) so that
        configuration or database issues surface at startup rather than
        at first query time.

        Args:
            settings: Configuration object.  Falls back to
                ``get_settings()`` singleton when *None*.  Useful for
                injecting mock settings in tests.
        """
        self.settings = settings or get_settings()

        self.collection_name = getattr(
            self.settings, 'CHROMA_COLLECTION_NAME', 'veriquery_text'
        )
        self.persist_dir = getattr(
            self.settings, 'CHROMA_PERSIST_DIR', './data/chroma'
        )
        self.dimension = getattr(
            self.settings, 'EMBEDDING_DIMENSION', 1536
        )

        self.client = None
        self.collection = None

        self._connect()

    def _validate_and_recreate_collection(self):
        """Validate collection dimension; recreate on mismatch.

        Reads the ``dimension`` key from the collection's metadata and
        compares it with ``self.dimension``.  A mismatch (caused by
        switching embedding models) triggers deletion of the old
        collection and creation of a new one.  Old vectors are lost and
        must be re-indexed.
        """
        collection_metadata = self.collection.metadata or {}
        stored_dimension = collection_metadata.get("dimension")

        if stored_dimension and stored_dimension != self.dimension:
            logger.warning(
                "ChromaDB collection dimension (%s) differs from config (%s), "
                "recreating collection",
                stored_dimension, self.dimension,
            )
            self.client.delete_collection(name=self.collection_name)
            self._create_collection()
            logger.info("Recreated ChromaDB collection with dimension=%s", self.dimension)
        elif not stored_dimension:
            logger.warning("Existing collection has no dimension metadata; validation skipped")

    def _create_collection(self):
        """Create a new ChromaDB collection with cosine distance metric."""
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "dimension": self.dimension,
            },
        )
        logger.info("Created ChromaDB collection: %s, dimension=%s", self.collection_name, self.dimension)

    def _check_and_fix_db_schema(self):
        """Detect and resolve ChromaDB schema incompatibilities.

        ChromaDB versions may have different SQLite schemas.  When an
        incompatible schema is detected the entire persist directory is
        deleted so that a fresh database is created on the next connection.
        """
        from pathlib import Path
        import gc

        persist_path = Path(self.persist_dir)
        chroma_sqlite = persist_path / "chroma.sqlite3"
        if not chroma_sqlite.exists():
            return

        try:
            import sqlite3

            conn = sqlite3.connect(str(chroma_sqlite))
            cursor = conn.cursor()

            cursor.execute("PRAGMA table_info(collections)")
            columns = [col[1] for col in cursor.fetchall()]
            conn.close()

            if not columns:
                return

            try:
                test_client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=Settings(allow_reset=True, anonymized_telemetry=False),
                )
                test_client.list_collections()
                del test_client
                gc.collect()
                return
            except Exception as compat_err:
                logger.warning("ChromaDB schema incompatible: %s", compat_err)

                try:
                    del test_client
                except Exception:
                    pass
                gc.collect()

                logger.warning("Removing old database for recreation")

                import shutil
                import time

                for attempt in range(5):
                    try:
                        if persist_path.exists():
                            shutil.rmtree(persist_path)
                            logger.info("Removed old ChromaDB database: %s", persist_path)
                        break
                    except Exception as rm_err:
                        if attempt < 4:
                            logger.warning("Retry %d: failed to remove ChromaDB dir: %s", attempt + 1, rm_err)
                            gc.collect()
                            time.sleep(2)
                        else:
                            logger.error("Failed to remove ChromaDB dir after 5 attempts: %s", rm_err)
                            raise RetrievalError(
                                f"Cannot remove incompatible ChromaDB database. "
                                f"Please manually delete: {persist_path}"
                            )

        except RetrievalError:
            raise
        except Exception as e:
            logger.warning("ChromaDB schema check failed: %s", e)

    def _connect(self):
        """Connect to ChromaDB and obtain (or create) the target collection.

        Flow:
            1. Check/fix database schema compatibility.
            2. Create a ``PersistentClient`` for local disk persistence.
            3. Try ``get_collection`` → validate dimension on existing data.
            4. Fall back to ``_create_collection`` when the collection
               does not yet exist.

        Raises:
            RetrievalError: If the connection cannot be established.
        """
        try:
            self._check_and_fix_db_schema()

            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(allow_reset=True, anonymized_telemetry=False),
            )

            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info("Loaded existing ChromaDB collection: %s", self.collection_name)
                self._validate_and_recreate_collection()
            except Exception:
                self._create_collection()

            logger.info("Connected to ChromaDB: %s", self.persist_dir)

        except Exception as e:
            logger.error("ChromaDB connection failed: %s", e)
            raise RetrievalError(f"Cannot connect to ChromaDB: {e}")

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Add document vectors to the ChromaDB collection.

        Although ChromaDB's Python SDK is synchronous, this method is
        declared ``async`` for API consistency with the upstream pipeline
        and to allow a future migration to an async-capable vector
        database without changing call sites.

        Args:
            documents: List of document dicts, each containing:
                - ``content`` (str): Original text chunk.
                - ``metadata`` (dict): Metadata with document_id,
                  filename, page, char_start, char_end, etc.
                - ``id`` (str, optional): Unique identifier; defaults to
                  a UUID4 when omitted.
            embeddings: Embedding vectors corresponding 1-to-1 with
                *documents*.  Each vector's length must equal
                ``self.dimension``.

        Returns:
            List of IDs for the successfully added documents.

        Raises:
            ValueError: If any embedding dimension does not match
                ``self.dimension``.
            RetrievalError: If ChromaDB fails to persist the data.
        """
        if not documents:
            return []

        texts = []
        metadatas = []
        ids = []

        for doc in documents:
            texts.append(doc.get('content', ''))
            metadatas.append(doc.get('metadata', {}))
            ids.append(doc.get('id', str(uuid.uuid4())))

        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.dimension:
                raise ValueError(
                    f"Document {i} embedding dimension {len(embedding)} "
                    f"does not match expected {self.dimension}"
                )
            if all(x == 0.0 for x in embedding):
                logger.warning("Document %d has a zero vector; it will never be retrieved", i)

        try:
            clean_embeddings = []
            for emb in embeddings:
                if hasattr(emb, 'tolist'):
                    clean_embeddings.append(emb.tolist())
                elif isinstance(emb, (list, tuple)):
                    clean_embeddings.append(list(emb))
                else:
                    clean_embeddings.append(emb)

            self.collection.add(
                embeddings=clean_embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info("Added %d vectors to ChromaDB", len(texts))
            return ids
        except Exception as e:
            logger.error("ChromaDB add failed: %s", e)
            raise RetrievalError(f"Failed to add vectors: {e}")

    def _build_filter_dict(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Convert a user-supplied filter dict into a ChromaDB ``where`` clause.

        Transformation rules:
            1. Only keys in ``SUPPORTED_FILTER_KEYS`` are processed.
            2. ``document_ids`` is normalised to ``document_id`` (the
               actual metadata key in ChromaDB).
            3. List/tuple/set values become ``{"$in": [...]}``.
            4. Scalar values become equality matches.
            5. ``None`` values are skipped.
            6. Multiple conditions are combined with ``"$and"``.

        Args:
            filter_dict: Raw filter from the caller, e.g.
                ``{"document_id": "doc1"}`` or
                ``{"document_ids": ["doc1", "doc2"]}``.

        Returns:
            A ChromaDB-compatible ``where`` dict, or ``None`` when no
            valid filters remain.
        """
        if not filter_dict:
            return None

        normalized_filters = {}

        for key, value in filter_dict.items():
            if key not in self.SUPPORTED_FILTER_KEYS:
                logger.debug("Skipping unsupported filter key: %s", key)
                continue

            normalized_key = "document_id" if key == "document_ids" else key

            if isinstance(value, (list, tuple, set)):
                values = [str(v) for v in value if v is not None]
                if values:
                    normalized_filters[normalized_key] = {"$in": values}
            elif value is not None:
                normalized_filters[normalized_key] = str(value)

        if not normalized_filters:
            return None

        if len(normalized_filters) == 1:
            return normalized_filters

        return {"$and": [{k: v} for k, v in normalized_filters.items()]}

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 50,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for the most similar vectors by cosine distance.

        ChromaDB's HNSW index returns cosine *distances*; these are
        converted to similarity scores via ``score = 1.0 - distance``,
        clamped to [0, 1] to guard against floating-point drift.

        The ``collection.query()`` response is a nested dict whose values
        are two-dimensional lists (outer = queries, inner = results).
        Since this method issues a single query, index ``[0]`` is used
        throughout.

        Args:
            query_embedding: Query vector produced by
                ``EmbeddingManager.embed_query()``.  Must match the
                collection's embedding dimension.
            top_k: Maximum number of results.  Callers (e.g.
                ``HybridRetriever``) often request ``top_k * 2`` to
                compensate for RRF deduplication.
            filter_dict: Optional metadata filter passed to
                ``_build_filter_dict``.

        Returns:
            Results sorted by descending similarity score.

        Raises:
            RetrievalError: On ChromaDB query failure.
        """
        try:
            query_embedding_list = (
                query_embedding.tolist()
                if hasattr(query_embedding, 'tolist')
                else list(query_embedding)
            )
            query_params = {
                'query_embeddings': [query_embedding_list],
                'n_results': top_k,
                'include': ['documents', 'metadatas', 'distances'],
            }

            where_clause = self._build_filter_dict(filter_dict)
            if where_clause:
                query_params['where'] = where_clause

            try:
                results = self.collection.query(**query_params)
            except TypeError as e:
                if "has no len()" in str(e):
                    logger.warning("ChromaDB where-clause compatibility error, falling back to unfiltered search: %s", e)
                    query_params_fallback = {
                        'query_embeddings': [query_embedding_list],
                        'n_results': top_k,
                        'include': ['documents', 'metadatas', 'distances'],
                    }
                    results = self.collection.query(**query_params_fallback)

                    if where_clause and results['ids'] and results['ids'][0]:
                        filter_key = None
                        filter_values = set()
                        if '$and' in where_clause:
                            for cond in where_clause['$and']:
                                for k, v in cond.items():
                                    if k == 'document_id':
                                        if isinstance(v, dict) and '$in' in v:
                                            filter_values = set(v['$in'])
                                        else:
                                            filter_values = {v}
                                        filter_key = k
                        elif 'document_id' in where_clause:
                            v = where_clause['document_id']
                            if isinstance(v, dict) and '$in' in v:
                                filter_values = set(v['$in'])
                            else:
                                filter_values = {v}
                            filter_key = 'document_id'

                        if filter_key and filter_values:
                            filtered_ids = []
                            filtered_docs = []
                            filtered_metas = []
                            filtered_dists = []
                            for i, id_ in enumerate(results['ids'][0]):
                                meta = results['metadatas'][0][i] or {}
                                if meta.get(filter_key) in filter_values:
                                    filtered_ids.append(id_)
                                    filtered_docs.append(results['documents'][0][i])
                                    filtered_metas.append(meta)
                                    filtered_dists.append(results['distances'][0][i])
                            results = {
                                'ids': [filtered_ids],
                                'documents': [filtered_docs],
                                'metadatas': [filtered_metas],
                                'distances': [filtered_dists],
                            }
                else:
                    raise

            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, id_ in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    score = max(0.0, min(1.0, 1.0 - distance))
                    metadata = results['metadatas'][0][i] or {}

                    search_results.append(VectorSearchResult(
                        id=id_,
                        score=score,
                        text=results['documents'][0][i],
                        metadata=metadata,
                    ))

            logger.info("ChromaDB search returned %d results", len(search_results))
            return search_results

        except Exception as e:
            logger.error("ChromaDB search failed: %s", e)
            raise RetrievalError(f"Search failed: {e}")

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all vectors belonging to a specific document.

        Used by ``cleanup_manager`` during document removal and by the
        document re-processing pipeline.

        Args:
            document_id: The document identifier matching the
                ``document_id`` key in each record's metadata.

        Returns:
            Number of deleted records.  Returns 0 (instead of raising)
            on failure so that cleanup of other stores (BM25, table)
            can continue.
        """
        try:
            try:
                result = self.collection.get(where={"document_id": document_id})
            except TypeError as e:
                if "has no len()" in str(e):
                    logger.warning("ChromaDB get filter compatibility error, falling back to full scan: %s", e)
                    all_data = self.collection.get(include=['metadatas'])
                    matched_ids = []
                    for i, id_ in enumerate(all_data['ids']):
                        meta = all_data['metadatas'][i] if all_data['metadatas'] else {}
                        if meta.get('document_id') == document_id:
                            matched_ids.append(id_)
                    result = {'ids': matched_ids}
                else:
                    raise

            ids_to_delete = result.get("ids", [])
            count = len(ids_to_delete)
            if count > 0:
                self.collection.delete(ids=ids_to_delete)
                logger.info("Deleted %d vectors for document_id=%s", count, document_id)
            return count
        except Exception as e:
            logger.error("Failed to delete vectors by document_id: %s", e)
            return 0


def create_vector_store(settings=None):
    """Factory function for creating a ChromaDBVectorStore instance.

    Decouples callers from the concrete implementation class, making it
    straightforward to swap in an alternative vector database (e.g.
    Milvus, Weaviate) without modifying downstream code.

    Args:
        settings: Configuration object; defaults to the global
            ``get_settings()`` singleton.  Pass a custom object to
            inject mock settings in tests.

    Returns:
        A connected ``ChromaDBVectorStore`` instance.
    """
    return ChromaDBVectorStore(settings)
