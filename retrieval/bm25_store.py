"""
BM25 Sparse Retrieval Store
============================

VeriQuery retrieval module component implementing the Sparse retrieval path
in the "Dense + Sparse + Structured" hybrid retrieval architecture.

Hybrid Retrieval Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     HybridRetriever (RRF Fusion)                    │
    │                                                                     │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
    │  │  VectorStore  │  │  BM25Store   │  │  TableStore   │              │
    │  │  (Dense)      │  │  (Sparse)    │  │  (Structured) │              │
    │  │  ChromaDB     │  │  ★ This ★    │  │  SQLite       │              │
    │  │  Semantic     │  │  Keyword     │  │  Table data   │              │
    │  └──────────────┘  └──────────────┘  └──────────────┘              │
    │        w=0.5            w=0.35           w=0.15                     │
    └─────────────────────────────────────────────────────────────────────┘

Core Responsibilities:
    1. BM25-based sparse retrieval (keyword matching) to complement vector
       retrieval for exact matches on part numbers, parameter names, etc.
    2. Chinese (jieba) and English tokenization for mixed-language datasheets.
    3. Domain-specific synonym expansion (e.g. "放大器" ↔ "amplifier").
    4. Stop-word filtering to improve retrieval precision.
    5. Index persistence via pickle serialization.

BM25 Algorithm (Okapi BM25, Robertson & Zaragoza 2009):

    score(D, Q) = Σ_{i=1}^{n} IDF(q_i) × (f(q_i, D) × (k1 + 1)) /
                                     (f(q_i, D) + k1 × (1 - b + b × |D|/avgdl))

    where:
        q_i        - i-th query term
        f(q_i, D)  - term frequency of q_i in document D
        |D|        - document length (term count)
        avgdl      - average document length in corpus
        k1         - term frequency saturation parameter (default 1.5)
        b          - length normalization parameter (default 0.75)
        IDF(q_i)   - inverse document frequency =
                     log((N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1)

Why BM25 alongside vector retrieval:
    - Dense retrieval excels at semantic understanding but may drift on exact
      keywords (e.g. part number "SN74HC04").
    - BM25 provides exact term matching, naturally suited for part numbers,
      parameter names, and other proper nouns.
    - The two are complementary: vector search understands "chip functionality",
      BM25 precisely matches "SN74HC04 pin definitions".
    - RRF fusion combines both for balanced semantic + lexical retrieval.

Data Flow:
    Document upload → document_processor chunking → add_documents() tokenize + index
                                                              ↓
    User query → search() tokenize → synonym expand → BM25 score → sort → results
                                                              ↓
                              HybridRetriever._bm25_retrieve() → RRF fusion

Dependencies:
    - rank_bm25: BM25Okapi implementation (pip install rank-bm25)
    - jieba: Chinese tokenizer (pip install jieba), optional with graceful fallback
    - pickle: Built-in serialization for index persistence
"""

import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from core import get_settings

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BM25SearchResult:
    """BM25 search result data class.

    Attributes:
        doc_id: Document unique identifier (e.g. "{document_id}_chunk_{index}").
        score: BM25 relevance score. Raw scores are unbounded; normalized to [0, 1]
               by HybridRetriever before RRF fusion.
        text: Original document text (un-tokenized), for display or LLM input.
        metadata: Document metadata including document_id, filename, page,
                  char_start/char_end, source, etc.
    """
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class BM25Store:
    """BM25 sparse retrieval store.

    Core implementation of the Sparse retrieval path in VeriQuery's hybrid
    architecture. Handles document addition, tokenization, BM25 indexing,
    keyword search, synonym expansion, and index persistence.

    Called by HybridRetriever._bm25_retrieve(); results are normalized and
    fused with VectorStore and TableStore via RRF.

    Core Data Structures (index-correlated lists):
        bm25:           BM25Okapi index object (None if not yet built).
        documents:      Original document text list.
        tokenized_docs: Tokenized document list (input for BM25 index).
        metadatas:      Per-document metadata list.
        doc_ids:        Document unique ID list.

    All five lists are position-correlated: documents[i] corresponds to
    tokenized_docs[i], metadatas[i], and doc_ids[i].

    Example:
        store = BM25Store()
        store.add_documents(["doc1 text", "doc2 text"], [{"id": "1"}, {"id": "2"}])
        results = store.search("query", top_k=5)
    """

    def __init__(self, settings=None, persist_path: str = None):
        """Initialize BM25 store.

        Initialization flow:
            1. Load settings → 2. Load stop words → 3. Init tokenizer
            → 4. Attempt to load existing index from persist_path.

        Args:
            settings: Global configuration object. Defaults to get_settings()
                      singleton. Allows injection for testing.
            persist_path: Path for index persistence (.pkl format). If specified
                          and file exists, existing index is auto-loaded.
        """
        self.settings = settings or get_settings()
        self.persist_path = persist_path

        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        self.metadatas = []
        self.doc_ids = []

        self._load_stopwords()
        self._init_tokenizer()

        if persist_path and Path(persist_path).exists():
            self._load()

    def _load_stopwords(self):
        """Load stop words table.

        Stop words are high-frequency but semantically empty words (e.g. "的",
        "the") that should be filtered to reduce index size and improve BM25
        scoring precision.

        Loading strategy (priority order):
            1. Load from configured stop words file (supports custom extension).
            2. Fall back to built-in default stop word set (~30 Chinese + English).
        """
        stopwords_file = getattr(self.settings, 'STOPWORDS_FILE', None)

        if stopwords_file and Path(stopwords_file).exists():
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    self.stopwords = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(self.stopwords)} stop words from {stopwords_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to load stop words file: {e}, using defaults")

        self.stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
            '会', '着', '没有', '看', '好', '自己', '这', '个', '那', '些',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }

    def _init_tokenizer(self):
        """Initialize tokenizer with strategy pattern.

        Strategy selection:
            - jieba available → jieba precise mode (prefix dictionary + Viterbi).
            - jieba unavailable → fallback to simple regex tokenization (whitespace
              and punctuation split, English-only effective).

        The selected tokenize function is bound to self.tokenize, so callers
        invoke self.tokenize(text) without needing to check which strategy is active.
        """
        if JIEBA_AVAILABLE:
            jieba.setLogLevel(logging.WARNING)
            self.tokenize = self._jieba_tokenize
            logger.info("Using jieba tokenizer")
        else:
            self.tokenize = self._simple_tokenize
            logger.warning("jieba not installed, using simple tokenizer")

    def _jieba_tokenize(self, text: str) -> List[str]:
        """jieba precise tokenization with stop word filtering.

        Processing pipeline:
            1. Input validation → 2. Regex cleanup → 3. Lowercase
            → 4. jieba precise cut → 5. Stop word + short token filtering

        Args:
            text: Raw text to tokenize, e.g. "NE5532是双运算放大器".

        Returns:
            Filtered token list, e.g. ["NE5532", "双", "运算", "放大器"].
        """
        if not text or not isinstance(text, str):
            return []

        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        text = text.lower()

        if not text.strip():
            return []

        try:
            tokens = jieba.lcut(text, cut_all=False)
        except Exception as e:
            logger.error(f"jieba tokenization failed: {e}")
            return self._simple_tokenize(text)

        if not tokens:
            return []

        filtered_tokens = [
            token for token in tokens
            if (len(token) > 1 and
                not token.isspace() and
                token not in self.stopwords and
                re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', token))
        ]

        return filtered_tokens

    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple whitespace/punctuation tokenization — fallback when jieba unavailable.

        Splits on non-word characters. Effective for English text; Chinese text
        will not be properly segmented (entire phrases treated as single tokens).

        Args:
            text: Raw text to tokenize.

        Returns:
            Filtered token list.
        """
        if not text or not isinstance(text, str):
            return []

        text = text.lower()

        if not text.strip():
            return []

        tokens = re.split(r'[^\w]+', text)

        filtered_tokens = [
            token for token in tokens
            if len(token) > 1 and not token.isspace() and token not in self.stopwords
        ]

        return filtered_tokens

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None,
                      doc_ids: List[str] = None) -> int:
        """Add documents to the BM25 index.

        Each call appends documents and rebuilds the entire BM25 index
        (BM25Okapi does not support incremental updates).

        Args:
            texts: Document text list (typically chunks from document_processor).
            metadatas: Metadata list corresponding to texts. Auto-generated as
                       empty dicts if None.
            doc_ids: Document ID list corresponding to texts. Auto-generated as
                     incrementing strings if None.

        Returns:
            Number of documents successfully added.

        Raises:
            RuntimeError: If rank_bm25 is not installed.
        """
        if not texts:
            return 0

        if BM25Okapi is None:
            raise RuntimeError("rank_bm25 not installed, BM25 retrieval unavailable")

        metadatas = metadatas or [{} for _ in texts]
        doc_ids = doc_ids or [str(i + len(self.documents)) for i in range(len(texts))]

        new_tokenized = [self.tokenize(text) for text in texts]

        self.documents.extend(texts)
        self.tokenized_docs.extend(new_tokenized)
        self.metadatas.extend(metadatas)
        self.doc_ids.extend(doc_ids)

        self._rebuild_index()

        return len(texts)

    def _rebuild_index(self):
        """Rebuild BM25 index from full tokenized_docs.

        BM25Okapi construction computes:
            1. Inverse document frequency (IDF) for each term.
            2. Corpus average document length (avgdl).
            3. Term frequency distribution per document.

        Raises:
            RuntimeError: If rank_bm25 is not installed.
        """
        if BM25Okapi is None:
            raise RuntimeError("rank_bm25 not installed, BM25 retrieval unavailable")

        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10,
               filter_func: callable = None) -> List[BM25SearchResult]:
        """Search documents using BM25 algorithm.

        Retrieval pipeline:
            1. Pre-check (index and documents existence)
            2. Query tokenization
            3. Synonym expansion (boost recall)
            4. BM25 scoring
            5. Optional metadata filtering
            6. Sort by score descending, take top_k
            7. Filter zero-score results, build return objects

        Args:
            query: User query text, e.g. "NE5532的供电电压范围".
            top_k: Maximum number of results to return. Defaults to 10.
                   HybridRetriever typically sets top_k*2 for broader candidates.
            filter_func: Optional filter function with signature
                         f(doc_id, metadata) -> bool. Return True to keep,
                         False to filter out.

        Returns:
            List of BM25SearchResult sorted by relevance score (descending).
        """
        if self.bm25 is None or not self.documents:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        expanded_tokens = self._expand_query_tokens(query_tokens)

        scores = self.bm25.get_scores(expanded_tokens)

        scored_indices = [(i, scores[i]) for i in range(len(scores))]

        if filter_func:
            scored_indices = [
                (i, s) for i, s in scored_indices
                if filter_func(self.doc_ids[i], self.metadatas[i])
            ]

        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:top_k]

        results = []
        for idx, score in top_indices:
            if score > 0:
                results.append(BM25SearchResult(
                    doc_id=self.doc_ids[idx],
                    score=float(score),
                    text=self.documents[idx],
                    metadata=self.metadatas[idx]
                ))

        return results

    def _expand_query_tokens(self, tokens: List[str]) -> List[str]:
        """Expand query tokens with domain-specific synonyms.

        Query expansion is a classic information retrieval technique to boost
        recall. Users may use different terms than documents (e.g. searching
        "放大器" but documents use "amplifier"). Synonym expansion bridges
        this gap for BM25's exact-match paradigm.

        The synonym map is bidirectional (Chinese ↔ English) to ensure
        matching regardless of query language.

        Args:
            tokens: Original tokenized query, e.g. ["放大器", "工作", "电压"].

        Returns:
            Expanded token list, e.g. ["放大器", "放大", "amplifier", "amplify",
            "工作", "电压"].
        """
        expanded = list(tokens)

        synonym_map = {
            "放大器": ["放大", "amplifier", "amplify"],
            "放大": ["放大器", "amplifier", "amplify"],
            "滤波器": ["滤波", "filter"],
            "滤波": ["滤波器", "filter"],
            "振荡器": ["振荡", "oscillator"],
            "振荡": ["振荡器", "oscillator"],
            "整流器": ["整流", "rectifier"],
            "整流": ["整流器", "rectifier"],
            "稳压器": ["稳压", "regulator", "voltage_regulator"],
            "稳压": ["稳压器", "regulator"],
            "电路": ["circuit"],
            "电路图": ["circuit", "schematic"],
            "电阻": ["resistor"],
            "电容": ["capacitor"],
            "电感": ["inductor"],
            "晶体管": ["transistor"],
            "二极管": ["diode"],
            "运算放大器": ["opamp", "op_amp", "operational_amplifier"],
            "opamp": ["运算放大器", "operational_amplifier"],
            "op_amp": ["运算放大器", "operational_amplifier"],
        }

        for token in tokens:
            if token in synonym_map:
                for synonym in synonym_map[token]:
                    if synonym not in expanded:
                        expanded.append(synonym)

        return expanded

    def delete_by_doc_ids(self, doc_ids: List[str]) -> int:
        """Delete documents by ID with dual matching strategy.

        BM25 internally stores chunk-level IDs (e.g. "abc12345_p3_c7"), but
        callers typically pass document-level IDs (e.g. "abc12345"). Three
        matching conditions are checked for each entry:
            1. Exact match: doc_id in to_delete (for batch chunk-level IDs).
            2. Metadata match: metadata["document_id"] in to_delete.
            3. Prefix match: doc_id starts with a to_delete entry + "_p".

        Args:
            doc_ids: Document IDs to delete (supports document-level or chunk-level).

        Returns:
            Number of actually deleted document chunks.
        """
        if not doc_ids:
            return 0

        to_delete = set(doc_ids)
        original_count = len(self.doc_ids)

        def _should_delete(idx: int) -> bool:
            chunk_id = self.doc_ids[idx]
            if chunk_id in to_delete:
                return True
            meta_doc_id = self.metadatas[idx].get("document_id", "") if idx < len(self.metadatas) else ""
            if meta_doc_id and meta_doc_id in to_delete:
                return True
            for del_id in to_delete:
                if chunk_id.startswith(del_id + "_p"):
                    return True
            return False

        indices_to_keep = [
            i for i in range(original_count)
            if not _should_delete(i)
        ]

        deleted_count = original_count - len(indices_to_keep)

        if deleted_count > 0:
            self.documents = [self.documents[i] for i in indices_to_keep]
            self.tokenized_docs = [self.tokenized_docs[i] for i in indices_to_keep]
            self.metadatas = [self.metadatas[i] for i in indices_to_keep]
            self.doc_ids = [self.doc_ids[i] for i in indices_to_keep]

            self._rebuild_index()

            if self.persist_path:
                try:
                    self.save()
                except Exception as e:
                    logger.warning(f"Failed to persist BM25 index after deletion: {e}")

        return deleted_count

    def _reset_state(self):
        """Reset all internal data structures to empty state.

        Called when index loading fails to ensure a consistent empty state,
        preventing residual data from causing errors in subsequent operations.
        """
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        self.metadatas = []
        self.doc_ids = []

    def save(self, path: str = None):
        """Save BM25 index data to file via pickle serialization.

        Saves the four core lists (documents, tokenized_docs, metadatas, doc_ids)
        but NOT the BM25Okapi object itself (it may contain unpicklable internals).
        The index is rebuilt from tokenized_docs on load via _rebuild_index().

        Args:
            path: Save path. Defaults to persist_path from initialization.
                  If both are None, no save is performed.
        """
        path = path or self.persist_path
        if not path:
            return

        data = {
            "documents": self.documents,
            "tokenized_docs": self.tokenized_docs,
            "metadatas": self.metadatas,
            "doc_ids": self.doc_ids
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index saved to: {path}")

    def _load(self):
        """Load BM25 index from file with layered error handling.

        Loading flow:
            1. Check file existence → 2. pickle deserialize → 3. Integrity check
            → 4. Consistency check → 5. Rebuild BM25 index

        Error handling (layered degradation):
            - FileNotFoundError: Reset state, await new documents.
            - PickleError: Corrupted file, reset state, need re-indexing.
            - ValueError: Incomplete/inconsistent data, reset state.
            - Other: Unknown error, reset state, system degrades to vector-only.
        """
        try:
            if not self.persist_path or not Path(self.persist_path).exists():
                logger.info(f"BM25 index file not found: {self.persist_path}")
                self.bm25 = None
                return

            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)

            required_keys = ["documents", "tokenized_docs", "metadatas", "doc_ids"]
            if not all(key in data for key in required_keys):
                raise ValueError("Index data incomplete, missing required fields")

            self.documents = data["documents"]
            self.tokenized_docs = data["tokenized_docs"]
            self.metadatas = data["metadatas"]
            self.doc_ids = data["doc_ids"]

            doc_count = len(self.documents)
            if not all(len(arr) == doc_count for arr in [self.tokenized_docs, self.metadatas, self.doc_ids]):
                raise ValueError(
                    f"Index data inconsistent: documents={doc_count}, "
                    f"tokenized_docs={len(self.tokenized_docs)}, "
                    f"metadatas={len(self.metadatas)}, doc_ids={len(self.doc_ids)}"
                )

            if BM25Okapi is not None:
                self._rebuild_index()
            else:
                logger.warning("rank_bm25 not installed, cannot build BM25 index")

            logger.info(f"BM25 index loaded: {len(self.documents)} documents")

        except FileNotFoundError:
            logger.warning(f"BM25 index file not found: {self.persist_path}")
            logger.info("BM25 retrieval unavailable until documents are added")
            self._reset_state()

        except pickle.PickleError as e:
            logger.error(f"BM25 index file corrupted: {e}")
            logger.info("BM25 retrieval unavailable, index needs to be recreated")
            self._reset_state()

        except ValueError as e:
            logger.error(f"BM25 index data validation failed: {e}")
            logger.info("BM25 retrieval unavailable, index needs to be recreated")
            self._reset_state()

        except Exception as e:
            logger.error(f"BM25 index load failed: {e}", exc_info=True)
            logger.info("BM25 retrieval unavailable, system degrades to vector-only")
            self._reset_state()
