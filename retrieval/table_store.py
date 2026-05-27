"""
Table Store — Structured Table Storage with Full-Text Search
=============================================================

Persistent storage and retrieval for PDF-extracted tabular data using
SQLite + FTS5. Part of VeriQuery's three-path hybrid retrieval pipeline
(Dense + Sparse + Structured).

Architecture Position:
    extraction/table_extractor.py  →  TableStore.add()       (write)
    retrieval/hybrid_retriever.py  →  TableStore.search()    (read)
    extraction/parameter_extractor  →  TableStore.get_tables_by_documents()  (read)
    core/cleanup_manager.py        →  TableStore.delete_by_document()       (delete)

Database Schema:
    Main table `tables`:
        table_id      TEXT PRIMARY KEY  — Unique table identifier
        document_id   TEXT NOT NULL     — Parent document ID
        filename      TEXT              — Source PDF filename
        page          INTEGER           — Page number (1-indexed)
        data          TEXT NOT NULL     — Raw table data (JSON 2D string array)
        header        TEXT              — Header row (JSON serialized)
        content       TEXT              — Full text (all cells space-joined, for FTS5/LIKE)
        row_count     INTEGER           — Number of rows
        column_count  INTEGER           — Number of columns
        source        TEXT              — Extraction source (e.g. "camelot_lattice")
        confidence    REAL              — Extraction confidence [0, 1]
        created_at    TIMESTAMP         — Creation timestamp

    Virtual table `tables_fts` (FTS5):
        rowid         — Maps to main table rowid (content table mode)
        header        — Header text for FTS5 MATCH
        content       — Full text for FTS5 MATCH

    Indexes:
        idx_tables_document  — ON tables(document_id)
        idx_tables_page      — ON tables(document_id, page)

Key Design Decisions:
    - SQLite over MySQL/PostgreSQL: Zero-config, file-level storage,
      built-in FTS5 full-text search, sufficient for single-machine deployment.
    - FTS5 with LIKE fallback: Graceful degradation when FTS5 module is
      unavailable (e.g. some Windows SQLite builds). FTS5 uses BM25 ranking;
      LIKE assigns uniform score=1.0.
    - Content table mode (content='tables'): FTS5 indexes reference the main
      table's data rather than storing a copy, saving disk space. Trade-off:
      FTS index must be manually synchronized on writes (DELETE old + INSERT new).
    - Async search via asyncio.to_thread(): SQLite's synchronous I/O is
      offloaded to a thread pool to avoid blocking FastAPI's event loop.
      Other methods (add, get_tables_by_documents) remain synchronous as
      they are called from sync contexts.
"""

import asyncio
import logging
import sqlite3
import json
import re
import uuid
import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

from core import get_settings
from core.sqlite_utils import get_safe_connection, check_database_health, repair_corrupted_database

logger = logging.getLogger(__name__)


@dataclass
class TableSearchResult:
    """Single table search result with relevance score.

    Attributes:
        table_id: Unique table identifier, e.g. "table_a1b2c3d4".
        document_id: Parent document ID.
        filename: Source PDF filename.
        page: Page number in the PDF (1-indexed).
        data: Raw table data as 2D string array.
        row_count: Number of rows (including header).
        column_count: Number of columns.
        source: Extraction source, e.g. "camelot_lattice" or "pdfplumber".
        confidence: Extraction confidence [0, 1]. Higher is better.
        score: Search relevance score. FTS5: BM25 score (abs value, higher
               is more relevant). LIKE: uniform 1.0.
    """
    table_id: str
    document_id: str
    filename: str
    page: int
    data: List[List[str]]
    row_count: int
    column_count: int
    source: str
    confidence: float
    score: float = 0


class TableStore:
    """Repository for structured table data with FTS5 full-text search.

    Implements the Repository Pattern: encapsulates all SQLite data access
    behind domain-oriented interfaces (add/search/get_tables_by_documents).

    Features:
        - FTS5 full-text search with BM25 ranking, automatic LIKE fallback.
        - Async search interface via asyncio.to_thread().
        - Per-document table queries with confidence filtering.
        - Automatic FTS index rebuild on corruption detection.
        - Thread-safe writes via threading.Lock.

    Thread Safety:
        SQLite operates in SERIALIZED mode by default. Each operation creates
        a new connection via _get_connection() and closes it after use,
        ensuring thread safety without a connection pool.
    """

    _QUERY_CLEAN_PATTERN = re.compile(
        r'[?？,，\(\)\[\]\{\}"\'\:\;\-\_\*\+\.\/\\\|\&\#\@\!\=\<\>]'
    )

    def __init__(self, db_path: str = None, settings=None):
        """Initialize table store with optional dependency injection.

        Args:
            db_path: SQLite database file path. Defaults to
                     settings.table_db_path or "./data/tables.db".
            settings: Configuration object. Defaults to get_settings() singleton.
        """
        try:
            self.settings = settings or get_settings()
            self.db_path = db_path or str(getattr(self.settings, 'table_db_path', "./data/tables.db"))
        except Exception as e:
            logger.warning(f"Settings load failed: {e}, using default path")
            self.db_path = "./data/tables.db"

        self._write_lock = threading.Lock()
        self._fts_corrupted = False
        self._init_database()

    def _init_database(self):
        """Initialize database: create main table, indexes, and FTS5 virtual table.

        Idempotent: uses IF NOT EXISTS for all DDL statements, safe to call
        repeatedly (e.g. on application restart).

        FTS5 availability is detected at runtime:
            - Success → self.fts_available = True (BM25 search path)
            - "no such module: fts5" → self.fts_available = False (LIKE fallback)
            - Other errors → propagated as exceptions
        """
        try:
            for ext in ["-wal", "-shm"]:
                f = f"{self.db_path}{ext}"
                if os.path.exists(f):
                    try:
                        os.remove(f)
                        logger.info(f"Cleaned residual file: {f}")
                    except Exception:
                        pass

            if not check_database_health(self.db_path):
                logger.warning(f"Database may be corrupted: {self.db_path}")
                if repair_corrupted_database(self.db_path):
                    logger.info(f"Database repaired: {self.db_path}")
                else:
                    logger.error(f"Database unrepairable, will delete and recreate: {self.db_path}")
                    from core.sqlite_utils import safe_delete_database
                    safe_delete_database(self.db_path)

            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tables (
                        table_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        filename TEXT,
                        page INTEGER,
                        data TEXT NOT NULL,
                        header TEXT,
                        content TEXT,
                        row_count INTEGER,
                        column_count INTEGER,
                        source TEXT,
                        confidence REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tables_document
                    ON tables(document_id)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tables_page
                    ON tables(document_id, page)
                """)

                try:
                    conn.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS tables_fts USING fts5(
                            header,
                            content,
                            content='tables',
                            content_rowid='rowid'
                        )
                    """)
                    self.fts_available = True
                    logger.info("FTS5 full-text search available")
                except sqlite3.OperationalError as e:
                    if "no such module: fts5" in str(e):
                        logger.warning("FTS5 unavailable, falling back to LIKE search")
                        self.fts_available = False
                    else:
                        raise

                conn.commit()

                if self.fts_available:
                    try:
                        orphan_count = conn.execute(
                            "SELECT COUNT(*) FROM tables_fts WHERE rowid NOT IN (SELECT rowid FROM tables)"
                        ).fetchone()[0]
                        if orphan_count > 0:
                            logger.warning(f"Found {orphan_count} orphan FTS entries, cleaning...")
                            conn.execute(
                                "DELETE FROM tables_fts WHERE rowid NOT IN (SELECT rowid FROM tables)"
                            )
                            conn.commit()
                            logger.info(f"Cleaned {orphan_count} orphan FTS entries")
                    except Exception as orphan_err:
                        logger.warning(f"FTS orphan cleanup failed: {orphan_err}")

            logger.info(f"Table store database initialized: {self.db_path}")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    @contextmanager
    def _get_connection(self):
        """Context manager for SQLite connections with WAL mode and auto-repair.

        Uses core.sqlite_utils.get_safe_connection() for robust connection
        management (WAL mode, PRAGMA optimization, corruption detection).

        Yields:
            sqlite3.Connection with row_factory set to sqlite3.Row.
        """
        with get_safe_connection(self.db_path) as conn:
            yield conn

    def add(self, table: Dict[str, Any]) -> str:
        """Add a table record to the database (INSERT OR REPLACE).

        If the table_id already exists, the old record is replaced. FTS5
        index is synchronized by deleting the old entry and inserting the new one
        (required by FTS5 content table mode).

        Args:
            table: Table data dict with keys:
                - table_id (str, optional): Auto-generated as "table_{uuid8}" if missing.
                - document_id (str): Parent document ID.
                - filename (str, optional): Source PDF filename.
                - page (int, optional): Page number.
                - data (List[List[str]]): 2D table data.
                - source (str, optional): Extraction source.
                - confidence (float, optional): Extraction confidence, default 1.0.

        Returns:
            The table_id of the inserted/updated record.
        """
        try:
            table_id = table.get("table_id") or f"table_{uuid.uuid4().hex[:8]}"
            data = table.get("data", [])

            header = json.dumps(data[0] if data else [], ensure_ascii=False)

            content = " ".join(
                str(cell) for row in data for cell in row if cell
            )

            with self._write_lock:
                with self._get_connection() as conn:
                    old_rowid = None
                    if self.fts_available:
                        row = conn.execute(
                            "SELECT rowid FROM tables WHERE table_id = ?",
                            (table_id,)
                        ).fetchone()
                        if row:
                            old_rowid = row[0]

                    if old_rowid is not None and self.fts_available:
                        try:
                            conn.execute(
                                "DELETE FROM tables_fts WHERE rowid = ?",
                                (old_rowid,)
                            )
                        except sqlite3.DatabaseError as fts_err:
                            err_lower = str(fts_err).lower()
                            if "malformed" in err_lower or "corrupt" in err_lower or "disk image" in err_lower:
                                logger.warning(f"FTS write error, marking as corrupted for rebuild: {fts_err}")
                                self._fts_corrupted = True
                                self.fts_available = False
                            else:
                                raise

                    conn.execute("""
                        INSERT OR REPLACE INTO tables
                        (table_id, document_id, filename, page, data, header, content,
                         row_count, column_count, source, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        table_id,
                        table.get("document_id"),
                        table.get("filename", ""),
                        table.get("page", 0),
                        json.dumps(data, ensure_ascii=False),
                        header,
                        content,
                        len(data),
                        len(data[0]) if data else 0,
                        table.get("source", ""),
                        table.get("confidence", 1.0)
                    ))

                    if self.fts_available:
                        try:
                            conn.execute("""
                                INSERT INTO tables_fts(rowid, header, content)
                                SELECT rowid, header, ? FROM tables WHERE table_id = ?
                            """, (content, table_id))
                        except sqlite3.DatabaseError as fts_err:
                            err_lower = str(fts_err).lower()
                            if "malformed" in err_lower or "corrupt" in err_lower or "disk image" in err_lower:
                                logger.warning(f"FTS write error, marking as corrupted for rebuild: {fts_err}")
                                self._fts_corrupted = True
                                self.fts_available = False
                            else:
                                logger.warning(f"FTS index update failed: {fts_err}")

                    conn.commit()

            if self._fts_corrupted and not self.fts_available:
                self._try_rebuild_fts_async()

            return table_id

        except Exception as e:
            logger.error(f"Failed to add table: {e}")
            raise

    def _try_rebuild_fts_async(self):
        """Attempt to rebuild the FTS5 index after corruption detection.

        Steps:
            1. Verify main database integrity via PRAGMA integrity_check.
            2. Drop the corrupted FTS virtual table.
            3. Recreate FTS virtual table and rebuild index from main table.
            4. On any failure, keep fts_available=False (LIKE fallback).
        """
        if not self._fts_corrupted:
            return

        if self._write_lock.locked():
            logger.info("FTS rebuild deferred: write lock held, will retry on next add()")
            return

        logger.info("Starting FTS index rebuild on new connection...")
        try:
            with self._write_lock:
                try:
                    with sqlite3.connect(self.db_path) as check_conn:
                        check_conn.execute("PRAGMA integrity_check").fetchone()
                except sqlite3.DatabaseError:
                    logger.warning("Main database file corrupted, attempting repair...")
                    if not repair_corrupted_database(self.db_path):
                        logger.error("Main database repair failed, FTS rebuild aborted")
                        return

                try:
                    with self._get_connection() as conn:
                        conn.execute("DROP TABLE IF EXISTS tables_fts")
                        conn.commit()
                        logger.info("Dropped old FTS index table")
                except Exception as drop_err:
                    logger.warning(f"Failed to drop old FTS table: {drop_err}")
                    try:
                        for ext in ["", "-wal", "-shm"]:
                            f = f"{self.db_path}{ext}"
                            if os.path.exists(f):
                                os.remove(f)
                        logger.info("Deleted corrupted database files, system will rebuild")
                    except Exception:
                        pass
                    self._init_database()
                    self._fts_corrupted = False
                    return

                try:
                    with self._get_connection() as conn:
                        conn.execute("""
                            CREATE VIRTUAL TABLE IF NOT EXISTS tables_fts USING fts5(
                                header, content, content='tables', content_rowid='rowid'
                            )
                        """)
                        conn.execute("""
                            INSERT INTO tables_fts(rowid, header, content)
                            SELECT rowid, header, content FROM tables
                        """)
                        conn.commit()
                        self.fts_available = True
                        self._fts_corrupted = False
                        logger.info("FTS index rebuilt successfully")
                except Exception as rebuild_err:
                    logger.warning(f"FTS rebuild failed, using LIKE search: {rebuild_err}")
                    self.fts_available = False
                    self._fts_corrupted = False
        except Exception as e:
            logger.error(f"FTS rebuild process error: {e}")
            self.fts_available = False
            self._fts_corrupted = False

    def _clean_query(self, query: str) -> List[str]:
        """Clean query string and return keyword list for FTS5/LIKE search.

        Removes special characters that could cause FTS5 MATCH syntax errors
        (punctuation, brackets, operators), then splits by whitespace.

        Args:
            query: Raw query string, e.g. "SN74HC04的供电电压是多少？"

        Returns:
            Cleaned keyword list, e.g. ["SN74HC04", "供电电压", "是多少"].
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return []

        clean_query = self._QUERY_CLEAN_PATTERN.sub(' ', clean_query)
        return [word for word in clean_query.split() if word.strip()]

    def _search_by_content(self, query: str, top_k: int = 10,
                           document_ids: List[str] = None) -> List[TableSearchResult]:
        """Dispatch table content search to FTS5 or LIKE strategy.

        Strategy selection based on self.fts_available:
            - True  → _search_by_content_fts() (BM25 ranking)
            - False → _search_by_content_like() (LIKE matching, score=1.0)

        This is the synchronous core; the async search() method wraps it
        via asyncio.to_thread().

        Args:
            query: Search query text.
            top_k: Maximum number of results. Defaults to 10.
            document_ids: Optional document ID list to scope the search.

        Returns:
            Search results sorted by relevance.
        """
        try:
            with self._get_connection() as conn:
                if self.fts_available:
                    return self._search_by_content_fts(conn, query, top_k, document_ids)
                else:
                    return self._search_by_content_like(conn, query, top_k, document_ids)
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            return []

    async def search(self, query: str, top_k: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Async table search — unified external interface.

        Wraps the synchronous _search_by_content() via asyncio.to_thread()
        to avoid blocking the event loop.

        Args:
            query: Search query text, e.g. "SN74HC04 供电电压".
            top_k: Maximum number of results. Defaults to 10.
                   hybrid_retriever typically passes 50 for RRF fusion.
            filters: Optional filter dict supporting:
                - "document_ids": List[str] — scope to these documents.
                - "document_id": str — single document (converted to list).

        Returns:
            List of result dicts with keys: id, table_data, score,
            document_id, filename, page, metadata, title.
        """
        document_ids = None
        if filters:
            if "document_ids" in filters and isinstance(filters["document_ids"], list):
                document_ids = [str(d) for d in filters["document_ids"]]
            elif "document_id" in filters:
                document_ids = [str(filters["document_id"])]

        results = await asyncio.to_thread(
            self._search_by_content, query, top_k, document_ids
        )

        return [
            {
                "id": r.table_id,
                "table_data": r.data,
                "score": r.score,
                "document_id": r.document_id,
                "filename": r.filename,
                "page": r.page,
                "metadata": {
                    "row_count": r.row_count,
                    "column_count": r.column_count,
                    "source": r.source,
                    "confidence": r.confidence,
                },
                "title": self._build_table_title(r)
            }
            for r in results
        ]

    @staticmethod
    def _build_table_title(result: TableSearchResult) -> str:
        """Build a human-readable title from table header row.

        Joins the first row's column names with " | ", limited to 6 columns.
        Falls back to "Table p{page}" if no header data is available.

        Args:
            result: Table search result object.

        Returns:
            Readable title, e.g. "参数 | 最小值 | 典型值 | 最大值 | 单位"
            or "Table p3" as fallback.
        """
        if result.data and result.data[0]:
            header_cells = [str(c).strip() for c in result.data[0] if c]
            if header_cells:
                title = " | ".join(header_cells[:6])
                if len(header_cells) > 6:
                    title += " ..."
                return title
        return f"Table p{result.page}"

    def _search_by_content_fts(self, conn, query: str, top_k: int, document_ids: List[str] = None):
        """FTS5 full-text search with BM25 relevance ranking.

        SQLite FTS5 bm25() returns negative values (more negative = more
        relevant). Results are ordered ascending by score, then abs() is
        applied in _rows_to_results() to normalize to positive values.

        Args:
            conn: SQLite connection (passed from _search_by_content).
            query: Search query text.
            top_k: Maximum number of results.
            document_ids: Optional document ID list to scope the search.

        Returns:
            List[TableSearchResult] sorted by BM25 relevance.
        """
        words = self._clean_query(query)
        if not words:
            return []

        fts_query = " ".join(words)

        try:
            if document_ids:
                placeholders = ','.join('?' * len(document_ids))
                sql = f"""
                    SELECT t.*, bm25(tables_fts) as score
                    FROM tables t
                    JOIN tables_fts f ON t.rowid = f.rowid
                    WHERE tables_fts MATCH ?
                    AND t.document_id IN ({placeholders})
                    ORDER BY score
                    LIMIT ?
                """
                params = [fts_query] + document_ids + [top_k]
            else:
                sql = """
                    SELECT t.*, bm25(tables_fts) as score
                    FROM tables t
                    JOIN tables_fts f ON t.rowid = f.rowid
                    WHERE tables_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                """
                params = [fts_query, top_k]

            cursor = conn.execute(sql, params)
            return self._rows_to_results(cursor.fetchall())

        except Exception as e:
            logger.error(f"FTS search failed: {e}, query: '{fts_query}'")
            return []

    def _search_by_content_like(self, conn, query: str, top_k: int, document_ids: List[str] = None):
        """LIKE fallback search when FTS5 is unavailable.

        Uses SQL LIKE '%word%' for substring matching on the content column.
        All results receive score=1.0 (no relevance ranking capability).

        Key differences from FTS5 search:
            - Substring matching vs. token matching (may over-match).
            - No BM25 ranking; score is uniform 1.0.
            - Full table scan O(n) vs. inverted index O(log n).
            - AND logic: all keywords must appear in content.

        Args:
            conn: SQLite connection.
            query: Search query text.
            top_k: Maximum number of results.
            document_ids: Optional document ID list to scope the search.

        Returns:
            List[TableSearchResult] with score=1.0 for all results.
        """
        words = self._clean_query(query)
        if not words:
            return []

        like_query = " AND ".join([f"(content LIKE ?)" for _ in words])
        like_params = [f"%{word}%" for word in words]

        try:
            if document_ids:
                placeholders = ','.join('?' * len(document_ids))
                sql = f"""
                    SELECT t.*, 1.0 as score
                    FROM tables t
                    WHERE {like_query}
                    AND t.document_id IN ({placeholders})
                    LIMIT ?
                """
                params = like_params + document_ids + [top_k]
            else:
                sql = f"""
                    SELECT t.*, 1.0 as score
                    FROM tables t
                    WHERE {like_query}
                    LIMIT ?
                """
                params = like_params + [top_k]

            cursor = conn.execute(sql, params)
            return self._rows_to_results(cursor.fetchall())

        except Exception as e:
            logger.error(f"LIKE search failed: {e}, query: '{query}'")
            return []

    def _rows_to_results(self, rows) -> List[TableSearchResult]:
        """Convert SQLite Row objects to TableSearchResult domain objects.

        Handles NULL values with sensible defaults and applies abs() to
        FTS5 BM25 scores (which are negative) for consistent "higher is
        better" semantics.

        Args:
            rows: sqlite3.Row list from cursor.fetchall().

        Returns:
            List[TableSearchResult] with normalized values.
        """
        results = []
        for row in rows:
            try:
                data = json.loads(row['data']) if row['data'] else []
            except json.JSONDecodeError:
                logger.warning(f"Table data parse failed: {row['table_id']}")
                data = []

            page_value = row['page'] or 1

            results.append(TableSearchResult(
                table_id=row['table_id'],
                document_id=row['document_id'],
                filename=row['filename'] or "",
                page=page_value,
                data=data,
                row_count=row['row_count'] or 0,
                column_count=row['column_count'] or 0,
                source=row['source'] or "",
                confidence=row['confidence'] or 1.0,
                score=abs(float(row['score'])) if 'score' in row else 0.0
            ))

        return results

    def get_tables_by_documents(self, document_ids: List[str], limit: int = 100,
                               min_confidence: float = 0.5) -> List[TableSearchResult]:
        """Retrieve all tables for specified documents (exact query, not search).

        Unlike search(), this performs no full-text matching. It queries all
        tables by document ID, filtered by minimum confidence, sorted by page.

        Primarily used by parameter_extractor.py Stage 1 for structured
        table lookup.

        Args:
            document_ids: Document ID list, e.g. ["doc_abc123", "doc_def456"].
            limit: Maximum number of results. Defaults to 100.
            min_confidence: Minimum confidence threshold [0, 1]. Defaults to 0.5.
                           Tables below this threshold (e.g. mechanical drawings)
                           are excluded.

        Returns:
            List[TableSearchResult] sorted by page number.
        """
        if not document_ids:
            return []

        try:
            with self._get_connection() as conn:
                placeholders = ','.join('?' * len(document_ids))
                sql = f"""
                    SELECT * FROM tables
                    WHERE document_id IN ({placeholders})
                    AND confidence >= ?
                    ORDER BY page
                    LIMIT ?
                """
                params = document_ids + [min_confidence, limit]
                cursor = conn.execute(sql, params)
                return self._rows_to_results(cursor.fetchall())
        except Exception as e:
            logger.error(f"Failed to get tables by documents: {e}")
            return []

    def delete_by_document(self, document_id: str) -> int:
        """Delete all table records for a given document.

        Also cleans up orphaned FTS index entries. Called by
        core/cleanup_manager.py during document cleanup.

        Args:
            document_id: Document ID whose tables should be deleted.

        Returns:
            Number of deleted records. 0 if no matching records or on error.
        """
        try:
            with self._write_lock:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "DELETE FROM tables WHERE document_id = ?",
                        (document_id,)
                    )
                    conn.commit()
                    deleted_count = cursor.rowcount

                    if deleted_count > 0 and self.fts_available:
                        try:
                            conn.execute(
                                "DELETE FROM tables_fts WHERE rowid NOT IN (SELECT rowid FROM tables)"
                            )
                            conn.commit()
                        except sqlite3.DatabaseError as fts_err:
                            err_lower = str(fts_err).lower()
                            if "malformed" in err_lower or "corrupt" in err_lower or "disk image" in err_lower:
                                logger.warning(f"FTS index error during delete, marking as corrupted: {fts_err}")
                                self._fts_corrupted = True
                                self.fts_available = False
                            else:
                                logger.warning(f"FTS cleanup failed: {fts_err}")

                    return deleted_count
        except sqlite3.DatabaseError as db_err:
            err_lower = str(db_err).lower()
            if "malformed" in err_lower or "corrupt" in err_lower or "disk image" in err_lower:
                logger.warning(f"Database corruption during delete, marking FTS for rebuild: {db_err}")
                self._fts_corrupted = True
                self.fts_available = False
                self._try_rebuild_fts_async()
                return 0
            logger.error(f"Failed to delete document tables: {db_err}")
            return 0
        except Exception as e:
            logger.error(f"Failed to delete document tables: {e}")
            return 0
