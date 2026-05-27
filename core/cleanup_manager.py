"""
Orphan data cleanup and document lifecycle management.

Detects and cleans orphan data (file deleted but index/cache/metadata remains),
maintaining data consistency across multiple storage backends including ChromaDB,
BM25, SQLite, and disk.

Cleanup targets:
  1. Orphan vectors — ChromaDB embeddings for deleted documents
  2. Orphan BM25 entries — keyword index entries for deleted documents
  3. Orphan table data — SQLite parameter tables for deleted documents
  4. Orphan image cache — rendered PNG pages in data/images/
  5. Orphan circuit images — circuit diagram PNGs in data/circuit_images/
  6. Orphan metadata — invalid records in documents_db.json

Usage:
  - Startup background cleanup: create_cleanup_manager() -> cleanup_orphan_data()
  - Document deletion: cleanup_document_indexes() + cleanup_document_files()
  - Duplicate detection: find_duplicate_documents()
"""

import logging
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CleanupStats:
    """Statistics for a cleanup operation.

    Attributes:
        orphan_documents: Number of orphan documents detected.
        cleaned_vectors: Number of vectors removed from ChromaDB.
        cleaned_bm25: Number of BM25 entries removed.
        cleaned_tables: Number of parameter tables removed.
        cleaned_images: Number of cached images removed from disk.
        cleaned_metadata: Number of metadata records removed from documents_db.json.
        errors: List of error messages from failed cleanup steps.
        duration_seconds: Total cleanup duration in seconds.
    """

    orphan_documents: int = 0
    cleaned_vectors: int = 0
    cleaned_bm25: int = 0
    cleaned_tables: int = 0
    cleaned_images: int = 0
    cleaned_metadata: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to a dict for JSON serialization."""
        return {
            "orphan_documents": self.orphan_documents,
            "cleaned_vectors": self.cleaned_vectors,
            "cleaned_bm25": self.cleaned_bm25,
            "cleaned_tables": self.cleaned_tables,
            "cleaned_images": self.cleaned_images,
            "cleaned_metadata": self.cleaned_metadata,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 2),
        }

    def __str__(self) -> str:
        total_cleaned = (
            self.cleaned_vectors + self.cleaned_bm25
            + self.cleaned_tables + self.cleaned_images + self.cleaned_metadata
        )
        return (
            f"Cleanup done: orphans={self.orphan_documents}, "
            f"vectors={self.cleaned_vectors}, bm25={self.cleaned_bm25}, "
            f"tables={self.cleaned_tables}, images={self.cleaned_images}, "
            f"metadata={self.cleaned_metadata}, total={total_cleaned}, "
            f"errors={len(self.errors)}, duration={self.duration_seconds:.2f}s"
        )


@dataclass
class CleanupConfig:
    """Configuration for cleanup timeouts, caching, and concurrency.

    Timeout hierarchy: vector(60s) < document(120s) < total(600s).

    Attributes:
        vector_cleanup_timeout: Timeout for a single vector store cleanup.
        document_cleanup_timeout: Timeout for cleaning one document's data.
        total_cleanup_timeout: Timeout for the entire cleanup task.
        cache_ttl_seconds: TTL for cached document metadata.
        enable_cache: Whether to enable metadata caching.
        max_concurrent_cleanups: Max concurrent document cleanups (Semaphore).
    """

    vector_cleanup_timeout: float = 60.0
    document_cleanup_timeout: float = 120.0
    total_cleanup_timeout: float = 600.0
    cache_ttl_seconds: int = 60
    enable_cache: bool = True
    max_concurrent_cleanups: int = 3


class CleanupManager:
    """Core engine for orphan data detection and cleanup.

    Cleanup order per document: indexes -> files -> metadata.
    This ensures each step can correctly reference the document before
    the associated data is removed.

    Thread safety:
        - _cache_lock: asyncio.Lock for concurrent metadata loading.
        - _semaphore: asyncio.Semaphore for concurrent cleanup limit.
    """

    def __init__(self, settings=None, config: Optional[CleanupConfig] = None):
        self.settings = settings or get_settings()
        self.config = config or CleanupConfig()
        self.documents_db: Dict[str, Dict] = {}
        self._cache_timestamp: Optional[float] = None
        self._cache_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_cleanups)

    def _parse_documents_data(self, raw_data: Dict) -> Dict[str, Dict]:
        """Parse raw document data, converting relative paths to absolute paths.

        Args:
            raw_data: Raw JSON data from documents_db.json.

        Returns:
            Parsed document dict with absolute filepaths, or empty dict
            if raw_data is not a dict.
        """
        if not isinstance(raw_data, dict):
            return {}

        base_dir = getattr(self.settings, "BASE_DIR", Path("."))
        parsed_data = {}

        for doc_id, doc in raw_data.items():
            parsed_doc = doc.copy()
            filepath = parsed_doc.get("filepath")
            if filepath:
                path_obj = Path(filepath)
                if not path_obj.is_absolute():
                    parsed_doc["filepath"] = str(base_dir / path_obj)
            parsed_data[doc_id] = parsed_doc

        return parsed_data

    async def _load_documents_db_async(self, force_reload: bool = False):
        """Load document metadata from documents_db.json with caching.

        Args:
            force_reload: If True, bypass cache and reload from disk.
        """
        async with self._cache_lock:
            if (
                not force_reload
                and self.config.enable_cache
                and self._cache_timestamp is not None
            ):
                elapsed = time.time() - self._cache_timestamp
                if elapsed < self.config.cache_ttl_seconds:
                    logger.debug(f"Using cached document db (loaded {elapsed:.1f}s ago)")
                    return

            try:
                meta_path = self.settings.DATA_DIR / "documents_db.json"
                if meta_path.exists():
                    raw = json.loads(meta_path.read_text(encoding="utf-8"))
                    self.documents_db = self._parse_documents_data(raw)
                    self._cache_timestamp = time.time()
                    logger.info(f"Loaded {len(self.documents_db)} document metadata entries")
                else:
                    self.documents_db = {}
            except Exception as e:
                logger.error(f"Failed to load document metadata: {e}")
                self.documents_db = {}

    def _identify_orphan_documents(self) -> List[str]:
        """Identify documents whose files no longer exist on disk.

        Returns:
            List of orphan document IDs.
        """
        orphan_docs = []

        for doc_id, doc in self.documents_db.items():
            filepath = Path(doc.get("filepath", ""))
            if not filepath.exists():
                orphan_docs.append(doc_id)
                logger.info(f"Orphan document found: {doc_id} (file missing: {filepath})")

        return orphan_docs

    async def _cleanup_vector_store(self, doc_id: str, vector_store) -> int:
        """Clean up vector store data for a document with timeout control.

        Args:
            doc_id: Document ID.
            vector_store: ChromaDB vector store instance.

        Returns:
            Number of vectors removed.
        """
        if vector_store is None:
            return 0

        try:
            def _sync_vector_delete():
                try:
                    result = vector_store.collection.get(where={"document_id": doc_id})
                except TypeError as e:
                    if "has no len()" in str(e):
                        all_data = vector_store.collection.get(include=["metadatas"])
                        matched_ids = []
                        for i, id_ in enumerate(all_data["ids"]):
                            meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
                            if meta.get("document_id") == doc_id:
                                matched_ids.append(id_)
                        result = {"ids": matched_ids}
                    else:
                        raise
                ids_to_delete = result.get("ids", [])
                count = len(ids_to_delete)
                if count > 0:
                    vector_store.collection.delete(ids=ids_to_delete)
                return count

            count = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, _sync_vector_delete),
                timeout=self.config.vector_cleanup_timeout,
            )
            logger.info(f"Cleaned vector store: {doc_id}, removed {count} vectors")
            return count
        except asyncio.TimeoutError:
            logger.error(f"Vector store cleanup timed out: {doc_id}")
            return 0
        except Exception as e:
            logger.error(f"Failed to clean vector store: {e}")
            return 0

    def _cleanup_bm25_store(self, doc_id: str, bm25_store) -> int:
        """Clean up BM25 index data for a document.

        Args:
            doc_id: Document ID.
            bm25_store: BM25 store instance.

        Returns:
            Number of BM25 entries removed.
        """
        if bm25_store is None:
            return 0

        try:
            total_entries = len(bm25_store.doc_ids) if hasattr(bm25_store, "doc_ids") else -1
            count = bm25_store.delete_by_doc_ids([doc_id])
            if count > 0:
                logger.info(f"Cleaned BM25 store: {doc_id}, removed {count} entries (total was {total_entries})")
            else:
                logger.warning(f"Cleaned BM25 store: {doc_id}, no matching entries (total: {total_entries})")
            return count
        except Exception as e:
            logger.error(f"Failed to clean BM25 store: {doc_id}, error: {e}", exc_info=True)
            return 0

    def _cleanup_table_store(self, doc_id: str, table_store) -> int:
        """Clean up table store data for a document.

        Args:
            doc_id: Document ID.
            table_store: SQLite table store instance.

        Returns:
            Number of tables removed.
        """
        if table_store is None:
            return 0

        try:
            count = table_store.delete_by_document(doc_id)
            logger.info(f"Cleaned table store: {doc_id}, removed {count} tables")
            return count
        except Exception as e:
            logger.error(f"Failed to clean table store: {e}")
            return 0

    def _cleanup_image_cache(self, doc_id: str) -> int:
        """Clean up rendered page images for a document.

        Args:
            doc_id: Document ID.

        Returns:
            Number of images removed.
        """
        try:
            images_dir = self.settings.IMAGE_DIR
            if not images_dir.exists():
                return 0

            count = 0
            for image_file in images_dir.glob(f"{doc_id}_page_*.png"):
                image_file.unlink(missing_ok=True)
                count += 1
                logger.info(f"Cleaned image cache: {doc_id}, deleted {image_file.name}")

            return count
        except Exception as e:
            logger.error(f"Failed to clean image cache: {e}")
            return 0

    def _cleanup_circuit_images(self, doc_id: str) -> int:
        """Clean up circuit diagram images for a document.

        Args:
            doc_id: Document ID.

        Returns:
            Number of circuit images removed.
        """
        try:
            circuit_dir = self.settings.DATA_DIR / "circuit_images" / doc_id
            if not circuit_dir.exists():
                return 0

            count = 0
            for image_file in circuit_dir.glob("*.png"):
                image_file.unlink(missing_ok=True)
                count += 1
                logger.info(f"Cleaned circuit image: {doc_id}, deleted {image_file.name}")

            if count > 0:
                try:
                    circuit_dir.rmdir()
                    logger.info(f"Removed circuit image directory: {circuit_dir}")
                except OSError:
                    pass

            return count
        except Exception as e:
            logger.error(f"Failed to clean circuit images: {e}")
            return 0

    async def _do_cleanup_document(self, doc_id: str, container, stats: CleanupStats):
        """Execute the full cleanup pipeline for a single document.

        Cleanup order: vectors -> BM25 -> tables -> images -> circuit images
                       -> visual index -> metadata.

        Args:
            doc_id: Document ID to clean up.
            container: Service container providing storage backend instances.
            stats: CleanupStats to update during cleanup.
        """
        if container is None:
            try:
                from api.dependencies import get_service_container
                container = get_service_container()
            except ImportError:
                logger.warning("Cannot import service container, skipping store cleanup")
                container = None

        if container and hasattr(container, "vector_store"):
            vector_store = getattr(container, "vector_store", None)
            if vector_store:
                stats.cleaned_vectors = await self._cleanup_vector_store(doc_id, vector_store)

        if container and hasattr(container, "bm25_store"):
            bm25_store = getattr(container, "bm25_store", None)
            if bm25_store:
                stats.cleaned_bm25 = self._cleanup_bm25_store(doc_id, bm25_store)

        if container and hasattr(container, "table_store"):
            table_store = getattr(container, "table_store", None)
            if table_store:
                stats.cleaned_tables = self._cleanup_table_store(doc_id, table_store)

        stats.cleaned_images += self._cleanup_image_cache(doc_id)
        stats.cleaned_images += self._cleanup_circuit_images(doc_id)

        if container and hasattr(container, "visual_indexer"):
            visual_indexer = getattr(container, "visual_indexer", None)
            if visual_indexer:
                try:
                    removed_count = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None, visual_indexer.remove_document_index, doc_id
                        ),
                        timeout=10.0,
                    )
                    if removed_count > 0:
                        logger.info(f"Cleaned visual index: {removed_count} entries (doc={doc_id})")
                        try:
                            await asyncio.wait_for(
                                asyncio.get_running_loop().run_in_executor(
                                    None, visual_indexer.flush_index_to_disk
                                ),
                                timeout=30.0,
                            )
                        except (asyncio.TimeoutError, Exception) as flush_err:
                            logger.warning(f"Visual index flush failed: {flush_err}")
                except asyncio.TimeoutError:
                    logger.warning(f"Visual index cleanup timed out (10s): {doc_id}")
                except Exception as e:
                    logger.warning(f"Visual index cleanup failed: {e}")

        try:
            from api.routers.documents import get_documents_db
            db = get_documents_db()
            if await db.exists(doc_id):
                await db.delete(doc_id)
                stats.cleaned_metadata = 1
                logger.info(f"Cleaned document metadata: {doc_id}")
        except Exception as e:
            logger.warning(f"Failed to clean document metadata: {e}")

    async def cleanup_orphan_document(self, doc_id: str, container=None) -> CleanupStats:
        """Clean up all data for a single orphan document with concurrency and timeout control.

        Args:
            doc_id: Orphan document ID.
            container: Optional service container.

        Returns:
            CleanupStats with detailed cleanup results.
        """
        async with self._semaphore:
            start_time = time.time()
            stats = CleanupStats()
            stats.orphan_documents = 1

            logger.info(f"Starting orphan document cleanup: {doc_id}")

            try:
                await asyncio.wait_for(
                    self._do_cleanup_document(doc_id, container, stats),
                    timeout=self.config.document_cleanup_timeout,
                )
                stats.duration_seconds = time.time() - start_time
                logger.info(f"Orphan document cleanup done: {doc_id}, took {stats.duration_seconds:.2f}s")
            except asyncio.TimeoutError:
                error_msg = f"Orphan document cleanup timed out: {doc_id}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
            except Exception as e:
                error_msg = f"Orphan document cleanup failed: {doc_id}, error: {e}"
                logger.error(error_msg, exc_info=True)
                stats.errors.append(error_msg)

            return stats

    async def cleanup_orphan_data(self, container=None) -> CleanupStats:
        """Clean up all orphan data with total timeout control.

        Main entry point for system startup cleanup. Scans all documents,
        identifies orphans, and cleans them concurrently.

        Args:
            container: Optional service container.

        Returns:
            Aggregated CleanupStats for all orphan documents.
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("Starting orphan data cleanup")
        logger.info("=" * 60)

        total_stats = CleanupStats()

        async def _do_cleanup():
            await self._load_documents_db_async(force_reload=True)

            orphan_docs = self._identify_orphan_documents()
            total_stats.orphan_documents = len(orphan_docs)

            if not orphan_docs:
                logger.info("No orphan documents found")
                return

            logger.info(f"Found {len(orphan_docs)} orphan documents, starting cleanup")

            tasks = [
                self.cleanup_orphan_document(doc_id, container)
                for doc_id in orphan_docs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    error_msg = f"Cleanup task exception: {result}"
                    logger.error(error_msg)
                    total_stats.errors.append(error_msg)
                else:
                    total_stats.cleaned_vectors += result.cleaned_vectors
                    total_stats.cleaned_bm25 += result.cleaned_bm25
                    total_stats.cleaned_tables += result.cleaned_tables
                    total_stats.cleaned_images += result.cleaned_images
                    total_stats.cleaned_metadata += result.cleaned_metadata
                    total_stats.errors.extend(result.errors)

            logger.info("=" * 60)
            logger.info(f"Orphan data cleanup complete: {total_stats}")
            logger.info("=" * 60)

        try:
            await asyncio.wait_for(_do_cleanup(), timeout=self.config.total_cleanup_timeout)
        except asyncio.TimeoutError:
            error_msg = "Total cleanup task timed out"
            logger.error(error_msg)
            total_stats.errors.append(error_msg)
            total_stats.duration_seconds = time.time() - start_time
        except Exception as e:
            error_msg = f"Orphan data cleanup failed: {e}"
            logger.error(error_msg, exc_info=True)
            total_stats.errors.append(error_msg)
            total_stats.duration_seconds = time.time() - start_time

        return total_stats

    async def cleanup_document_indexes(
        self,
        document_id: str,
        container,
        include_logs: bool = True,
    ) -> Dict[str, int]:
        """Clean up index data (vectors/BM25/tables/visual) for a document.

        This is the first step of the two-step document deletion process.
        Use cleanup_document_files() for the second step.

        Args:
            document_id: Document ID.
            container: Service container with storage backend instances.
            include_logs: If True, log detailed info; otherwise only warnings.

        Returns:
            Dict with cleanup counts for each index type.
        """
        cleanup = {
            "vector": 0,
            "bm25": 0,
            "tables": 0,
            "visual_index": 0,
        }

        try:
            vector_store = getattr(container, "vector_store", None)
            if vector_store:
                cleanup["vector"] = await asyncio.wait_for(
                    self._cleanup_vector_store(document_id, vector_store),
                    timeout=15.0,
                )
        except asyncio.TimeoutError:
            if include_logs:
                logger.warning(f"Vector index cleanup timed out (15s), skipping: {document_id}")
        except Exception as e:
            if include_logs:
                logger.warning(f"Vector index cleanup failed: {e}")

        try:
            bm25_store = getattr(container, "bm25_store", None)
            if bm25_store:
                cleanup["bm25"] = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None, self._cleanup_bm25_store, document_id, bm25_store
                    ),
                    timeout=30.0,
                )
        except asyncio.TimeoutError:
            if include_logs:
                logger.warning(f"BM25 index cleanup timed out (30s), skipping: {document_id}")
        except Exception as e:
            if include_logs:
                logger.warning(f"BM25 index cleanup failed: {e}")

        try:
            table_store = getattr(container, "table_store", None)
            if table_store:
                cleanup["tables"] = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None, self._cleanup_table_store, document_id, table_store
                    ),
                    timeout=15.0,
                )
        except asyncio.TimeoutError:
            if include_logs:
                logger.warning(f"Table index cleanup timed out (15s), skipping: {document_id}")
        except Exception as e:
            if include_logs:
                logger.warning(f"Table index cleanup failed: {e}")

        try:
            visual_indexer = getattr(container, "visual_indexer", None)
            if visual_indexer:
                cleanup["visual_index"] = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None, visual_indexer.remove_document_index, document_id
                    ),
                    timeout=10.0,
                )
                if include_logs:
                    logger.info(f"Cleaned visual index: {cleanup['visual_index']} entries")
                if cleanup["visual_index"] > 0:
                    try:
                        await asyncio.wait_for(
                            asyncio.get_running_loop().run_in_executor(
                                None, visual_indexer.flush_index_to_disk
                            ),
                            timeout=30.0,
                        )
                        if include_logs:
                            logger.info(f"Visual index flushed: {document_id}")
                    except asyncio.TimeoutError:
                        if include_logs:
                            logger.warning(f"Visual index flush timed out (30s): {document_id}")
                    except Exception as flush_err:
                        if include_logs:
                            logger.warning(f"Visual index flush failed: {flush_err}")
        except asyncio.TimeoutError:
            if include_logs:
                logger.warning(f"Visual index cleanup timed out (10s), skipping: {document_id}")
        except Exception as e:
            if include_logs:
                logger.warning(f"Visual index cleanup failed: {e}")

        return cleanup

    def cleanup_document_files(
        self,
        document_id: str,
        doc: Dict,
        include_logs: bool = True,
    ) -> Dict[str, int]:
        """Clean up file data (PDF + image cache + circuit images) for a document.

        This is the second step of the two-step document deletion process.

        Args:
            document_id: Document ID.
            doc: Document data dict containing at least "filepath".
            include_logs: Whether to log detailed info.

        Returns:
            Dict with cleanup counts for each file type.
        """
        cleanup = {
            "images": 0,
            "circuit_images": 0,
        }

        try:
            filepath = Path(doc.get("filepath", ""))
            if filepath.exists():
                filepath.unlink()
                if include_logs:
                    logger.info(f"Deleted file: {filepath}")
        except Exception as e:
            if include_logs:
                logger.warning(f"Failed to delete file: {e}")

        cleanup["images"] = self._cleanup_image_cache(document_id)
        if include_logs and cleanup["images"] > 0:
            logger.info(f"Cleaned image cache: {cleanup['images']} files")

        circuit_count = self._cleanup_circuit_images(document_id)
        cleanup["circuit_images"] = circuit_count
        if include_logs and circuit_count > 0:
            logger.info(f"Cleaned circuit images: {circuit_count} files")

        return cleanup

    async def find_duplicate_documents(self, filename: str) -> List[str]:
        """Find documents with the same filename.

        Args:
            filename: Filename to search for (without path).

        Returns:
            List of doc_ids with matching filename.
        """
        await self._load_documents_db_async()

        duplicates = []
        for doc_id, doc in self.documents_db.items():
            if doc.get("filename") == filename:
                duplicates.append(doc_id)

        return duplicates

    def cleanup_stale_backups(self, max_backups: int = 5) -> int:
        """Remove old database backups, keeping only the most recent ones.

        Args:
            max_backups: Maximum number of backups to retain.

        Returns:
            Number of backups removed.
        """
        backup_dir = Path("data/db_backups")
        if not backup_dir.exists():
            return 0

        backups = sorted(
            backup_dir.glob("*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        removed = 0
        for old_backup in backups[max_backups:]:
            try:
                old_backup.unlink()
                removed += 1
            except Exception:
                pass
        if removed > 0:
            logger.info(f"Cleaned stale backups: {removed} (kept latest {max_backups})")
        return removed

    async def scan_orphan_uploads(self) -> List[str]:
        """Scan uploads directory for files not tracked in documents_db.json.

        Returns:
            List of paths to orphan upload files.
        """
        upload_dir = Path("data/uploads")
        if not upload_dir.exists():
            return []

        await self._load_documents_db_async()
        valid_ids = set(self.documents_db.keys())

        orphans = []
        for f in upload_dir.glob("*_*"):
            if f.suffix.lower() == ".pdf":
                doc_id = f.stem.split("_", 1)[0] if "_" in f.stem else ""
                if doc_id and doc_id not in valid_ids:
                    orphans.append(str(f))

        if orphans:
            logger.warning(f"Found {len(orphans)} orphan upload files: {[Path(o).name for o in orphans]}")
            for f in orphans:
                try:
                    Path(f).unlink()
                    logger.info(f"Deleted orphan upload: {Path(f).name}")
                except Exception as e:
                    logger.warning(f"Failed to delete orphan upload: {e}")

        return orphans


def create_cleanup_manager(
    settings=None, config: Optional[CleanupConfig] = None
) -> CleanupManager:
    """Factory function to create a CleanupManager instance.

    Args:
        settings: Global settings object. Uses get_settings() if None.
        config: Cleanup configuration. Uses CleanupConfig() defaults if None.

    Returns:
        A new CleanupManager instance.
    """
    return CleanupManager(settings, config)
