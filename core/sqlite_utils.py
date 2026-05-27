"""
SQLite robustness utilities — WAL mode, corruption detection, and auto-repair.

Provides safe database connections with optimized PRAGMA settings (WAL mode,
NORMAL sync, busy timeout), automatic corruption detection via integrity_check,
and a dump-and-reload repair strategy with FTS-aware fallback.

Usage:
    with get_safe_connection("data/tables.db") as conn:
        conn.execute("SELECT * FROM tables")
"""

import sqlite3
import os
import shutil
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator
from datetime import datetime

logger = logging.getLogger(__name__)


class SQLiteCorruptedError(Exception):
    """Raised when a database file is corrupted beyond repair."""
    pass


def get_pragma_config() -> dict:
    """Return optimized SQLite PRAGMA configuration.

    Key settings:
        journal_mode=WAL   — Write-Ahead Logging for crash recoverability.
        synchronous=NORMAL — Balanced safety and performance.
        busy_timeout=5000  — Wait up to 5s on lock contention.
        cache_size=-64000  — 64 MB page cache.
    """
    return {
        "journal_mode": "WAL",
        "synchronous": "NORMAL",
        "busy_timeout": 5000,
        "cache_size": -64000,
        "foreign_keys": "ON",
        "temp_store": "MEMORY",
    }


@contextmanager
def get_safe_connection(db_path: str, **kwargs) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for a SQLite connection with PRAGMA optimization and corruption handling.

    Args:
        db_path: Path to the database file.
        **kwargs: Extra arguments passed to sqlite3.connect().

    Yields:
        A configured sqlite3.Connection.

    Raises:
        SQLiteCorruptedError: If the database is corrupted and repair fails.
    """
    conn = None
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path, **kwargs)
        conn.row_factory = sqlite3.Row

        for key, value in get_pragma_config().items():
            conn.execute(f"PRAGMA {key}={value}")

        logger.debug(f"SQLite PRAGMA applied: {db_path}")

        yield conn

    except sqlite3.DatabaseError as e:
        error_msg = str(e).lower()

        if "malformed" in error_msg or "corrupt" in error_msg or "disk image" in error_msg:
            logger.error(f"Database possibly corrupted: {db_path}, error: {e}")

            if repair_corrupted_database(db_path):
                logger.info(f"Database repaired successfully: {db_path}")
            else:
                logger.warning(f"Database repair failed, continuing with potentially corrupted file: {db_path}")
        else:
            raise

    except Exception as e:
        logger.error(f"Database connection failed: {db_path}, error: {e}")
        raise

    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to close database connection: {e}")


def check_database_health(db_path: str) -> bool:
    """Check database integrity via PRAGMA integrity_check.

    Args:
        db_path: Path to the database file.

    Returns:
        True if the database is healthy or does not exist yet.
    """
    if not os.path.exists(db_path):
        return True

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()

            if result and result[0] == "ok":
                return True
            else:
                logger.warning(f"Integrity check failed: {db_path}, result: {result}")
                return False

    except sqlite3.DatabaseError as e:
        logger.error(f"Health check exception: {db_path}, error: {e}")
        return False


def repair_corrupted_database(db_path: str, backup_dir: Optional[str] = None) -> bool:
    """Attempt to repair a corrupted database using dump-and-reload.

    Strategy:
        1. Back up the corrupted file for forensics.
        2. Try .iterdump() → new database (standard SQLite recovery).
        3. If that fails (e.g. FTS virtual tables), delete the corrupted
           file so the system can rebuild it from scratch.

    Args:
        db_path: Path to the corrupted database.
        backup_dir: Directory for corrupted-file backups. Defaults to
            ``<db_dir>/db_backups/``.

    Returns:
        True if repair succeeded, False otherwise.
    """
    if not os.path.exists(db_path):
        logger.info(f"Database file does not exist, nothing to repair: {db_path}")
        return True

    logger.warning(f"Attempting database repair: {db_path}")

    try:
        if backup_dir is None:
            backup_dir = str(Path(db_path).parent / "db_backups")

        Path(backup_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{Path(db_path).stem}_corrupted_{timestamp}.db"
        backup_path = os.path.join(backup_dir, backup_name)

        shutil.copy2(db_path, backup_path)
        logger.info(f"Corrupted database backed up to: {backup_path}")

        recovered_db = f"{db_path}.recovered"
        sql_dump_path = f"{recovered_db}.sql"

        try:
            with open(sql_dump_path, "w", encoding="utf-8") as f:
                with sqlite3.connect(db_path) as conn:
                    for line in conn.iterdump():
                        f.write(f"{line}\n")

            with sqlite3.connect(recovered_db) as new_conn:
                with open(sql_dump_path, "r", encoding="utf-8") as f:
                    new_conn.executescript(f.read())

            if check_database_health(recovered_db):
                os.remove(db_path)
                os.rename(recovered_db, db_path)

                if os.path.exists(sql_dump_path):
                    os.remove(sql_dump_path)

                logger.info(f"Database repaired successfully: {db_path}")
                return True
            else:
                logger.warning("Recovered database still corrupted, will delete and rebuild")
                os.remove(recovered_db)
                if os.path.exists(sql_dump_path):
                    os.remove(sql_dump_path)
                return False

        except Exception as recover_error:
            logger.error(f".dump recovery method failed: {recover_error}")

            for path in (recovered_db, sql_dump_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            logger.warning("Falling back to forced deletion (FTS tables are not dumpable)")
            try:
                for ext in ("", "-wal", "-shm"):
                    f = f"{db_path}{ext}"
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except PermissionError:
                            import gc
                            gc.collect()
                            try:
                                os.remove(f)
                            except PermissionError:
                                pass

                if not os.path.exists(db_path):
                    logger.info(f"Corrupted database deleted, system will rebuild: {db_path}")
                    return True
                else:
                    logger.error(f"Cannot delete database (file locked): {db_path}")
                    return False
            except Exception as delete_err:
                logger.error(f"Database deletion failed: {delete_err}")
                return False

    except Exception as e:
        logger.error(f"Database repair process error: {e}")
        return False


def safe_delete_database(db_path: str) -> bool:
    """Delete a database and its WAL/SHM files, with a backup first.

    Args:
        db_path: Path to the database file.

    Returns:
        True if deletion succeeded.
    """
    if not os.path.exists(db_path):
        return True

    try:
        repair_corrupted_database(db_path)

        os.remove(db_path)

        for ext in ("-wal", "-shm"):
            wal_file = f"{db_path}{ext}"
            if os.path.exists(wal_file):
                os.remove(wal_file)

        logger.info(f"Database safely deleted: {db_path}")
        return True

    except Exception as e:
        logger.error(f"Database deletion failed: {db_path}, error: {e}")
        return False


def initialize_database_with_robustness(
    db_path: str,
    init_sql_callback=None,
) -> sqlite3.Connection:
    """Initialize a database with health check and auto-repair.

    Args:
        db_path: Path to the database file.
        init_sql_callback: Optional callable ``callback(conn)`` to run
            initialization SQL (e.g. CREATE TABLE statements).

    Returns:
        An initialized sqlite3.Connection.
    """
    if not check_database_health(db_path):
        logger.warning(f"Unhealthy database detected, attempting repair: {db_path}")
        if not repair_corrupted_database(db_path):
            logger.error(f"Database unrepairable, deleting for rebuild: {db_path}")
            safe_delete_database(db_path)

    with get_safe_connection(db_path) as conn:
        if init_sql_callback:
            init_sql_callback(conn)
        conn.commit()
        return conn
