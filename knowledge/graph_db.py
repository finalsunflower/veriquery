"""
Knowledge Graph Database Initialization and Management.

This module defines the SQLite schema and lifecycle for the VeriQuery knowledge
graph. It is the "schema layer" in the knowledge module pipeline:

    graph_db.py (schema) → chip_importer.py (data) → graph_query.py (queries)

The database contains three core tables:
  - chips: basic info (chip_id PK, name, family, manufacturer, supply_voltage, ...)
  - pins: pin details (pin_id PK, chip_id FK→chips, pin_number, pin_name, ...)
  - parameters: electrical specs (param_id PK, chip_id FK→chips, pin_id FK→pins, ...)

Table relationships: chips(1)→(N)pins, chips(1)→(N)parameters, pins(1)→(N)parameters.

SQLite is chosen over Neo4j/MySQL for its zero-dependency, single-file deployment,
and sufficient performance for the dataset size (dozens of chips).
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict
from core.sqlite_utils import get_safe_connection, check_database_health, repair_corrupted_database

logger = logging.getLogger(__name__)


class KnowledgeGraphDB:
    """SQLite knowledge graph database manager with singleton pattern.

    Uses __new__-based singleton to ensure only one instance per db_path.
    The _initialized flag prevents __init__ from re-running on subsequent
    calls to KnowledgeGraphDB(db_path).

    Database connections are short-lived: each operation acquires a fresh
    connection via _get_connection() and closes it with a ``with`` block,
    avoiding lock contention in SQLite's single-writer model.
    """

    _instances: Dict[str, 'KnowledgeGraphDB'] = {}
    _default_db_path: str = "./data/knowledge_graph.db"

    def __new__(cls, db_path: str = None):
        """Return the singleton instance for the given db_path.

        Creates a new instance on first access; returns the cached one
        thereafter. Sets ``_initialized = False`` on newly created instances
        so that ``__init__`` can perform first-time setup.
        """
        if db_path is None:
            db_path = cls._default_db_path
        if db_path not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[db_path] = instance
            instance._initialized = False
        return cls._instances[db_path]

    def __init__(self, db_path: str = None):
        """Initialize the database (create tables and indexes) once.

        Guarded by ``_initialized`` to prevent re-execution when the
        singleton is returned by ``__new__``.
        """
        if self._initialized:
            return

        self.db_path = db_path if db_path is not None else self._default_db_path
        self._ensure_db_directory()
        self._init_database()
        self._initialized = True
    
    def _ensure_db_directory(self):
        """Create the parent directory of the database file if it does not exist."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"数据库目录: {db_dir}")

    def _init_database(self):
        """Create the chips, pins, and parameters tables and their indexes."""
        logger.info(f"初始化知识图谱数据库: {self.db_path}")

        if not check_database_health(self.db_path):
            logger.warning(f"⚠️ 检测到知识图谱数据库可能损坏: {self.db_path}")
            if repair_corrupted_database(self.db_path):
                logger.info(f"✅ 知识图谱数据库已修复: {self.db_path}")
            else:
                logger.error(f"❌ 知识图谱数据库无法修复，将删除重建: {self.db_path}")
                from core.sqlite_utils import safe_delete_database
                safe_delete_database(self.db_path)

        with self._get_connection() as conn:
            self._create_chips_table(conn)
            self._create_pins_table(conn)
            self._create_parameters_table(conn)
            conn.commit()
            logger.info("知识图谱数据库表结构初始化完成")
    
    def _get_connection(self):
        """Return a configured SQLite connection with WAL mode and PRAGMA tuning."""
        conn = None
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            pragma_config = {
                "journal_mode": "WAL",
                "synchronous": "NORMAL",
                "busy_timeout": 5000,
                "cache_size": -64000,
                "foreign_keys": "ON",
            }
            for key, value in pragma_config.items():
                conn.execute(f"PRAGMA {key}={value}")

            return conn
        except Exception as e:
            if conn:
                conn.close()
            raise e
    
    def _create_chips_table(self, conn: sqlite3.Connection):
        """Create the chips table and its indexes."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chips (
                chip_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                full_name TEXT,
                family TEXT,
                manufacturer TEXT,
                supply_voltage REAL,
                package TEXT,
                process TEXT,
                pin_count INTEGER,
                description TEXT,
                datasheet_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_chips_name ON chips(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chips_family ON chips(family)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chips_manufacturer ON chips(manufacturer)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chips_process ON chips(process)")
        logger.debug("芯片表创建完成")
    
    def _create_pins_table(self, conn: sqlite3.Connection):
        """Create the pins table and its indexes."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pins (
                pin_id TEXT PRIMARY KEY,
                chip_id TEXT NOT NULL,
                pin_number INTEGER,
                pin_name TEXT NOT NULL,
                function_type TEXT,
                direction TEXT,
                alternate_functions TEXT,
                electrical_params TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chip_id) REFERENCES chips(chip_id) ON DELETE CASCADE
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_pins_chip ON pins(chip_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pins_name ON pins(pin_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pins_function ON pins(function_type)")
        logger.debug("引脚表创建完成")
    
    def _create_parameters_table(self, conn: sqlite3.Connection):
        """Create the parameters table and its indexes."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parameters (
                param_id TEXT PRIMARY KEY,
                chip_id TEXT,
                pin_id TEXT,
                param_name TEXT NOT NULL,
                param_value REAL,
                unit TEXT,
                condition TEXT,
                source TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chip_id) REFERENCES chips(chip_id) ON DELETE CASCADE,
                FOREIGN KEY (pin_id) REFERENCES pins(pin_id) ON DELETE CASCADE
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_parameters_chip ON parameters(chip_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_parameters_pin ON parameters(pin_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_parameters_name ON parameters(param_name)")
        logger.debug("参数表创建完成")
    
    def get_stats(self) -> Dict[str, int]:
        """Return row counts for each table.

        Returns:
            Dict mapping table names to row counts, e.g.
            ``{'chips': 15, 'pins': 120, 'parameters': 350}``.
        """
        stats = {'chips': 0, 'pins': 0, 'parameters': 0}

        try:
            with self._get_connection() as conn:
                for table in stats.keys():
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cursor.fetchone()[0]
        except Exception as e:
            logger.warning(f"获取数据库统计信息失败: {e}")

        return stats

    def is_initialized(self) -> bool:
        """Check whether the database contains data (chips table has rows)."""
        try:
            stats = self.get_stats()
            return stats.get('chips', 0) > 0
        except Exception:
            return False

def auto_init_knowledge_graph(
    db_path: str = None
) -> bool:
    """Initialize the knowledge graph: create tables and import chip data.

    This is a two-step process:
      1. KnowledgeGraphDB(db_path) — create the schema (chips/pins/parameters).
      2. import_common_chip_data(db_path) — populate predefined chip data.

    ``chip_importer`` is imported lazily inside this function to avoid circular
    imports and reduce startup cost.

    Returns:
        True on success, False on failure (errors are logged internally).
    """
    if db_path is None:
        db_path = "./data/knowledge_graph.db"

    try:
        logger.info("=" * 60)
        logger.info("开始自动初始化知识图谱数据库")
        logger.info("=" * 60)

        logger.info("步骤1/2: 初始化数据库表结构...")
        db = KnowledgeGraphDB(db_path)
        logger.info("✅ 数据库表结构初始化完成")

        logger.info("步骤2/2: 导入常见芯片数据...")
        from .chip_importer import import_common_chip_data
        import_common_chip_data(db_path)
        logger.info("✅ 常见芯片数据导入完成")

        logger.info("=" * 60)
        logger.info("知识图谱数据库初始化完成！")
        logger.info("=" * 60)

        stats = db.get_stats()
        logger.info(f"数据库统计: {stats}")

        return True

    except Exception as e:
        logger.error(f"❌ 知识图谱数据库初始化失败: {e}", exc_info=True)
        return False


def ensure_knowledge_graph_initialized(
    db_path: str = None
) -> bool:
    """Ensure the knowledge graph database is ready for queries.

    Checks whether the database file exists and contains data; triggers
    ``auto_init_knowledge_graph`` only when necessary so that a normal
    restart skips the import step.
    """
    if db_path is None:
        db_path = "./data/knowledge_graph.db"

    try:
        db_file = Path(db_path)
        if not db_file.exists():
            logger.info(f"知识图谱数据库不存在: {db_path}")
            logger.info("开始自动初始化...")
            return auto_init_knowledge_graph(db_path)

        db = KnowledgeGraphDB(db_path)

        if not db.is_initialized():
            logger.info("知识图谱数据库存在但无数据，开始导入数据...")
            return auto_init_knowledge_graph(db_path)

        stats = db.get_stats()
        logger.info(f"知识图谱数据库已初始化: {stats}")
        return True

    except Exception as e:
        logger.error(f"确保知识图谱初始化失败: {e}", exc_info=True)
        logger.info("尝试重新初始化...")
        try:
            return auto_init_knowledge_graph(db_path)
        except Exception as retry_error:
            logger.error(f"重新初始化失败: {retry_error}")
            return False


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    success = auto_init_knowledge_graph()

    if success:
        logger.info("知识图谱数据库自动初始化成功！")
        sys.exit(0)
    else:
        logger.error("知识图谱数据库自动初始化失败！")
        sys.exit(1)
