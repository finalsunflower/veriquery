"""
Knowledge Graph Query Engine.

Provides read-only queries against the VeriQuery SQLite knowledge graph,
which stores chip electrical parameters in three tables:

    chips → pins → parameters

The pipeline is: graph_db.py (schema) → chip_importer.py (data) → graph_query.py (queries).

Primary consumers:
  - agents/comparison_node.py — multi-dimension chip comparison scoring
  - agents/erc_node.py — electrical rule compatibility checks

Core method: ``query_chip_parameters(chip_name)`` returns a dict of electrical
parameters for a given chip, using a three-level fallback strategy:

  Level 1 — exact match on ``chips.name``
  Level 2 — normalized name + LIKE on ``name`` / ``full_name``
  Level 3 — relaxed LIKE for 74-series variants (74HC/74HCT/74LS/…)

This ensures hits regardless of whether the user supplies a manufacturer prefix
(SN74HC04 vs 74HC04) or a sub-family variant.
"""
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

_CHIP_PREFIXES = frozenset([
    'SN', 'CD', 'MC', 'LM', 'NE', 'UA', 'ULN', 'TLC', 'TIP', 'IRF', '2N', 'BC', 'BD'
])


class SQLiteGraphQueryEngine:
    """Read-only query engine for the VeriQuery knowledge graph database.

    Uses short-lived SQLite connections (one per query) with
    ``row_factory = sqlite3.Row`` for dict-style column access.
    Deduplicates parameters by keeping the highest-confidence value
    (SQL orders by ``confidence DESC``).
    """

    def __init__(self, db_path: str = "./data/knowledge_graph.db"):
        """Initialize the query engine.

        Args:
            db_path: Path to the SQLite database file created by
                     ``graph_db.py`` and populated by ``chip_importer.py``.
        """
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Warn if the database file is missing."""
        if not Path(self.db_path).exists():
            logger.warning(f"知识图谱数据库不存在: {self.db_path}")
            logger.info("请先运行 init_knowledge_graph.py 初始化数据库")

    def _get_connection(self):
        """Return a SQLite connection with ``row_factory`` set to ``Row``."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _parse_json_field(value: Optional[str], default: Any = None) -> Any:
        """Safely parse a JSON string stored in a database field.

        Returns ``default`` (or an empty list) when *value* is empty or
        malformed, instead of raising.
        """
        if not value:
            return default if default is not None else []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"JSON解析失败: {value}")
            return default if default is not None else []

    @staticmethod
    def _normalize_chip_name(chip_name: str) -> str:
        """Strip common manufacturer prefixes from a chip name.

        E.g. ``SN74HC04`` → ``74HC04``, ``LM358`` → ``358``, ``NE555`` → ``555``.
        Returns the upper-cased original if no known prefix matches.
        """
        if not chip_name:
            return chip_name

        name = chip_name.upper().strip()

        for prefix in _CHIP_PREFIXES:
            if name.startswith(prefix) and len(name) > len(prefix):
                remaining = name[len(prefix):]
                if remaining and (remaining[0].isdigit() or
                                 remaining.startswith('74') or 
                                 remaining.startswith('40')):
                    return remaining

        return name

    def _execute_chip_query_with_fallback(
        self,
        base_query: str,
        chip_name: str,
        use_table_alias: bool = True
    ) -> List[sqlite3.Row]:
        """Execute a chip query with three-level fallback.

        Level 1 — exact match on ``name``.
        Level 2 — normalized name + LIKE on ``name`` / ``full_name``.
        Level 3 — relaxed LIKE for 74-series variants.

        Args:
            base_query: SQL with a ``WHERE name = ?`` (or ``c.name = ?``) placeholder.
            chip_name: Chip name to search for.
            use_table_alias: Whether the SQL uses table alias ``c.``.

        Returns:
            List of matching rows (may be empty).
        """
        normalized_name = self._normalize_chip_name(chip_name)
        name_col = "c.name" if use_table_alias else "name"
        full_name_col = "c.full_name" if use_table_alias else "full_name"

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Level 1: exact match
            cursor.execute(base_query, (chip_name,))
            rows = cursor.fetchall()
            if rows:
                return rows

            # Level 2: normalized name + LIKE
            fallback_query = base_query.replace(
                f"WHERE {name_col} = ?",
                f"WHERE {name_col} = ? OR {name_col} LIKE ? OR {full_name_col} = ? OR {full_name_col} LIKE ?"
            )
            cursor.execute(
                fallback_query,
                (normalized_name, f"%{normalized_name}%", chip_name, f"%{chip_name}%")
            )
            rows = cursor.fetchall()
            if rows:
                return rows

            # Level 3: 74-series relaxed LIKE
            if normalized_name.startswith("74"):
                query_74 = base_query.replace(
                    f"WHERE {name_col} = ?",
                    f"WHERE {name_col} LIKE ? OR {full_name_col} LIKE ?"
                )
                cursor.execute(query_74, (f"%{normalized_name}%", f"%{normalized_name}%"))
                rows = cursor.fetchall()

            return rows
    
    def query_chip_parameters(self, chip_name: str) -> Dict[str, Dict[str, Any]]:
        """Query all electrical parameters for a chip.

        Primary consumer of :class:`SQLiteGraphQueryEngine`.  Returns a dict
        keyed by parameter name; when duplicate names exist the row with the
        highest ``confidence`` is kept (SQL already orders by
        ``confidence DESC``).

        Args:
            chip_name: Chip name (case-insensitive, prefix-tolerant).

        Returns:
            ``{param_name: {value, unit, condition, source, confidence}}``
            or an empty dict when no match is found.
        """
        query = """
            SELECT p.param_name, p.param_value, p.unit, p.condition, p.source, p.confidence
            FROM parameters p
            JOIN chips c ON p.chip_id = c.chip_id
            WHERE c.name = ?
            ORDER BY p.confidence DESC
        """

        rows = self._execute_chip_query_with_fallback(query, chip_name)

        result = {}
        for row in rows:
            param_name = row['param_name']
            if param_name not in result:
                result[param_name] = {
                    'value': row['param_value'],
                    'unit': row['unit'],
                    'condition': row['condition'] if 'condition' in row.keys() else '',
                    'source': row['source'],
                    'confidence': row['confidence']
                }

        return result

    def get_stats(self) -> Dict[str, int]:
        """Return row counts for the three knowledge-graph tables.

        Useful for startup checks and diagnostics.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            tables = ['chips', 'pins', 'parameters']
            stats = {}

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

            return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = SQLiteGraphQueryEngine()
    stats = engine.get_stats()
    print("知识图谱统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
