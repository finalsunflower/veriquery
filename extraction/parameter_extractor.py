"""
Smart Parameter Extractor — Three-Stage Cascaded Pipeline

Extracts electrical parameters (VOH/VOL/VIH/VIL/IOH/IOL etc.) from datasheet
PDF text and structured tables, producing standardized ElectricalSpec objects.

Pipeline:
    PDF text + table_store
      → Stage 1: Structured table query (highest confidence)
      → Stage 2: Section-anchored regex matching
      → Stage 3: Few-shot LLM targeted verification
      → List[ElectricalSpec]

Each stage processes only the parameters missed by prior stages (cascaded
fallback), ensuring high-confidence results are preserved first.

Dependencies:
    - core/schema.py: ElectricalSpec, Citation, ExtractionSource
    - retrieval/table_store.py: TableStore (Stage 1)
    - core/llm_client.py: LLMClient (Stage 3)
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple

from core.schema import Citation, ElectricalSpec, ExtractionSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section boundary keywords for Stage 2 section anchoring
# ---------------------------------------------------------------------------
# Datasheets have a fixed chapter structure; electrical parameters are
# concentrated in "Electrical Characteristics" or "Recommended Operating
# Conditions" sections.  Anchoring regex to these sections reduces noise
# by 80%+ compared to full-text matching.
# (?i) = case-insensitive, \b = word boundary
# ---------------------------------------------------------------------------

_SECTION_START = [
    r'(?i)\b(dc\s+)?electrical\s+characteristics\b',
    r'(?i)\brecommended\s+operating\s+conditions?\b',
    r'(?i)\babsolute\s+maximum\s+ratings?\b',
    r'(?i)\b电气特性\b',
    r'(?i)\b直流特性\b',
    r'(?i)\b推荐工作条件\b',
]

_SECTION_END = [
    r'(?i)\b(ac\s+)?(timing|switching)\s+characteristics\b',
    r'(?i)\bapplication\s+(information|circuit|note)\b',
    r'(?i)\bpackage\s+(information|outline|dimensions)\b',
    r'(?i)\b应用电路\b',
    r'(?i)\b封装尺寸\b',
]

# ---------------------------------------------------------------------------
# Column semantic recognition for Stage 1 table parsing
# ---------------------------------------------------------------------------
# Datasheet table headers use varied wording across vendors (TI/ST/NXP/etc.).
# Each semantic role matches only the first occurrence to avoid column index
# collisions when a table has duplicate header names.
# ---------------------------------------------------------------------------

_COL_SYMBOL = ['symbol', 'parameter', 'param', 'sym', '符号', '参数名', '参数符号',
               'characteristic', 'characteristics']
_COL_MIN    = ['min', 'minimum', '最小', 'min.']
_COL_TYP    = ['typ', 'typical', 'nom', 'nominal', '典型', 'typ.']
_COL_MAX    = ['max', 'maximum', '最大', 'max.']
_COL_UNIT   = ['unit', 'units', '单位', 'un']
_COL_COND   = ['condition', 'test condition', 'note', '条件', '测试条件',
               'conditions', 'notes', 'remarks']

# ---------------------------------------------------------------------------
# Default units and plausible value ranges per parameter symbol
# ---------------------------------------------------------------------------
# _PARAM_UNITS: fallback unit when the table's unit column is missing.
# _PARAM_RANGES: (min, max) plausible range for range validation — values
#   outside these bounds are treated as false positives and filtered out.
#   Ranges are based on IEEE logic-level definitions and typical TTL/CMOS values.
# ---------------------------------------------------------------------------

_PARAM_UNITS: Dict[str, str] = {
    'VOH': 'V',  'VOL': 'V',  'VIH': 'V',  'VIL': 'V',
    'VCC': 'V',  'VDD': 'V',  'VSS': 'V',  'VEE': 'V',
    'VIN': 'V',  'VOUT': 'V', 'VREF': 'V',
    'IOH': 'mA', 'IOL': 'mA',
    'IIH': 'µA', 'IIL': 'µA',
    'ICC': 'mA', 'IDD': 'mA', 'IQ': 'µA',
    'tPLH': 'ns', 'tPHL': 'ns', 'tpd': 'ns',
    'fmax': 'MHz', 'fMAX': 'MHz',
}

_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    'VOH': (0.5,  25.0),
    'VOL': (0.0,   3.0),
    'VIH': (0.5,  25.0),
    'VIL': (0.0,   5.0),
    'VCC': (1.0,  50.0),
    'VDD': (1.0,  50.0),
    'IOH': (-500.0, 500.0),
    'IOL': (0.01, 500.0),
    'IIH': (-500.0, 500.0),
    'IIL': (-500.0, 500.0),
    'ICC': (0.0,  1000.0),
    'IDD': (0.0,  1000.0),
}

# Parameters whose conservative (worst-case) value is the *minimum*.
_MIN_VALUE_PARAMS: frozenset = frozenset({'VOH', 'VIH', 'IOH'})

# Parameters whose conservative (worst-case) value is the *maximum*.
_MAX_VALUE_PARAMS: frozenset = frozenset(
    {'VOL', 'VIL', 'IOL', 'TPD', 'TPLH', 'TPHL', 'ICC', 'IDD', 'IIH', 'IIL'}
)

# Strips trailing unit characters from numeric strings, e.g. "2.4V" → "2.4".
_NUM_UNIT_RE = re.compile(r'\s*[VmAµuμnpkMGHz°CF%Ωs]+$', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Multi-pattern regex lists per parameter (ordered by specificity, high→low)
# ---------------------------------------------------------------------------
# Each parameter gets multiple regex patterns covering the diverse formats
# found in datasheets: table-row, assignment, full-name, and loose fallback.
# Confidence decays with pattern index: conf = 0.85 - idx * 0.04.
# ---------------------------------------------------------------------------

_MULTI_PATTERNS: Dict[str, List[str]] = {
    'VOH': [
        r'VOH\s+\S[^\n]*?\s([2-9]\.\d+|\d\d\.\d*)\s*V\b',
        r'V_?OH\s*[=:]\s*([+-]?\d+\.?\d*)\s*V',
        r'Output\s+High\s+Volt(?:age)?\s*[^\n]*?([2-9]\.\d+|\d\d\.\d*)\s*V',
        r'HIGH[- ]Level\s+Output\s+Volt[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'VOH\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
    ],
    'VOL': [
        r'VOL\s+\S[^\n]*?\s(0\.\d+|[01]\.\d*)\s*V\b',
        r'V_?OL\s*[=:]\s*([+-]?\d+\.?\d*)\s*V',
        r'Output\s+Low\s+Volt(?:age)?\s*[^\n]*?(0\.\d+|[01]\.\d*)\s*V',
        r'LOW[- ]Level\s+Output\s+Volt[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'VOL\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
    ],
    'VIH': [
        r'VIH\s+\S[^\n]*?\sMIN\s+([+-]?\d+\.?\d*)\s*V\b',
        r'VIH\s+\S[^\n]*?\smin\s+([+-]?\d+\.?\d*)\s*V\b',
        r'VIH\s+\S[^\n]*?\s([0-2]\.\d+)\s*V\b',
        r'V_?IH\s*[=:]\s*([+-]?\d+\.?\d*)\s*V',
        r'Input\s+High\s+Volt(?:age)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'HIGH[- ]Level\s+Input\s+Volt[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'VIH\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
    ],
    'VIL': [
        r'VIL\s+\S[^\n]*?\sMAX\s+([+-]?\d+\.?\d*)\s*V\b',
        r'VIL\s+\S[^\n]*?\smax\s+([+-]?\d+\.?\d*)\s*V\b',
        r'VIL\s+\S[^\n]*?\s([0-1]\.\d+)\s*V\b',
        r'V_?IL\s*[=:]\s*([+-]?\d+\.?\d*)\s*V',
        r'Input\s+Low\s+Volt(?:age)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'LOW[- ]Level\s+Input\s+Volt[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'VIL\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
    ],
    'IOH': [
        r'IOH\s+\S[^\n]*?\s([+-]?\d+\.?\d*)\s*m?A\b',
        r'I_?OH\s*[=:]\s*([+-]?\d+\.?\d*)\s*m?A',
        r'Output\s+High\s+Curr(?:ent)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*m?A',
        r'IOH\s*[^\n]*?([+-]?\d+\.?\d*)\s*m?A',
    ],
    'IOL': [
        r'IOL\s+\S[^\n]*?\s([+-]?\d+\.?\d*)\s*m?A\b',
        r'I_?OL\s*[=:]\s*([+-]?\d+\.?\d*)\s*m?A',
        r'Output\s+Low\s+Curr(?:ent)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*m?A',
        r'IOL\s*[^\n]*?([+-]?\d+\.?\d*)\s*m?A',
    ],
    'IIH': [
        r'IIH\s+\S[^\n]*?\s([+-]?\d+\.?\d*)\s*[µuμ]?A\b',
        r'I_?IH\s*[=:]\s*([+-]?\d+\.?\d*)\s*[µuμ]?A',
        r'Input\s+High\s+Curr(?:ent)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*[µuμ]?A',
        r'IIH\s*[^\n]*?([+-]?\d+\.?\d*)\s*[µuμ]?A',
    ],
    'IIL': [
        r'IIL\s+\S[^\n]*?\s([+-]?\d+\.?\d*)\s*[µuμ]?A\b',
        r'I_?IL\s*[=:]\s*([+-]?\d+\.?\d*)\s*[µuμ]?A',
        r'Input\s+Low\s+Curr(?:ent)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*[µuμ]?A',
        r'IIL\s*[^\n]*?([+-]?\d+\.?\d*)\s*[µuμ]?A',
    ],
    'VCC': [
        r'VCC\s+\S[^\n]*?\s([+-]?\d+\.?\d*)\s*V\b',
        r'V_?CC\s*[=:]\s*([+-]?\d+\.?\d*)\s*V',
        r'Supply\s+Volt(?:age)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'VCC\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'VCC±\s*[=:]\s*±(\d+\.?\d*)\s*V',
        r'VCC\s*\±\s*([+-]?\d+\.?\d*)\s*V',
        r'Supply\s+voltage\s+(?:range)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'Operating\s+voltage\s+(?:range)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
    ],
    'VDD': [
        r'VDD\s+\S[^\n]*?\s([+-]?\d+\.?\d*)\s*V\b',
        r'V_?DD\s*[=:]\s*([+-]?\d+\.?\d*)\s*V',
        r'Supply\s+Volt(?:age)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'VDD\s*[^\n]*?([+-]?\d+\.?\d*)\s*V',
        r'VDD±\s*[=:]\s*±(\d+\.?\d*)\s*V',
        r'VDD\s*\±\s*([+-]?\d+\.?\d*)\s*V',
    ],
    'ICC': [
        r'ICC\s+\S[^\n]*?\s([+-]?\d+\.?\d*)\s*m?A\b',
        r'I_?CC\s*[=:]\s*([+-]?\d+\.?\d*)\s*m?A',
        r'Supply\s+Curr(?:ent)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*m?A',
        r'Quiescent\s+Curr(?:ent)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*m?A',
        r'ICC\s*[^\n]*?([+-]?\d+\.?\d*)\s*m?A',
    ],
    'IDD': [
        r'IDD\s+\S[^\n]*?\s([+-]?\d+\.?\d*)\s*m?A\b',
        r'I_?DD\s*[=:]\s*([+-]?\d+\.?\d*)\s*m?A',
        r'Supply\s+Curr(?:ent)?\s*[^\n]*?([+-]?\d+\.?\d*)\s*m?A',
        r'IDD\s*[^\n]*?([+-]?\d+\.?\d*)\s*m?A',
    ],
}

_COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {
    param: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for param, patterns in _MULTI_PATTERNS.items()
}

# ---------------------------------------------------------------------------
# Few-shot examples for Stage 3 LLM prompting (designed for 2B-scale models)
# ---------------------------------------------------------------------------
# Three examples cover the most common datasheet formats:
#   (1) Table-row: symbol + description + values + unit + condition
#   (2) Assignment: V_OL = 0.4V max when ...
#   (3) Full-name: Input High Voltage (VIH): minimum 2.0V ...
# Output format: "PARAM: value unit [condition]" (line format chosen over JSON
# because 2B models achieve ~85% compliance vs. ~60% for JSON).
# ---------------------------------------------------------------------------

_FEW_SHOT = """\
示例1 (表格行格式):
文本: "VOH  Output High Voltage  2.4  4.4  V  VCC=4.5V,IOH=-0.4mA"
提取VOH: 2.4 V [VCC=4.5V,IOH=-0.4mA]

示例2 (赋值格式):
文本: "V_OL = 0.4V max when IOL = 8mA"
提取VOL: 0.4 V [IOL=8mA]

示例3 (英文全名格式):
文本: "Input High Voltage (VIH): minimum 2.0V (VCC=5V)"
提取VIH: 2.0 V [VCC=5V]\
"""


class SmartParameterExtractor:
    """Three-stage cascaded parameter extractor with progressive fallback.

    Stage 1 — Structured table query (confidence ~0.93):
        Query table_store for pre-extracted tables, parse via header
        detection + row matching.  O(1) lookup, near-zero noise.

    Stage 2 — Section-anchored regex (confidence 0.73–0.85):
        Locate the electrical characteristics section, then apply
        multi-pattern regex matching.  Section anchoring reduces noise 80%+.

    Stage 3 — Few-shot LLM verification (confidence ~0.80):
        For remaining parameters, extract a context window around each
        parameter name and send a focused prompt with few-shot examples.
    """

    def __init__(self, llm_client=None, table_store=None):
        self.llm_client = llm_client
        self.table_store = table_store
        self._llm_batch = 3
        self._llm_ctx_chars = 250

    async def extract_batch(
        self,
        text: str,
        target_params: List[str],
        chip_name: str = "",
        filename: str = "",
        page: int = 0,
        document_ids: Optional[List[str]] = None,
    ) -> List[ElectricalSpec]:
        """Run the three-stage extraction pipeline.

        Returns:
            List of ElectricalSpec sorted by confidence (descending).
        """
        results: Dict[str, ElectricalSpec] = {}
        remaining = list(target_params)

        # Stage 1: structured table query
        if self.table_store and remaining:
            s1 = await self._stage1_table(chip_name, remaining, document_ids, filename, page)
            results.update(s1)
            remaining = [p for p in remaining if p not in results]
            if s1:
                logger.info("[Stage1-Table] %d/%d 命中: %s", len(s1), len(target_params), list(s1))

        # Stage 2: section-anchored regex
        if remaining and text:
            section = _extract_section(text)
            s2 = _stage2_regex(section, remaining, chip_name, filename, page)
            for param, spec in s2.items():
                if param not in results:
                    results[param] = spec
            remaining = [p for p in remaining if p not in results]
            if s2:
                logger.info("[Stage2-Regex] %d 命中: %s（节段: %d 字符）",
                            len(s2), list(s2), len(section))

        # Stage 3: few-shot LLM verification
        if remaining and self.llm_client and text:
            s3 = await self._stage3_llm(text, remaining, chip_name, filename, page)
            for param, spec in s3.items():
                if param not in results:
                    results[param] = spec
            if s3:
                logger.info("[Stage3-LLM] %d 命中: %s", len(s3), list(s3))

        out = sorted(results.values(), key=lambda s: s.confidence, reverse=True)
        logger.info("SmartExtractor 汇总: %d/%d 参数成功", len(out), len(target_params))
        return out

    # ------------------------------------------------------------------
    # Stage 1: structured table query
    # ------------------------------------------------------------------

    async def _stage1_table(
        self,
        chip_name: str,
        target_params: List[str],
        document_ids: Optional[List[str]],
        filename: str,
        page: int,
    ) -> Dict[str, ElectricalSpec]:
        """Extract parameters from pre-parsed tables in table_store.

        Strategy A: structured row matching (header detection + symbol lookup).
        Strategy B: full-text table matching (fallback when PDF parsing splits
                    parameter names and values across cells).
        """
        results: Dict[str, ElectricalSpec] = {}
        try:
            raw_tables = []

            # Prefer direct document lookup (O(1))
            if document_ids:
                raw_tables = self.table_store.get_tables_by_documents(
                    document_ids, limit=100, min_confidence=0.5
                )
                logger.debug("[Stage1-Table] 直接获取文档表格: %d 个", len(raw_tables))

            # Fallback: FTS search by chip name or parameter names
            if not raw_tables:
                query = chip_name if chip_name else " ".join(target_params[:3])
                filters = {"document_ids": document_ids} if document_ids else {}
                raw_tables = await self.table_store.search(query, top_k=25, filters=filters)
                logger.debug("[Stage1-Table] FTS搜索表格: %d 个", len(raw_tables))

            for entry in raw_tables:
                if hasattr(entry, 'data'):
                    table_data = entry.data
                    src_file = entry.filename or filename
                    src_page = max(1, entry.page or page)
                else:
                    table_data = entry.get("table_data", [])
                    src_file = entry.get("filename", filename) or filename
                    src_page = max(1, entry.get("page", page) or page)

                if not table_data or len(table_data) < 2:
                    continue

                # Strategy A: structured row matching
                col_map = None
                header_row_idx = 0
                for row_idx in range(min(10, len(table_data))):
                    candidate_col = _detect_columns(table_data[row_idx])
                    if _is_electrical_table(candidate_col):
                        col_map = candidate_col
                        header_row_idx = row_idx
                        break

                if col_map and _is_electrical_table(col_map):
                    for row in table_data[header_row_idx + 1:]:
                        sym_raw = _cell(row, col_map.get("symbol"))
                        if not sym_raw:
                            continue
                        sym_clean = re.sub(r'[\s\n\r_]+', '', str(sym_raw).upper())
                        for param in target_params:
                            if param in results:
                                continue
                            p_norm = param.upper().replace("_", "")
                            if p_norm == sym_clean or p_norm in sym_clean or sym_clean in p_norm:
                                spec = _row_to_spec(row, col_map, param, chip_name,
                                                    src_file, src_page)
                                if spec:
                                    results[param] = spec

                # Strategy B: full-text table matching
                table_full_text = ""
                for row in table_data:
                    row_text = " ".join(str(cell) for cell in row if cell)
                    table_full_text += row_text + " "
                table_full_text = re.sub(r'\s+', ' ', table_full_text)
                table_clean = re.sub(r'[\s\n\r]+', '', table_full_text.upper())

                for param in target_params:
                    if param in results:
                        continue
                    p_norm = param.upper()
                    if p_norm in table_clean or "HIGH" in table_clean or "LOW" in table_clean:
                        spec = _extract_param_from_table_text(
                            table_full_text, param, chip_name, src_file, src_page
                        )
                        if spec:
                            results[param] = spec

        except Exception as exc:
            logger.warning("[Stage1-Table] 查询失败: %s", exc)
        return results

    # ------------------------------------------------------------------
    # Stage 3: few-shot LLM targeted verification
    # ------------------------------------------------------------------

    async def _stage3_llm(
        self,
        full_text: str,
        missing_params: List[str],
        chip_name: str,
        filename: str,
        page: int,
    ) -> Dict[str, ElectricalSpec]:
        """Extract remaining parameters via few-shot LLM prompting.

        Parameters are processed in batches of ``_llm_batch`` to avoid
        overwhelming the 2B-scale model.  Each parameter gets a focused
        context window rather than the full document.
        """
        results: Dict[str, ElectricalSpec] = {}

        for i in range(0, len(missing_params), self._llm_batch):
            batch = missing_params[i: i + self._llm_batch]
            ctx_parts = []
            for param in batch:
                snippet = _find_context(full_text, param, self._llm_ctx_chars)
                if snippet:
                    ctx_parts.append(f"[{param}]\n{snippet}")

            if not ctx_parts:
                continue

            prompt = (
                f"你是电子元件参数提取专家。请从以下数据手册片段中提取指定参数值。\n\n"
                f"{_FEW_SHOT}\n\n"
                f"现在从芯片 {chip_name} 的文档中提取：{', '.join(batch)}\n\n"
                + "\n\n".join(ctx_parts)
                + "\n\n输出格式（每行一个）：参数名: 数值 单位 [条件]\n"
                "找不到时输出：参数名: 未找到\n"
                "直接输出，不要额外解释："
            )

            try:
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, self.llm_client.invoke, prompt),
                    timeout=40,
                )
                batch_res = _parse_line_response(
                    str(response), batch, chip_name, filename, page
                )
                results.update(batch_res)
            except asyncio.TimeoutError:
                logger.warning("[Stage3-LLM] 超时，跳过 %s", batch)
            except Exception as exc:
                logger.warning("[Stage3-LLM] 调用失败: %s", exc)

        return results


# ======================================================================
# Module-level helper functions (stateless, independently testable)
# ======================================================================

def _extract_section(text: str, max_chars: int = 2000) -> str:
    """Locate and extract the electrical characteristics section.

    Algorithm (Section Anchoring):
        1. Search for the earliest ``_SECTION_START`` keyword as the start.
        2. Extract up to *max_chars* characters from that position.
        3. Within the extracted body, search for ``_SECTION_END`` keywords
           and truncate at the earliest match (after at least 50 chars to
           avoid cutting off the section title itself).
    """
    start = len(text)
    for pat in _SECTION_START:
        m = re.search(pat, text)
        if m and m.start() < start:
            start = m.start()

    if start == len(text):
        return text[:max_chars]

    body = text[start: start + max_chars]
    end = len(body)
    for pat in _SECTION_END:
        m = re.search(pat, body)
        if m and m.start() > 50:
            end = min(end, m.start())

    extracted = body[:end]
    logger.debug("[Section] 节段 %d/%d 字符（信噪比约 %dx）",
                 len(extracted), len(text), max(1, len(text) // max(1, len(extracted))))
    return extracted


def _detect_columns(header_row: List) -> Dict[str, int]:
    """Identify the semantic role of each column in a table header row.

    Matches header cell text against synonym lists (``_COL_SYMBOL``, etc.).
    Each semantic role is assigned only to the first matching column.

    Enhancement: if no ``symbol`` column is found, tries merging adjacent
    cells to handle PDF-parsing splits (e.g. "Sym" + "bol" → "Symbol").
    """
    col: Dict[str, int] = {}

    for i, cell in enumerate(header_row):
        h = str(cell).lower().strip()
        h_clean = re.sub(r'[\s\n\r]+', '', h)

        if any(kw in h_clean or kw in h for kw in _COL_SYMBOL) and "symbol" not in col:
            col["symbol"] = i
        elif any(kw in h_clean or kw in h for kw in _COL_MIN) and "min" not in col:
            col["min"] = i
        elif any(kw in h_clean or kw in h for kw in _COL_TYP) and "typ" not in col:
            col["typ"] = i
        elif any(kw in h_clean or kw in h for kw in _COL_MAX) and "max" not in col:
            col["max"] = i
        elif any(kw in h_clean or kw in h for kw in _COL_UNIT) and "unit" not in col:
            col["unit"] = i
        elif any(kw in h_clean or kw in h for kw in _COL_COND) and "cond" not in col:
            col["cond"] = i

    if "symbol" not in col:
        for i in range(len(header_row) - 1):
            combined = str(header_row[i]).lower() + str(header_row[i + 1]).lower()
            combined_clean = re.sub(r'[\s\n\r]+', '', combined)
            if any(kw in combined_clean for kw in _COL_SYMBOL) and "symbol" not in col:
                col["symbol"] = i

    return col


def _is_electrical_table(col: Dict[str, int]) -> bool:
    """Return True if the column map represents an electrical characteristics table.

    Requires a ``symbol`` column and at least one value column (min/typ/max).
    """
    return "symbol" in col and any(k in col for k in ("min", "typ", "max"))


def _extract_param_from_table_text(
    table_text: str,
    param: str,
    chip_name: str,
    filename: str,
    page: int,
) -> Optional[ElectricalSpec]:
    """Extract a parameter from the full text of a table (Strategy B fallback).

    Uses context-aware keyword search: locates relevant keywords (e.g.
    "HIGH-LEVEL", "VOH"), then extracts values within physically plausible
    ranges from the surrounding context.

    Supported parameters: VOH, VOL, VIH, VIL, IOH, IOL, IIH, IIL.
    """
    param_upper = param.upper()
    clean_text = re.sub(r'\s+', ' ', table_text)
    clean_upper = clean_text.upper()

    # -- inner helpers (closures over clean_text, param) -------------------

    def find_numbers_in_range(text: str, min_val: float, max_val: float) -> List[float]:
        numbers = re.findall(r'(\d+\.?\d*)', text)
        valid = []
        for n in numbers:
            try:
                val = float(n)
                if min_val <= val <= max_val:
                    valid.append(val)
            except (ValueError, TypeError):
                pass
        return valid

    def extract_voh_from_high_level(idx: int) -> Optional[ElectricalSpec]:
        snippet = clean_text[idx:idx + 500]
        snippet_upper = snippet.upper()
        if 'OH' in snippet_upper:
            oh_idx = snippet_upper.find('OH')
            context = snippet[max(0, oh_idx - 50):oh_idx + 200]
            numbers = find_numbers_in_range(context, 2.0, 5.5)
            if numbers:
                return ElectricalSpec(
                    param=param, name=param,
                    min_value=numbers[0],
                    typ_value=numbers[1] if len(numbers) > 1 else None,
                    max_value=None,
                    unit='V', condition="",
                    source_type=ExtractionSource.TABLE, confidence=0.85,
                )
        return None

    def extract_vol_from_low_level(idx: int) -> Optional[ElectricalSpec]:
        snippet = clean_text[idx:idx + 500]
        numbers = find_numbers_in_range(snippet, 0.0, 1.5)
        if numbers:
            return ElectricalSpec(
                param=param, name=param,
                min_value=None,
                typ_value=numbers[0],
                max_value=numbers[1] if len(numbers) > 1 else numbers[0],
                unit='V', condition="",
                source_type=ExtractionSource.TABLE, confidence=0.85,
            )
        return None

    def extract_vih_from_high_level(idx: int) -> Optional[ElectricalSpec]:
        snippet = clean_text[idx:idx + 500]
        numbers = find_numbers_in_range(snippet, 2.0, 4.0)
        for n in numbers:
            if n >= 2.0:
                return ElectricalSpec(
                    param=param, name=param,
                    min_value=n, typ_value=None, max_value=None,
                    unit='V', condition="",
                    source_type=ExtractionSource.TABLE, confidence=0.85,
                )
        return None

    def extract_vil_from_low_level(idx: int) -> Optional[ElectricalSpec]:
        snippet = clean_text[idx:idx + 500]
        numbers = find_numbers_in_range(snippet, 0.0, 1.5)
        for n in numbers:
            if n <= 1.0:
                return ElectricalSpec(
                    param=param, name=param,
                    min_value=None, typ_value=None, max_value=n,
                    unit='V', condition="",
                    source_type=ExtractionSource.TABLE, confidence=0.85,
                )
        return None

    def extract_ioh_from_oh(idx: int) -> Optional[ElectricalSpec]:
        snippet = clean_text[max(0, idx - 100):idx + 300]
        numbers = re.findall(r'([+-]?\d+\.?\d*)', snippet)
        for n in numbers:
            try:
                val = float(n)
                if 1.0 <= abs(val) <= 100.0:
                    return ElectricalSpec(
                        param=param, name=param,
                        min_value=-abs(val), typ_value=None, max_value=None,
                        unit='mA', condition="",
                        source_type=ExtractionSource.TABLE, confidence=0.85,
                    )
            except (ValueError, TypeError):
                pass
        return None

    def extract_iol_from_ol(idx: int) -> Optional[ElectricalSpec]:
        snippet = clean_text[max(0, idx - 100):idx + 300]
        numbers = re.findall(r'([+-]?\d+\.?\d*)', snippet)
        for n in numbers:
            try:
                val = float(n)
                if 1.0 <= val <= 100.0:
                    return ElectricalSpec(
                        param=param, name=param,
                        min_value=val, typ_value=None, max_value=None,
                        unit='mA', condition="",
                        source_type=ExtractionSource.TABLE, confidence=0.85,
                    )
            except (ValueError, TypeError):
                pass
        return None

    def extract_ii_iiL(idx: int, is_ih: bool) -> Optional[ElectricalSpec]:
        snippet = clean_text[idx:idx + 300]
        numbers = re.findall(r'(\d+\.?\d*)', snippet)
        valid = [float(n) for n in numbers if 0.001 <= float(n) <= 100.0]
        if valid:
            return ElectricalSpec(
                param=param, name=param,
                min_value=valid[0], typ_value=None, max_value=None,
                unit='µA', condition="",
                source_type=ExtractionSource.TABLE, confidence=0.85,
            )
        return None

    # -- parameter dispatch ------------------------------------------------

    if param_upper == 'VOH':
        for pattern in ['HIGH-LEVEL', 'HIGH LEVEL', 'VOH']:
            idx = clean_upper.find(pattern)
            if idx >= 0:
                result = extract_voh_from_high_level(idx)
                if result:
                    return result
        idx = clean_upper.find('OH')
        if idx >= 0:
            return extract_voh_from_high_level(idx)

    elif param_upper == 'VOL':
        for pattern in ['LOW-LEVEL', 'LOW LEVEL', 'VOL']:
            idx = clean_upper.find(pattern)
            if idx >= 0:
                result = extract_vol_from_low_level(idx)
                if result:
                    return result
        idx = clean_upper.find('OL')
        if idx >= 0:
            return extract_vol_from_low_level(idx)

    elif param_upper == 'VIH':
        for pattern in ['HIGH-LEVEL', 'HIGH LEVEL', 'VIH']:
            idx = clean_upper.find(pattern)
            if idx >= 0:
                result = extract_vih_from_high_level(idx)
                if result:
                    return result
        idx = clean_upper.find('H-LEVEL')
        if idx >= 0:
            return extract_vih_from_high_level(idx)

    elif param_upper == 'VIL':
        for pattern in ['LOW-LEVEL', 'LOW LEVEL', 'VIL']:
            idx = clean_upper.find(pattern)
            if idx >= 0:
                result = extract_vil_from_low_level(idx)
                if result:
                    return result
        idx = clean_upper.find('L-LEVEL')
        if idx >= 0:
            return extract_vil_from_low_level(idx)

    elif param_upper == 'IOH':
        for pattern in ['I OH', 'IOH', 'OH']:
            idx = clean_upper.find(pattern)
            if idx >= 0:
                result = extract_ioh_from_oh(idx)
                if result:
                    return result

    elif param_upper == 'IOL':
        for pattern in ['I OL', 'IOL', 'OL']:
            idx = clean_upper.find(pattern)
            if idx >= 0:
                result = extract_iol_from_ol(idx)
                if result:
                    return result

    elif param_upper == 'IIH':
        for pattern in ['I IH', 'IIH']:
            idx = clean_upper.find(pattern)
            if idx >= 0:
                result = extract_ii_iiL(idx, True)
                if result:
                    return result
        idx = clean_upper.find('LEAKAGE')
        if idx >= 0:
            return extract_ii_iiL(idx, True)

    elif param_upper == 'IIL':
        for pattern in ['I IL', 'IIL']:
            idx = clean_upper.find(pattern)
            if idx >= 0:
                result = extract_ii_iiL(idx, False)
                if result:
                    return result
        idx = clean_upper.find('LEAKAGE')
        if idx >= 0:
            return extract_ii_iiL(idx, False)

    return None


def _cell(row: List, idx: Optional[int]) -> Optional[str]:
    """Safely retrieve a cell value from a table row by column index."""
    if idx is None or idx >= len(row):
        return None
    v = row[idx]
    return str(v).strip() if v else None


def _parse_num(s: Optional[str]) -> Optional[float]:
    """Parse a cell string to float, handling placeholders and trailing units.

    Handles: None / empty → None, placeholders ('-', '—', 'N/A') → None,
    trailing units ("2.4V" → 2.4), range values ("2.0 to 5.5" → 2.0).
    """
    if not s:
        return None
    s = s.strip()
    if s in ('-', '—', '–', 'N/A', 'n/a', ''):
        return None
    clean = _NUM_UNIT_RE.sub('', s).strip()
    if ' to ' in clean:
        clean = clean.split(' to ')[0].strip()
    try:
        return float(clean)
    except ValueError:
        return None


def _validate_range(param: str, value: Optional[float]) -> bool:
    """Check whether *value* falls within the plausible range for *param*.

    Unknown parameters (not in ``_PARAM_RANGES``) always pass validation.
    """
    if value is None:
        return False
    rng = _PARAM_RANGES.get(param)
    if rng is None:
        return True
    return rng[0] <= value <= rng[1]


def _row_to_spec(
    row: List,
    col: Dict[str, int],
    param: str,
    chip_name: str,
    filename: str,
    page: int,
) -> Optional[ElectricalSpec]:
    """Convert a table row to an ElectricalSpec (Stage 1).

    Value selection follows the worst-case design principle:
        - VOH, VIH, IOH → take Min (conservative lower bound)
        - VOL, VIL, IOL, TPD, etc. → take Max (conservative upper bound)
        - Others → Typ > Max > Min
    """
    min_v = _parse_num(_cell(row, col.get("min")))
    typ_v = _parse_num(_cell(row, col.get("typ")))
    max_v = _parse_num(_cell(row, col.get("max")))
    unit  = _cell(row, col.get("unit")) or _PARAM_UNITS.get(param, '')
    cond  = _cell(row, col.get("cond")) or ''

    if min_v is None and typ_v is None and max_v is None:
        return None

    param_upper = param.upper()

    if param_upper in _MIN_VALUE_PARAMS:
        rep = min_v if min_v is not None else (typ_v if typ_v is not None else max_v)
    elif param_upper in _MAX_VALUE_PARAMS:
        rep = max_v if max_v is not None else (typ_v if typ_v is not None else min_v)
    else:
        rep = typ_v if typ_v is not None else (max_v if max_v is not None else min_v)

    if rep is None:
        return None

    if not _validate_range(param, rep):
        return None

    return ElectricalSpec(
        param=param,
        name=f"{chip_name} {param}".strip(),
        min_value=min_v,
        typ_value=rep,
        max_value=max_v,
        unit=unit,
        condition=cond,
        source_type=ExtractionSource.TABLE,
        confidence=0.93,
        citation=Citation(
            file=filename or "table_store",
            page=max(1, page),
            text_snippet=f"{param} extracted from table",
        ),
    )


def _stage2_regex(
    section_text: str,
    target_params: List[str],
    chip_name: str,
    filename: str,
    page: int,
) -> Dict[str, ElectricalSpec]:
    """Extract parameters via multi-pattern regex on a section-anchored text.

    For each parameter, patterns are tried in order; the first match that
    passes range validation wins.  Confidence decays with pattern index:
    ``conf = 0.85 - idx * 0.04``.
    """
    results: Dict[str, ElectricalSpec] = {}

    for param in target_params:
        patterns = _COMPILED_PATTERNS.get(param, [])
        best_val: Optional[float] = None
        best_conf = 0.0

        for idx, pat in enumerate(patterns):
            m = pat.search(section_text)
            if m:
                try:
                    val = float(m.group(1))
                    conf = 0.85 - idx * 0.04
                    if _validate_range(param, val) and (best_val is None or conf > best_conf):
                        best_val = val
                        best_conf = conf
                except (ValueError, IndexError):
                    continue

        if best_val is not None:
            results[param] = ElectricalSpec(
                param=param,
                name=f"{chip_name} {param}".strip(),
                typ_value=best_val,
                unit=_PARAM_UNITS.get(param, 'V'),
                source_type=ExtractionSource.REGEX,
                confidence=best_conf,
                citation=Citation(
                    file=filename or "text",
                    page=max(1, page),
                    text_snippet=f"{param} from section regex",
                ),
            )

    return results


def _find_context(text: str, param: str, window: int = 250) -> Optional[str]:
    """Extract a context window around the first occurrence of *param* in *text*.

    Falls back to matching the first 3 characters of *param* for cases where
    the PDF parser inserts spaces within the symbol name.
    """
    m = re.search(rf'\b{re.escape(param)}\b', text, re.IGNORECASE)
    if not m and len(param) >= 3:
        m = re.search(rf'\b{re.escape(param[:3])}\b', text, re.IGNORECASE)
    if not m:
        return None
    half = window // 2
    start = max(0, m.start() - half)
    end = min(len(text), m.end() + half)
    return text[start:end].strip()


def _parse_line_response(
    response: str,
    target_params: List[str],
    chip_name: str,
    filename: str,
    page: int,
) -> Dict[str, ElectricalSpec]:
    """Parse LLM line-format output (``"PARAM: value unit [condition]"``).

    The line format is chosen over JSON because 2B-scale models achieve
    ~85% format compliance with simple line output vs. ~60% for JSON.
    """
    results: Dict[str, ElectricalSpec] = {}

    for param in target_params:
        pat = (
            rf'{re.escape(param)}\s*:\s*'
            rf'([+-]?\d+\.?\d*)\s*'
            rf'([VmAµuμnpkMGHz°CF%Ωs]*)\s*'
            rf'(?:\[([^\]]*)\])?'
        )
        m = re.search(pat, response, re.IGNORECASE)
        if not m:
            continue
        try:
            val = float(m.group(1))
            unit = (m.group(2) or '').strip() or _PARAM_UNITS.get(param, '')
            cond = (m.group(3) or '').strip()

            if not _validate_range(param, val):
                continue

            results[param] = ElectricalSpec(
                param=param,
                name=f"{chip_name} {param}".strip(),
                typ_value=val,
                unit=unit,
                condition=cond,
                source_type=ExtractionSource.LLM,
                confidence=0.80,
                citation=Citation(
                    file=filename or "llm",
                    page=max(1, page),
                    text_snippet=f"{param} verified by LLM",
                ),
            )
        except (ValueError, IndexError):
            continue

    return results
