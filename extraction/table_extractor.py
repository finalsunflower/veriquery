"""
Datasheet Table Extractor — Three-Layer Architecture

Intelligently extracts structured table data from datasheet PDFs using a
three-layer pipeline:

    Layer 1 — DocumentPreScanner: page-level pre-scanning and classification
    Layer 2 — IntelligentRouter:  strategy-based extraction with fallback
    Layer 3 — DatasheetOptimizer:  post-extraction optimization

Pipeline:
    PDF path
      → DocumentPreScanner.scan_document() → DocumentProfile
      → IntelligentRouter.extract_with_strategy() → List[TableExtractionResult]
      → DatasheetOptimizer.optimize_results() → List[TableExtractionResult]
      → downstream: retrieval/table_store.py

Two extraction engines are supported:
    - Camelot Lattice: line-border detection (for bordered tables)
    - pdfplumber: text-alignment inference (for borderless tables)

Dependencies:
    - PyMuPDF (fitz): PDF parsing and drawing-element extraction (Layer 1)
    - camelot: lattice-mode table extraction (Layer 2, optional)
    - pdfplumber: text-based table extraction (Layer 2)
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

logger = logging.getLogger(__name__)


class PageType(Enum):
    """Classification of PDF page types for extraction strategy selection."""

    COVER = "cover"
    TOC = "toc"
    PARAMETER_TABLE = "parameter_table"
    PIN_TABLE = "pin_table"
    TIMING_DIAGRAM = "timing_diagram"
    TEXT_ONLY = "text_only"


class ExtractionStrategy(Enum):
    """Table extraction strategy for a given page."""

    CAMELOT_LATTICE = "camelot_lattice"
    PDFPLUMBER = "pdfplumber"
    SKIP = "skip"


@dataclass
class TableExtractionResult:
    """Structured result for a single extracted table."""

    data: List[List[str]]
    page: int
    confidence: float
    source: str
    column_count: int
    row_count: int
    empty_cells: int
    bbox: tuple = None
    metadata: dict = None
    table_type: str = "unknown"
    header_keywords: List[str] = field(default_factory=list)
    is_cross_page: bool = False


@dataclass
class PageProfile:
    """Analysis result for a single PDF page (output of Layer 1)."""

    page_number: int
    text_density: float
    has_clear_borders: bool
    has_table_regions: bool
    page_type: PageType
    recommended_strategy: ExtractionStrategy


@dataclass
class DocumentProfile:
    """Pre-scan result for an entire PDF document."""

    total_pages: int
    page_profiles: List[PageProfile]


# ---------------------------------------------------------------------------
# Layer 1: Document-level pre-scanner
# ---------------------------------------------------------------------------

class DocumentPreScanner:
    """Single-pass PDF scanner that produces a PageProfile for every page.

    Design: "Scan-then-Extract" — open the PDF once, classify each page by
    text density, border detection, table-region detection, and keyword
    matching, then recommend an extraction strategy per page.

    Uses PyMuPDF (fitz) because ``get_drawings()`` provides direct access to
    line elements needed for border detection, and its page rendering is
    significantly faster than pdfplumber.
    """

    DATASHEET_KEYWORDS = {
        'parameter': [
            'parameter', 'symbol', 'min', 'max', 'typ', 'unit', 'condition', 'test',
            'value', 'rating', 'characteristics',
            '参数', '符号', '最小值', '最大值', '典型值', '单位', '条件', '测试',
            '数值', '额定值', '特性', '电气特性',
        ],
        'pin': [
            'pin', 'pinout', 'pin-out', 'pin configuration', 'pin description',
            'symbol', 'description',
            '引脚', '管脚', '引脚配置', '引脚描述', '功能',
        ],
        'electrical': [
            'electrical characteristics', 'recommended operating', 'absolute maximum',
            'dc characteristics', 'ac characteristics',
            '电气特性', '推荐工作条件', '绝对最大额定值', '直流特性', '交流特性',
            '工作条件', '绝对最大值',
        ],
        'timing': [
            'timing', 'timing diagram', 'waveform', 'clock',
            '时序', '时序图', '波形', '时钟',
        ],
        'cover': [
            'datasheet', 'rev', 'revision', 'preliminary', 'production',
            '数据手册', '版本', '修订版', '初步的', '生产版',
        ],
    }

    def scan_document(self, pdf_path: str) -> DocumentProfile:
        """Scan the entire PDF and return a DocumentProfile."""
        start_time = time.time()
        logger.info(f"开始文档级预扫描: {pdf_path}")

        try:
            import fitz
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count

            page_profiles = []
            table_page_count = 0

            for page_num in range(total_pages):
                profile = self._analyze_page(doc, page_num)
                page_profiles.append(profile)
                if profile.page_type in [PageType.PARAMETER_TABLE, PageType.PIN_TABLE]:
                    table_page_count += 1

            doc.close()

            document_profile = DocumentProfile(
                total_pages=total_pages,
                page_profiles=page_profiles,
            )

            scan_time = time.time() - start_time
            logger.info(
                f"文档预扫描完成: {total_pages}页, 含表页面{table_page_count}页, 耗时{scan_time:.2f}s"
            )
            return document_profile

        except Exception as e:
            logger.error(f"文档预扫描失败: {e}")
            raise

    def _analyze_page(self, doc, page_num: int) -> PageProfile:
        """Analyze a single page and return its PageProfile."""
        page = doc[page_num]

        text = page.get_text()
        rect = page.rect
        area = rect.width * rect.height
        text_density = len(text) / area if area > 0 else 0

        has_borders = False
        has_table_regions = False

        try:
            drawings = page.get_drawings()
            h_lines = sum(1 for d in drawings
                          if d.get("items") and len(d["items"]) > 0
                          and d["items"][0][0] == "l"
                          and abs(d["items"][0][1][3] - d["items"][0][1][1]) < 2)
            v_lines = sum(1 for d in drawings
                          if d.get("items") and len(d["items"]) > 0
                          and d["items"][0][0] == "l"
                          and abs(d["items"][0][1][2] - d["items"][0][1][0]) < 2)
            has_borders = h_lines >= 3 and v_lines >= 3
        except Exception:
            pass

        try:
            tables_found = page.find_tables()
            has_table_regions = len(tables_found.tables) > 0 if tables_found else False
        except Exception:
            pass

        page_type = self._classify_page_type(text, has_borders, has_table_regions)
        strategy = self._recommend_strategy(page_type, has_borders, has_table_regions, text_density)

        return PageProfile(
            page_number=page_num + 1,
            text_density=text_density,
            has_clear_borders=has_borders,
            has_table_regions=has_table_regions,
            page_type=page_type,
            recommended_strategy=strategy,
        )

    def _classify_page_type(self, text: str, has_borders: bool,
                            has_table_regions: bool) -> PageType:
        """Classify a page based on text content and structural features."""
        text_lower = text.lower()

        cover_kw = self.DATASHEET_KEYWORDS['cover']
        if any(kw in text_lower for kw in cover_kw) and len(text) < 500:
            return PageType.COVER

        if any(kw in text_lower for kw in ['contents', 'table of contents']):
            return PageType.TOC

        timing_kw = self.DATASHEET_KEYWORDS['timing']
        if any(kw in text_lower for kw in timing_kw):
            return PageType.TIMING_DIAGRAM

        if has_table_regions or has_borders:
            electrical_kw = self.DATASHEET_KEYWORDS['electrical']
            param_kw = self.DATASHEET_KEYWORDS['parameter']
            pin_kw = self.DATASHEET_KEYWORDS['pin']

            if any(kw in text_lower for kw in electrical_kw + param_kw):
                return PageType.PARAMETER_TABLE
            if any(kw in text_lower for kw in pin_kw):
                return PageType.PIN_TABLE

        if len(text) > 200 and not has_table_regions:
            return PageType.TEXT_ONLY

        return PageType.TEXT_ONLY

    def _recommend_strategy(self, page_type: PageType, has_borders: bool,
                            has_table_regions: bool, text_density: float) -> ExtractionStrategy:
        """Select the best extraction strategy for a page.

        Priority:
            1. Non-table pages → SKIP
            2. Borders + table regions → CAMELOT_LATTICE (strongest signal)
            3. Table regions + sufficient density → PDFPLUMBER
            4. Borders only → CAMELOT_LATTICE (region detection may have missed)
            5. Fallback → PDFPLUMBER (most general)
        """
        if page_type in [PageType.TOC, PageType.COVER, PageType.TEXT_ONLY,
                         PageType.TIMING_DIAGRAM]:
            return ExtractionStrategy.SKIP

        if has_borders and has_table_regions:
            return ExtractionStrategy.CAMELOT_LATTICE

        if has_table_regions and text_density > 0.0005:
            return ExtractionStrategy.PDFPLUMBER

        if has_borders:
            return ExtractionStrategy.CAMELOT_LATTICE

        return ExtractionStrategy.PDFPLUMBER


# ---------------------------------------------------------------------------
# Layer 2: Intelligent extraction router
# ---------------------------------------------------------------------------

class IntelligentRouter:
    """Strategy-based table extractor with automatic fallback.

    Routes each page to the recommended extractor (Camelot or pdfplumber)
    based on the Layer 1 PageProfile.  When Camelot fails, automatically
    falls back to pdfplumber (but not vice-versa, since pdfplumber is the
    more general extractor).
    """

    def __init__(self, settings=None):
        self.settings = settings
        self._camelot_available = self._check_camelot()
        self._pdfplumber_available = self._check_pdfplumber()

    def _check_camelot(self) -> bool:
        try:
            import camelot  # noqa: F401
            return True
        except ImportError:
            logger.warning("Camelot未安装，将使用pdfplumber")
            return False

    def _check_pdfplumber(self) -> bool:
        try:
            import pdfplumber  # noqa: F401
            return True
        except ImportError:
            logger.warning("pdfplumber未安装")
            return False

    def extract_with_strategy(self, pdf_path: str,
                              page_profile: PageProfile) -> List[TableExtractionResult]:
        """Extract tables from a page using the recommended strategy.

        Fallback: CAMELOT_LATTICE → PDFPLUMBER on failure.
        """
        strategy = page_profile.recommended_strategy

        if strategy == ExtractionStrategy.SKIP:
            logger.debug(f"页面 {page_profile.page_number} 跳过提取")
            return []

        results = []

        if strategy == ExtractionStrategy.CAMELOT_LATTICE:
            results = self._extract_camelot_lattice(pdf_path, page_profile)
            if not results and self._pdfplumber_available:
                logger.info(f"Camelot提取失败，降级到pdfplumber: 页面 {page_profile.page_number}")
                results = self._extract_pdfplumber(pdf_path, page_profile)

        elif strategy == ExtractionStrategy.PDFPLUMBER:
            results = self._extract_pdfplumber(pdf_path, page_profile)

        return results

    def _extract_camelot_lattice(self, pdf_path: str,
                                 page_profile: PageProfile) -> List[TableExtractionResult]:
        """Extract tables using Camelot's lattice mode (line-border detection)."""
        if not self._camelot_available:
            return []

        try:
            import camelot

            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_profile.page_number),
                flavor='lattice',
                suppress_stdout=True,
            )

            results = []
            for i, table in enumerate(tables):
                df = table.df
                data = df.values.tolist()

                empty_cells = sum(1 for row in data for cell in row if not str(cell).strip())
                accuracy = table.accuracy / 100.0 if hasattr(table, 'accuracy') else 0.8

                result = TableExtractionResult(
                    data=data,
                    page=page_profile.page_number,
                    confidence=accuracy,
                    source="camelot_lattice",
                    column_count=len(df.columns),
                    row_count=len(df),
                    empty_cells=empty_cells,
                    bbox=table._bbox if hasattr(table, '_bbox') else None,
                    metadata={"flavor": "lattice", "table_index": i},
                )
                results.append(result)

            logger.info(f"Camelot lattice提取页面 {page_profile.page_number}: {len(results)}个表格")
            return results

        except Exception as e:
            logger.warning(f"Camelot lattice提取失败: {e}")
            return []

    def _extract_pdfplumber(self, pdf_path: str,
                            page_profile: PageProfile) -> List[TableExtractionResult]:
        """Extract tables using pdfplumber (text-alignment inference)."""
        if not self._pdfplumber_available:
            return []

        try:
            import pdfplumber

            results = []
            with pdfplumber.open(pdf_path) as pdf:
                if page_profile.page_number > len(pdf.pages):
                    return []

                page = pdf.pages[page_profile.page_number - 1]

                table_settings = {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 10,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 2,
                }

                tables = page.extract_tables(table_settings)

                for i, table_data in enumerate(tables):
                    if table_data:
                        results.append(self._create_result(
                            table_data, page_profile.page_number, i, None, "pdfplumber"
                        ))

            logger.info(f"pdfplumber提取页面 {page_profile.page_number}: {len(results)}个表格")
            return results

        except Exception as e:
            logger.warning(f"pdfplumber提取失败: {e}")
            return []

    def _create_result(self, data: List[List], page_num: int, index: int,
                       bbox: tuple, source: str) -> TableExtractionResult:
        """Create a TableExtractionResult from raw table data.

        Confidence formula: 0.75 * (1 - empty_ratio * 0.5)
        """
        cleaned_data = []
        for row in data:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_data.append(cleaned_row)

        empty_cells = sum(1 for row in cleaned_data for cell in row if not cell)
        total_cells = len(cleaned_data) * len(cleaned_data[0]) if cleaned_data and cleaned_data[0] else 1
        empty_ratio = empty_cells / total_cells
        confidence = 0.75 * (1 - empty_ratio * 0.5)

        return TableExtractionResult(
            data=cleaned_data,
            page=page_num,
            confidence=confidence,
            source=source,
            column_count=len(cleaned_data[0]) if cleaned_data else 0,
            row_count=len(cleaned_data),
            empty_cells=empty_cells,
            bbox=bbox,
            metadata={"table_index": index},
        )


# ---------------------------------------------------------------------------
# Layer 3: Datasheet-specific post-extraction optimizer
# ---------------------------------------------------------------------------

class DatasheetOptimizer:
    """Post-extraction optimizer for datasheet tables.

    Pipeline (in order):
        1. Table type classification
        2. Header keyword extraction
        3. Quality score calculation
        4. Cross-page table merging
        5. Pin sequence validation
        6. Duplicate table removal
    """

    TABLE_TYPE_PATTERNS = {
        'electrical_characteristics': [
            r'electrical\s*characteristics',
            r'dc\s*characteristics',
            r'ac\s*characteristics',
            r'recommended\s*operating',
        ],
        'pin_configuration': [
            r'pin\s*configuration',
            r'pin\s*description',
            r'pinout',
            r'pin-out',
        ],
        'absolute_maximum': [
            r'absolute\s*maximum',
        ],
        'timing': [
            r'timing\s*characteristics',
            r'switching\s*characteristics',
        ],
        'package_dimensions': [
            r'package\s*(information|dimensions|outline)',
        ],
    }

    HEADER_KEYWORDS = {
        'parameter': ['parameter', 'symbol', 'param', 'characteristic', '参数', '符号'],
        'value': ['min', 'max', 'typ', 'typical', 'minimum', 'maximum',
                  '最小', '最大', '典型'],
        'unit': ['unit', 'units', '单位'],
        'condition': ['condition', 'test condition', 'note', '条件', '测试条件'],
        'pin': ['pin', 'pinout', '引脚', '管脚'],
        'description': ['description', 'function', '描述', '功能'],
    }

    def optimize_results(self, results: List[TableExtractionResult],
                         doc_profile: DocumentProfile) -> List[TableExtractionResult]:
        """Run the full optimization pipeline on extracted results."""
        if not results:
            return []

        for result in results:
            self._classify_table_type(result)
            self._extract_header_keywords(result)
            self._calculate_quality_score(result)

        results = self._merge_cross_page_tables(results, doc_profile)
        results = self._validate_pin_sequence(results)
        results = self._deduplicate_tables(results)

        return results

    def _classify_table_type(self, result: TableExtractionResult) -> TableExtractionResult:
        """Classify a table into a known datasheet table type.

        Strategy:
            1. Detect package_dimensions first (low-value, suppress confidence)
            2. Match section-title patterns via regex
            3. Fallback: infer from header column names
        """
        if not result.data:
            return result

        header_text = " ".join(str(cell).lower() for cell in result.data[0]) if result.data else ""
        first_col_text = " ".join(
            str(row[0]).lower() for row in result.data[1:5] if row
        ) if len(result.data) > 1 else ""

        combined_text = header_text + " " + first_col_text

        package_keywords = [
            'seating plane', 'package dimensions', 'mechanical data', 'package information',
            'outline', 'drawing', 'mm', '[', ']', 'typ', 'pin 1 id area',
            'mechanical', 'package', 'dimensions (mm)', 'inches', 'millimeters',
        ]

        if any(kw in combined_text for kw in package_keywords):
            result.table_type = 'package_dimensions'
            result.metadata = result.metadata or {}
            result.metadata['table_type'] = 'package_dimensions'
            result.confidence = min(result.confidence, 0.3)
            return result

        for table_type, patterns in self.TABLE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    result.table_type = table_type
                    result.metadata = result.metadata or {}
                    result.metadata['table_type'] = table_type
                    return result

        if any(kw in header_text for kw in ['pin', 'symbol', 'description']):
            result.table_type = 'pin_configuration'
        elif any(kw in header_text for kw in ['parameter', 'min', 'max', 'typ']):
            result.table_type = 'electrical_characteristics'

        return result

    def _extract_header_keywords(self, result: TableExtractionResult) -> TableExtractionResult:
        """Extract semantic keywords from the header row.

        Format: "category:keyword", e.g. "parameter:min", "value:max", "unit:V".
        Semantic normalization maps synonyms (Min/Minimum/min) to a single key.
        """
        if not result.data or not result.data[0]:
            return result

        header = result.data[0]
        keywords_found = []

        for cell in header:
            cell_lower = str(cell).lower().strip()
            for category, kws in self.HEADER_KEYWORDS.items():
                for kw in kws:
                    if kw in cell_lower:
                        keywords_found.append(f"{category}:{kw}")
                        break

        result.header_keywords = keywords_found

        if keywords_found:
            result.confidence = min(result.confidence + 0.05, 1.0)

        return result

    def _calculate_quality_score(self, result: TableExtractionResult) -> TableExtractionResult:
        """Compute a multi-dimensional quality score and update confidence.

        Formula:
            quality_score = fill_rate * 0.4
                          + header_quality * 0.3
                          + confidence * 0.2
                          + type_bonus

        Weights: fill_rate (40%), header_quality (30%), original confidence (20%),
        type_bonus (10%).  Weights sum to 1.0, keeping score in [0, 1].
        """
        if not result.data:
            return result

        total_cells = result.row_count * result.column_count
        if total_cells == 0:
            return result

        fill_rate = 1 - (result.empty_cells / total_cells)
        header_quality = 1.0 if result.header_keywords else 0.5
        type_bonus = 0.1 if result.table_type != 'unknown' else 0

        quality_score = (
            fill_rate * 0.4
            + header_quality * 0.3
            + result.confidence * 0.2
            + type_bonus
        )

        result.confidence = min(quality_score, 1.0)
        return result

    def _merge_cross_page_tables(self, results: List[TableExtractionResult],
                                 doc_profile: DocumentProfile) -> List[TableExtractionResult]:
        """Merge cross-page continuation tables using greedy clustering.

        Two tables are merged when all of the following hold:
            1. Adjacent pages (|page1 - page2| == 1)
            2. Same table type (not 'unknown')
            3. Column counts differ by at most 1
            4. Header Jaccard similarity > 0.7
        """
        if len(results) < 2:
            return results

        merged_results = []
        used_indices = set()
        group_id = 0

        for i, table1 in enumerate(results):
            if i in used_indices:
                continue

            cluster = [table1]
            used_indices.add(i)

            for j, table2 in enumerate(results):
                if j in used_indices or j == i:
                    continue
                if self._should_merge_tables(table1, table2):
                    cluster.append(table2)
                    used_indices.add(j)

            if len(cluster) > 1:
                merged = self._merge_table_cluster(cluster, group_id)
                merged_results.append(merged)
                group_id += 1
            else:
                merged_results.append(table1)

        return merged_results

    def _should_merge_tables(self, table1: TableExtractionResult,
                             table2: TableExtractionResult) -> bool:
        """Return True if two tables should be merged as a cross-page continuation."""
        if abs(table1.page - table2.page) != 1:
            return False

        if table1.table_type != table2.table_type:
            return False

        if table1.table_type == 'unknown':
            return False

        if abs(table1.column_count - table2.column_count) > 1:
            return False

        if table1.data and table2.data:
            header1 = set(str(cell).lower() for cell in table1.data[0])
            header2 = set(str(cell).lower() for cell in table2.data[0])

            if header1 == header2 and len(header1) > 0:
                return True

            similarity = len(header1 & header2) / max(len(header1 | header2), 1)
            return similarity > 0.7

        return False

    def _merge_table_cluster(self, cluster: List[TableExtractionResult],
                             group_id: int) -> TableExtractionResult:
        """Merge a cluster of cross-page tables into a single table.

        Keeps only the first header row; subsequent tables skip their header.
        Confidence is set to the maximum in the cluster.
        """
        sorted_cluster = sorted(cluster, key=lambda t: t.page)

        merged_data = []
        header_added = False

        for table in sorted_cluster:
            if not header_added and table.data:
                merged_data.append(table.data[0])
                header_added = True

            if table.data:
                start_idx = 1 if header_added else 0
                merged_data.extend(table.data[start_idx:])

        first_table = sorted_cluster[0]

        merged = TableExtractionResult(
            data=merged_data,
            page=first_table.page,
            confidence=max(t.confidence for t in cluster),
            source=f"merged_{len(cluster)}_tables",
            column_count=first_table.column_count,
            row_count=len(merged_data),
            empty_cells=sum(1 for row in merged_data for cell in row if not str(cell).strip()),
            bbox=first_table.bbox,
            metadata={
                "merged_from_pages": [t.page for t in sorted_cluster],
                "merge_group": group_id,
            },
            table_type=first_table.table_type,
            header_keywords=first_table.header_keywords,
            is_cross_page=True,
        )

        logger.info(f"合并跨页表格: 页面 {[t.page for t in sorted_cluster]} -> {merged.row_count}行")
        return merged

    def _validate_pin_sequence(self, results: List[TableExtractionResult]) -> List[TableExtractionResult]:
        """Validate pin-number continuity in pin_configuration tables.

        Gaps in the pin sequence reduce confidence proportionally:
            confidence *= (1 - gap_ratio * 0.5)
        """
        for result in results:
            if result.table_type != 'pin_configuration':
                continue

            if not result.data or len(result.data) < 2:
                continue

            pin_numbers = []
            for row in result.data[1:]:
                if row:
                    first_cell = str(row[0]).strip()
                    match = re.match(r'^(\d+)$', first_cell)
                    if match:
                        pin_numbers.append(int(match.group(1)))

            if pin_numbers:
                pin_numbers.sort()
                gap_count = sum(
                    1 for i in range(1, len(pin_numbers))
                    if pin_numbers[i] - pin_numbers[i - 1] > 1
                )
                if gap_count > 0:
                    gap_ratio = gap_count / len(pin_numbers)
                    result.confidence *= (1 - gap_ratio * 0.5)
                    logger.debug(
                        f"引脚序列检测到{gap_count}个间隙，置信度调整为{result.confidence:.2f}"
                    )

        return results

    def _deduplicate_tables(self, results: List[TableExtractionResult]) -> List[TableExtractionResult]:
        """Remove duplicate tables based on content signatures."""
        if len(results) < 2:
            return results

        unique_results = []
        seen_signatures = set()

        for result in results:
            signature = self._calculate_table_signature(result)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_results.append(result)
            else:
                logger.debug(f"去除重复表格: 页面 {result.page}")

        return unique_results

    def _calculate_table_signature(self, result: TableExtractionResult) -> str:
        """Compute a deduplication signature: page_colCount_rowCount_firstRow_lastRow."""
        if not result.data:
            return f"empty_{result.page}"

        first_row = tuple(str(cell)[:20] for cell in result.data[0]) if result.data[0] else ()
        last_row = tuple(str(cell)[:20] for cell in result.data[-1]) if result.data[-1] else ()

        return f"{result.page}_{result.column_count}_{result.row_count}_{first_row}_{last_row}"


# ---------------------------------------------------------------------------
# Facade: main entry point with LRU caching
# ---------------------------------------------------------------------------

class DatasheetTableExtractor:
    """Facade that wires the three layers together and adds LRU caching.

    Usage::

        extractor = DatasheetTableExtractor()
        results = extractor.extract("path/to/datasheet.pdf")

    Cache key = pdf_path + mtime + page_range, so file overwrites invalidate
    stale entries.  Maximum cache size is 32 entries.
    """

    MAX_CACHE_SIZE = 32

    def __init__(self, settings=None):
        self.settings = settings
        self.pre_scanner = DocumentPreScanner()
        self.router = IntelligentRouter(settings)
        self.optimizer = DatasheetOptimizer()

        self._result_cache: Dict[str, List[TableExtractionResult]] = {}
        self._cache_order: List[str] = []

    def _get_cache_key(self, pdf_path: str, pages: List[int] = None) -> str:
        """Build a cache key from path, mtime, and page range."""
        import os
        try:
            mtime = os.path.getmtime(pdf_path)
        except OSError:
            mtime = 0
        pages_str = ','.join(map(str, pages)) if pages else 'all'
        return f"{pdf_path}_{mtime}_{pages_str}"

    def _evict_cache_if_needed(self):
        """Evict oldest cache entries when the cache exceeds MAX_CACHE_SIZE."""
        while len(self._result_cache) > self.MAX_CACHE_SIZE:
            oldest_key = self._cache_order.pop(0)
            self._result_cache.pop(oldest_key, None)
            logger.debug(f"缓存淘汰: {oldest_key}")

    def extract(self, pdf_path: str, pages: List[int] = None) -> List[TableExtractionResult]:
        """Extract and optimize tables from a datasheet PDF.

        Args:
            pdf_path: Absolute path to the PDF file.
            pages: Optional list of 1-based page numbers to extract.
                   None means extract all non-SKIP pages.

        Returns:
            Optimized table extraction results.
        """
        start_time = time.time()

        cache_key = self._get_cache_key(pdf_path, pages)
        if cache_key in self._result_cache:
            if cache_key in self._cache_order:
                self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            logger.info(f"从缓存获取结果: {pdf_path}")
            return self._result_cache[cache_key]

        logger.info(f"开始提取表格: {pdf_path}")

        doc_profile = self.pre_scanner.scan_document(pdf_path)

        if pages:
            page_profiles = [
                p for p in doc_profile.page_profiles if p.page_number in pages
            ]
        else:
            page_profiles = [
                p for p in doc_profile.page_profiles
                if p.recommended_strategy != ExtractionStrategy.SKIP
            ]

        all_results = []
        for page_profile in page_profiles:
            page_results = self.router.extract_with_strategy(pdf_path, page_profile)
            all_results.extend(page_results)

        all_results = self.optimizer.optimize_results(all_results, doc_profile)

        self._evict_cache_if_needed()
        self._result_cache[cache_key] = all_results
        self._cache_order.append(cache_key)

        elapsed_time = time.time() - start_time
        logger.info(f"表格提取完成: {len(all_results)}个表格, 耗时{elapsed_time:.2f}s")

        return all_results
