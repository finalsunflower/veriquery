"""
Document processor for the VeriQuery ingestion layer.

This module handles PDF document parsing, image extraction with CLIP-based
filtering, chart noise cleaning, table extraction orchestration, and visual
embedding creation. It serves as the entry point for all documents before
they enter the retrieval and reasoning pipeline.

Main components:
    - PageContent / ParseResult: Data containers for per-page and per-document results
    - _PDFParser: Internal PDF parsing engine (PyMuPDF + CLIP filtering)
    - EnhancedDocumentProcessor: Core orchestrator (parse + tables + visual embeddings)
    - ProcessingResult: Unified result type for downstream API consumption
    - IngestionPipeline: External entry point with document type detection
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import hashlib
import time
import fitz
import io
import re

from core.config import get_settings
from core.exceptions import ProcessingError
from extraction.table_extractor import DatasheetTableExtractor
from .image_indexer import create_visual_indexer

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Parsed result for a single PDF page.

    Attributes:
        page_number: 1-based page number.
        text: Cleaned text content (chart noise removed).
        tables: Table list (empty at parse stage; filled by table_extractor).
        images: Image info dicts with keys: image_id, source, data, is_circuit, filter_reason.
        metadata: Page-level metadata (table_count, image_count, text_length, page_width/height/area).
        has_error: Whether this page failed to parse.
        error_message: Error detail when has_error is True.
    """

    page_number: int
    text: str
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    has_error: bool = False
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.tables is None:
            self.tables = []
        if self.images is None:
            self.images = []
        if self.metadata is None:
            self.metadata = {}

    def get_pil_image(self):
        """Return the PIL Image for this page, or None if unavailable.

        Uses a two-level fallback: cached _pil_image → first image's raw data → None.
        PIL Image is stored as a private attribute rather than a dataclass field
        because it is not serializable and consumes significant memory.
        """
        try:
            from PIL.Image import Image as PILImage

            if hasattr(self, '_pil_image') and self._pil_image:
                return self._pil_image

            if self.images:
                first_image = self.images[0]
                if 'data' in first_image:
                    return PILImage.open(io.BytesIO(first_image['data']))

            return None

        except Exception as e:
            logger.warning(f"获取页面{self.page_number}图片失败: {e}")
            return None

    def set_pil_image(self, pil_image):
        """Cache a PIL Image for this page (used by visual embedding later)."""
        self._pil_image = pil_image


@dataclass
class ParseResult:
    """Parsed result for an entire PDF document.

    Attributes:
        document_id: MD5 hash of the file path.
        filename: Original file name.
        page_count: Total number of pages.
        pages: List of PageContent objects.
        tables: Aggregated table list (empty at parse stage).
        images: Aggregated image info list.
        metadata: Document-level metadata (title, author, dates, etc.).
        status: "success" | "partial" | "error".
        error_message: Error detail when status is "error".
        warnings: Warning messages.
        error_pages: 1-based page numbers that failed to parse.
        image_count: Total number of extracted images.
    """

    document_id: str
    filename: str
    page_count: int
    pages: List[PageContent]
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    status: str = "success"
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    error_pages: List[int] = field(default_factory=list)
    image_count: int = 0

    def __post_init__(self):
        if self.pages is None:
            self.pages = []
        if self.tables is None:
            self.tables = []
        if self.images is None:
            self.images = []
        if self.metadata is None:
            self.metadata = {}


class _PDFParser:
    """Internal PDF parsing engine based on PyMuPDF (fitz).

    Extracts text, bitmap images, and vector drawings from each page.
    Vector drawings are rendered to images and filtered via CLIP to retain
    only circuit diagrams / schematics. Chart noise in extracted text is
    cleaned using regex-based pattern matching.

    The underscore prefix marks this as a module-internal class not exported
    via __init__.py; it is used exclusively by EnhancedDocumentProcessor.
    """

    _figure_pattern = re.compile(r'Figure\s+\d+', re.IGNORECASE)
    _axis_label_pattern = re.compile(r'\(V\)\s*$')
    _layout_code_pattern = re.compile(r'[A-Z]\d{3}\s*')

    def __init__(self, settings=None, visual_indexer_provider=None):
        """Initialize the PDF parser.

        Args:
            settings: Configuration object (defaults to get_settings()).
            visual_indexer_provider: Factory function for lazy initialization of
                the visual indexer. Deferred to avoid loading GPU models at startup.
        """
        self.settings = settings or get_settings()
        self.clip_threshold = self.settings.CLIP_THRESHOLD
        self.render_dpi = self.settings.PDF_RENDER_DPI
        self._visual_indexer_provider = visual_indexer_provider

    def _cluster_drawings_by_y(self, drawings: list, page_rect,
                               gap_threshold: float = 80.0,
                               min_drawings: int = 3) -> List[Tuple[Any, list]]:
        """Cluster vector drawings by Y-axis position into distinct regions.

        Whole-page rendering can cause CLIP to misclassify when text/tables
        dominate the page. By clustering drawings into separate regions, each
        region can be rendered and classified independently.

        Algorithm:
            1. Bucket drawings by Y-midpoint (bucket width = 50pt).
            2. Merge adjacent buckets within gap_threshold into groups.
            3. Compute a clipping rectangle (with margin) for each group.
            4. Filter out groups with fewer than min_drawings elements.

        Args:
            drawings: List of drawing dicts from page.get_drawings().
            page_rect: Page rectangle (fitz.Rect).
            gap_threshold: Max Y-gap to consider buckets adjacent (default 80pt).
            min_drawings: Minimum drawings per region (default 3).

        Returns:
            List of (clip_rect, region_drawings) tuples.
        """
        if not drawings:
            return []

        if len(drawings) <= 5:
            all_x0 = min(d['rect'].x0 for d in drawings)
            all_y0 = min(d['rect'].y0 for d in drawings)
            all_x1 = max(d['rect'].x1 for d in drawings)
            all_y1 = max(d['rect'].y1 for d in drawings)
            margin = 20
            clip_rect = fitz.Rect(
                max(0, all_x0 - margin),
                max(0, all_y0 - margin),
                min(page_rect.width, all_x1 + margin),
                min(page_rect.height, all_y1 + margin)
            )
            return [(clip_rect, drawings)]

        bucket_width = 50
        y_buckets: Dict[int, list] = {}
        for d in drawings:
            r = d['rect']
            y_mid = int((r.y0 + r.y1) / 2 / bucket_width) * bucket_width
            if y_mid not in y_buckets:
                y_buckets[y_mid] = []
            y_buckets[y_mid].append(d)

        sorted_buckets = sorted(y_buckets.items())
        merged_groups = []
        current_group = [sorted_buckets[0]]

        for i in range(1, len(sorted_buckets)):
            prev_y = current_group[-1][0]
            curr_y = sorted_buckets[i][0]
            if curr_y - prev_y <= gap_threshold:
                current_group.append(sorted_buckets[i])
            else:
                merged_groups.append(current_group)
                current_group = [sorted_buckets[i]]
        merged_groups.append(current_group)

        regions = []
        for group in merged_groups:
            group_drawings = []
            for _, bucket_drawings in group:
                group_drawings.extend(bucket_drawings)

            if len(group_drawings) < min_drawings:
                continue

            all_x0 = min(d['rect'].x0 for d in group_drawings)
            all_y0 = min(d['rect'].y0 for d in group_drawings)
            all_x1 = max(d['rect'].x1 for d in group_drawings)
            all_y1 = max(d['rect'].y1 for d in group_drawings)

            margin = 20
            clip_rect = fitz.Rect(
                max(0, all_x0 - margin),
                max(0, all_y0 - margin),
                min(page_rect.width, all_x1 + margin),
                min(page_rect.height, all_y1 + margin)
            )
            regions.append((clip_rect, group_drawings))

        if not regions:
            all_x0 = min(d['rect'].x0 for d in drawings)
            all_y0 = min(d['rect'].y0 for d in drawings)
            all_x1 = max(d['rect'].x1 for d in drawings)
            all_y1 = max(d['rect'].y1 for d in drawings)
            margin = 20
            clip_rect = fitz.Rect(
                max(0, all_x0 - margin),
                max(0, all_y0 - margin),
                min(page_rect.width, all_x1 + margin),
                min(page_rect.height, all_y1 + margin)
            )
            regions.append((clip_rect, drawings))

        return regions

    def _get_render_matrix(self, base_dpi: float = 72.0) -> fitz.Matrix:
        """Compute the scaling matrix for PDF page rendering.

        PDF pages use 72 DPI by default; this scales to the configured
        render_dpi (default 200) for higher-resolution output.

        Args:
            base_dpi: PDF native DPI (standard: 72).

        Returns:
            fitz.Matrix for use with page.get_pixmap(matrix=...).
        """
        scale = self.render_dpi / base_dpi
        return fitz.Matrix(scale, scale)

    def parse(self, filepath: str, document_id: str) -> ParseResult:
        """Parse a PDF file and return structured results.

        Processes each page independently: extracts text (with noise cleaning),
        images (bitmap + vector drawings with CLIP filtering), and metadata.
        Per-page failures are captured gracefully without aborting the entire
        document parse.

        Args:
            filepath: Absolute path to the PDF file.
            document_id: Unique document identifier (MD5 hash).

        Returns:
            ParseResult with status "success", "partial", or "error".
        """
        doc = None
        try:
            doc = fitz.open(filepath)

            filename = Path(filepath).name
            page_count = doc.page_count

            logger.info(f"开始解析PDF: {filename}, 页数: {page_count}")

            pages = []
            all_tables = []
            all_images = []
            error_pages = []
            warnings = []

            for page_num in range(page_count):
                try:
                    page = doc.load_page(page_num)

                    raw_text = page.get_text()
                    text = self._clean_chart_noise(raw_text)
                    page_tables = []

                    page_images, rendered_pixmap = self._extract_images(
                        page, document_id, filename, page_num
                    )
                    all_images.extend(page_images)

                    if rendered_pixmap is not None:
                        pil_image = self._pixmap_to_pil(rendered_pixmap, cleanup=True)
                        rendered_pixmap = None
                    elif page_images:
                        pil_image = self._render_page_as_image(page)
                    else:
                        pil_image = None

                    page_rect = page.rect
                    page_content = PageContent(
                        page_number=page_num + 1,
                        text=text,
                        tables=page_tables,
                        images=page_images,
                        metadata={
                            "table_count": len(page_tables),
                            "image_count": len(page_images),
                            "text_length": len(text),
                            "page_width": float(page_rect.width),
                            "page_height": float(page_rect.height),
                            "page_area": float(page_rect.width * page_rect.height)
                        }
                    )

                    if pil_image:
                        page_content.set_pil_image(pil_image)

                    pages.append(page_content)

                    logger.debug(f"解析页面 {page_num + 1}: 图像数 {len(page_images)}")

                except Exception as e:
                    logger.error(f"解析页面 {page_num + 1} 失败: {e}")
                    error_pages.append(page_num + 1)
                    warnings.append(f"页面 {page_num + 1} 解析失败: {str(e)}")

                    error_page = PageContent(
                        page_number=page_num + 1,
                        text="",
                        tables=[],
                        images=[],
                        metadata={"error": str(e)},
                        has_error=True,
                        error_message=str(e)
                    )
                    pages.append(error_page)

            doc_metadata = self._extract_metadata(doc)

            if error_pages:
                if len(error_pages) == page_count:
                    status = "error"
                else:
                    status = "partial"
            else:
                status = "success"

            result = ParseResult(
                document_id=document_id,
                filename=filename,
                page_count=page_count,
                pages=pages,
                tables=all_tables,
                images=all_images,
                metadata=doc_metadata,
                status=status,
                warnings=warnings,
                error_pages=error_pages,
                image_count=len(all_images)
            )

            logger.info(f"PDF解析完成: {filename}, 状态: {status}, 总图像数: {len(all_images)}, 错误页面: {len(error_pages)}")

            return result

        except Exception as e:
            logger.error(f"PDF解析失败 {filepath}: {e}")
            return ParseResult(
                document_id=document_id,
                filename=Path(filepath).name,
                page_count=0,
                pages=[],
                tables=[],
                images=[],
                metadata={"error": str(e), "error_type": type(e).__name__},
                status="error",
                error_message=str(e),
                warnings=[f"PDF解析完全失败: {str(e)}"],
                error_pages=[]
            )
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as e:
                    logger.warning(f"关闭PDF文档失败: {e}")

    def _extract_images(self, page, document_id: str, filename: str,
                        page_num: int) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
        """Extract all images from a PDF page via dual-channel approach.

        Channel 1 - Bitmap extraction:
            page.get_images() retrieves embedded raster images directly.

        Channel 2 - Vector drawing extraction:
            page.get_drawings() detects vector instructions, which are rendered
            to images and classified via CLIP. Only circuit diagrams / schematics
            are retained; chip packages, pin diagrams, etc. are filtered out.

        Args:
            page: PyMuPDF Page object.
            document_id: Document unique ID (for image_id generation).
            filename: File name (recorded in image info).
            page_num: 0-based page index (for image_id generation).

        Returns:
            Tuple of (image_info_list, rendered_pixmap). The pixmap is returned
            for external reuse when vector drawings were rendered.
        """
        images = []
        rendered_pixmap = None

        try:
            image_list = page.get_images(full=True)
            logger.debug(f"页面 {page_num + 1} get_images返回: {len(image_list)}个位图图像")

            for i, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)

                    # Skip CMYK color space (requires conversion; tobytes("png") would fail)
                    if pix.n - pix.alpha < 4:
                        image_id = f"{document_id}_image_{page_num}_{i}"
                        image_bytes = pix.tobytes("png")

                        pil_image = self._pixmap_to_pil(pix, cleanup=False)
                        is_circuit = None
                        filter_reason = ""
                        clip_type = None
                        clip_confidence = 0.0

                        if pil_image and pix.width >= 100 and pix.height >= 100:
                            filter_result, confidence, reason = self._classify_image(pil_image)
                            is_circuit = (filter_result == "PASS")
                            filter_reason = reason
                            clip_type = self._last_clip_type if hasattr(self, '_last_clip_type') else None
                            clip_confidence = confidence

                            if filter_result == "REJECT" and confidence > 0.70:
                                logger.debug(f"页面 {page_num + 1} 位图{i} CLIP过滤: {reason}")
                                del pix
                                if pil_image:
                                    pil_image.close()
                                continue
                            elif filter_result == "REJECT":
                                is_circuit = None

                        image_info = {
                            "image_id": image_id,
                            "document_id": document_id,
                            "filename": filename,
                            "page": page_num + 1,
                            "xref": xref,
                            "width": pix.width,
                            "height": pix.height,
                            "colorspace": pix.colorspace.name if pix.colorspace else "unknown",
                            "source": "pymupdf_bitmap",
                            "data": image_bytes,
                            "is_circuit": is_circuit,
                            "filter_reason": filter_reason,
                            "clip_type": clip_type,
                            "clip_confidence": clip_confidence,
                        }

                        images.append(image_info)
                        if pil_image:
                            pil_image.close()

                    del pix

                except Exception as e:
                    logger.warning(f"提取位图图像失败 (页面 {page_num + 1}, 图像 {i}): {e}")

            drawings = page.get_drawings()
            logger.debug(f"页面 {page_num + 1} get_drawings返回: {len(drawings)}个矢量绘图")

            if drawings:
                try:
                    drawing_regions = self._cluster_drawings_by_y(drawings, page.rect)
                    logger.debug(f"页面 {page_num + 1} 矢量绘图聚类为{len(drawing_regions)}个区域")

                    for region_idx, (clip_rect, region_drawings) in enumerate(drawing_regions):
                        try:
                            pix = page.get_pixmap(
                                matrix=self._get_render_matrix(),
                                clip=clip_rect,
                                alpha=False
                            )
                            pil_image = self._pixmap_to_pil(pix, cleanup=False)
                            if pil_image is None:
                                logger.warning(f"页面 {page_num + 1} 区域{region_idx}矢量绘图转PIL失败")
                                del pix
                                continue

                            filter_result, confidence, filter_reason = self._classify_image(pil_image)
                            logger.debug(
                                f"页面 {page_num + 1} 区域{region_idx} CLIP分类: "
                                f"{filter_result}, 置信度: {confidence:.3f}, 原因: {filter_reason}"
                            )

                            image_id = f"{document_id}_drawing_{page_num}_{region_idx}"
                            image_bytes = pix.tobytes("png")

                            is_circuit = (filter_result == "PASS")

                            image_info = {
                                "image_id": image_id,
                                "document_id": document_id,
                                "filename": filename,
                                "page": page_num + 1,
                                "xref": 0,
                                "width": pix.width,
                                "height": pix.height,
                                "colorspace": pix.colorspace.name if pix.colorspace else "unknown",
                                "source": "pymupdf_drawing",
                                "data": image_bytes,
                                "is_circuit": is_circuit,
                                "filter_reason": filter_reason,
                                "clip_confidence": confidence,
                                "filter_result": filter_result,
                                "drawing_count": len(region_drawings),
                            }

                            if filter_result == "PASS":
                                images.append(image_info)
                                logger.info(
                                    f"页面 {page_num + 1} 区域{region_idx} 矢量绘图: "
                                    f"{pix.width}x{pix.height} [PASS - {filter_reason}]"
                                )
                            else:
                                logger.info(
                                    f"页面 {page_num + 1} 区域{region_idx} 矢量绘图: "
                                    f"[REJECT - {filter_reason}]"
                                )

                            if region_idx == 0:
                                rendered_pixmap = pix
                            else:
                                del pix

                        except Exception as region_err:
                            logger.warning(f"渲染矢量绘图区域{region_idx}失败 (页面 {page_num + 1}): {region_err}")

                except Exception as e:
                    logger.warning(f"渲染矢量绘图失败 (页面 {page_num + 1}): {e}")

        except Exception as e:
            logger.warning(f"页面 {page_num + 1} 图像提取失败: {e}")

        logger.debug(
            f"页面 {page_num + 1} 提取图片: "
            f"共{len(image_list)}个位图，"
            f"{len(drawings) if 'drawings' in locals() else 0}个绘图，"
            f"保留{len(images)}个图像"
        )
        return images, rendered_pixmap

    def _extract_metadata(self, doc) -> Dict[str, Any]:
        """Extract PDF document metadata (title, author, dates, etc.)."""
        try:
            metadata = doc.metadata
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "format": metadata.get("format", "PDF"),
                "encryption": metadata.get("encryption", None)
            }
        except Exception as e:
            logger.warning(f"提取元数据失败: {e}")
            return {"error": str(e)}

    def _pixmap_to_pil(self, pix, cleanup: bool = True) -> Optional[Any]:
        """Convert a PyMuPDF Pixmap to a PIL Image.

        The conversion path is: Pixmap → PNG bytes → PIL Image.
        When cleanup=True, the Pixmap is deleted after conversion to release
        C-level memory (PyMuPDF Pixmap objects are not fully managed by
        Python's garbage collector).

        Args:
            pix: PyMuPDF Pixmap object.
            cleanup: Whether to delete the Pixmap after conversion.

        Returns:
            PIL.Image.Image, or None on failure.
        """
        try:
            from PIL import Image

            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))

            if cleanup:
                del pix

            return pil_image

        except Exception as e:
            logger.warning(f"Pixmap转PIL图片失败: {e}")
            if cleanup and 'pix' in locals():
                del pix
            return None

    def _render_page_as_image(self, page) -> Optional[Any]:
        """Render an entire PDF page as a PIL Image at the configured DPI."""
        try:
            mat = self._get_render_matrix()
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return self._pixmap_to_pil(pix, cleanup=True)
        except Exception as e:
            logger.warning(f"页面{page.number + 1}渲染失败: {e}")
            return None

    def _get_visual_indexer(self):
        """Return the visual indexer instance via the lazy factory function."""
        return self._visual_indexer_provider()

    def _classify_image(self, pil_image) -> Tuple[str, float, str]:
        """Classify an image using CLIP to determine if it is a circuit diagram.

        Decision rules:
            - circuit_diagram / application_circuit / test_circuit / block_diagram → PASS
            - schema → PASS
            - chip_package / pin_diagram → REJECT
            - pcb_layout / mechanical_drawing / table / chart / etc. → REJECT
            - Unknown types → PASS if confidence >= clip_threshold, else REJECT

        Args:
            pil_image: PIL Image to classify.

        Returns:
            Tuple of (filter_result, confidence, filter_reason).
            filter_result is "PASS", "REJECT", or "ERROR".
        """
        try:
            visual_indexer = self._get_visual_indexer()
            image_type, confidence = visual_indexer.classify_image_with_clip(pil_image)
            self._last_clip_type = image_type

            if image_type == "circuit_diagram":
                return "PASS", confidence, f"电路图得分高: {confidence:.2f}"
            elif image_type in ["application_circuit", "test_circuit", "block_diagram"]:
                return "PASS", confidence, f"检测为{image_type}（应用/测试/方框电路）"
            elif image_type == "schema":
                return "PASS", confidence, "检测为原理图"
            elif image_type in ["chip_package", "pin_diagram"]:
                return "REJECT", confidence, f"检测为{image_type}，不是目标图像"
            elif image_type in ["pcb_layout", "mechanical_drawing", "package_materials",
                                "table", "chart", "timing_diagram", "logo", "photo",
                                "truth_table", "state_diagram", "pinout_diagram"]:
                return "REJECT", confidence, f"检测为{image_type}，非电路原理图"
            elif image_type == "other":
                return "REJECT", confidence, "未分类图像，保守过滤"
            else:
                result = "REJECT" if confidence < self.clip_threshold else "PASS"
                return result, confidence, f"未知类型图像"

        except Exception as e:
            logger.warning(f"CLIP分类失败: {e}")
            return "ERROR", 0.0, f"分类异常: {str(e)}"

    def _clean_chart_noise(self, text: str) -> str:
        """Remove chart noise from PDF-extracted text while preserving technical parameters.

        Uses a state-machine approach: when a chart area is detected (via figure
        captions, high digit-density lines, or axis labels), subsequent lines are
        skipped until the chart area ends (empty line with low line count, or
        exceeding the 15-line limit).

        Filtering rules:
            1. "Figure N" markers → enter chart area
            2. Lines with >70% digit words (≥5 words) → chart data rows
            3. Axis labels like "(V)" at line end → chart area
            4. Layout codes (e.g. "A001") → stripped from retained lines

        Args:
            text: Raw text from page.get_text().

        Returns:
            Cleaned text with chart noise removed.
        """
        lines = text.split('\n')
        cleaned_lines = []

        in_chart_area = False
        chart_line_count = 0

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if chart_line_count > 0 and chart_line_count < 10:
                    in_chart_area = False
                chart_line_count = 0
                continue

            if self._figure_pattern.search(line_stripped):
                in_chart_area = True
                chart_line_count = 0
                continue

            words = line_stripped.split()
            if len(words) >= 5:
                digit_count = sum(
                    1 for w in words
                    if w.replace('±', '').replace('+', '').replace('-', '').replace('.', '').isdigit()
                )
                if digit_count / len(words) > 0.7:
                    in_chart_area = True
                    chart_line_count += 1
                    continue

            if self._axis_label_pattern.search(line_stripped):
                in_chart_area = True
                chart_line_count += 1
                continue

            if in_chart_area:
                chart_line_count += 1
                if chart_line_count > 15:
                    in_chart_area = False
                    chart_line_count = 0
                continue

            cleaned_line = self._layout_code_pattern.sub('', line_stripped)

            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)

        return '\n'.join(cleaned_lines).strip()


class EnhancedDocumentProcessor:
    """Core document processing orchestrator.

    Coordinates three subsystems in sequence:
        1. _PDFParser: PDF parsing (text + images + metadata)
        2. DatasheetTableExtractor: Structured table extraction
        3. TrueColPaliIndexer: Visual embedding creation (CLIP + ColPali)

    Table extraction uses dedicated engines (Camelot/pdfplumber) rather than
    PyMuPDF's limited table capabilities, following the principle of using
    each tool for what it does best.
    """

    def __init__(self, settings=None):
        """Initialize the document processor.

        Args:
            settings: Configuration object (defaults to get_settings()).
        """
        self.settings = settings or get_settings()
        self._pdf_parser = _PDFParser(settings, visual_indexer_provider=self._get_visual_indexer)
        self.table_extractor = DatasheetTableExtractor(settings=settings)
        self.colpali_indexer = None

        logger.info("增强版文档处理器初始化完成")

    def _get_visual_indexer(self):
        """Return the visual indexer instance (lazy singleton).

        Deferred initialization avoids loading ~2GB of GPU models at startup.
        """
        if self.colpali_indexer is None:
            logger.info("初始化TrueColPaliIndexer...")
            self.colpali_indexer = create_visual_indexer()
            logger.info("TrueColPaliIndexer初始化成功")
        return self.colpali_indexer

    def optimize_for_document_type(self, doc_type: str) -> Dict[str, Any]:
        """Return processing options based on document type.

        Visual embeddings are currently handled by a background pipeline
        (_index_circuits_background), so ColPali is disabled here for all
        document types to avoid duplicate processing.

        Args:
            doc_type: "datasheet", "manual", or "default".

        Returns:
            Dict with enable_colpali and require_visual_embeddings flags.
        """
        return {
            "enable_colpali": False,
            "require_visual_embeddings": False,
        }

    async def process_document(self, file_path: Union[str, Path],
                               options: Dict[str, Any] = None,
                               document_id: str = None) -> Dict[str, Any]:
        """Process a document through the full pipeline.

        Steps:
            1. _parse_pdf(): Extract text, images, metadata.
            2. _extract_tables(): Extract structured tables.
            3. _create_visual_embeddings(): Create visual embeddings (if enabled).

        Error handling strategy:
            - PDF parse failure → raise ProcessingError (cannot continue).
            - Table extraction failure → return empty list (non-critical).
            - Visual embedding failure → raise or warn based on require_visual_embeddings.

        Args:
            file_path: Path to the PDF file.
            options: Processing options from optimize_for_document_type().
            document_id: Optional document ID override.

        Returns:
            Dict with document_info, processed_pages, extracted_tables, image_count.

        Raises:
            ProcessingError: On unrecoverable processing failure.
        """
        file_path = Path(file_path)
        options = options or {}

        logger.info(f"开始完整文档处理: {file_path.name}")

        start_time = time.time()

        try:
            parsed_data = self._parse_pdf(file_path)

            tables = self._extract_tables(file_path)

            visual_embeddings = self._create_visual_embeddings(
                file_path, parsed_data, options, document_id=document_id
            )

            if visual_embeddings and visual_embeddings.get("status") == "failed":
                error_msg = visual_embeddings.get("error", "未知错误")
                error_type = visual_embeddings.get("error_type", "unknown")

                require_visual = options.get('require_visual_embeddings', False)

                if require_visual:
                    logger.error(f"视觉嵌入创建失败且配置要求必须成功: {error_msg}")
                    raise ProcessingError(
                        f"视觉嵌入创建失败（类型: {error_type}）: {error_msg}\n"
                        f"请检查ColPali配置和模型可用性，或设置enable_colpali=False禁用视觉嵌入"
                    )
                else:
                    logger.warning(f"视觉嵌入创建失败，但配置允许继续处理: {error_msg}")
                    logger.warning(f"错误类型: {error_type}")

            processing_time = time.time() - start_time
            logger.info(f"文档处理完成: {file_path.name}, "
                        f"耗时: {processing_time:.2f}s, "
                        f"提取表格: {len(tables)}个")

            return {
                "document_info": parsed_data['document_info'],
                "processed_pages": parsed_data['pages'],
                "extracted_tables": tables,
                "image_count": parsed_data.get('image_count', 0),
            }

        except ProcessingError:
            raise
        except Exception as e:
            logger.error(f"文档处理失败 {file_path.name}: {e}")
            raise ProcessingError(f"Document processing failed: {e}") from e

    def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Parse a PDF file and return structured data.

        Generates a deterministic document_id from the file path via MD5,
        ensuring idempotent uploads of the same file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dict with document_info, pages, and image_count.
        """
        try:
            document_id = hashlib.md5(str(file_path).encode()).hexdigest()
            parse_result = self._pdf_parser.parse(str(file_path), document_id)
            document_info = {
                "document_id": parse_result.document_id,
                "filename": parse_result.filename,
                "page_count": parse_result.page_count,
                "title": parse_result.metadata.get("title", ""),
                "author": parse_result.metadata.get("author", ""),
                "creation_date": parse_result.metadata.get("creation_date", ""),
                "metadata": parse_result.metadata,
                "status": parse_result.status,
                "warnings": parse_result.warnings,
                "error_message": parse_result.error_message
            }
            return {
                "document_info": document_info,
                "pages": parse_result.pages,
                "image_count": parse_result.image_count
            }
        except Exception as e:
            logger.error(f"PDF解析失败 {file_path.name}: {e}")
            raise

    def _extract_tables(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract structured tables from a PDF using DatasheetTableExtractor.

        Table extraction failure is non-critical: an empty list is returned
        so that text and image data remain available.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of table dicts with data, page, source, confidence, dimensions, etc.
        """
        start_time = time.time()

        try:
            table_results = self.table_extractor.extract(str(file_path))

            tables = []
            for table in table_results:
                table_dict = {
                    "data": table.data,
                    "page": table.page,
                    "source": table.source,
                    "confidence": table.confidence,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "empty_cells": table.empty_cells,
                    "metadata": getattr(table, 'metadata', {}),
                    "table_type": getattr(table, 'table_type', 'unknown'),
                    "header_keywords": getattr(table, 'header_keywords', []),
                    "is_cross_page": getattr(table, 'is_cross_page', False),
                    "bbox": getattr(table, 'bbox', None)
                }
                tables.append(table_dict)

            extract_time = time.time() - start_time
            logger.debug(f"表格提取完成: {len(tables)}个表格, 耗时={extract_time:.2f}s")

            return tables

        except Exception as e:
            logger.error(f"表格提取失败: {e}")
            return []

    def _create_visual_embeddings(self, file_path: Path, parsed_data: Dict[str, Any],
                                  options: Dict[str, Any],
                                  document_id: str = None) -> Optional[Dict[str, Any]]:
        """Create visual embeddings for the document using ColPali architecture.

        ColPali generates multi-vector patch embeddings from rendered page images,
        enabling visual retrieval (e.g. circuit diagram search) that traditional
        text-only embeddings cannot support.

        Pre-loaded PIL Images from the parse stage are reused to avoid redundant
        PDF re-opening and page rendering.

        Args:
            file_path: Path to the PDF file.
            parsed_data: Output from _parse_pdf(), containing pages list.
            options: Processing options with enable_colpali flag.
            document_id: Optional document ID override.

        Returns:
            Dict with status ("success"/"disabled"/"failed") and related fields.
        """
        enable_colpali = options.get('enable_colpali', True)

        if not enable_colpali:
            logger.info("视觉嵌入创建已禁用")
            return {
                "status": "disabled",
                "document_id": file_path.stem,
                "note": "视觉嵌入创建已禁用"
            }

        try:
            visual_indexer = self._get_visual_indexer()

            if not document_id:
                document_id = file_path.stem

            logger.info(f"开始创建视觉嵌入: {file_path.name}")

            pages = parsed_data.get('pages', [])
            preloaded_images = []

            for page in pages:
                if hasattr(page, 'get_pil_image'):
                    pil_image = page.get_pil_image()
                    if pil_image:
                        preloaded_images.append(pil_image)
                    else:
                        logger.warning(f"页面{page.page_number}没有预加载的图像")
                elif hasattr(page, '_pil_image') and page._pil_image:
                    preloaded_images.append(page._pil_image)

            if preloaded_images:
                logger.info(f"使用预加载的图像: {len(preloaded_images)}页（避免重复解析PDF）")
            else:
                logger.warning(f"没有找到预加载的图像，将重新解析PDF")

            embeddings = visual_indexer.index_document(
                document_path=file_path,
                metadata={
                    'file_path': str(file_path),
                    'document_id': document_id,
                    'pages_count': len(parsed_data.get('pages', []))
                },
                preloaded_images=preloaded_images if preloaded_images else None
            )

            if preloaded_images:
                preloaded_images.clear()
                import gc
                gc.collect()
                if 'torch' in dir():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            logger.info(f"视觉嵌入创建成功: {file_path.name}")
            return embeddings

        except Exception as e:
            logger.error(f"视觉嵌入创建失败: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "document_id": file_path.stem,
                "error_type": type(e).__name__,
                "note": "视觉嵌入创建失败"
            }


class ProcessingResult:
    """Unified result type for downstream API consumption.

    Unlike ParseResult (internal, fine-grained, contains raw image data),
    ProcessingResult is the external-facing summary with aggregate counts
    and no large binary payloads. Chunk and circuit counts are populated
    downstream during the indexing stage.

    Attributes:
        status: "success" or "error".
        page_count: Total pages processed.
        chunk_count: Text chunk count (populated downstream, initially 0).
        table_count: Number of extracted tables.
        image_count: Number of extracted images.
        circuit_count: Circuit diagram count (populated downstream, initially 0).
        warnings: Warning messages.
        error_message: Error detail.
        processed_pages: List of PageContent objects.
        extracted_tables: List of table dicts.
    """

    def __init__(self, status: str, page_count: int = 0,
                 table_count: int = 0, image_count: int = 0,
                 warnings: list = None, error_message: str = "",
                 processed_pages: list = None, extracted_tables: list = None):
        self.status = status
        self.page_count = page_count
        self.chunk_count = 0
        self.table_count = table_count
        self.image_count = image_count
        self.circuit_count = 0
        self.warnings = warnings or []
        self.error_message = error_message
        self.processed_pages = processed_pages or []
        self.extracted_tables = extracted_tables or []


def _build_processing_result(result: Dict[str, Any], status: str = "success",
                             error_message: str = "") -> ProcessingResult:
    """Build a ProcessingResult from the processor's return dict.

    Args:
        result: Output dict from EnhancedDocumentProcessor.process_document().
        status: "success" or "error".
        error_message: Error message for error status.

    Returns:
        ProcessingResult instance.
    """
    if status == "error":
        return ProcessingResult(
            status="error",
            page_count=0,
            table_count=0,
            image_count=0,
            warnings=[],
            error_message=error_message,
            processed_pages=[],
            extracted_tables=[]
        )

    doc_info = result.get("document_info", {})
    pages = result.get("processed_pages", [])
    return ProcessingResult(
        status="success",
        page_count=doc_info.get("page_count", len(pages)),
        table_count=len(result.get("extracted_tables", [])),
        image_count=result.get("image_count", 0),
        warnings=doc_info.get("warnings", []),
        error_message=doc_info.get("error_message", ""),
        processed_pages=pages,
        extracted_tables=result.get("extracted_tables", [])
    )


class IngestionPipeline:
    """Document ingestion pipeline — the system's external entry point.

    Implements the Facade pattern: wraps EnhancedDocumentProcessor with
    document type detection and option generation, providing a simplified
    interface for the API layer.

    Call chain:
        API → IngestionPipeline.process() → detect doc type → get options
        → EnhancedDocumentProcessor.process_document() → _build_processing_result()
    """

    def __init__(self, settings=None):
        """Initialize the ingestion pipeline.

        Args:
            settings: Configuration object (defaults to get_settings()).
        """
        self.processor = EnhancedDocumentProcessor(settings)

    async def process(self, file_path: str, document_id: str = None,
                      filename: str = None) -> ProcessingResult:
        """Process a document through the ingestion pipeline.

        Determines document type from the filename heuristically, then
        delegates to EnhancedDocumentProcessor.

        Args:
            file_path: Path to the PDF file.
            document_id: Optional document ID.
            filename: Original filename (used for type detection).

        Returns:
            ProcessingResult with success or error status.
        """
        try:
            doc_type = (
                "datasheet" if filename and ('数据手册' in filename or 'datasheet' in filename.lower())
                else "manual" if filename and filename.lower().endswith('.pdf')
                else "default"
            )
            options = self.processor.optimize_for_document_type(doc_type)

            result = await self.processor.process_document(file_path, options, document_id=document_id)
            return _build_processing_result(result)

        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            return _build_processing_result({}, status="error", error_message=str(e))
