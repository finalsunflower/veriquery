"""
Circuit retrieval router - Multi-modal circuit diagram search API.

Provides circuit diagram retrieval using ColPali vision-language model
with query enhancement and domain-aware re-ranking. Endpoints:
  - GET  /image/by-path      → Fetch circuit image by file path
  - GET  /{circuit_id}/image → Fetch circuit image by ID
  - POST /search             → Search circuits with enhanced query + visual retrieval + re-ranking
"""

import asyncio
import logging
import json
import time
import hashlib
import re
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.dependencies import get_service_container
from core.config import get_settings
from ingestion.image_indexer import TrueColPaliIndexer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/circuit", tags=["circuit"])

SEARCH_CONFIG = {
    "top_k_multiplier": 3,
    "max_search_time": 30.0,
    "cache_ttl": 300,
    "max_expanded_keywords": 12,
}

CIRCUIT_TYPE_KEYWORDS = TrueColPaliIndexer._CIRCUIT_TYPE_KEYWORDS


class SimpleCache:
    """Lightweight TTL cache for search results."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expire_time = self._cache[key]
            if time.time() < expire_time:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        self._cache[key] = (value, time.time() + self._ttl)

    def clear(self):
        self._cache.clear()


_search_cache = SimpleCache(SEARCH_CONFIG["cache_ttl"])


def _generate_cache_key(query: str, top_k: int, doc_ids: tuple) -> str:
    """Generate an MD5-based cache key from query parameters."""
    key_str = f"{query}:{top_k}:{doc_ids}"
    return hashlib.md5(key_str.encode()).hexdigest()


class CircuitSearchRequest(BaseModel):
    """Request model for circuit search."""

    query: str
    top_k: int = 10
    document_ids: List[str] = Field(default_factory=list)


class CircuitSearchResponse(BaseModel):
    """Response model for circuit search."""

    success: bool = True
    circuits: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    search_time: float = 0.0
    cached: bool = False


@router.get("/image/by-path")
async def get_image_by_path(image_path: str):
    """Fetch a circuit image by its file path.

    Args:
        image_path: URL-encoded relative image path.

    Returns:
        FileResponse with the image file.
    """
    settings = get_settings()
    try:
        full_path = _resolve_image_path(image_path, settings)
        return FileResponse(
            path=str(full_path),
            media_type="image/png",
            filename=full_path.name
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取图片失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{circuit_id}/image")
async def get_circuit_image(circuit_id: str, container=Depends(get_service_container)):
    """Fetch a circuit image by its ID.

    Looks up the image path from vector store metadata, then serves the file.

    Args:
        circuit_id: Unique circuit identifier in the vector store.
        container: ServiceContainer via dependency injection.

    Returns:
        FileResponse with the circuit image.
    """
    settings = get_settings()
    try:
        circuit_info = await _get_circuit_info(circuit_id, container)
        image_path = circuit_info.get("image_path", "")

        if not image_path:
            raise HTTPException(status_code=404, detail="电路图像不存在")

        full_path = _resolve_image_path(image_path, settings)
        return FileResponse(
            path=str(full_path),
            media_type="image/png",
            filename=f"circuit_{circuit_id}.png"
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取电路图像失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取电路图像失败: {str(e)}")


@router.post("/search", response_model=CircuitSearchResponse)
async def search_circuits(request: CircuitSearchRequest, container=Depends(get_service_container)):
    """Search circuits using ColPali visual retrieval with query enhancement and re-ranking.

    Flow: cache check → query enhancement → visual retrieval → result conversion
          → re-ranking → cache write → response.

    Args:
        request: CircuitSearchRequest with query, top_k, and optional document_ids.
        container: ServiceContainer via dependency injection.

    Returns:
        CircuitSearchResponse with matching circuits, timing, and cache status.
    """
    start_time = time.time()

    try:
        if not hasattr(container, 'retriever') or container.retriever is None:
            raise HTTPException(status_code=503, detail="检索系统未初始化")

        if not request.query:
            raise HTTPException(status_code=400, detail="搜索请求缺少query")

        doc_ids_tuple = tuple(sorted(request.document_ids)) if request.document_ids else ()
        cache_key = _generate_cache_key(request.query, request.top_k, doc_ids_tuple)

        cached_result = _search_cache.get(cache_key)
        if cached_result:
            cached_result["cached"] = True
            cached_result["search_time"] = time.time() - start_time
            logger.info(f"电路搜索缓存命中: {request.query}")
            return CircuitSearchResponse(**cached_result)

        try:
            circuits = await _execute_search_with_timeout(
                request, container, SEARCH_CONFIG["max_search_time"]
            )
        except TimeoutError:
            logger.warning(f"搜索超时，返回部分结果: {request.query}")
            circuits = []

        search_time = time.time() - start_time
        result = {
            "success": True,
            "circuits": circuits[:request.top_k],
            "total_count": len(circuits),
            "search_time": search_time,
            "cached": False
        }

        if circuits:
            _search_cache.set(cache_key, result.copy())

        logger.info(f"电路搜索完成: query={request.query}, results={len(circuits)}, time={search_time:.2f}s")
        return CircuitSearchResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"电路搜索异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"电路搜索系统错误: {str(e)}")


async def _execute_search_with_timeout(
    request: CircuitSearchRequest,
    container,
    timeout: float
) -> List[Dict[str, Any]]:
    """Execute search with a timeout guard.

    Args:
        request: Circuit search request.
        container: ServiceContainer providing search services.
        timeout: Maximum search time in seconds.

    Returns:
        List of circuit info dicts.

    Raises:
        TimeoutError: If search exceeds the timeout.
    """
    try:
        return await asyncio.wait_for(
            _execute_search_core(request, container),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise TimeoutError("搜索超时")


async def _execute_search_core(request: CircuitSearchRequest, container) -> List[Dict[str, Any]]:
    """Core search logic: query enhancement → visual retrieval → conversion → re-ranking.

    Args:
        request: Circuit search request.
        container: ServiceContainer providing visual_indexer and other services.

    Returns:
        Ranked list of circuit info dicts.
    """
    enhanced_query = _enhance_search_query(request.query)
    document_ids = request.document_ids if request.document_ids else None

    if not hasattr(container, 'visual_indexer') or container.visual_indexer is None:
        logger.error("视觉索引器未初始化")
        return []

    try:
        base_threshold = 0.08
        has_chip_model = bool(re.search(r'[a-zA-Z]*\d{2,}[a-zA-Z0-9]*', enhanced_query))
        dynamic_threshold = 0.05 if has_chip_model else base_threshold

        logger.info(f"动态阈值: 基础={base_threshold}, 芯片型号检测={has_chip_model}, 最终={dynamic_threshold}")

        visual_results = container.visual_indexer.search_documents(
            query=enhanced_query,
            top_k=request.top_k * SEARCH_CONFIG["top_k_multiplier"],
            score_threshold=dynamic_threshold,
            document_ids=document_ids
        )
    except Exception as e:
        logger.error(f"视觉检索失败: {e}")
        visual_results = []

    circuits = _convert_visual_results(visual_results, request.query)

    if not circuits:
        logger.info("visual_indexer返回空结果，尝试降级使用vector_store搜索")
        circuits = await _fallback_vector_store_search(request, container, enhanced_query, document_ids)

    circuits = _rank_circuits(circuits, request.query)
    return circuits


async def _fallback_vector_store_search(
    request: CircuitSearchRequest,
    container,
    enhanced_query: str,
    document_ids: Optional[List[str]]
) -> List[Dict[str, Any]]:
    """Fallback search using visual_indexer metadata when embedding search returns empty.

    Args:
        request: Circuit search request.
        container: ServiceContainer.
        enhanced_query: Enhanced query string.
        document_ids: Optional document ID filter.

    Returns:
        List of circuit info dicts from metadata search.
    """
    circuits = []

    try:
        visual_indexer = getattr(container, 'visual_indexer', None)
        if not visual_indexer:
            logger.warning("降级搜索跳过: visual_indexer未初始化")
            return []

        if not hasattr(visual_indexer, '_search_with_metadata'):
            logger.warning("降级搜索跳过: visual_indexer不支持元数据搜索")
            return []

        logger.info(f"使用visual_indexer元数据搜索: query='{enhanced_query}'")

        metadata_results = visual_indexer._search_with_metadata(
            query=enhanced_query,
            top_k=request.top_k * 3,
            score_threshold=0.08,
            document_ids=document_ids
        )

        if not metadata_results:
            logger.info("元数据搜索返回空结果")
            return []

        logger.info(f"元数据搜索返回{len(metadata_results)}条结果")

        for result in metadata_results:
            metadata = result.metadata if hasattr(result, 'metadata') else {}

            image_path = metadata.get('image_path', '')
            if not image_path:
                continue

            circuit_info = {
                'circuit_id': result.image_id if hasattr(result, 'image_id') else f"circuit_{len(circuits)}",
                'document_id': metadata.get('document_id', ''),
                'page_number': metadata.get('page', 0),
                'image_path': image_path,
                'caption': metadata.get('caption', ''),
                'circuit_type': metadata.get('circuit_type', ''),
                'component_types': metadata.get('component_types', ''),
                'component_count': metadata.get('component_count', 0),
                'confidence': metadata.get('overall_confidence', 0.5),
                'score': result.score if hasattr(result, 'score') else 0.0,
                'is_circuit': metadata.get('is_circuit', True),
            }
            circuits.append(circuit_info)

        logger.info(f"降级搜索返回{len(circuits)}条电路图结果")

    except Exception as e:
        logger.error(f"降级搜索失败: {e}", exc_info=True)

    return circuits


def _extract_query_keywords(query: str) -> set:
    """Extract meaningful keywords from a user query for relevance scoring.

    Strategies:
    1. Chip model detection (e.g. NE5532, LM358)
    2. Chinese term segmentation
    3. English word extraction
    4. Synonym expansion via circuit type synonym table

    Args:
        query: User query string.

    Returns:
        Set of lowercase keywords.
    """
    query_lower = query.lower().strip()

    STOP_WORDS = {
        '的', '是', '在', '和', '与', '或', '及', '等', '中', '上', '下',
        '电路', '原理图', 'schematic', 'diagram', 'circuit', 'circuits',
        '图', '图片', '图像', '查找', '搜索', '检索', '显示', '给我',
        '一个', '这个', '那个', '什么', '如何', '怎么', '哪', '几',
        '请', '帮忙', '需要', '想要', '看看', '找', '要'
    }

    keywords = set()

    chip_patterns = re.findall(r'\b[a-zA-Z]{0,}\d{2,}[a-zA-Z0-9]{0,}\b', query_lower)
    for pattern in chip_patterns:
        if len(pattern) >= 3 and pattern not in STOP_WORDS:
            keywords.add(pattern)

    chinese_terms = re.split(r'[\s,，、;；:：]+', query_lower)
    for term in chinese_terms:
        term = term.strip()
        if len(term) >= 1 and term not in STOP_WORDS:
            keywords.add(term)

    chinese_continuous = re.findall(r'[\u4e00-\u9fff]{2,}', query_lower)
    for phrase in chinese_continuous:
        for i in range(len(phrase)):
            for j in range(i + 2, len(phrase) + 1):
                sub = phrase[i:j]
                if sub not in STOP_WORDS and len(sub) >= 2:
                    keywords.add(sub)

    english_words = re.findall(r'[a-zA-Z]{2,}', query_lower)
    for word in english_words:
        if word not in STOP_WORDS:
            keywords.add(word)

    _CIRCUIT_TYPE_SYNONYMS = {
        '放大': {'amplifier', 'opamp', '运放', 'gain', '增益', '放大器'},
        '滤波': {'filter', '低通', '高通', '带通', 'LPF', 'HPF', 'BPF'},
        '电源': {'power', '供电', '稳压', 'regulator', 'LDO', 'DC-DC'},
        '振荡': {'oscillator', '晶振', 'crystal', '时钟', 'clock'},
        '驱动': {'driver', '电机驱动', 'LED驱动', 'H桥', 'PWM'},
        '比较': {'comparator', '比较器', '阈值', 'threshold'},
        '定时': {'timer', '555', '延时', 'delay'},
        '整流': {'rectifier', '桥式', '半波', '全波'},
        '保护': {'protection', 'ESD', 'TVS', '过流', '过压'},
        '接口': {'interface', 'UART', 'SPI', 'I2C', 'USB', 'CAN'},
        '复位': {'reset', 'POR', '看门狗', 'watchdog'},
        '稳压': {'regulator', 'LDO', '线性稳压', '开关稳压'},
    }

    expanded_keywords = set()
    for kw in list(keywords):
        for type_key, synonyms in _CIRCUIT_TYPE_SYNONYMS.items():
            if type_key in kw or kw in type_key:
                expanded_keywords.update(synonyms)
    keywords.update(expanded_keywords)

    return keywords


def _calculate_relevance_score(
    query: str,
    query_terms: set,
    caption: str,
    circuit_types: list,
    components: list,
    visual_score: float,
    filename: str = ""
) -> float:
    """Calculate multi-dimensional relevance score between a query and a search result.

    Scoring dimensions (weighted sum, total 0~1):
    1. Caption match (0.30)
    2. Circuit type match (0.20)
    3. Component match (0.15)
    4. Visual score (0.15)
    5. Filename match (0.20)

    Args:
        query: Lowercase query string.
        query_terms: Extracted keyword set.
        caption: Result caption text (lowercase).
        circuit_types: Circuit type list from metadata.
        components: Component list from metadata.
        visual_score: MaxSim visual similarity score.
        filename: Source PDF filename.

    Returns:
        Relevance score in [0, 1].
    """
    if not query_terms:
        return min(visual_score, 1.0) * 0.5

    score = 0.0

    caption_score = 0.0
    if caption and query_terms:
        if query in caption:
            caption_score = 1.0
        else:
            matched_terms = sum(1 for term in query_terms if term in caption)
            caption_score = min(matched_terms / len(query_terms), 1.0)
            if matched_terms >= 2:
                caption_score = min(caption_score * 1.2, 1.0)
    score += caption_score * 0.30

    type_score = 0.0
    if circuit_types and query_terms:
        type_list = []
        for ct in circuit_types:
            if isinstance(ct, str):
                type_list.append(ct.lower())
            elif isinstance(ct, dict) and 'type' in ct:
                type_list.append(ct['type'].lower())

        type_text = ' '.join(type_list)
        matched_types = sum(1 for term in query_terms if term in type_text)
        if matched_types > 0:
            type_score = min(matched_types / max(len(query_terms), 1), 1.0)
    score += type_score * 0.20

    component_score = 0.0
    if components and query_terms:
        comp_list = []
        for c in components:
            if isinstance(c, str):
                comp_list.append(c.lower())
            elif isinstance(c, dict) and 'type' in c:
                comp_list.append(c['type'].lower())

        comp_text = ' '.join(comp_list)
        matched_components = sum(1 for term in query_terms if term in comp_text)
        if matched_components > 0:
            component_score = min(matched_components / max(len(query_terms), 1), 1.0)
    score += component_score * 0.15

    normalized_visual = min(visual_score / 1.5, 1.0) if visual_score > 0 else 0.0
    score += normalized_visual * 0.15

    filename_score = 0.0
    if filename and query_terms:
        filename_lower = filename.lower().replace('.pdf', '')
        if query in filename_lower:
            filename_score = 1.0
        else:
            matched_fn = sum(1 for term in query_terms if term in filename_lower and len(term) >= 2)
            if matched_fn > 0:
                filename_score = min(matched_fn / max(len(query_terms), 1), 1.0)

                _chip_synonyms = {
                    'ne5532': ['5532', 'ne5532', 'sa5532', 'se5532'],
                    'lm358': ['358', 'lm358'],
                    'tl072': ['072', 'tl072'],
                    'lm741': ['741', 'lm741'],
                    'lm324': ['324', 'lm324'],
                    'lm386': ['386', 'lm386'],
                    '555': ['555', 'ne555', 'lm555'],
                    'lm317': ['317', 'lm317'],
                    'lm7805': ['7805', 'lm7805'],
                }
                for chip, synonyms in _chip_synonyms.items():
                    if any(s in filename_lower for s in synonyms):
                        if any(s in query for s in synonyms):
                            filename_score = max(filename_score, 0.8)
                            break
    score += filename_score * 0.20

    return min(score, 1.0)


def _convert_visual_results(visual_results, query: str) -> List[Dict[str, Any]]:
    """Convert ColPaliSearchResult objects to standardized circuit info dicts.

    Applies multi-layer filtering:
    1. Image path existence check
    2. Non-circuit keyword filtering (package info, mechanical drawings, etc.)
    3. is_circuit and confidence validation
    4. Query relevance scoring with dynamic threshold

    Args:
        visual_results: List of ColPaliSearchResult objects.
        query: Original query string for relevance validation.

    Returns:
        List of filtered and enriched circuit info dicts.
    """
    circuits = []

    MIN_CONFIDENCE_THRESHOLD = 0.08
    MIN_RELEVANCE_SCORE = 0.02
    MIN_RESULTS_RELEVANCE_AVG = 0.05

    BLOCKED_PAGES = set()
    VERIFIED_CIRCUIT_PAGES = set()

    NON_CIRCUIT_KEYWORDS = [
        'package material', 'package information', 'tape and reel',
        'mechanical drawing', 'outline dimension',
        'land pattern', 'solder mask', 'footprint', 'pad layout',
        'board layout', 'pcb layout', 'physical dimension',
        'reel dimension', 'cavity', 'quadrant assignment',
        'pin 1 orientation', 'spool hole', 'pocket quadrant',
        '封装材料', '封装信息', '机械尺寸', '焊盘', '阻焊层',
        'pack materials page',
        'stencil', 'solder paste', 'paste example', 'laser cutting',
        'aperture', 'thickness', 'based on', 'scale example',
        '钢网', '焊膏', '开口', '锡膏',
        'example stencil', 'small outline', 'trapezoidal wall',
        'rounded corner', 'assembly site', 'revision history',
        'ordering information', 'document history', 'datasheet cover'
    ]

    query_lower = query.lower().strip()
    query_terms = _extract_query_keywords(query)
    logger.info(f"查询相关性验证: 原始查询='{query}', 提取关键词={query_terms}")

    filtered_count = 0
    filter_reasons = {'confidence': 0, 'is_circuit': 0, 'blocked_page': 0, 'keywords': 0, 'irrelevant': 0, 'no_image_path': 0}
    relevance_scores = []

    for result in visual_results:
        metadata = result.metadata or {}

        image_path = metadata.get('image_path', '')
        if not image_path:
            filtered_count += 1
            filter_reasons['no_image_path'] += 1
            continue

        page_num = metadata.get('page', 1)
        caption = metadata.get('caption', '').lower()

        if page_num in BLOCKED_PAGES:
            filtered_count += 1
            filter_reasons['blocked_page'] += 1
            continue

        non_circuit_matches = sum(1 for kw in NON_CIRCUIT_KEYWORDS if kw.lower() in caption)
        if non_circuit_matches >= 2:
            filtered_count += 1
            filter_reasons['keywords'] += 1
            continue

        confidence = metadata.get('confidence', 0.0)
        is_circuit = metadata.get('is_circuit', True)

        if not is_circuit:
            circuit_types_check = _parse_json_field(metadata.get('circuit_type', []))
            valid_circuit_types = [ct for ct in circuit_types_check if ct and ct != 'non_circuit']
            if valid_circuit_types:
                is_circuit = True
                logger.info(f"is_circuit覆盖: page={page_num}, 有有效电路类型={valid_circuit_types}")
            elif confidence >= 0.15:
                is_circuit = True
                logger.info(f"is_circuit覆盖: page={page_num}, 置信度={confidence:.2f}>=0.15")
            else:
                filtered_count += 1
                filter_reasons['is_circuit'] += 1
                continue

        if confidence < MIN_CONFIDENCE_THRESHOLD:
            filtered_count += 1
            filter_reasons['confidence'] += 1
            continue

        path_obj = Path(image_path)
        if not path_obj.exists():
            fixed_path = _fix_image_path(image_path, metadata)
            if fixed_path and Path(fixed_path).exists():
                image_path = fixed_path
                metadata['image_path'] = fixed_path
            else:
                logger.debug(f"跳过不存在的图片: {image_path}")
                continue

        relevance_score = _calculate_relevance_score(
            query=query_lower,
            query_terms=query_terms,
            caption=caption,
            circuit_types=_parse_json_field(metadata.get('circuit_type', [])),
            components=_parse_json_field(metadata.get('components', [])),
            visual_score=result.score if hasattr(result, 'score') else 0.5,
            filename=metadata.get('filename', '')
        )

        relevance_scores.append(relevance_score)

        if relevance_score < MIN_RELEVANCE_SCORE:
            filtered_count += 1
            filter_reasons['irrelevant'] += 1
            logger.warning(f"相关性不足: page={page_num}, relevance={relevance_score:.3f} < {MIN_RELEVANCE_SCORE}")
            continue

        circuit_types = _parse_json_field(metadata.get('circuit_type', []))
        components = _parse_json_field(metadata.get('components', []))
        caption = metadata.get('caption', '')

        circuit_info = {
            "circuit_id": str(result.image_id) if hasattr(result, 'image_id') else str(result.page_num),
            "name": caption or f"电路图 (第{metadata.get('page', 1)}页)",
            "description": caption,
            "filename": metadata.get('filename', ''),
            "page": metadata.get('page', 1),
            "score": result.score if hasattr(result, 'score') else 0.5,
            "relevance_score": relevance_score,
            "metadata": metadata,
            "image_path": image_path,
            "image_id": metadata.get('image_id', ''),
            "document_id": metadata.get('document_id', ''),
            "circuit_type": circuit_types,
            "components": components,
            "caption": caption,
            "figure_label": metadata.get('figure_label', ''),
        }

        circuits.append(circuit_info)

    if circuits and relevance_scores:
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        max_relevance = max(relevance_scores)

        logger.info(f"结果集相关性分析: 平均={avg_relevance:.3f}, 最高={max_relevance:.3f}, 结果数={len(circuits)}")

        if avg_relevance < 0.03 and max_relevance < 0.08:
            logger.warning(f"结果集整体相关性过低: avg={avg_relevance:.3f}, 丢弃全部{len(circuits)}个结果")
            circuits = []
            filter_reasons['irrelevant'] += len(circuits)

    if filtered_count > 0:
        logger.warning(f"质量过滤统计: 总过滤={filtered_count}条 | 原因分布: {filter_reasons}")
    logger.info(f"视觉检索过滤统计: 总输入={len(visual_results)}, 有效输出={len(circuits)}, 过滤={filtered_count}")
    return circuits


def _parse_json_field(field_value) -> List[Any]:
    """Safely parse a metadata field that may be a JSON string or native list.

    ChromaDB only stores str/int/float/bool in metadata, so lists are
    serialized as JSON strings and must be deserialized on read.

    Args:
        field_value: Value to parse (str, list, or None).

    Returns:
        Parsed list, or empty list on failure.
    """
    if not field_value:
        return []
    try:
        if isinstance(field_value, str):
            return json.loads(field_value)
        return list(field_value) if hasattr(field_value, '__iter__') else []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _fix_image_path(image_path: str, metadata: dict) -> str:
    """Fix a broken image path caused by document_id changes.

    Searches for the image filename under the circuit_images directory
    when the original path no longer exists.

    Args:
        image_path: Original (possibly broken) image path.
        metadata: Metadata dict containing document_id and filename.

    Returns:
        Corrected path string, or empty string if not found.
    """
    try:
        path_obj = Path(image_path)
        filename = path_obj.name

        if not filename:
            return ""

        base_dir = Path(__file__).parent.parent / "data" / "circuit_images"
        if not base_dir.exists():
            return ""

        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / filename
                if candidate.exists():
                    logger.info(f"路径修正: {image_path} -> {candidate}")
                    return str(candidate)

        doc_id = metadata.get('document_id', '')
        target_dir = base_dir / doc_id
        if target_dir.exists():
            candidate = target_dir / filename
            if candidate.exists():
                logger.info(f"路径修正(按doc_id): {image_path} -> {candidate}")
                return str(candidate)

        return ""
    except Exception as e:
        logger.warning(f"路径修正失败: {e}")
        return ""


def _resolve_image_path(image_path: str, settings) -> Path:
    """Resolve an image path by trying multiple base directories.

    Args:
        image_path: Relative or absolute path (may be URL-encoded).
        settings: Settings object providing BASE_DIR and DATA_DIR.

    Returns:
        Resolved absolute Path.

    Raises:
        FileNotFoundError: If the image file cannot be found.
    """
    if not image_path:
        raise FileNotFoundError("图像路径为空")

    decoded_path = urllib.parse.unquote(image_path)

    paths_to_try = [
        Path(decoded_path),
        Path(settings.BASE_DIR) / decoded_path,
        Path(settings.DATA_DIR) / decoded_path,
    ]

    for full_path in paths_to_try:
        if full_path.exists():
            return full_path

    raise FileNotFoundError(f"图像文件不存在: {image_path}")


def _enhance_search_query(query: str) -> str:
    """Enhance a user query with domain-specific keyword expansion.

    Strategies:
    1. Chip model detection → append "{chip}电路", "{chip}运放", etc.
    2. Circuit function keyword detection → expand with synonyms.
    3. Generic query → append "电路" suffix as fallback.

    Args:
        query: User query string.

    Returns:
        Space-separated enhanced query string.
    """
    if not query:
        return query

    query_lower = query.lower()
    expanded = [query]

    chip_keywords = [
        "ne5532", "lm358", "tl072", "lm741", "lm324", "lm386", "555", "lm317", "lm7805",
        "5532", "358", "072", "741", "324", "386", "317", "7805",
        "opa", "ad8605", "tl431", "tl082", "tl084", "opa2134",
        "ad823", "lt1013", "mc34063", "uc3842"
    ]

    query_chips = [chip for chip in chip_keywords if chip in query_lower]

    if query_chips:
        for chip in query_chips:
            if "电路" not in query_lower:
                expanded.append(f"{chip}电路")
            if "放大器" not in query_lower and "放大" in query_lower:
                expanded.append(f"{chip}放大器")
            if "运放" not in query_lower and ("放大" in query_lower or "amplifier" in query_lower):
                expanded.append(f"{chip}运放")
            expanded.append(chip)
    else:
        circuit_function_keywords = {
            '复位': ['reset', 'POR', 'NRST', '看门狗', 'watchdog', '上电复位'],
            '启动': ['boot', 'BOOT0', '引导', 'bootstrap'],
            '滤波': ['filter', 'LPF', 'HPF', 'BPF', '低通', '高通', '带通'],
            '电源': ['power', 'LDO', '稳压', 'regulator', 'DC-DC', '供电'],
            '振荡': ['oscillator', '晶振', 'crystal', '时钟', 'clock', 'VCO'],
            '放大': ['amplifier', '运放', 'opamp', 'gain', '增益'],
            '驱动': ['driver', '电机驱动', 'LED驱动', 'H桥', 'PWM'],
            '转换': ['ADC', 'DAC', '模数', '数模', '采样', '电平转换', 'level'],
            '比较': ['comparator', '比较器', '阈值', 'threshold'],
            '定时': ['timer', '555', '延时', 'delay'],
            '接口': ['UART', 'SPI', 'I2C', 'USB', 'CAN', '串口'],
            '保护': ['ESD', 'TVS', '保险丝', '过流', '过压'],
            '去耦': ['decoupling', 'bypass', '旁路', '退耦'],
            '整流': ['rectifier', '桥式', '整流'],
            '逻辑': ['logic', '门电路', '触发器', 'flip-flop'],
            '传感': ['sensor', '传感器', '温度', '压力'],
        }

        detected_functions = []
        for func_name, keywords in circuit_function_keywords.items():
            if func_name in query_lower or any(kw.lower() in query_lower for kw in keywords):
                detected_functions.append(func_name)
                expanded.extend(keywords[:3])

        if not detected_functions and "电路" not in query_lower:
            expanded.append(query + "电路")

    seen = set()
    unique = []
    for kw in expanded:
        lower_kw = kw.lower()
        if lower_kw not in seen and len(lower_kw) > 1:
            seen.add(lower_kw)
            unique.append(kw)

    return " ".join(unique[:SEARCH_CONFIG["max_expanded_keywords"]])


def _rank_circuits(circuits: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Re-rank circuits using domain-aware bonus scoring.

    Bonus strategies:
    1. Chip model exact match in caption/filename: +0.15
    2. Circuit type keyword match in caption: +0.05/keyword
    3. Circuit type keyword match in circuit_type field: +0.08/keyword
    4. Total bonus capped at 0.3; final score capped at 1.0

    Args:
        circuits: List of circuit info dicts (pre-sorted by visual similarity).
        query: Original user query string.

    Returns:
        Re-sorted list by combined_score descending.
    """
    if not circuits:
        return circuits

    query_lower = query.lower()

    query_type_keywords = []
    for circuit_type, keywords in CIRCUIT_TYPE_KEYWORDS.items():
        if any(kw.lower() in query_lower for kw in keywords):
            query_type_keywords.extend(keywords)

    chip_keywords = [
        "ne5532", "lm358", "tl072", "lm741", "lm324", "opa", "ad8605",
        "5532", "358", "072", "741", "324", "555", "386", "317", "7805",
        "tl431", "tl082", "tl084", "opa2134", "ad823", "lt1013",
        "mc34063", "uc3842"
    ]

    for circuit in circuits:
        original_score = circuit.get("score", 0.0)
        caption = circuit.get("caption", "").lower()
        circuit_types = circuit.get("circuit_type", [])

        if isinstance(circuit_types, str):
            try:
                circuit_types = json.loads(circuit_types)
            except (json.JSONDecodeError, TypeError):
                circuit_types = []

        bonus = 0.0

        for chip in chip_keywords:
            if chip in query_lower and (chip in caption or chip in circuit.get("filename", "").lower()):
                bonus += 0.15
                break

        if query_type_keywords:
            types_text = " ".join(str(t).lower() for t in circuit_types)
            for kw in query_type_keywords:
                kw_lower = kw.lower()
                if kw_lower in caption:
                    bonus += 0.05
                if kw_lower in types_text:
                    bonus += 0.08

        circuit["combined_score"] = min(original_score + min(bonus, 0.3), 1.0)

    circuits.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return circuits


async def _get_circuit_info(circuit_id: str, container) -> Dict[str, Any]:
    """Fetch circuit metadata from vector store by circuit ID.

    Args:
        circuit_id: Unique circuit identifier in ChromaDB.
        container: ServiceContainer providing vector_store.

    Returns:
        Dict with circuit_id and image_path.

    Raises:
        HTTPException 503: Vector store not initialized.
        HTTPException 404: Circuit ID not found.
        HTTPException 500: Internal query error.
    """
    vector_store = getattr(container, 'vector_store', None)
    if vector_store is None:
        raise HTTPException(status_code=503, detail="向量存储未初始化")

    try:
        data = vector_store.collection.get(
            ids=[circuit_id],
            include=['metadatas']
        )

        if not data.get('ids'):
            raise HTTPException(status_code=404, detail=f"电路ID不存在: {circuit_id}")

        metadata = data['metadatas'][0] if data.get('metadatas') else {}
        return {
            "circuit_id": circuit_id,
            "image_path": metadata.get("image_path", "")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取电路信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取电路信息失败: {str(e)}")
