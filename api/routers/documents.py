"""
Document management router - Upload, process, query, and delete PDF documents.

Provides document lifecycle management including PDF upload, async processing
pipeline (parsing → chunking → vector/BM25/table indexing → circuit indexing),
status tracking, page image rendering, and device search.

Endpoints:
  GET    /documents/                          → List all documents
  POST   /documents/upload                    → Upload PDF document
  GET    /documents/{document_id}             → Get document info with progress
  DELETE /documents/{document_id}             → Delete document and all associated data
  GET    /documents/{document_id}/pages/{page}/image → Render PDF page as PNG
  POST   /documents/search                    → Search devices by keyword
"""

import logging
import uuid
import json
import re
import gc
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="veriquery_inference")

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core import get_settings
from api.dependencies import get_service_container
from core.cleanup_manager import create_cleanup_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_cleanup_manager_instance = None


def get_cleanup_manager():
    """Get or create the CleanupManager singleton."""
    global _cleanup_manager_instance
    if _cleanup_manager_instance is None:
        _cleanup_manager_instance = create_cleanup_manager()
    return _cleanup_manager_instance


class DocumentDB:
    """Lightweight JSON-file-based document metadata store.

    Provides async-safe CRUD operations with asyncio.Lock for concurrency
    protection and immediate persistence after every write.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._db: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self):
        try:
            if self.storage_path.exists():
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                self._db = data
                logger.info(f"文档数据库已加载: {len(self._db)} 个文档")
            else:
                self._db = {}
                logger.info("文档数据库文件不存在，创建新数据库")
        except Exception as e:
            logger.error(f"加载文档数据库失败: {e}")
            self._db = {}

    async def _save(self):
        try:
            self.storage_path.write_text(
                json.dumps(self._db, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"保存文档数据库失败: {e}")

    async def get(self, document_id: str) -> Optional[Dict]:
        async with self._lock:
            return self._db.get(document_id)

    async def set(self, document_id: str, data: Dict):
        async with self._lock:
            self._db[document_id] = data
            await self._save()

    async def update(self, document_id: str, data: Dict):
        async with self._lock:
            if document_id in self._db:
                self._db[document_id].update(data)
                await self._save()

    async def delete(self, document_id: str) -> Optional[Dict]:
        async with self._lock:
            if document_id in self._db:
                doc = self._db.pop(document_id)
                await self._save()
                return doc
            return None

    async def get_all(self) -> Dict[str, Dict]:
        async with self._lock:
            return dict(self._db)

    async def exists(self, document_id: str) -> bool:
        async with self._lock:
            return document_id in self._db

    async def keys(self) -> List[str]:
        async with self._lock:
            return list(self._db.keys())

    def __len__(self) -> int:
        return len(self._db)


_documents_db: Optional[DocumentDB] = None


def get_documents_db() -> DocumentDB:
    """Get or create the DocumentDB singleton."""
    global _documents_db
    if _documents_db is None:
        settings = get_settings()
        storage_path = settings.DATA_DIR / "documents_db.json"
        _documents_db = DocumentDB(storage_path)
    return _documents_db


def _calculate_optimal_batch_size(settings, chunk_count: int) -> int:
    """Calculate optimal embedding batch size based on GPU memory and chunk count.

    Adjusts batch size downward when GPU memory is limited or chunk count is high
    to prevent OOM errors during batch embedding.

    Args:
        settings: Global settings with BATCH_SIZE_INDEXING/MIN/MAX parameters.
        chunk_count: Total number of chunks to process.

    Returns:
        Clamped batch size within [MIN_BATCH_SIZE, MAX_BATCH_SIZE].
    """
    base_batch_size = int(getattr(settings, "BATCH_SIZE_INDEXING", 10))
    min_batch_size = int(getattr(settings, "MIN_BATCH_SIZE", 1))
    max_batch_size = int(getattr(settings, "MAX_BATCH_SIZE", 8))

    if chunk_count > 100:
        base_batch_size = min(base_batch_size, 4)
    elif chunk_count > 50:
        base_batch_size = min(base_batch_size, 6)

    batch_size = base_batch_size

    try:
        from core.memory_manager import get_memory_manager
        memory_manager = get_memory_manager()

        if memory_manager:
            mem_stats = memory_manager.get_memory_stats()
            free_memory = mem_stats.free_memory

            memory_thresholds = [
                (1.0, 2),
                (2.0, 4),
                (3.0, 6),
            ]

            for threshold, max_size in memory_thresholds:
                if free_memory < threshold:
                    batch_size = min(batch_size, max_size)
                    break
            else:
                batch_size = min(batch_size, max_batch_size)

            logger.info(f"优化内存使用：可用内存{free_memory:.2f}GB，分块数{chunk_count}，批处理大小调整为{batch_size}")
    except Exception as e:
        logger.warning(f"动态批处理调整失败: {e}")

    return max(min_batch_size, min(batch_size, max_batch_size))


async def _update_doc_status(doc_id: str, db: DocumentDB, **updates):
    """Update document status with existence check and logging.

    Silently skips if the document no longer exists (e.g., deleted during processing).
    """
    if await db.exists(doc_id):
        await db.update(doc_id, updates)
        stage = updates.get("stage", "")
        progress = updates.get("progress", "")
        if stage or progress != "":
            logger.info(f"文档 {doc_id} 状态更新: {stage}, {progress}%")


async def _ensure_memory_available(threshold_percent: float = 70.0, context: str = ""):
    """Ensure sufficient GPU memory is available, triggering cleanup if needed.

    Cleanup strategy (light to heavy):
      1. PyTorch cache cleanup + defragmentation
      2. Unload inactive models

    Args:
        threshold_percent: GPU utilization threshold to trigger cleanup.
        context: Description for log messages.
    """
    try:
        from core.memory_manager import get_memory_manager
        memory_manager = get_memory_manager()

        if not memory_manager:
            return

        mem_stats = memory_manager.get_memory_stats()
        context_str = f"({context}) " if context else ""
        logger.info(f"{context_str}GPU内存状态: 使用率={mem_stats.utilization_percent:.1f}%, 可用={mem_stats.free_memory:.2f}GB")

        if mem_stats.utilization_percent > threshold_percent:
            logger.warning(f"{context_str}内存使用率过高 ({mem_stats.utilization_percent:.1f}%), 执行清理")
            memory_manager.cleanup(aggressive=True)

            try:
                from core.model_manager import model_manager
                model_manager.unload_inactive_models()
                logger.info(f"{context_str}已卸载不活跃模型")
            except Exception as unload_error:
                logger.warning(f"{context_str}卸载模型失败: {unload_error}")

            mem_stats_after = memory_manager.get_memory_stats()
            logger.info(f"{context_str}清理后GPU内存: 使用率={mem_stats_after.utilization_percent:.1f}%, 可用={mem_stats_after.free_memory:.2f}GB")
    except Exception as e:
        logger.warning(f"内存检查失败: {e}")


def _split_sentences_with_spans(text: str) -> List[Dict[str, Any]]:
    """Split text at sentence boundaries, recording character offsets for each sentence.

    Splits on Chinese/English sentence-ending punctuation (。！？!?) and
    English period followed by whitespace or end-of-string (to avoid splitting
    decimal numbers like "3.5V").

    Args:
        text: Input text (typically a PDF page).

    Returns:
        List of dicts with "text", "start", and "end" keys. Empty list for empty input.
    """
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    pattern = re.compile(r".+?(?:[。！？!?]|\.(?=\s|$))")
    sentences = []
    last_end = 0
    for match in pattern.finditer(text):
        sentence = match.group().strip()
        if sentence:
            sentences.append({
                "text": sentence,
                "start": match.start(),
                "end": match.end()
            })
        last_end = match.end()

    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            start = text.find(remaining, last_end)
            sentences.append({
                "text": remaining,
                "start": start,
                "end": start + len(remaining)
            })

    return sentences


def _build_chunks_from_sentences(sentences: List[Dict[str, Any]],
                                 chunk_size: int,
                                 overlap: int,
                                 settings=None) -> List[Dict[str, Any]]:
    """Assemble sentences into fixed-size chunks using a sliding window strategy.

    Overlap between adjacent chunks ensures that information spanning chunk
    boundaries is not lost. Overly long single sentences are split into
    sub-chunks with step = chunk_size - overlap.

    Args:
        sentences: Sentence list from _split_sentences_with_spans.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap characters between adjacent chunks.
        settings: Settings for MAX_CHUNKS_PER_PAGE and MAX_TOTAL_CHUNKS limits.

    Returns:
        List of chunk dicts with "text", "start", and "end" keys.

    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size.
    """
    if not sentences:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    if overlap >= chunk_size:
        raise ValueError("overlap不能大于等于chunk_size")

    chunks: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    current_len = 0

    if settings is None:
        settings = get_settings()
    max_chunks_per_page = getattr(settings, 'MAX_CHUNKS_PER_PAGE', 10000)
    max_total_chunks = getattr(settings, 'MAX_TOTAL_CHUNKS', 100000)

    def flush_current():
        if not current:
            return None
        text = " ".join(s["text"] for s in current).strip()
        if not text:
            return None
        return {
            "text": text,
            "start": current[0]["start"],
            "end": current[-1]["end"],
        }

    i = 0
    while i < len(sentences):
        if len(chunks) >= max_total_chunks:
            logger.warning(f"达到最大chunk限制 {max_total_chunks}，停止分块")
            break

        sent = sentences[i]
        sent_text = sent["text"]
        sent_len = len(sent_text)

        if not current:
            if sent_len > chunk_size:
                step = max(chunk_size - overlap, 1)
                base_start = sent["start"]
                pos = 0
                chunks_created = 0
                while pos < sent_len and chunks_created < max_chunks_per_page:
                    sub_text = sent_text[pos:pos + chunk_size]
                    chunks.append({
                        "text": sub_text,
                        "start": base_start + pos,
                        "end": base_start + pos + len(sub_text)
                    })
                    pos += step
                    chunks_created += 1
                i += 1
                continue

            current = [sent]
            current_len = sent_len
            i += 1
            continue

        if current_len + 1 + sent_len <= chunk_size:
            current.append(sent)
            current_len += 1 + sent_len
            i += 1
            continue

        chunk = flush_current()
        if chunk:
            chunks.append(chunk)

        if overlap > 0 and current:
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current):
                s_len = len(s["text"])
                if overlap_len + s_len <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += s_len
                else:
                    break
            current = overlap_sents
            current_len = sum(len(s["text"]) for s in current) + max(0, len(current) - 1)
        else:
            current = []
            current_len = 0

    chunk = flush_current()
    if chunk:
        chunks.append(chunk)

    return chunks


def _extract_page_text_and_number(page) -> Tuple[str, int]:
    """Extract text content and page number from a page object (dict or object).

    Args:
        page: Page data as dict or object with text/page_number attributes.

    Returns:
        (text, page_number) tuple. Defaults to page 1 if number is missing.
    """
    if isinstance(page, dict):
        text = page.get("text", "")
        page_number = page.get("page_number") or page.get("page") or page.get("page_index")
    else:
        text = getattr(page, "text", "")
        page_number = getattr(page, "page_number", None) or getattr(page, "page", None)

    if not page_number:
        page_number = 1

    return text or "", int(page_number)


def _build_chunks_from_pages(pages: List[Any], settings, document_id: str, filename: str) -> List[Dict[str, Any]]:
    """Convert all PDF pages into structured text chunks with IDs and metadata.

    For each page: extract text → split sentences → sliding-window chunking →
    assign chunk IDs ({doc_id}_p{page}_c{index}) and metadata.

    Args:
        pages: PDF page list from IngestionPipeline.process().
        settings: Global settings for CHUNK_SIZE and CHUNK_OVERLAP.
        document_id: Document unique identifier.
        filename: Original file name.

    Returns:
        List of chunk dicts with id/content/metadata fields.
    """
    chunk_size = int(getattr(settings, "CHUNK_SIZE", 800))
    overlap = int(getattr(settings, "CHUNK_OVERLAP", 200))

    all_chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    for page in pages:
        text, page_number = _extract_page_text_and_number(page)
        sentences = _split_sentences_with_spans(text)
        page_chunks = _build_chunks_from_sentences(sentences, chunk_size, overlap, settings)

        for chunk in page_chunks:
            chunk_id = f"{document_id}_p{page_number}_c{chunk_index}"
            all_chunks.append({
                "id": chunk_id,
                "content": chunk["text"],
                "metadata": {
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number,
                    "char_start": chunk["start"],
                    "char_end": chunk["end"],
                    "source": "pdf"
                }
            })
            chunk_index += 1

    return all_chunks


def _extract_circuit_types_from_text(text: str) -> List[str]:
    """Extract circuit types from VLM raw response text as a fallback strategy.

    Used when structured parsing fails (circuit_type=[]), keyword matching
    ensures classification information is not lost.

    Args:
        text: VLM raw response text.

    Returns:
        List of matched circuit type strings.
    """
    text_lower = text.lower()
    found_types = []

    type_keywords = {
        "amplifier": ["amplifier", "op-amp", "opamp", "放大", "运放", "增益", "gain", "放大器", "放大电路", "信号放大", "功放", "差分", "differential", "反相", "inverting", "同相", "non-inverting", "反馈", "feedback", "电压跟随", "follower", "偏置", "bias", "ne5532", "lm358", "tl072", "lm741"],
        "filter": ["filter", "滤波", "低通", "高通", "带通", "lpf", "hpf", "bpf", "有源滤波", "sallen", "巴特沃斯", "butterworth", "陷波", "notch", "抗混叠", "anti-aliasing"],
        "oscillator": ["oscillator", "crystal", "hse", "hsi", "晶振", "振荡", "时钟", "clock", "vco", "压控振荡", "正弦波", "方波", "谐振", "resonant", "波形发生"],
        "power": ["power supply", "vdd", "vss", "vcc", "gnd", "regulator", "ldo", "dc-dc", "电源", "供电", "稳压", "接地", "ground", "去耦", "decoupling", "旁路", "bypass", "纹波", "ripple", "buck", "boost"],
        "rectifier": ["rectifier", "整流", "桥式", "半波", "全波", "二极管桥", "倍压"],
        "reset": ["reset", "nrst", "复位", "重启", "上电复位", "por", "看门狗", "watchdog"],
        "clock": ["clock", "timing", "clk", "时钟", "时钟分配", "clock buffer"],
        "driver": ["driver", "驱动", "电机驱动", "led驱动", "mosfet驱动", "h桥", "h-bridge", "pwm", "栅极驱动", "gate driver"],
        "comparator": ["comparator", "比较器", "lm393", "lm339", "阈值", "threshold", "迟滞", "hysteresis", "电压比较"],
        "timer": ["timer", "定时", "555", "ne555", "延时", "delay", "单稳态", "脉冲"],
        "protection": ["protection", "保护", "esd", "tvs", "过流", "overcurrent", "过压", "保险丝", "fuse", "限流", "浪涌", "surge"],
        "interface": ["interface", "接口", "uart", "spi", "i2c", "rs232", "rs485", "can", "usb", "电平转换", "收发器", "transceiver"],
        "sensor": ["sensor", "传感器", "检测", "测量", "温度", "压力", "光照", "霍尔", "热电偶", "信号采集", "变送器"],
        "regulator": ["regulator", "稳压", "ldo", "线性稳压", "开关稳压", "7805", "7812", "lm317", "tl431", "基准电压", "zener", "齐纳"],
        "buffer": ["buffer", "缓冲", "电压跟随器", "voltage follower", "单位增益", "阻抗变换", "线路驱动"],
        "logic": ["logic", "逻辑", "门电路", "gate", "触发器", "flip-flop", "cmos", "ttl"],
        "boot": ["boot", "startup", "启动", "引导", "bootstrap"],
        "decoupling": ["decoupling", "bypass", "去耦", "滤波电容", "旁路", "退耦"],
        "adc_dac": ["adc", "dac", "模数转换", "数模转换", "采样", "sampling", "参考电压", "reference"],
        "charge_pump": ["charge pump", "电荷泵", "电压反转", "倍压", "负电压"],
        "current_sense": ["current sense", "电流检测", "电流采样", "分流器", "shunt"],
        "emi_filter": ["emi", "emc", "共模电感", "磁珠", "ferrite", "噪声滤波", "电磁兼容"],
        "battery_mgmt": ["battery", "电池", "bms", "充电", "charging", "锂电池", "li-ion"],
        "pll": ["pll", "锁相环", "phase-locked", "频率合成", "鉴相器"],
        "sample_hold": ["sample and hold", "采样保持", "峰值检测", "模拟开关", "多路复用"],
    }

    for circuit_type, keywords in type_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                found_types.append(circuit_type)
                break

    return found_types


async def _index_document_content(document_id: str, filename: str, result, container, settings) -> Dict[str, int]:
    """Core indexing function: write parsed PDF content into vector/BM25/table stores.

    Flow: text chunking → batch embedding → vector store → BM25 store → table store.

    GPU memory is managed with dynamic batch sizing, pre-batch memory checks,
    and post-batch cache cleanup to prevent OOM errors.

    Args:
        document_id: Document unique identifier.
        filename: File name.
        result: IngestionPipeline.process() result with pages, tables, and stats.
        container: ServiceContainer providing index components.
        settings: Global settings.

    Returns:
        Dict with chunk_count, table_count, circuit_count, pages, and visual_indexer.

    Raises:
        RuntimeError: If pages are empty, components uninitialized, or GPU OOM.
    """
    logger.info(f"开始索引文档内容: {document_id} ({filename})")

    await _ensure_memory_available(threshold_percent=50.0, context="索引开始前")

    pages = getattr(result, "processed_pages", []) or []
    extracted_tables = getattr(result, "extracted_tables", []) or []

    logger.info(f"文档处理结果: 页面数={len(pages)}, 表格数={len(extracted_tables)}")

    if not pages:
        raise RuntimeError("文档处理结果缺少页面内容，无法进行索引")

    vector_store = getattr(container, "vector_store", None)
    bm25_store = getattr(container, "bm25_store", None)
    table_store = getattr(container, "table_store", None)
    embedding_manager = getattr(container, "embedding_manager", None)
    visual_indexer = getattr(container, "visual_indexer", None)

    logger.info(f"索引组件检查: vector_store={vector_store is not None}, bm25_store={bm25_store is not None}, table_store={table_store is not None}, embedding_manager={embedding_manager is not None}, visual_indexer={visual_indexer is not None}")

    if not vector_store or not bm25_store or not table_store or not embedding_manager:
        raise RuntimeError("索引组件未完全初始化，无法进行完整索引")

    try:
        chunks = _build_chunks_from_pages(pages, settings, document_id, filename)
        logger.info(f"生成文本分块: {len(chunks)} 个")

        if not chunks:
            raise RuntimeError("未生成任何文本分块，无法进行索引")

        db = get_documents_db()
        await _update_doc_status(document_id, db, stage="building_chunks", progress=40)

        batch_size = _calculate_optimal_batch_size(settings, len(chunks))

        logger.info(f"开始向量索引: 批处理大小={batch_size}, 总分块数={len(chunks)}")

        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [item["content"] for item in batch]

            batch_idx = i // batch_size

            current_progress = 45 + int((batch_idx / total_batches) * 27)
            await _update_doc_status(document_id, db,
                stage="indexing_vectors",
                progress=current_progress,
                stage_detail=f"正在嵌入第 {batch_idx + 1}/{total_batches} 批"
            )

            if batch_idx % 3 == 0:
                try:
                    from core.memory_manager import get_memory_manager
                    memory_manager = get_memory_manager()

                    if memory_manager:
                        mem_stats = memory_manager.get_memory_stats()
                        if mem_stats.utilization_percent > 50:
                            logger.info(f"批处理前主动清理内存：使用率{mem_stats.utilization_percent:.1f}%")
                            memory_manager.cleanup(aggressive=True)
                except Exception as e:
                    logger.warning(f"批处理前内存清理失败: {e}")

            logger.info(f"正在嵌入第 {i//batch_size + 1} 批文本 (共 {(len(chunks) + batch_size - 1)//batch_size} 批)")

            loop = asyncio.get_event_loop()
            try:
                embeddings = await loop.run_in_executor(
                    _inference_executor,
                    lambda t=texts: embedding_manager.embed_batch(t, use_cache=False)
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    logger.error(f"嵌入过程OOM: {e}")
                    raise RuntimeError(f"GPU内存不足，请减小batch_size或增加显存: {e}") from e
                else:
                    raise

            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()

            logger.info(f"正在添加第 {i//batch_size + 1} 批向量到存储")
            await vector_store.add_documents(batch, embeddings=embeddings)

            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"批处理后内存清理失败: {e}")

            logger.info(f"第 {i//batch_size + 1} 批向量添加完成")

        logger.info(f"向量索引完成")

        await _update_doc_status(document_id, db, stage="vector_indexed", progress=72)

        logger.info(f"开始BM25索引")
        bm25_texts = [item["content"] for item in chunks]
        bm25_metadatas = [item["metadata"] for item in chunks]
        bm25_ids = [item["id"] for item in chunks]
        bm25_store.add_documents(bm25_texts, bm25_metadatas, bm25_ids)
        bm25_store.save()
        logger.info(f"BM25索引完成")

        await _update_doc_status(document_id, db, stage="bm25_indexed", progress=80)

        logger.info(f"开始表格索引")
        table_count = 0
        for table in extracted_tables:
            table_payload = dict(table)
            table_payload["document_id"] = document_id
            table_payload["filename"] = filename
            table_store.add(table_payload)
            table_count += 1

        logger.info(f"表格索引完成: {table_count} 个表格")

        await _update_doc_status(document_id, db, stage="tables_indexed", progress=88)

        return {
            "chunk_count": len(chunks),
            "table_count": table_count,
            "circuit_count": 0,
            "pages": pages,
            "visual_indexer": visual_indexer,
        }

    except MemoryError as e:
        logger.error(f"内存不足，文档索引失败: {e}")
        logger.error(f"尝试减少CHUNK_SIZE或BATCH_SIZE_INDEXING配置")
        raise RuntimeError(f"内存不足，无法完成文档索引。建议减少CHUNK_SIZE或BATCH_SIZE_INDEXING配置值。") from e
    except Exception as e:
        logger.error(f"文档索引过程中发生错误: {e}", exc_info=True)
        raise


class DocumentInfo(BaseModel):
    """Single document status model for list and detail responses."""

    document_id: str
    filename: str
    status: str = "pending"
    page_count: int = 0
    file_size: int = 0
    upload_time: str = ""
    error_message: str = ""
    warnings: List[str] = Field(default_factory=list)
    chunk_count: int = 0
    table_count: int = 0
    circuit_count: int = 0
    image_count: int = 0
    processed_time: str = ""
    progress: int = 0
    stage: str = ""
    stage_detail: str = ""


class DocumentListResponse(BaseModel):
    """Document list response model."""
    success: bool = True
    documents: List[DocumentInfo] = Field(default_factory=list)
    total: int = 0


class UploadResponse(BaseModel):
    """Document upload response model."""
    success: bool = True
    document_id: str
    filename: str
    message: str = "上传成功"


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents with their current status."""
    try:
        db = get_documents_db()
        all_docs = await db.get_all()

        doc_list = []
        for doc_id, doc in all_docs.items():
            doc_info = DocumentInfo(
                document_id=doc_id,
                filename=doc.get("filename", ""),
                status=doc.get("status", "unknown"),
                page_count=doc.get("page_count", 0),
                file_size=doc.get("file_size", 0),
                upload_time=doc.get("upload_time", ""),
                error_message=doc.get("error_message", ""),
                warnings=doc.get("warnings", []),
                chunk_count=doc.get("chunk_count", 0),
                table_count=doc.get("table_count", 0),
                circuit_count=doc.get("circuit_count", 0),
                image_count=doc.get("image_count", 0),
                processed_time=doc.get("processed_time", ""),
                progress=doc.get("progress", 0),
                stage=doc.get("stage", ""),
                stage_detail=doc.get("stage_detail", ""),
            )
            doc_list.append(doc_info)

        return DocumentListResponse(documents=doc_list, total=len(doc_list))

    except Exception as e:
        logger.error(f"获取文档列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF document for async processing.

    Validates file type and size, saves to disk, checks for duplicate filenames,
    registers in DocumentDB, and schedules background processing. Returns
    immediately with a document_id for progress polling.

    Args:
        background_tasks: FastAPI background task queue.
        file: Uploaded PDF file.

    Returns:
        UploadResponse with document_id and filename.
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支持PDF文件")

        content = await file.read()
        file_size = len(content)

        if file_size > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件太大，最大支持100MB")

        doc_id = str(uuid.uuid4())[:8]

        settings = get_settings()
        upload_dir = settings.UPLOAD_DIR
        upload_dir.mkdir(parents=True, exist_ok=True)

        filepath = upload_dir / f"{doc_id}_{file.filename}"
        with open(filepath, "wb") as f:
            f.write(content)

        cleanup_manager = get_cleanup_manager()

        doc_ids_to_cleanup = await cleanup_manager.find_duplicate_documents(file.filename)

        if doc_ids_to_cleanup:
            logger.warning(f"发现同名文件: {file.filename}, 旧文档ID: {doc_ids_to_cleanup}")
            try:
                container = get_service_container()
                for existing_doc_id in doc_ids_to_cleanup:
                    await cleanup_manager.cleanup_orphan_document(existing_doc_id, container)
                    logger.info(f"已清理同名文件的旧数据: {existing_doc_id}")
            except Exception as cleanup_error:
                logger.error(f"清理同名文件旧数据失败: {cleanup_error}")

        db = get_documents_db()
        await db.set(doc_id, {
            "filename": file.filename,
            "filepath": str(filepath),
            "status": "uploading",
            "progress": 0,
            "stage": "uploading",
            "page_count": 0,
            "file_size": file_size,
            "upload_time": datetime.now().isoformat(),
        })

        logger.info(f"文档上传成功: {file.filename} -> {doc_id}, 开始后台处理")

        background_tasks.add_task(_process_document_delayed, doc_id, file.filename, str(filepath))

        return UploadResponse(document_id=doc_id, filename=file.filename)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"文档上传失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


async def _process_document_delayed(doc_id: str, filename: str, filepath: str):
    """Background task entry point: lazily initialize dependencies and process document.

    Args:
        doc_id: Document unique identifier.
        filename: File name.
        filepath: File storage path.
    """
    try:
        from api.dependencies import get_service_container
        container = get_service_container()
        settings = get_settings()

        await _process_document_async(doc_id, filename, filepath, container, settings)
    except Exception as e:
        logger.error(f"延迟处理文档失败: {doc_id}, {filename}, {e}", exc_info=True)
        db = get_documents_db()
        await _update_doc_status(doc_id, db,
            status="error",
            stage="error",
            progress=0,
            error_message=f"文档处理初始化失败: {str(e)}",
            processed_time=datetime.now().isoformat(),
        )


async def _index_circuits_background(doc_id: str, filename: str, pages: list, visual_indexer, container, settings):
    """Background circuit diagram indexing: ColPali embedding → VLM analysis → metadata update.

    Uses CLIP pre-filtering results from the PDF parsing stage to skip
    non-circuit images, then runs ColPali embedding and VLM deep analysis
    only on candidate circuit diagrams.

    Args:
        doc_id: Document unique identifier.
        filename: File name.
        pages: PDF page data with image info.
        visual_indexer: Visual indexer instance.
        container: ServiceContainer.
        settings: Global settings.
    """
    try:
        logger.info(f"开始后台电路索引(三阶段流水线-优化版): {doc_id}, 页数={len(pages)}")

        if not visual_indexer:
            logger.warning(f"电路索引跳过: visual_indexer未初始化")
            return

        candidate_images = []
        total_images = 0
        skipped_no_image = 0
        skipped_non_circuit = 0

        for page in pages:
            if isinstance(page, dict):
                page_num = page.get("page_number", 0)
                images = page.get("images", []) or []
                page_text = page.get("text", "") or ""
            else:
                page_num = getattr(page, "page_number", 0)
                images = getattr(page, "images", []) or []
                page_text = getattr(page, "text", "") or ""

            if not images:
                skipped_no_image += 1
                continue

            for img_idx, image_info in enumerate(images):
                total_images += 1
                image_data = image_info.get("data")
                if not image_data:
                    continue

                is_circuit = image_info.get("is_circuit")
                clip_type = image_info.get("clip_type", "")
                clip_confidence = image_info.get("clip_confidence", 0.0)

                if is_circuit is False:
                    skipped_non_circuit += 1
                    logger.debug(f"跳过非电路图: 页{page_num}-图{img_idx}, clip_type={clip_type}")
                    continue

                if clip_type in ["chip_package", "pin_diagram", "pcb_layout",
                                 "mechanical_drawing", "package_materials",
                                 "table", "logo", "photo", "truth_table",
                                 "state_diagram", "pinout_diagram"]:
                    skipped_non_circuit += 1
                    logger.debug(f"跳过已知非电路类型: 页{page_num}-图{img_idx}, clip_type={clip_type}")
                    continue

                candidate_images.append({
                    'page_num': page_num,
                    'img_idx': img_idx,
                    'image_info': image_info,
                    'image_data': image_data,
                    'document_id': doc_id,
                    'filename': filename,
                    'page_text': page_text,
                    'clip_type': clip_type,
                    'clip_confidence': clip_confidence,
                    'is_circuit': is_circuit,
                })

        logger.info(f"图像统计: 总数={total_images}, 候选电路图={len(candidate_images)}, "
                   f"跳过无图页={skipped_no_image}, 跳过非电路={skipped_non_circuit}")

        if not candidate_images:
            logger.info(f"没有候选电路图需要处理")
            db = get_documents_db()
            await _update_doc_status(doc_id, db, status="ready", stage="completed", progress=100, circuit_count=0, warnings=[])
            return

        db = get_documents_db()
        await _update_doc_status(doc_id, db, status="circuit_indexing", stage="colpali_embedding", progress=5, warnings=[])

        preloaded_images = []
        preloaded_page_numbers = []
        for task in candidate_images:
            img_data = task.get('image_data')
            if img_data:
                try:
                    from PIL import Image
                    import io
                    if isinstance(img_data, bytes):
                        img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    else:
                        img = img_data
                    preloaded_images.append(img)
                    preloaded_page_numbers.append(task.get('page_num', 0))
                except Exception as e:
                    logger.warning(f"图像转换失败: {e}")

        if preloaded_images and hasattr(visual_indexer, 'index_document'):
            try:
                logger.info(f"阶段0: ColPali嵌入生成，共{len(preloaded_images)}张候选电路图")

                document_path = settings.UPLOAD_DIR / f"{doc_id}_{filename}"

                index_result = visual_indexer.index_document(
                    document_path=document_path,
                    metadata={'document_id': doc_id, 'filename': filename},
                    preloaded_images=preloaded_images,
                    page_numbers=preloaded_page_numbers if preloaded_page_numbers else None
                )
                logger.info(f"ColPali嵌入生成完成: {index_result}")

                preloaded_images.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"ColPali嵌入生成失败: {e}", exc_info=True)

        images_dir = Path(settings.DATA_DIR) / "circuit_images" / doc_id
        images_dir.mkdir(parents=True, exist_ok=True)

        vlm_tasks = []
        for task in candidate_images:
            image_filename = f"page{task['page_num']}_img{task['img_idx']}.png"
            image_path = str(images_dir / image_filename)

            if not Path(image_path).exists():
                with open(image_path, "wb") as f:
                    f.write(task['image_data'])

            vlm_tasks.append({
                **task,
                'image_path': image_path,
            })

        await _update_doc_status(doc_id, db, status="circuit_indexing", stage="vlm_analysis", progress=30, warnings=[])

        vlm_results = await _phase2_vlm_analysis(vlm_tasks, visual_indexer, settings, doc_id, db, len(candidate_images))

        await _update_doc_status(doc_id, db, status="circuit_indexing", stage="embedding_generation", progress=70, warnings=[])

        circuit_count = await _phase3_embedding_storage(vlm_results, None, None, None,
                                                        visual_indexer, doc_id, db, len(candidate_images))

        db = get_documents_db()
        await _update_doc_status(doc_id, db, status="ready", stage="completed", progress=100,
                                circuit_count=circuit_count, warnings=[])
        logger.info(f"后台电路索引完成: {doc_id}, 电路数={circuit_count}, 状态: ready")

        vlm_tasks.clear()
        vlm_results.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"后台电路索引失败: {doc_id}, {e}", exc_info=True)
        db = get_documents_db()
        await _update_doc_status(doc_id, db, status="error", stage="error", progress=0,
                                error_message=f"电路索引失败: {str(e)}", processed_time=datetime.now().isoformat())


async def _phase2_vlm_analysis(clip_results: list, visual_indexer, settings, doc_id, db, total_images: int) -> list:
    """Phase 2: VLM deep analysis of candidate circuit diagrams.

    Loads VLM model, analyzes each candidate image, then unloads the model
    to free GPU memory.

    Args:
        clip_results: Candidate image tasks with image_path and clip metadata.
        visual_indexer: Visual indexer with VLM analysis capability.
        settings: Global settings.
        doc_id: Document ID for progress updates.
        db: DocumentDB for status updates.
        total_images: Total candidate count for progress calculation.

    Returns:
        List of confirmed circuit diagram tasks with VLM analysis results.
    """
    if not clip_results:
        logger.info("阶段1跳过: 没有候选电路图需要VLM分析")
        return []

    logger.info(f"阶段1: VLM深度分析开始，共 {len(clip_results)} 张候选电路图")

    vlm_results = []
    processed = 0

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _inference_executor,
            lambda: visual_indexer._load_model(model_type="qwen")
        )
        logger.info("VLM模型加载完成")

        for task in clip_results:
            try:
                image_path = task['image_path']
                clip_type = task.get('clip_type', None)
                clip_confidence = task.get('clip_confidence', 0.0)

                loop = asyncio.get_event_loop()
                analysis = await loop.run_in_executor(
                    _inference_executor,
                    lambda p=image_path, ct=clip_type: visual_indexer.analyze_circuit_image(
                        p, annotation_type="auto", clip_type=ct
                    )
                )

                if analysis and analysis.get("success"):
                    circuit_data = analysis.get("analysis", {})
                    is_circuit = analysis.get("is_circuit", False)

                    if is_circuit:
                        vlm_results.append({
                            **task,
                            'vlm_analysis': circuit_data,
                            'is_circuit': True
                        })
                        logger.info(f"  VLM确认电路图: 页{task['page_num']}-图{task['img_idx']}")
                    else:
                        logger.info(f"  VLM判定非电路图: 页{task['page_num']}-图{task['img_idx']}")
                else:
                    logger.warning(f"  VLM分析失败: 页{task['page_num']}-图{task['img_idx']}")

                processed += 1
                if processed % 3 == 0:
                    progress = int(30 + (processed / len(clip_results)) * 30)
                    await _update_doc_status(doc_id, db, status="circuit_indexing", stage="vlm_analysis",
                                            progress=progress, warnings=[])

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"  VLM处理失败: 页{task['page_num']}-图{task['img_idx']}, {e}")

        if hasattr(visual_indexer, 'unload_model'):
            visual_indexer.unload_model()
            logger.info("VLM模型已释放")
        elif hasattr(visual_indexer, '_colpali_model') and visual_indexer._colpali_model is not None:
            del visual_indexer._colpali_model
            visual_indexer._colpali_model = None
            visual_indexer._qwen_loaded = False
            logger.info("VLM模型已释放")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"阶段1完成: VLM确认 {len(vlm_results)} 张电路图")

    except Exception as e:
        logger.error(f"阶段2 VLM分析失败: {e}", exc_info=True)

    return vlm_results


async def _phase3_embedding_storage(vlm_results: list, vector_store, bm25_store,
                                    embedding_manager, visual_indexer, doc_id, db, total_images: int) -> int:
    """Phase 3: Update visual_indexer metadata with VLM analysis results.

    Circuit retrieval uses the independent visual_indexer (ColPali multi-vector
    embeddings), not the text/table stores. This phase only updates metadata
    (caption, circuit_type, components) and persists the index to disk.

    Args:
        vlm_results: Confirmed circuit diagrams with VLM analysis.
        vector_store: Unused (kept for interface compatibility).
        bm25_store: Unused (kept for interface compatibility).
        embedding_manager: Unused (kept for interface compatibility).
        visual_indexer: Visual indexer for metadata updates.
        doc_id: Document ID.
        db: DocumentDB for progress updates.
        total_images: Total count for progress calculation.

    Returns:
        Number of successfully indexed circuit diagrams.
    """
    if not vlm_results:
        logger.info("阶段3跳过: 没有电路图需要存储")
        return 0

    logger.info(f"阶段3: 更新visual_indexer元数据，共 {len(vlm_results)} 张电路图")

    circuit_count = 0
    processed = 0

    try:
        for task in vlm_results:
            try:
                image_path = task['image_path']
                vlm_analysis = task.get('vlm_analysis', {})

                circuit_type = vlm_analysis.get('circuit_type', 'unknown')
                if isinstance(circuit_type, str):
                    circuit_type = [circuit_type]
                elif not isinstance(circuit_type, list):
                    circuit_type = []

                components = vlm_analysis.get('components', [])
                raw_response = vlm_analysis.get('raw_response', '')

                if raw_response:
                    clean_response = raw_response
                    template_patterns = [
                        r'关键信息摘要\s*[:：]?\s*',
                        r'根据您提供的图片内容[，,]?\s*',
                        r'以下是详细的分析和回答\s*[:：]?\s*',
                        r'根据图片内容[，,]?\s*',
                    ]
                    for pattern in template_patterns:
                        clean_response = re.sub(pattern, '', clean_response, flags=re.IGNORECASE)

                    clean_response = re.sub(r'^#{1,6}\s*', '', clean_response, flags=re.MULTILINE)
                    clean_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_response)
                    clean_response = re.sub(r'\*([^*]+)\*', r'\1', clean_response)
                    clean_response = re.sub(r'\s+', ' ', clean_response).strip()

                    if len(clean_response) > 200:
                        last_period = clean_response[:200].rfind('。')
                        last_newline = clean_response[:200].rfind('\n')
                        last_semicolon = clean_response[:200].rfind('；')
                        cut_point = max(last_period, last_newline, last_semicolon)
                        if cut_point > 100:
                            caption = clean_response[:cut_point + 1]
                        else:
                            caption = clean_response[:200]
                    else:
                        caption = clean_response
                else:
                    caption = f"{' '.join(circuit_type)} 电路图" if circuit_type else "电路图"

                confidence = vlm_analysis.get('overall_confidence', 0.5)
                figure_label = vlm_analysis.get('figure_label', '')

                if visual_indexer and hasattr(visual_indexer, 'update_page_metadata'):
                    try:
                        visual_indexer.update_page_metadata(
                            document_id=doc_id,
                            page_num=task['page_num'],
                            metadata_updates={
                                'caption': caption,
                                'circuit_type': circuit_type,
                                'components': components,
                                'figure_label': figure_label,
                                'clip_type': task.get('clip_type', ''),
                                'clip_confidence': task.get('clip_confidence', 0.0),
                                'overall_confidence': confidence,
                                'is_circuit': True,
                            }
                        )
                        logger.debug(f"更新元数据: 页{task['page_num']}, 类型={circuit_type}, 图号={figure_label}")
                    except Exception as meta_err:
                        logger.warning(f"  visual_indexer元数据更新失败: {meta_err}")

                circuit_count += 1
                processed += 1

                if processed % 3 == 0:
                    progress = int(60 + (processed / len(vlm_results)) * 35)
                    await _update_doc_status(doc_id, db, status="circuit_indexing", stage="embedding_generation",
                                            progress=progress, circuit_count=circuit_count, warnings=[])

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"  元数据更新失败: 页{task['page_num']}-图{task['img_idx']}, {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if visual_indexer and hasattr(visual_indexer, 'flush_index_to_disk'):
            try:
                visual_indexer.flush_index_to_disk()
                logger.info("visual_indexer元数据已持久化到磁盘")
            except Exception as flush_err:
                logger.warning(f"visual_indexer持久化失败: {flush_err}")

        logger.info(f"阶段2完成: 更新 {circuit_count} 张电路图元数据")

    except Exception as e:
        logger.error(f"阶段3 元数据更新失败: {e}", exc_info=True)

    return circuit_count


async def _process_document_async(doc_id: str, filename: str, filepath: str, container, settings):
    """Two-phase document processing: core indexing (sync) + circuit indexing (background).

    Phase 1 (blocking): PDF parsing → chunking → vector/BM25/table indexing.
    Once complete, document status is set to "ready" for immediate use.

    Phase 2 (background): Circuit diagram indexing via asyncio.create_task,
    which continues without blocking the user.

    Args:
        doc_id: Document unique identifier.
        filename: File name.
        filepath: File storage path.
        container: ServiceContainer.
        settings: Global settings.
    """
    try:
        logger.info(f"开始异步处理文档: {doc_id}, {filename}")

        await _ensure_memory_available(threshold_percent=70.0, context="文档处理前")

        db = get_documents_db()
        await _update_doc_status(doc_id, db, status="processing", stage="parsing_pdf", progress=5)

        await _update_doc_status(doc_id, db, stage="parsing_pdf", progress=10)

        pipeline = container.ingestion_pipeline
        result = await pipeline.process(filepath, document_id=doc_id, filename=filename)
        logger.info(f"文档 {doc_id} 处理完成: {result.status}")

        if result.status != "success":
            raise RuntimeError(getattr(result, "error_message", "文档处理未成功"))

        await _update_doc_status(doc_id, db, stage="parsing_complete", progress=30)
        await _update_doc_status(doc_id, db, stage="indexing_vectors", progress=35)

        index_stats = await _index_document_content(
            document_id=doc_id,
            filename=filename,
            result=result,
            container=container,
            settings=settings
        )
        logger.info(f"文档 {doc_id} 核心索引完成: {index_stats}")

        await _update_doc_status(doc_id, db,
            status="ready",
            stage="completed",
            progress=100,
            page_count=getattr(result, 'page_count', 0),
            chunk_count=index_stats.get("chunk_count", getattr(result, 'chunk_count', 0)),
            table_count=index_stats.get("table_count", getattr(result, 'table_count', 0)),
            circuit_count=0,
            image_count=getattr(result, 'image_count', 0),
            error_message="",
            warnings=[],
            processed_time=datetime.now().isoformat(),
        )
        logger.info(f"文档核心处理完成: {filename} -> {doc_id}, 状态: ready (文本/表格/BM25索引已就绪, 电路索引将在后台进行)")

        pages = index_stats.get("pages", [])
        visual_indexer = index_stats.get("visual_indexer")

        if visual_indexer and pages:
            lightweight_pages = []
            for p in pages:
                lp = {
                    "page_number": getattr(p, "page_number", 0),
                    "images": getattr(p, "images", []) or [],
                    "text": getattr(p, "text", "") or "",
                }
                lightweight_pages.append(lp)

            index_stats["pages"] = None

            if hasattr(result, 'processed_pages'):
                for pp in result.processed_pages:
                    if hasattr(pp, '_pil_image'):
                        pp._pil_image = None
                        del pp._pil_image
                    if hasattr(pp, '_pil_images') and pp._pil_images:
                        pp._pil_images.clear()

                result.processed_pages.clear()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            asyncio.create_task(_index_circuits_background(doc_id, filename, lightweight_pages, visual_indexer, container, settings))
            logger.info(f"电路索引任务已提交到后台，文档 {doc_id} 核心处理完成，用户可立即使用")

    except Exception as e:
        error_msg = f"文档摄取失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        db = get_documents_db()
        await _update_doc_status(doc_id, db,
            status="error",
            stage="error",
            progress=0,
            error_message=error_msg,
            processed_time=datetime.now().isoformat(),
        )


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """Get single document info including processing progress.

    Used by the frontend for progress polling after upload.
    Polling stops when status is "ready" or "error".
    """
    db = get_documents_db()

    if not await db.exists(document_id):
        raise HTTPException(status_code=404, detail="文档不存在")

    doc = await db.get(document_id)
    return DocumentInfo(
        document_id=document_id,
        filename=doc.get("filename", ""),
        status=doc.get("status", "unknown"),
        page_count=doc.get("page_count", 0),
        file_size=doc.get("file_size", 0),
        upload_time=doc.get("upload_time", ""),
        error_message=doc.get("error_message", ""),
        warnings=doc.get("warnings", []),
        chunk_count=doc.get("chunk_count", 0),
        table_count=doc.get("table_count", 0),
        circuit_count=doc.get("circuit_count", 0),
        image_count=doc.get("image_count", 0),
        processed_time=doc.get("processed_time", ""),
        progress=doc.get("progress", 0),
        stage=doc.get("stage", ""),
        stage_detail=doc.get("stage_detail", ""),
    )


async def _cleanup_all_background(document_id: str, doc: dict, container, cleanup_manager):
    """Background task: async cleanup of document files and indexes.

    All cleanup operations run in the background to avoid blocking the
    event loop. File cleanup runs in a thread executor; index cleanup
    runs as async operations.
    """
    try:
        file_cleanup = await asyncio.get_running_loop().run_in_executor(
            None, cleanup_manager.cleanup_document_files, document_id, doc, True
        )
        logger.info(f"后台文件清理完成: {document_id}, 结果: {file_cleanup}")
    except Exception as e:
        logger.error(f"后台文件清理失败: {document_id}, 错误: {e}", exc_info=True)

    try:
        index_cleanup = await cleanup_manager.cleanup_document_indexes(
            document_id, container, include_logs=True
        )
        logger.info(f"后台索引清理完成: {document_id}, 结果: {index_cleanup}")
    except Exception as e:
        logger.error(f"后台索引清理失败: {document_id}, 错误: {e}", exc_info=True)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    container=Depends(get_service_container)
):
    """Delete a document and all associated data (files, vectors, indexes).

    Uses a three-phase deletion strategy:
      1. (Sync) Remove DocumentDB record — microsecond-level
      2. (Background) Clean up files (PDF, images, circuit diagrams)
      3. (Background) Clean up indexes (vector, BM25, table, visual)

    The API returns immediately after phase 1. Remaining cleanup runs
    in the background to avoid blocking the event loop.
    """
    db = get_documents_db()

    logger.info(f"收到删除请求: document_id={document_id}")

    if not await db.exists(document_id):
        logger.warning(f"文档不存在: {document_id}")
        raise HTTPException(status_code=404, detail=f"文档不存在: {document_id}")

    try:
        doc = await db.delete(document_id)
        logger.info(f"从数据库移除文档: {document_id}, 文件名: {doc.get('filename')}")

        cleanup_manager = get_cleanup_manager()

        background_tasks.add_task(
            _cleanup_all_background,
            document_id, doc, container, cleanup_manager
        )

        logger.info(f"文档删除已受理: {document_id}, 文件+索引清理已加入后台任务队列")
        return {
            "success": True,
            "document_id": document_id,
            "cleanup": {"status": "background"},
            "message": "文档已删除，关联数据清理正在后台执行"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档删除失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/pages/{page}/image")
async def get_page_image(
    document_id: str,
    page: int,
    services = Depends(get_service_container)
):
    """Render a PDF page as PNG image with on-demand caching.

    Checks for a cached image first; if not found, renders the PDF page
    using PyMuPDF at 150 DPI and saves the result for future requests.

    Args:
        document_id: Document unique identifier.
        page: Page number (1-based).
        services: ServiceContainer via dependency injection.

    Returns:
        FileResponse with PNG image.
    """
    try:
        db = get_documents_db()

        if not await db.exists(document_id):
            raise HTTPException(status_code=404, detail="文档未找到")

        doc = await db.get(document_id)
        if doc.get("status") != "ready":
            raise HTTPException(status_code=400, detail="文档尚未处理完成")

        if page < 1 or page > doc.get("page_count", 0):
            raise HTTPException(status_code=404, detail="页码超出范围")

        settings = get_settings()
        image_path = settings.IMAGE_DIR / f"{document_id}_page_{page}.png"

        if not image_path.exists():
            try:
                filepath = doc.get("filepath", "")
                if filepath:
                    original_path = Path(settings.BASE_DIR) / filepath
                else:
                    original_path = Path(settings.DATA_DIR) / "uploads" / f"{document_id}_{doc['filename']}"

                if original_path.exists():
                    def _render_page_sync(pdf_path: str, page_num: int, out_path: Path) -> None:
                        import fitz
                        pdf_doc = fitz.open(pdf_path)
                        try:
                            if page_num > len(pdf_doc):
                                raise ValueError("页码超出范围")
                            pix = pdf_doc[page_num - 1].get_pixmap(dpi=150)
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            pix.save(str(out_path))
                        finally:
                            pdf_doc.close()

                    try:
                        await asyncio.to_thread(
                            _render_page_sync, str(original_path), page, image_path
                        )
                    except ValueError:
                        raise HTTPException(status_code=404, detail="页码超出范围")
                else:
                    raise HTTPException(status_code=404, detail=f"原始文档文件不存在: {original_path}")
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"动态生成页面图像失败: {e}")
                raise HTTPException(status_code=404, detail="页面图像不可用")

        return FileResponse(
            path=str(image_path),
            media_type="image/png",
            filename=f"{doc['filename']}_page_{page}.png"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取页面图像失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取页面图像失败")


class DeviceSearchRequest(BaseModel):
    """Device search request model with keyword and filter support."""
    query: str = Field(default="", description="搜索关键词")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤器")


class DeviceSearchResponse(BaseModel):
    """Device search response model."""
    success: bool = True
    devices: List[Dict[str, Any]] = Field(default_factory=list)
    message: str = ""


@router.post("/search", response_model=DeviceSearchResponse)
async def search_devices(request: DeviceSearchRequest):
    """Search devices by keyword across uploaded documents.

    Lightweight file-name/metadata search (not semantic vector search).
    Supports direct substring matching and Chinese-to-English keyword
    mapping via settings.DEVICE_SEARCH_KEYWORD_MAPPINGS.

    Args:
        request: Search request with query and filters.

    Returns:
        DeviceSearchResponse with matching device list.
    """
    query = (request.query or "").strip().lower()
    filters = request.filters or {}
    logger.info(f"设备搜索请求: query='{request.query}', filters={filters}")

    settings = get_settings()
    db = get_documents_db()

    try:
        all_docs = await db.get_all()
    except Exception as e:
        logger.error(f"读取文档数据库失败: {e}")
        raise HTTPException(status_code=500, detail="文档数据库读取失败")

    devices = []
    keyword_mappings = settings.DEVICE_SEARCH_KEYWORD_MAPPINGS

    for doc_id, meta in all_docs.items():
        filename = meta.get("filename", "")
        name = filename.rsplit(".", 1)[0] if "." in filename else filename
        if not filename:
            continue

        filename_lower = filename.lower()
        name_lower = name.lower()

        if query:
            direct_match = query in filename_lower or query in name_lower

            keyword_match = False
            if not direct_match:
                for cn_keyword, en_keywords in keyword_mappings.items():
                    if cn_keyword in query:
                        if any(kw in filename_lower for kw in en_keywords):
                            keyword_match = True
                            break

            if not direct_match and not keyword_match:
                continue

        match_filters = True
        for filter_key, filter_value in filters.items():
            filter_value_str = str(filter_value).lower()

            if filter_key == "status":
                if meta.get("status", "unknown") != filter_value:
                    match_filters = False
                    break
            elif filter_key == "file_type":
                file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                if file_ext != filter_value_str:
                    match_filters = False
                    break
            elif filter_key in meta:
                meta_value = str(meta.get(filter_key, "")).lower()
                if filter_value_str not in meta_value:
                    match_filters = False
                    break

        if not match_filters:
            continue

        devices.append({
            "name": name,
            "part_number": name,
            "document_id": doc_id,
            "filename": filename,
            "status": meta.get("status", "unknown"),
        })

    logger.info(f"设备搜索完成: 找到 {len(devices)} 个匹配设备")
    return DeviceSearchResponse(
        success=True,
        devices=devices,
        message="" if devices else "未找到匹配设备"
    )
