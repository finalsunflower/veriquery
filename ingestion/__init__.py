"""
VeriQuery 文档入库模块

文档摄取与处理系统：
- document_processor: 入库总流水线（已合并PDF解析）
- image_indexer:      图像向量化索引（统一版：CLIP+ColPali+Qwen3.5-2B）
                      已合并circuit_captioner功能：元件知识库+相关性评分
"""

import logging

logger = logging.getLogger(__name__)

from .document_processor import (
    EnhancedDocumentProcessor,
    IngestionPipeline,
    ProcessingResult,
)
from .image_indexer import TrueColPaliIndexer, create_visual_indexer, get_visual_indexer

__all__ = [
    'EnhancedDocumentProcessor',
    'IngestionPipeline',
    'ProcessingResult',
    'TrueColPaliIndexer',
    'create_visual_indexer',
    'get_visual_indexer',
]
