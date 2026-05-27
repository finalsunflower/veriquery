"""
检索模块

包含向量存储、表格存储、文档存储、嵌入服务等组件。
"""

try:
    from .vector_store import (
        ChromaDBVectorStore, 
        create_vector_store
    )
    from .table_store import TableStore, TableSearchResult
    from .embeddings import EmbeddingManager
    from .bm25_store import BM25Store
    from .hybrid_retriever import HybridRetriever
    
    __all__ = [
        "ChromaDBVectorStore",
        "create_vector_store",
        "TableStore",
        "TableSearchResult",
        "BM25Store",
        "EmbeddingManager",
        "HybridRetriever",
    ]
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"检索模块导入失败: {e}")
    raise ImportError(f"检索模块关键组件导入失败: {e}") from e