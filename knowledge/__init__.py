"""
知识图谱模块

提供知识图谱数据库和查询功能，用于：
- 芯片参数查询
- 引脚信息查询
- 模块信息查询
- 兼容性查询
- 知识补全
"""
from .graph_db import (
    KnowledgeGraphDB,
    auto_init_knowledge_graph,
    ensure_knowledge_graph_initialized,
)
from .graph_query import SQLiteGraphQueryEngine

__all__ = [
    "KnowledgeGraphDB",
    "auto_init_knowledge_graph",
    "ensure_knowledge_graph_initialized",
    "SQLiteGraphQueryEngine",
]
