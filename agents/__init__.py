"""
VeriQuery Agent 编排模块

工作流路由（通过意图识别自动分发）：
- 意图路由节点（自动识别用户意图）
- 文本检索增强（RAG问答）
- 引脚分析节点（芯片引脚定义）

专用功能（API层直接调用，无需意图路由）：
- ERC 检查节点（电气规则兼容性）
- 参数对比节点（多器件比较）
- 电路检索节点（电路图搜索）
"""

from .workflow_graph import (
    UnifiedAgentWorkflow,
    create_workflow,
)

__all__ = [
    "UnifiedAgentWorkflow",
    "create_workflow",
]
