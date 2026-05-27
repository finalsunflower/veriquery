"""
Agentic RAG 工作流引擎 - 意图路由与检索增强

本文件是Agent层的工作流编排引擎，构建和运行基于LangGraph的有向图工作流，
实现"用户输入 → 意图识别 → 条件路由 → 对应处理 → 响应输出"的完整流程。

工作流拓扑(DAG):
    START → intent_router ─┬─ intent="qa" ──→ text_retrieval → response_generation → END
                           └─ intent="pinout" → pinout → END

路由逻辑:
    - intent="qa"(默认) → text_retrieval → response_generation → END
    - intent="pinout"   → pinout → END

注意: ERC检查、参数对比为专用功能，由API层直接调用对应节点，无需经过意图路由。
原因: 这些功能的输入是结构化参数(如driver_chip/receiver_chip)，不需要从自然语言推断意图。

设计要点:
    - LangGraph声明式图构建: 可视化、可扩展、状态管理清晰
    - DAG无环设计: 保证工作流一定能终止
    - 异步执行: 检索和LLM生成均为I/O密集操作，async避免阻塞事件循环
    - 优雅降级: 工作流异常时返回降级结果，而非抛出异常
    - 编译一次执行多次: compiled_graph是只读的，线程安全，可并发调用
"""

import logging
import time
from typing import Dict, Any

from langgraph.graph import StateGraph, END, START

from core.config import get_settings
from core import create_initial_state
from .workflow_nodes import (
    UnifiedAgentNodes,
    create_unified_agent_nodes,
)

logger = logging.getLogger(__name__)


class UnifiedAgentWorkflow:
    """意图驱动型工作流编排器，负责构建LangGraph有向图并提供异步执行入口。

    职责:
        1. 初始化所有工作流节点(通过create_unified_agent_nodes工厂函数)
        2. 构建有向图(定义节点和边)
        3. 编译图为可执行对象
        4. 提供异步执行入口(ainvoke)
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.agent_nodes = create_unified_agent_nodes(settings)
        self.graph = self._build_unified_graph()
        logger.info("Agent工作流引擎初始化完成")

    def _route_by_intent(self, state: Dict[str, Any]) -> str:
        """根据意图路由到对应节点，作为LangGraph条件边的路由函数。

        路由映射:
            - intent="qa"(默认) → "text_retrieval"
            - intent="pinout"   → "pinout"
            - 未知意图          → 兜底到"text_retrieval"
        """
        intent = state.get("intent", "qa")

        route_map = {
            "qa": "text_retrieval",
            "pinout": "pinout",
        }

        route = route_map.get(intent, "text_retrieval")
        logger.info(f"路由决策: intent={intent} -> node={route}")
        return route

    def _build_unified_graph(self) -> StateGraph:
        """构建工作流有向图: 意图路由 → QA检索/引脚分析。

        图拓扑:
            START → intent_router ─┬─ intent="qa" ──→ text_retrieval → response_generation → END
                                   └─ intent="pinout" → pinout → END
        """
        workflow = StateGraph(dict)

        # 添加节点
        workflow.add_node("intent_router", self.agent_nodes.intent_router_node)
        workflow.add_node("text_retrieval", self.agent_nodes.text_retrieval_node)
        workflow.add_node("response_generation", self.agent_nodes.response_generation_node)
        workflow.add_node("pinout", self.agent_nodes.pinout_node)

        # 起始边
        workflow.add_edge(START, "intent_router")

        # 条件边: 根据意图路由
        workflow.add_conditional_edges(
            "intent_router",
            self._route_by_intent,
            {
                "text_retrieval": "text_retrieval",
                "pinout": "pinout",
            }
        )

        # 固定边: QA路径
        workflow.add_edge("text_retrieval", "response_generation")
        workflow.add_edge("response_generation", END)

        # 固定边: pinout路径(短路，跳过response_generation)
        workflow.add_edge("pinout", END)

        # 编译图
        compiled_graph = workflow.compile()
        logger.info(f"工作流编译成功: {type(compiled_graph)}")

        return compiled_graph

    async def ainvoke(
        self,
        question: str = None,
        session_id: str = None,
        user_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """异步运行工作流，本类的唯一对外接口。

        Args:
            question: 用户问题字符串
            session_id: 会话ID
            user_context: 用户上下文(含selected_document_ids等)

        Returns:
            状态字典，包含:
                - final_response: 最终答案文本
                - citations: 引用来源列表
                - intent: 识别的意图
                - processing_time: 处理耗时(秒)
                - error: 错误信息(正常时为None)
        """
        start_time = time.time()

        initial_state = create_initial_state(query=question or "")
        initial_state["session_id"] = session_id or "default"

        if user_context:
            initial_state["user_context"] = user_context

        try:
            result = await self.graph.ainvoke(initial_state)
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            logger.info(f"工作流执行完成，耗时: {processing_time:.2f}s, 意图: {result.get('intent', 'unknown')}")
            return result

        except Exception as e:
            logger.error(f"工作流执行失败: {e}", exc_info=True)
            return {
                "error": str(e),
                "final_response": f"处理请求时发生错误: {str(e)}",
                "processing_time": time.time() - start_time,
            }


def create_workflow(settings=None) -> UnifiedAgentWorkflow:
    """创建Agent工作流引擎实例的工厂函数。"""
    return UnifiedAgentWorkflow(settings)
