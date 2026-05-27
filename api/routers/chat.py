"""
Chat router - Core API endpoints for VeriQuery intelligent Q&A system.

Provides synchronous and streaming chat endpoints that:
1. Accept natural language queries from the frontend
2. Execute the Agent workflow (RAG retrieval-augmented generation)
3. Perform quality validation and post-processing on results
4. Return answers with citation information
"""

import logging
import time
import json
import uuid
import re as _re
from typing import AsyncGenerator, Tuple, List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core import AgentState
from api.dependencies import get_service_container, ServiceContainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat queries."""

    query: str = Field(..., min_length=1, max_length=2000, description="用户查询内容")
    document_ids: List[str] = Field(default_factory=list, description="选中的文档ID列表")
    session_id: Optional[str] = Field(default=None, description="会话ID，不传则自动生成")


class CitationInfo(BaseModel):
    """Citation information for RAG traceability."""

    file: str = Field(..., description="来源文件名")
    page: int = Field(..., description="页码")
    text_snippet: str = Field(default="", description="文本片段")
    section: Optional[str] = Field(default=None, description="章节标题")


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""

    success: bool = Field(default=True, description="是否成功")
    response: str = Field(..., description="AI响应内容")
    citations: List[CitationInfo] = Field(default_factory=list, description="引用信息列表")
    processing_time: float = Field(default=0, description="处理时间(秒)")
    session_id: str = Field(default="", description="会话ID")


class AgentResult(BaseModel):
    """Intermediate result extracted from AgentState, decoupled from internal workflow fields."""

    response: str = Field(..., description="AI响应内容")
    citations: List[CitationInfo] = Field(default_factory=list, description="引用信息列表")


def _generate_session_id() -> str:
    """Generate a unique session ID based on UUID4 prefix."""
    return str(uuid.uuid4())[:16]


def _extract_results_from_state(result_state: AgentState) -> AgentResult:
    """Extract final response and citations from the Agent workflow state dict.

    Args:
        result_state: Agent state dict after workflow execution,
            containing 'final_response' and 'citations' keys.

    Returns:
        AgentResult with extracted response and structured citations.
    """
    response = result_state.get("final_response", "系统处理异常")
    raw_citations = result_state.get("citations", [])

    citations = [
        CitationInfo(
            file=c.get("file", "未知文档"),
            page=c.get("page", 1),
            text_snippet=c.get("text_snippet", ""),
            section=c.get("section")
        )
        for c in raw_citations
    ]

    return AgentResult(response=response, citations=citations)


def _split_into_chunks(text: str, chunk_size: int = 16) -> List[str]:
    """Split text into chunks for streaming, preferring punctuation boundaries.

    Args:
        text: The text to split.
        chunk_size: Target chunk size in characters. Defaults to 16.

    Returns:
        List of text chunks, each a semantically coherent segment.
    """
    if not text:
        return []

    BREAK_CHARS = set("。！？.!?\n,，；;：:")
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            best = end
            for i in range(end, max(start, end - 4), -1):
                if text[i] in BREAK_CHARS:
                    best = i + 1
                    break
            end = best

        chunks.append(text[start:end])
        start = end

    return chunks


async def _execute_agent_workflow(
    query: str,
    doc_ids: List[str],
    session_id: str,
    container: ServiceContainer
) -> Tuple[Optional[AgentResult], Optional[str]]:
    """Execute the Agent workflow and return results or an error message.

    Args:
        query: User query text.
        doc_ids: Selected document IDs for filtered retrieval.
        session_id: Session ID for logging and context tracking.
        container: ServiceContainer providing the Agent workflow graph.

    Returns:
        A tuple of (AgentResult, None) on success, or (None, error_message) on failure.
    """
    try:
        graph = container.veriquery_graph
        if graph is None:
            logger.error("Agent图未初始化，系统配置错误")
            return None, "系统初始化失败，请联系管理员"

        user_context = {"selected_document_ids": doc_ids}
        logger.info(f"执行Agent工作流 - session_id: {session_id}, docs: {len(doc_ids)}")

        result_state = await graph.ainvoke(
            question=query,
            session_id=session_id,
            user_context=user_context
        )

        result = _extract_results_from_state(result_state)
        return result, None

    except Exception as e:
        logger.error(f"Agent处理失败: {e}", exc_info=True)
        return None, f"Agent处理异常: {str(e)}"


def _validate_response_quality(response_text: str) -> Tuple[List[str], bool]:
    """Run quality checks on the LLM response text.

    Checks for:
    1. Error keywords indicating LLM internal failures (precise matching to avoid
       false positives on technical terms like "error amplifier" or "failed bit")
    2. Role marker leakage (assistant:/user:/system:)
    3. Repetitive content via similarity-hash deduplication

    Args:
        response_text: Raw LLM response text.

    Returns:
        Tuple of (quality_issues_list, is_error_response).
        quality_issues_list may be empty if no issues found.
        is_error_response is True if error keywords are detected.
    """
    precise_error_patterns = [
        r'traceback\s*\(most\s+recent\s+call\s+last\)',
        r'model_kwargs.*not\s+used\s+by\s+the\s+model',
        r'处理请求时发生错误',
        r'生成响应时发生错误',
        r'LLM生成答案失败',
        r'RuntimeError',
        r'ValueError.*model',
        r'ImportError',
        r'ModuleNotFoundError',
        r'OSError.*model',
        r'CUDA\s+out\s+of\s+memory',
        r'连接.*超时',
        r'服务.*不可用',
    ]
    is_error_response = any(
        _re.search(pat, response_text, _re.IGNORECASE) for pat in precise_error_patterns
    )
    quality_issues = []
    if is_error_response:
        quality_issues.append("检测到内部错误信息")
    return quality_issues, is_error_response


def _clean_role_markers(response_text: str) -> Tuple[str, List[str]]:
    """Detect and clean leaked role markers from LLM output.

    Args:
        response_text: Response text potentially containing role markers.

    Returns:
        Tuple of (cleaned_text, quality_issues).
    """
    quality_issues = []
    if not response_text or len(response_text) < 5:
        quality_issues.append("回答过短")
    elif "assistant" in response_text.lower() or "user:" in response_text.lower():
        quality_issues.append("包含角色标记泄漏")
        for stop_word in ["assistant", "user:", "system:", "\n\n\n"]:
            idx = response_text.lower().find(stop_word)
            if idx > 0:
                response_text = response_text[:idx].strip()
                break
    return response_text, quality_issues


def _deduplicate_sentences(response_text: str) -> Tuple[str, List[str]]:
    """Deduplicate repetitive sentences using similarity-hash prefix matching.

    Args:
        response_text: Response text to deduplicate.

    Returns:
        Tuple of (deduped_text, quality_issues).
    """
    quality_issues = []
    sentences = _re.split(r'(?<=[。！？.!?])|\n', response_text)
    seen = set()
    unique_sentences = []

    for sent in sentences:
        sent_stripped = sent.strip()
        if len(sent_stripped) > 6:
            sent_normalized = _re.sub(r'\s+', '', sent_stripped.lower())
            if len(sent_normalized) > 4:
                similarity_hash = sent_normalized[:20] if len(sent_normalized) > 20 else sent_normalized
                if similarity_hash not in seen:
                    seen.add(similarity_hash)
                    unique_sentences.append(sent_stripped)

    deduped_response = '\n'.join(unique_sentences[:15])

    original_len = len(response_text.replace(' ', ''))
    deduped_len = len(deduped_response.replace(' ', ''))
    if original_len > 20 and deduped_len < original_len * 0.5:
        quality_issues.append("存在大量重复内容")

    return deduped_response, quality_issues


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, container=Depends(get_service_container)):
    """Synchronous chat endpoint - returns complete response with citations.

    POST /api/v1/chat/

    Args:
        request: ChatRequest with query, optional document_ids, and optional session_id.
        container: ServiceContainer injected by FastAPI.

    Returns:
        ChatResponse with answer, citations, processing time, and session ID.
    """
    start_time = time.time()
    session_id = request.session_id or _generate_session_id()

    try:
        logger.info(f"[session:{session_id}] 处理查询: {request.query[:50]}...")
        logger.info(f"[session:{session_id}] 选中文档数量: {len(request.document_ids)}")

        result, error = await _execute_agent_workflow(
            request.query, request.document_ids, session_id, container
        )

        if error:
            return ChatResponse(
                success=False,
                response=error,
                processing_time=time.time() - start_time,
                session_id=session_id
            )

        response_text = result.response.strip()

        _, is_error_response = _validate_response_quality(response_text)
        if is_error_response:
            logger.error(f"[session:{session_id}] 检测到错误响应: {response_text[:100]}")
            return ChatResponse(
                success=False,
                response="⚠️ 抱歉，系统在生成回答时遇到技术问题，请稍后重试。如问题持续存在，请联系管理员。",
                processing_time=time.time() - start_time,
                session_id=session_id,
                citations=[]
            )

        if not response_text or len(response_text.strip()) < 5:
            logger.warning(f"[session:{session_id}] 响应为空或过短，可能是LLM仅输出了思考过程")
            return ChatResponse(
                success=False,
                response="⚠️ 抱歉，系统未能生成有效回答，请尝试重新表述问题。",
                processing_time=time.time() - start_time,
                session_id=session_id,
                citations=[]
            )

        response_text, quality_issues = _clean_role_markers(response_text)

        deduped_response, dedup_issues = _deduplicate_sentences(response_text)
        quality_issues.extend(dedup_issues)

        has_tech_content = bool(_re.search(
            r'[±]?\d+\.?\d*\s*[VvµuMmGgKk]?[AaWwHhZz]|'
            r'[±]?\d+\.?\d*\s*(?:V|A|mA|µA|kHz|MHz|GHz|°C|℃|Ω|kΩ|MΩ|pF|nF|µF|mm|ns|µs|ms|dB|%)',
            response_text
        ))

        if quality_issues and len(deduped_response.strip()) < 10 and not has_tech_content:
            logger.warning(f"[session:{session_id}] 回答质量不合格: {quality_issues}")
            return ChatResponse(
                success=False,
                response="⚠️ 抱歉，系统生成的回答未能通过质量检测，请尝试重新提问或换个方式描述问题。",
                processing_time=time.time() - start_time,
                session_id=session_id,
                citations=[]
            )
        elif quality_issues:
            logger.info(f"[session:{session_id}] 回答已自动优化: {quality_issues}")
            if len(deduped_response.strip()) >= 10 or has_tech_content:
                response_text = deduped_response

        processing_time = time.time() - start_time

        return ChatResponse(
            response=response_text,
            citations=result.citations,
            processing_time=processing_time,
            success=True,
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"[session:{session_id}] 聊天处理失败: {e}", exc_info=True)
        return ChatResponse(
            success=False,
            response=f"处理请求时发生错误: {str(e)}",
            processing_time=time.time() - start_time,
            session_id=session_id
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest, container=Depends(get_service_container)):
    """Streaming chat endpoint - pushes response chunks via SSE-style messages.

    POST /api/v1/chat/stream

    Message types:
        - start:  Signals stream begin with query and session_id.
        - chunk:  Contains a text segment, index, total count, and is_complete flag.
        - complete: Full response with citations and metadata.
        - error:  Error message if any stage fails.

    Args:
        request: ChatRequest with query, optional document_ids, and optional session_id.
        container: ServiceContainer injected by FastAPI.

    Returns:
        StreamingResponse with async generator yielding JSON messages.
    """
    start_time = time.time()
    session_id = request.session_id or _generate_session_id()

    async def generate_stream() -> AsyncGenerator[str, None]:
        """Async generator yielding SSE-style JSON messages for streaming response."""
        try:
            logger.info(f"[session:{session_id}] 流式处理查询: {request.query[:50]}...")

            yield json.dumps({
                "type": "start",
                "data": {"query": request.query, "session_id": session_id}
            }) + "\n"

            result, error = await _execute_agent_workflow(
                request.query, request.document_ids, session_id, container
            )

            if error:
                yield json.dumps({
                    "type": "error",
                    "data": {"message": error, "session_id": session_id}
                }) + "\n"
                return

            response_text = result.response.strip()

            _, is_error_response = _validate_response_quality(response_text)
            if is_error_response:
                logger.error(f"[session:{session_id}] 流式检测到错误响应: {response_text[:100]}")
                yield json.dumps({
                    "type": "error",
                    "data": {
                        "message": "⚠️ 抱歉，系统在生成回答时遇到技术问题，请稍后重试。",
                        "session_id": session_id
                    }
                }) + "\n"
                return

            if not response_text or len(response_text.strip()) < 5:
                logger.warning(f"[session:{session_id}] 流式响应为空或过短，可能是LLM仅输出了思考过程")
                yield json.dumps({
                    "type": "error",
                    "data": {
                        "message": "⚠️ 抱歉，系统未能生成有效回答，请尝试重新表述问题。",
                        "session_id": session_id
                    }
                }) + "\n"
                return

            response_text, quality_issues = _clean_role_markers(response_text)

            deduped_response, dedup_issues = _deduplicate_sentences(response_text)
            quality_issues.extend(dedup_issues)

            has_tech_content = bool(_re.search(
                r'[±]?\d+\.?\d*\s*[VvµuMmGgKk]?[AaWwHhZz]|'
                r'[±]?\d+\.?\d*\s*(?:V|A|mA|µA|kHz|MHz|GHz|°C|℃|Ω|kΩ|MΩ|pF|nF|µF|mm|ns|µs|ms|dB|%)',
                response_text
            ))

            if quality_issues and len(deduped_response.strip()) < 10 and not has_tech_content:
                logger.warning(f"[session:{session_id}] 流式回答质量不合格: {quality_issues}, original_len={len(response_text)}, deduped_len={len(deduped_response)}")
                yield json.dumps({
                    "type": "error",
                    "data": {
                        "message": "⚠️ 抱歉，系统生成的回答未能通过质量检测，请尝试重新提问。",
                        "session_id": session_id
                    }
                }) + "\n"
                return
            elif quality_issues:
                logger.info(f"[session:{session_id}] 流式回答已自动优化: {quality_issues}")
                if len(deduped_response.strip()) >= 10 or has_tech_content:
                    response_text = deduped_response
                else:
                    logger.info(f"[session:{session_id}] 去重后过短但有技术内容，保留原始响应")

            chunks = _split_into_chunks(response_text)
            for i, chunk in enumerate(chunks):
                yield json.dumps({
                    "type": "chunk",
                    "data": {
                        "chunk": chunk,
                        "index": i,
                        "total": len(chunks),
                        "is_complete": i == len(chunks) - 1
                    }
                }) + "\n"

            processing_time = time.time() - start_time

            yield json.dumps({
                "type": "complete",
                "data": {
                    "response": response_text,
                    "citations": [c.model_dump() for c in result.citations],
                    "processing_time": processing_time,
                    "session_id": session_id,
                    "success": True
                }
            }) + "\n"

        except Exception as e:
            logger.error(f"[session:{session_id}] 流式聊天处理失败: {e}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "data": {
                    "message": f"处理请求时发生错误: {str(e)}",
                    "processing_time": time.time() - start_time,
                    "session_id": session_id
                }
            }) + "\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"X-Session-ID": session_id}
    )
