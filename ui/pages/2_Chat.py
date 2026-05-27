"""
Intelligent Q&A Page — EC-VeriQuery

Provides a ChatGPT-style conversational interface for RAG-based question answering
over uploaded electronic component datasheets. Supports streaming SSE responses,
citation tracing with page-image links, and message metadata badges.

Key features:
    - Chat-style Q&A with real-time streaming via SSE (Server-Sent Events)
    - Citation tracing: document source, page number, and screenshot links
    - Message metadata badges: citation count, processing time, quality status
    - Custom CSS for bubble-style conversation layout (user blue / assistant green)
    - Fixed-height scrollable chat container with persistent input area

Dependencies:
    - theme.py: Academic-style CSS variables and empty-state component
    - sidebar_nav.py: Sidebar navigation and document selector
    - api_client.py: HTTP REST API client with SSE streaming support
"""

import streamlit as st
import sys
import os
import html
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.set_page_config(
    page_title="智能问答 - EC-VeriQuery",
    page_icon="💬",
    layout="wide",
)

try:
    from theme import apply_academic_theme
    from sidebar_nav import render_sidebar_nav
    from api_client import (
        get_documents,
        is_api_connected,
        check_api_connection,
        chat_query_stream,
        get_page_image_url,
    )
except ImportError as e:
    st.error(f"组件导入失败: {e}")
    st.stop()

apply_academic_theme()
check_api_connection()

with st.sidebar:
    render_sidebar_nav()

st.markdown("""
<style>
.block-container {
    max-width: 100% !important;
    padding-top: 2.5rem !important;
}
section.main > div { padding-top: 2.5rem !important; }

.qa-page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0 0.7rem 0;
    margin-bottom: 0.85rem;
    border-bottom: 2px solid #2563eb;
}

.qa-page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0;
    padding: 0;
    white-space: nowrap;
    letter-spacing: -0.02em;
}

.qa-page-meta {
    font-size: 0.8rem;
    color: #64748b;
    white-space: nowrap;
}

.qa-helper-card {
    margin: 0.1rem 0 0.4rem 0;
    padding: 0.7rem 1rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    color: #475569;
    font-size: 0.86rem;
    line-height: 1.55;
}

.qa-helper-card strong {
    color: #1e293b;
}

.qa-chat-container {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #ffffff;
    min-height: 380px;
    padding: 0.6rem 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    overflow-y: auto;
    margin-bottom: 0.3rem;
    margin-top: 0.2rem;
}

.qa-chat-fixed-height {
    height: calc(100vh - 310px);
    min-height: 380px;
    max-height: calc(100vh - 310px);
    overflow-y: auto;
    padding-right: 0.3rem;
    position: relative;
}

.chat-scroll-container {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    padding: 0.3rem 0;
}

.qa-msg-wrapper {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.qa-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 280px;
    color: #94a3b8;
}

.qa-empty-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.qa-empty-text {
    font-size: 0.95rem;
    color: #64748b;
}

.qa-input-divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 0.3rem 0 0.5rem 0;
}

.qa-count {
    text-align: center;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

.chat-message {
    display: flex;
    gap: 0.5rem;
    padding: 0.35rem 0.5rem;
    border-radius: 8px;
}

.chat-message-user {
    flex-direction: row-reverse;
}

.chat-avatar {
    font-size: 1.3rem;
    min-width: 1.5rem;
    text-align: center;
    padding-top: 0.15rem;
}

.chat-content {
    max-width: 85%;
    min-width: 60%;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.chat-message-user .chat-content {
    align-items: flex-end;
    min-width: auto;
}

.chat-message-assistant .chat-content {
    align-items: flex-start;
}

.chat-bubble {
    padding: 0.55rem 0.8rem;
    border-radius: 10px;
    font-size: 0.875rem;
    line-height: 1.6;
    word-break: break-word;
}

.chat-bubble-user {
    background: #ffffff;
    color: #1e293b;
    border: 2px solid #3b82f6;
    border-bottom-right-radius: 3px;
}

.chat-bubble-assistant {
    background: #f0fdf4;
    color: #1e293b;
    border: 2px solid #10b981;
    border-bottom-left-radius: 3px;
    min-height: 2rem;
}

.chat-citations {
    margin-top: 0.4rem;
    padding: 0.5rem 0.7rem;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-radius: 8px;
    border-left: 4px solid #2563eb;
    box-shadow: 0 1px 3px rgba(37, 99, 235, 0.1);
}

.chat-citations-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #1e40af;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

.chat-citation-item {
    font-size: 0.82rem;
    color: #1e40af;
    padding: 0.25rem 0;
    font-weight: 600;
    transition: all 0.2s;
    display: block;
}

.chat-citation-item a {
    color: #1d4ed8 !important;
    text-decoration: none !important;
    margin-left: 0.5rem;
    display: inline-block;
}

.chat-citation-item a:hover {
    text-decoration: underline !important;
}

.chat-citation-item:hover {
    color: #1d4ed8;
}

.chat-timestamp {
    font-size: 0.65rem;
    color: #94a3b8;
    margin-top: 0.1rem;
}

.chat-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.25rem;
}

.chat-meta-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    border: 1px solid #dbeafe;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 0.7rem;
    line-height: 1.2;
}

.chat-text {
    word-wrap: break-word;
    overflow-wrap: break-word;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 0.95rem;
    line-height: 1.6;
    color: inherit;
}

.chat-message [data-testid="stMarkdownPre"],
.chat-message pre.st-emotion-cache-12sf6ke,
.chat-message .stCode,
.chat-message .stCode pre,
.chat-message .stCode code,
.chat-message div[style*="background-color: transparent"] code {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

.chat-message [data-testid="stCodeCopyButton"] {
    display: none !important;
    visibility: hidden !important;
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

[data-testid="stMarkdownContainer"] .qa-chat-fixed-height {
    overflow-y: auto !important;
    height: calc(100vh - 310px) !important;
    max-height: calc(100vh - 310px) !important;
}

[data-testid="stMarkdownContainer"] {
    overflow: visible !important;
}

.chat-message,
.chat-message-user,
.chat-message-assistant {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
}

.chat-bubble-assistant {
    background: #f0fdf4 !important;
    color: #1e293b !important;
    border: 2px solid #10b981 !important;
    border-bottom-left-radius: 3px !important;
    min-height: 2rem !important;
}

.chat-bubble-assistant .chat-text {
    color: #1e293b !important;
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
}

.chat-bubble-user {
    background: #eff6ff !important;
    color: #1e293b !important;
    border: 2px solid #3b82f6 !important;
    border-bottom-right-radius: 3px !important;
}

.chat-bubble-user .chat-text {
    color: #1e293b !important;
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


def _build_doc_info(documents: list, doc_ids: list) -> str:
    """
    Build a human-readable document info string for the page header.

    Matches selected doc IDs to their filenames from the full document list,
    showing only ``status == "ready"`` documents. Falls back to a count-based
    display if filename resolution fails.
    """
    if not doc_ids:
        return "当前文档: 未选择"

    try:
        ready_docs = [d for d in documents if d.get("status") == "ready"]
        selected_names = [
            d.get("filename", "未知")
            for d in ready_docs
            if d.get("document_id", d.get("id", "")) in doc_ids
        ]
        if selected_names:
            return f"当前文档: {', '.join(selected_names)}"
    except Exception:
        pass

    return f"当前文档: {len(doc_ids)} 份文档"


def _build_message_meta(result: dict, citations: list) -> dict:
    """
    Build message metadata (badge labels and citation visibility flag).

    Returns a dict with:
        - ``show_citations`` (bool): whether to render the citation card
        - ``badges`` (list[str], optional): capsule-style badge labels
          (e.g. "引用 3", "耗时 2.5s", "⚠️ 质量检测未通过")
    """
    meta = {}
    badges = []

    is_success = result.get("success", True)
    meta["show_citations"] = is_success and len(citations or []) > 0

    citation_count = len(citations or [])
    if citation_count and is_success:
        badges.append(f"引用 {citation_count}")

    processing_time = result.get("processing_time", 0) or 0
    if processing_time:
        badges.append(f"耗时 {processing_time:.1f}s")

    if not is_success:
        badges.append("⚠️ 质量检测未通过")

    if badges:
        meta["badges"] = badges

    return meta


api_connected = is_api_connected()
if not api_connected:
    st.warning("⚠️ 后端API服务未连接，请先启动后端服务")
    st.code("cd d:\\veriquery && python -m api.main", language="bash")
    col_r, col_i = st.columns([1, 3])
    with col_r:
        if st.button("🔄 重新连接", use_container_width=True):
            for k in ["api_check_ts", "api_connected", "api_url"]:
                st.session_state.pop(k, None)
            st.rerun()
    with col_i:
        st.info("💡 如果后端服务已启动，点击'重新连接'按钮")
    st.stop()

documents = get_documents()
ready_docs = [d for d in documents if d.get("status") == "ready"]
doc_ids = st.session_state.get("selected_doc_ids", [])
doc_info = _build_doc_info(documents, doc_ids)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_answer" not in st.session_state:
    st.session_state.processing_answer = False
if "last_question" not in st.session_state:
    st.session_state.last_question = ""


def build_message_html(role: str, content: str,
                       citations=None, timestamp: str = "",
                       message_meta=None, documents=None) -> str:
    """
    Build the complete HTML string for a single chat message.

    Includes avatar, bubble, optional citation card, metadata badges, and
    timestamp. All dynamic content is escaped via ``html.escape()`` to prevent
    XSS. Newlines in content are converted to ``<br>`` tags.
    """
    bubble_cls = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
    msg_cls = f"chat-message chat-message-{role}"
    icon = "👤" if role == "user" else "🤖"
    safe = html.escape(content).replace("\n", "<br>")

    cit_html = ""
    if citations and role == "assistant" and message_meta and message_meta.get("show_citations", False):
        seen_citations = set()
        unique_citations = []
        for c in citations:
            raw_file = c.get("file", c.get("document", "未知文档"))
            page = c.get("page", 1)
            dedup_key = f"{raw_file}_{page}"
            if dedup_key not in seen_citations:
                seen_citations.add(dedup_key)
                unique_citations.append(c)

        cit_html = '<div class="chat-citations"><div class="chat-citations-title">📚 信息来源</div>'
        for c in unique_citations:
            raw_file = c.get("file", c.get("document", "未知文档"))
            fname = html.escape(raw_file)
            page = c.get("page", 1)

            page_img_url = ""
            if documents:
                for d in documents:
                    did = d.get("document_id", d.get("id", ""))
                    if d.get("filename") == raw_file and did:
                        page_img_url = get_page_image_url(did, page)
                        break

            cit_html += f'<div class="chat-citation-item">📄 {fname} · 第 {page} 页'
            if page_img_url:
                cit_html += f' <a href="{page_img_url}" target="_blank" rel="noopener">📷 查看截图</a>'
            cit_html += '</div>'

        cit_html += '</div>'

    meta_html = ""
    if role == "assistant" and message_meta and message_meta.get("badges"):
        badges_html = "".join(
            f'<span class="chat-meta-badge">{b}</span>'
            for b in message_meta["badges"]
        )
        meta_html = f'<div class="chat-meta">{badges_html}</div>'

    ts_html = f'<div class="chat-timestamp">{html.escape(timestamp)}</div>' if timestamp else ""

    return f"""
    <div class="{msg_cls}">
        <div class="chat-avatar">{icon}</div>
        <div class="chat-content">
            <div class="chat-bubble {bubble_cls}"><div class="chat-text">{safe}</div></div>
            {cit_html}
            {meta_html}
            {ts_html}
        </div>
    </div>
    """


st.markdown(f"""
<div class="qa-page-header">
    <div class="qa-page-title">💬 智能问答</div>
    <div class="qa-page-meta">{doc_info}</div>
</div>
""", unsafe_allow_html=True)

chat_container = st.container(border=False)
with chat_container:
    if st.session_state.chat_history:
        chat_area = st.container(height=520)
        with chat_area:
            for c in st.session_state.chat_history:
                msg_html = build_message_html(
                    role=c["role"],
                    content=c["content"],
                    citations=c.get("citations", []),
                    timestamp=c.get("timestamp", ""),
                    message_meta=c.get("message_meta", {}),
                    documents=documents,
                )
                st.markdown(msg_html, unsafe_allow_html=True)
        st.markdown("""
<script>
(function() {
    var containers = window.parent.document.querySelectorAll('[data-testid="stVerticalBlock"]');
    for (var i = containers.length - 1; i >= 0; i--) {
        var scrollEl = containers[i];
        if (scrollEl.scrollHeight > scrollEl.clientHeight) {
            scrollEl.scrollTop = scrollEl.scrollHeight;
            break;
        }
    }
})();
</script>""", unsafe_allow_html=True)
    elif not ready_docs:
        empty_area = st.container(height=520)
        with empty_area:
            st.markdown("""
<div class="qa-empty">
  <div class="qa-empty-icon">📂</div>
  <div class="qa-empty-text">请先上传文档，再进行智能问答</div>
</div>""", unsafe_allow_html=True)
        if st.button("📄 前往上传文档", type="primary", key="goto_upload"):
            st.switch_page("pages/1_Documents.py")
    elif not doc_ids:
        empty_area = st.container(height=520)
        with empty_area:
            st.markdown("""
<div class="qa-empty">
  <div class="qa-empty-icon">☑️</div>
  <div class="qa-empty-text">请在左侧侧边栏勾选要查询的文档</div>
</div>""", unsafe_allow_html=True)
    else:
        empty_area = st.container(height=520)
        with empty_area:
            st.markdown("""
<div class="qa-empty">
  <div class="qa-empty-icon">💬</div>
  <div class="qa-empty-text">在下方输入问题，开始智能问答</div>
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="qa-input-divider">', unsafe_allow_html=True)

col_q, col_send, col_clr = st.columns([8, 1, 1])

with col_q:
    question = st.text_input(
        "问题",
        placeholder="输入您的问题，例如：STM32F103 的工作电压范围和典型功耗是多少？",
        key="chat_question_input",
        value=st.session_state.last_question if st.session_state.processing_answer else "",
        label_visibility="collapsed",
    )

with col_send:
    send_btn = st.button(
        "发送",
        type="primary",
        use_container_width=True,
        disabled=(
            not (question or "").strip()
            or st.session_state.processing_answer
            or not doc_ids
        ),
        key="send_btn",
    )

with col_clr:
    clr_btn = st.button(
        "清空",
        type="secondary",
        use_container_width=True,
        key="clear_btn",
    )

if st.session_state.chat_history:
    st.markdown(
        f'<div class="qa-count">共 {len(st.session_state.chat_history)} 条对话</div>',
        unsafe_allow_html=True,
    )

if clr_btn:
    st.session_state.chat_history = []
    st.rerun()

if send_btn and (question or "").strip() and doc_ids and not st.session_state.processing_answer:
    print(f"[CHAT-DEBUG] 发送按钮触发: question='{(question or '').strip()[:30]}', doc_ids={doc_ids}, processing={st.session_state.processing_answer}")
    st.session_state.chat_history.append({
        "role": "user",
        "content": question.strip(),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    st.session_state.processing_answer = True
    st.session_state.last_question = question.strip()
    st.rerun()

if st.session_state.processing_answer and st.session_state.last_question:
    print(f"[CHAT-DEBUG] 进入处理分支: question='{st.session_state.last_question[:30]}', doc_ids={doc_ids}, len(history)={len(st.session_state.chat_history)}")
    _ph = st.empty()
    _resp = [""]
    _cites = [[]]

    _ph.info("🤖 正在分析文档，请稍候...")

    def _stream_cb(data):
        try:
            t = data.get("type")
            d = data.get("data", {}) or {}
            if t == "start":
                pass
            elif t in ("chunk", "token"):
                chunk = d.get("chunk") or d.get("token", "")
                if chunk:
                    _resp[0] += chunk
            elif t == "complete":
                _cites[0] = d.get("citations") or []
            elif t == "error":
                _cites[0] = []
                logger.warning(f"Stream error: {d.get('message', '未知错误')}")
        except Exception as cb_ex:
            logger.warning(f"Chat stream callback error: {cb_ex}")

    try:
        result = chat_query_stream(st.session_state.last_question, doc_ids, callback=_stream_cb)
        print(f"[CHAT-DEBUG] chat_query_stream返回: type={type(result).__name__}, success={result.get('success') if result else 'None'}, response_len={len(result.get('response','')) if result else 0}")
    except Exception as stream_ex:
        logger.error(f"chat_query_stream exception: {stream_ex}")
        result = {"success": False, "error": f"请求异常: {str(stream_ex)}", "response": "", "citations": []}

    try:
        if result and result.get("success"):
            resp_text = result.get("response", _resp[0])
            citations = result.get("citations", _cites[0]) if result.get("success") else []
            if not resp_text and _resp[0]:
                resp_text = _resp[0]
            if resp_text:
                message_meta = _build_message_meta(result, citations)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": resp_text,
                    "sources": [c.get("file", c.get("document", "未知")) for c in citations],
                    "citations": citations,
                    "message_meta": message_meta,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "processing_time": result.get("processing_time", 0),
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "⚠️ 抱歉，未能获取到有效回答，请稍后重试或尝试其他问题",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message_meta": {"badges": ["⚠️ 获取失败"], "show_citations": False},
                })
        elif result and (result.get("error") or not result.get("success", True)):
            error_msg = result.get("error") or result.get("response", "查询失败")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ {error_msg}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message_meta": {"badges": ["❌ 查询失败"], "show_citations": False},
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "⚠️ 抱歉，系统未返回有效响应，请稍后重试",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
    except Exception as post_ex:
        logger.error(f"Post-processing error: {post_ex}")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"⚠️ 响应处理异常: {str(post_ex)}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

    st.session_state.processing_answer = False
    st.session_state.last_question = ""
    try:
        _ph.empty()
    except Exception:
        pass
    st.rerun()
