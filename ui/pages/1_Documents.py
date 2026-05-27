"""
Document Management Page — EC-VeriQuery

Provides the primary user entry point for managing electronic component datasheets.
Users can upload PDF documents, track processing progress, search for devices,
and manage the document lifecycle (view, navigate, delete).

Key features:
    - PDF upload with asynchronous backend processing pipeline
    - Real-time progress tracking via polling (1s interval, 5min timeout)
    - Device search across indexed documents
    - Document list with status badges, statistics, and action buttons
    - Two-step delete confirmation to prevent accidental data loss
    - API connection health check with manual reconnect

Dependencies:
    - theme.py: Academic-style CSS variables and empty-state component
    - sidebar_nav.py: Sidebar navigation and document selector
    - api_client.py: HTTP REST API client with connection pooling and TTL cache
"""

import streamlit as st
import sys
import os
from urllib.parse import urlparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from theme import apply_academic_theme, empty_state
    from sidebar_nav import render_sidebar_nav
    from api_client import (
        get_api_url,
        get_documents,
        is_api_connected,
        check_api_connection,
        upload_document,
        delete_document,
        search_devices,
        get_document_status,
        _fetch_documents_cached,
    )
except ImportError as e:
    st.error(f"组件导入失败: {e}")
    st.stop()

st.set_page_config(
    page_title="文档管理 - EC-VeriQuery",
    page_icon="📄",
    layout="wide",
)

apply_academic_theme()

import time as time_module

_DELETE_GRACE_PERIOD = 60
_UPLOAD_GRACE_PERIOD = 30

is_recent_delete = (
    "last_delete_time" in st.session_state
    and time_module.time() - st.session_state.get("last_delete_time", 0) < _DELETE_GRACE_PERIOD
)

is_recent_upload = (
    "last_upload_time" in st.session_state
    and time_module.time() - st.session_state.get("last_upload_time", 0) < _UPLOAD_GRACE_PERIOD
)

if is_recent_delete or is_recent_upload:
    st.session_state.api_connected = True
    st.session_state.api_check_ts = time_module.monotonic()
else:
    check_api_connection(force_refresh=True)

with st.sidebar:
    render_sidebar_nav()

st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem !important;
    }
    section.main > div { padding-top: 2.5rem !important; }

    .documents-page-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.55rem 0 0.7rem 0;
        margin-bottom: 0.85rem;
        border-bottom: 2px solid var(--primary);
    }
    .documents-page-title {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    .search-container {
        display: flex !important;
        align-items: center !important;
        gap: 1rem !important;
        min-height: 2.75rem !important;
    }

    .search-input-wrapper,
    .search-button-wrapper {
        display: flex !important;
        align-items: center !important;
        min-height: 2.75rem !important;
        height: 2.75rem !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .search-input-wrapper > div,
    .search-button-wrapper > div {
        min-height: 2.75rem !important;
        height: 2.75rem !important;
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .search-input-wrapper input,
    .search-button-wrapper button {
        min-height: 2.75rem !important;
        height: 2.75rem !important;
        margin: 0 !important;
        padding: 0.5rem 0.75rem !important;
        box-sizing: border-box !important;
        line-height: 1.5 !important;
        vertical-align: middle !important;
    }

    .stTextInput > label {
        display: none !important;
    }

    .documents-summary-card {
        padding: 0.9rem 1rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .documents-summary-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }

    .documents-summary-label {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="documents-page-header">
    <div class="documents-page-title">📄 文档管理</div>
</div>
""", unsafe_allow_html=True)

api_connected = st.session_state.get("api_connected", False)

_STAGE_LABELS = {
    "uploading": "📤 上传文件中",
    "parsing_pdf": "📄 解析PDF结构",
    "parsing_complete": "📄 PDF解析完成",
    "indexing_vectors": "🔍 建立向量索引",
    "building_chunks": "📦 构建文本分块",
    "vector_indexed": "✅ 向量索引完成",
    "bm25_indexed": "✅ BM25索引完成",
    "tables_indexed": "✅ 表格索引完成",
    "saving_metadata": "💾 保存元数据",
    "clip_filtering": "🔍 CLIP图像过滤",
    "vlm_analysis": "🧠 VLM电路分析",
    "embedding_generation": "🔢 生成嵌入",
    "completed": "✅ 处理完成",
    "error": "❌ 处理失败",
}


def _reset_api_connection_state():
    """Clear API connection session state to force a fresh health check."""
    for key in ("api_connected", "api_url", "api_check_ts"):
        st.session_state.pop(key, None)


def _set_current_document_context(doc_id: str, filename: str = ""):
    """
    Sync the current document to the sidebar selector for cross-page context.

    Uses a deferred-sync strategy: writes to ``_pending_sidebar_selection`` so
    that ``render_sidebar_nav()`` can pick it up as the ``default`` parameter
    for ``st.multiselect`` on the next render cycle (direct assignment to a
    widget-owned key is disallowed by Streamlit).
    """
    if not doc_id:
        return

    st.session_state.selected_doc_ids = [doc_id]
    st.session_state._pending_sidebar_selection = [doc_id]


def _get_api_endpoint_hint() -> str:
    """
    Return the netloc portion of the configured API URL for display in diagnostics.

    Falls back to the raw URL string if parsing fails.
    """
    api_url = get_api_url()
    parsed = urlparse(api_url)
    return parsed.netloc or api_url

st.markdown("""
<style>
    .documents-toolbar-note {
        padding: 0.55rem 0.9rem;
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    .document-card {
        padding: 1.1rem 1.25rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .document-card.highlighted {
        background: #fffbeb;
        border: 2px solid #f59e0b;
    }

    .document-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .document-card-title {
        font-size: 1.02rem;
        font-weight: 600;
        color: var(--text-primary);
        word-break: break-word;
    }

    .document-card-meta {
        font-size: 0.84rem;
        color: var(--text-secondary);
        margin-bottom: 0.35rem;
    }

    .document-card-stats {
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    .document-card-note {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.65rem;
    }
</style>
""", unsafe_allow_html=True)

col_search1, col_search2, col_search3 = st.columns([3, 1, 1])

with col_search1:
    device_search_query = st.text_input(
        "设备搜索",
        placeholder="输入设备名称或关键词，如：74HC04、反相器、逻辑门...",
        key="device_search_input",
        label_visibility="collapsed",
    )

with col_search2:
    search_device_btn = st.button("搜索", type="primary", use_container_width=True)

with col_search3:
    if st.button("🔄 刷新", use_container_width=True):
        _fetch_documents_cached.clear()
        st.rerun()

if search_device_btn and device_search_query and api_connected:
    with st.spinner("正在搜索设备..."):
        search_result = search_devices(device_search_query)
        if search_result and not search_result.get("error"):
            st.session_state.device_search_result = search_result
        elif search_result and search_result.get("error"):
            st.error(f"搜索失败: {search_result.get('error')}")

if "device_search_result" in st.session_state:
    search_result = st.session_state.device_search_result
    devices = search_result.get("devices", [])

    if devices:
        st.markdown(f"**找到 {len(devices)} 个匹配设备**")

        for dev in devices:
            col_dev1, col_dev2, col_dev3 = st.columns([3, 2, 1])

            with col_dev1:
                st.markdown(f"📄 **{dev.get('name', dev.get('part_number', '未知'))}**")

            with col_dev2:
                st.caption(f"文档: {dev.get('filename', '')}")

            with col_dev3:
                if st.button("查看", key=f"view_dev_{dev.get('document_id', '')}", use_container_width=True):
                    st.session_state.highlight_doc_id = dev.get('document_id', '')
                    del st.session_state.device_search_result
                    st.rerun()

        if st.button("清除搜索结果", key="clear_search"):
            del st.session_state.device_search_result
            st.rerun()
    else:
        st.info("未找到匹配设备")
        if st.button("清除", key="clear_search_empty"):
            del st.session_state.device_search_result
            st.rerun()

st.markdown("**📤 上传文档**")

if not api_connected:
    api_endpoint_hint = _get_api_endpoint_hint()
    st.warning("⚠️ 后端API服务未连接，请先启动后端服务")
    st.code(f"cd d:\\veriquery && python -m api.main\n# 当前UI连接地址: {api_endpoint_hint}", language="bash")

    col_refresh_conn, col_info = st.columns([1, 3])
    with col_refresh_conn:
        if st.button("🔄 重新连接", use_container_width=True):
            _reset_api_connection_state()
            _fetch_documents_cached.clear()
            try:
                check_api_connection(force_refresh=True)
            except Exception:
                pass
            st.rerun()

    with col_info:
        st.info(
            f"💡 如果后端服务已启动，点击'重新连接'按钮。如果连接仍然失败，请检查：\n\n"
            f"1. 后端服务是否正常运行\n"
            f"2. 当前配置地址 `{api_endpoint_hint}` 是否可访问\n"
            f"3. 防火墙是否阻止连接\n"
            f"4. 本机网络与端口映射是否正常"
        )

else:
    uploaded_file = st.file_uploader(
        "选择PDF文件",
        type=["pdf"],
        help="支持PDF格式的电子器件数据手册，建议文件大小不超过50MB",
        label_visibility="visible",
    )

    if uploaded_file:
        if st.button("🚀 开始上传", type="primary", use_container_width=True):
            st.session_state.last_upload_time = time_module.time()
            st.session_state.api_connected = True
            st.session_state.api_check_ts = time_module.monotonic()

            with st.spinner("正在上传文档..."):
                result = upload_document(uploaded_file)

            if result and not result.get("error"):
                doc_id = result.get('document_id')
                st.success(f"✅ **上传成功！** 文档ID: `{doc_id}`")

                st.session_state.last_upload_time = time_module.time()
                st.session_state.api_connected = True
                st.session_state.api_check_ts = time_module.monotonic()

                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(0, text="开始处理...")

                import time

                max_attempts = 300
                attempt = 0

                while attempt < max_attempts:
                    doc_info = get_document_status(doc_id)

                    if doc_info:
                        progress = doc_info.get("progress", 0)
                        stage = doc_info.get("stage", "")
                        status = doc_info.get("status", "")
                        stage_detail = doc_info.get("stage_detail", "")

                        stage_text = _STAGE_LABELS.get(stage, stage)
                        if stage_detail:
                            progress_text = f"{stage_text} - {stage_detail} ({progress}%)"
                        else:
                            progress_text = f"{stage_text}... {progress}%"
                        progress_bar.progress(progress / 100, text=progress_text)

                        if status == "ready":
                            progress_placeholder.empty()
                            status_placeholder.success("✅ **处理完成！** 文档已就绪，可以开始使用问答功能。")
                            _fetch_documents_cached.clear()
                            st.rerun()
                            break
                        elif status == "error":
                            progress_placeholder.empty()
                            error_msg = doc_info.get("error_message", "未知错误")
                            status_placeholder.error(f"❌ **处理失败**\n\n错误信息: {error_msg}")
                            break
                    else:
                        pass

                    attempt += 1
                    time.sleep(1)

                else:
                    progress_placeholder.empty()
                    status_placeholder.warning("⚠️ 处理超时，请稍后刷新页面查看文档状态。")
                    _fetch_documents_cached.clear()

            else:
                error_msg = result.get("error", "未知错误") if result else "未知错误"
                st.error(f"❌ 上传失败: {error_msg}")


def render_documents_list():
    """Render the full document list with summary cards, status badges, and action buttons."""
    highlight_doc_id = st.session_state.get("highlight_doc_id")
    selected_doc_ids = st.session_state.get("selected_doc_ids", [])

    documents = get_documents()

    if documents is None:
        if not api_connected:
            st.warning("⚠️ 无法获取文档列表：后端服务未连接")
        else:
            st.error("❌ 获取文档列表失败，请稍后重试")
        return

    total = len(documents)
    ready_count = sum(1 for d in documents if d.get("status") == "ready")
    processing_count = sum(1 for d in documents if d.get("status") in ["processing", "uploading", "pending", "circuit_indexing"])
    error_count = sum(1 for d in documents if d.get("status") == "error")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    for col, value, label in [
        (col_s1, total, "文档总数"),
        (col_s2, ready_count, "已就绪"),
        (col_s3, processing_count, "处理中"),
        (col_s4, error_count, "异常"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="documents-summary-card">
                    <div class="documents-summary-value">{value}</div>
                    <div class="documents-summary-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not documents:
        empty_state("暂无文档", "📭")
        st.info("💡 **提示**: 请上传PDF格式的电子器件规格书开始使用")
    else:
        with st.container():
            for idx, doc in enumerate(documents):
                doc_id = doc.get("document_id", "")
                if not doc_id:
                    doc_id = doc.get("id", doc.get("doc_id", ""))
                if not doc_id:
                    doc_id = f"doc_{idx}"

                filename = doc.get("filename", "未知文档")
                status = doc.get("status", "unknown")
                upload_time = doc.get("upload_time", "")
                page_count = doc.get("page_count", 0)
                file_size = doc.get("file_size", 0)
                chunk_count = doc.get("chunk_count", 0)
                table_count = doc.get("table_count", 0)
                circuit_count = doc.get("circuit_count", 0)

                if circuit_count == -1:
                    circuit_display = "⏳ 索引中"
                elif circuit_count == 0:
                    circuit_display = "0 电路图"
                else:
                    circuit_display = f"{circuit_count} 电路图"
                error_message = doc.get("error_message", "")

                is_highlighted = highlight_doc_id == doc_id
                is_selected = doc_id in selected_doc_ids
                is_ready = status == "ready"

                status_classes = {
                    "ready": "status-success",
                    "processing": "status-warning",
                    "uploading": "status-info",
                    "pending": "status-info",
                    "partial": "status-warning",
                    "circuit_indexing": "status-warning",
                    "error": "status-error",
                }
                status_labels = {
                    "ready": "✅ 就绪",
                    "processing": "⏳ 处理中",
                    "uploading": "📤 上传中",
                    "pending": "⏳ 等待中",
                    "partial": "⚠️ 电路索引中",
                    "circuit_indexing": "🔌 电路索引中",
                    "error": "❌ 错误",
                }

                with st.container():
                    selection_note = " | 已设为当前文档" if is_selected else ""
                    st.markdown(
                        f"""
                        <div class="document-card {'highlighted' if is_highlighted else ''}">
                            <div class="document-card-header">
                                <div class="document-card-title">📄 {filename}</div>
                                <span class="status-badge {status_classes.get(status, 'status-processing')}">
                                    {status_labels.get(status, status)}
                                </span>
                            </div>
                            <div class="document-card-meta">
                                {page_count} 页 | {file_size/1024/1024:.1f} MB | {upload_time}{selection_note}
                            </div>
                            <div class="document-card-stats">
                                📝 {chunk_count} 文本块 | 📊 {table_count} 表格 | 🔌 {circuit_display}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    progress = doc.get("progress", 0)
                    stage = doc.get("stage", "")

                    if status in ["processing", "uploading", "circuit_indexing"] and progress > 0:
                        stage_text = _STAGE_LABELS.get(stage, stage)
                        if status == "circuit_indexing":
                            st.progress(progress / 100, text=f"🔌 {stage_text}... {progress}%")
                        else:
                            st.progress(progress / 100, text=f"{stage_text}... {progress}%")
                    elif status == "error" and error_message:
                        st.caption(f"失败原因: {error_message}")
                    elif status == "partial" and stage == "indexing_circuits":
                        st.caption("⏳ 电路图索引正在进行中，完成后将自动更新...")
                        st.progress(progress / 100, text=f"电路索引... {progress}%")
                    elif is_ready:
                        pass
                    else:
                        st.caption("文档尚未就绪，暂不开放分析功能。")

                    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

                    with col_btn1:
                        if st.button(
                            "💬 问答",
                            key=f"chat_{idx}_{doc_id}",
                            use_container_width=True,
                            disabled=not is_ready,
                        ):
                            _set_current_document_context(doc_id, filename)
                            st.switch_page("pages/2_Chat.py")

                    with col_btn2:
                        if st.button(
                            "📍 引脚",
                            key=f"pinout_{idx}_{doc_id}",
                            use_container_width=True,
                            disabled=not is_ready,
                        ):
                            _set_current_document_context(doc_id, filename)
                            st.switch_page("pages/3_Pinout.py")

                    with col_btn3:
                        if st.button("🗑️ 删除", key=f"delete_{idx}_{doc_id}", use_container_width=True):
                            st.session_state.delete_doc_id = doc_id
                            st.session_state.delete_doc_filename = filename
                            st.session_state.show_delete_confirm = True

                    if st.session_state.get("show_delete_confirm", False) and st.session_state.get("delete_doc_id") == doc_id:
                        st.markdown(f"""
                        <div style="padding: 1rem; background: #fee2e2; border: 1px solid #fecaca; border-radius: 8px; margin: 1rem 0;">
                            <div style="font-weight: 600; color: #991b1b; margin-bottom: 0.5rem;">⚠️ 确认删除</div>
                            <div style="color: #991b1b; margin-bottom: 1rem;">
                                确定要删除文档 <strong>{filename}</strong> 吗？<br>
                                此操作将删除文档文件及其所有相关数据（包括向量索引、表格、图像缓存等）。
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        col_confirm, col_cancel = st.columns([1, 1])
                        with col_cancel:
                            if st.button("❌ 取消", key=f"cancel_delete_{idx}_{doc_id}"):
                                for key in ("delete_doc_id", "delete_doc_filename", "show_delete_confirm"):
                                    st.session_state.pop(key, None)
                                st.rerun()

                        with col_confirm:
                            if st.button("✅ 确认删除", key=f"confirm_delete_{doc_id}", type="primary"):
                                import time

                                st.session_state.last_delete_time = time.time()
                                st.session_state.deleting_doc_id = doc_id

                                result = delete_document(doc_id)
                                if result and result.get("success"):
                                    st.success(f"✅ 文档 '{filename}' 删除成功")
                                    for key in ("delete_doc_id", "delete_doc_filename", "show_delete_confirm"):
                                        st.session_state.pop(key, None)
                                    _fetch_documents_cached.clear()
                                    st.session_state.pop("cached_documents", None)
                                    selected = st.session_state.get("selected_doc_ids", [])
                                    if doc_id in selected:
                                        st.session_state.selected_doc_ids = [d for d in selected if d != doc_id]

                                    st.session_state.api_connected = True
                                    st.session_state.api_check_ts = time.monotonic()
                                    st.rerun()
                                else:
                                    error_msg = result.get("error", "未知错误") if result else "未知错误"
                                    st.error(f"❌ 删除失败: {error_msg}")
                                    st.session_state.pop("deleting_doc_id", None)

        if highlight_doc_id:
            if st.button("清除高亮", key="clear_highlight"):
                st.session_state.highlight_doc_id = None
                st.rerun()

render_documents_list()
