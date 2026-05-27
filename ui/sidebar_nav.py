"""
Sidebar Navigation Component — EC-VeriQuery

Shared sidebar renderer invoked by every page in the Streamlit multi-page
application.  Provides four visual zones:

    1. Brand header  — system name, icon, subtitle
    2. Page nav      — seven st.page_link entries
    3. Doc selector  — st.multiselect for choosing active documents
    4. Status footer — API connection, document counts, selection scope

Cross-page state synchronisation:
    st.multiselect on_change → _sync_doc_selection()
    → writes st.session_state.selected_doc_ids
    → read by all feature pages

Dependencies:
    - api_client.get_documents(): lazy-loaded document list
    - theme.py: CSS classes (sb-header, sb-brand, sb-footer, …)
"""
import streamlit as st

DOC_SELECT_KEY = "sidebar_doc_multiselect"


def _sync_doc_selection():
    """Synchronise the multiselect widget value to ``selected_doc_ids``.

    Bound as the ``on_change`` callback of the document multiselect widget.
    Decouples the widget's internal key (``DOC_SELECT_KEY``) from the
    canonical state key (``selected_doc_ids``) consumed by feature pages.
    """
    selected = st.session_state.get(DOC_SELECT_KEY, [])
    st.session_state.selected_doc_ids = [str(d) for d in list(selected)] if selected else []


def render_sidebar_nav():
    """Render the full sidebar used across all application pages.

    Layout zones (top → bottom):
        Brand header → page navigation links → document multiselect → status footer
    """
    st.markdown("""
    <div class="sb-header">
        <div class="sb-brand">
            <div class="sb-brand-icon">🌻</div>
            <div class="sb-brand-text">
                <h3>EC-VeriQuery</h3>
                <p>电子硬件规格书智能问答系统</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    documents = st.session_state.get("cached_documents")
    if documents is None:
        try:
            from api_client import get_documents
            documents = get_documents()
            st.session_state.cached_documents = documents
        except Exception:
            documents = []
            st.warning("文档列表加载失败，请检查API服务连接")

    ready_docs = [d for d in documents if d.get("status") == "ready"]

    nav_items = [
        ("🧭 导航",    "app.py"),
        ("📄 文档管理", "pages/1_Documents.py"),
        ("💬 智能问答", "pages/2_Chat.py"),
        ("📍 引脚分析", "pages/3_Pinout.py"),
        ("⚡ ERC检查",  "pages/4_ERC.py"),
        ("📊 参数对比", "pages/5_Compare.py"),
        ("🔌 电路检索", "pages/6_Circuit.py"),
    ]

    for label, page in nav_items:
        st.page_link(page, label=label, use_container_width=True)

    st.markdown('<div class="sb-nav-divider"></div>', unsafe_allow_html=True)

    doc_options = {doc.get("document_id", doc.get("id", f"doc_{i}")): doc.get('filename', '未知文档')
                  for i, doc in enumerate(ready_docs)} if ready_docs else {}

    current_selected = st.session_state.get("selected_doc_ids", [])
    valid_selected = [d for d in current_selected if d in doc_options]

    if DOC_SELECT_KEY not in st.session_state:
        st.session_state[DOC_SELECT_KEY] = valid_selected

    pending = st.session_state.pop("_pending_sidebar_selection", None)
    if pending is not None:
        pending_valid = [d for d in pending if d in doc_options]
        st.session_state.pop(DOC_SELECT_KEY, None)
        st.session_state.selected_doc_ids = pending_valid
        valid_selected = pending_valid

    if doc_options:
        st.multiselect(
            "📋 选择分析的文档",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options.get(x, x),
            placeholder="选择文档...",
            key=DOC_SELECT_KEY,
            on_change=_sync_doc_selection
        )
    elif documents:
        st.info("暂无就绪文档，请先上传并等待处理完成")
    else:
        st.info("未检测到文档，请先上传文档")

    api_connected  = st.session_state.get("api_connected", False)
    doc_count      = len(documents) if documents else 0
    ready_count    = len(ready_docs) if ready_docs else 0
    selected_count = len(st.session_state.get("selected_doc_ids", []))
    status_class   = "online" if api_connected else "offline"
    status_text    = "已连接" if api_connected else "未连接"

    st.markdown(f"""
    <div class="sb-footer">
        <span class="sb-footer-title">系统状态</span>
        <div class="sb-footer-item">
            <span class="sb-footer-label">🔌 API 服务</span>
            <span class="sb-footer-value {status_class}">● {status_text}</span>
        </div>
        <div class="sb-footer-item">
            <span class="sb-footer-label">📁 文档总数</span>
            <span class="sb-footer-value">{doc_count} 份</span>
        </div>
        <div class="sb-footer-item">
            <span class="sb-footer-label">✅ 就绪文档</span>
            <span class="sb-footer-value">{ready_count} 份</span>
        </div>
        <div class="sb-footer-item">
            <span class="sb-footer-label">🧾 当前范围</span>
            <span class="sb-footer-value">{selected_count} 份</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
