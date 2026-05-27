"""
Circuit Search Page — EC-VeriQuery

Provides an interactive interface for searching circuit diagrams within uploaded
electronic component datasheets. Users enter natural language queries (e.g.
"NE5532 amplifier circuit", "filter", "power supply") and the backend performs
TrueColPali visual retrieval with intelligent re-ranking, returning the most
relevant circuit figures along with metadata.

Search pipeline:
    User query → api_client.search_circuits() → POST /api/v1/circuit/search
    → query augmentation (synonym expansion)
    → TrueColPaliIndexer.search() (ColPali MaxSim visual retrieval)
    → intelligent re-ranking (circuit-type + chip-model bonus)
    → JSON response {circuits, search_time, cached}

Image URL resolution follows a three-tier fallback strategy:
    1. image_path → direct file path (most precise, cropped circuit figure)
    2. circuit_id → database lookup (standard method)
    3. filename + page → full-page screenshot (fallback)

Result caching via session_state survives Streamlit script re-executions.
A context fingerprint (query, top_k, sorted doc_ids) is used to detect
stale cache entries when the user changes search parameters.

Dependencies:
    - theme.py: Academic-style CSS variables and similarity bar component
    - sidebar_nav.py: Sidebar navigation and document selector
    - api_client.py: HTTP REST API client for circuit search endpoint
"""
import html
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.set_page_config(page_title="电路检索 - EC-VeriQuery", page_icon="🔌", layout="wide")

try:
    from theme import apply_academic_theme, render_similarity_bar
    from sidebar_nav import render_sidebar_nav
    from api_client import (
        check_api_connection,
        get_circuit_image_by_path_url,
        get_circuit_image_url,
        get_document_id_by_filename,
        get_documents,
        get_page_image_url,
        is_api_connected,
        search_circuits,
    )
except ImportError as e:
    st.error(f"组件导入失败: {e}")
    st.stop()


CIRCUIT_TYPE_EXAMPLES = "输入电路类型关键词，如：放大电路、滤波器、复位电路、启动电路、电源电路、NE5532、LM358、STM32、运放、典型应用等"

RESULT_KEY = "circuit_result"
RESULT_CONTEXT_KEY = "circuit_result_context"
SELECTED_RESULT_KEY = "circuit_selected_index"
QUERY_KEY = "circuit_query_value"

PAGE_CSS = """
<style>
.block-container {
    max-width: 100% !important;
    padding-top: 2.5rem !important;
}
section.main > div { padding-top: 2.5rem !important; }

.circuit-page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: 0.55rem 0 0.7rem 0;
    margin-bottom: 0.8rem;
    border-bottom: 2px solid var(--primary);
    flex-wrap: wrap;
}
.circuit-page-title {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.circuit-page-meta {
    font-size: 0.84rem;
    color: var(--text-secondary);
    text-align: right;
}

.circuit-helper-card,
.circuit-search-card,
.circuit-guide-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.95rem 1.05rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.circuit-helper-card {
    margin-bottom: 0.9rem;
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.65;
}

.circuit-search-card {
    margin-bottom: 0.9rem;
}

.circuit-inline-note {
    color: var(--text-secondary);
    font-size: 0.82rem;
    margin-top: 0.45rem;
    line-height: 1.55;
}

.circuit-detail-title,
.circuit-guide-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.65rem;
}

.circuit-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.35rem;
}
.circuit-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.22rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    border: 1px solid var(--border);
    background: #f8fafc;
    color: #475569;
}
.circuit-chip.primary {
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #1e40af;
}

.circuit-kv {
    display: grid;
    grid-template-columns: 108px 1fr;
    gap: 0.55rem 0.8rem;
    margin-top: 0.9rem;
    align-items: start;
}
.circuit-kv-label {
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.circuit-kv-value {
    color: var(--text-primary);
    font-size: 0.9rem;
    line-height: 1.6;
    word-break: break-word;
}

.circuit-evidence-box {
    margin-top: 0.6rem;
    padding: 0.5rem 0.7rem;
    border-radius: 6px;
    background: #fffbeb;
    border: 1px solid #fde68a;
}
.circuit-evidence-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #92400e;
    margin-bottom: 0.2rem;
}
.circuit-evidence-text {
    font-size: 0.82rem;
    color: #78350f;
    line-height: 1.5;
}

.circuit-image-container {
    margin-left: 30px;
    margin-top: -1rem;
}

.circuit-guide-card {
    margin-top: 1rem;
}
.circuit-guide-card ul {
    margin: 0.45rem 0 0 1rem;
    color: var(--text-secondary);
    font-size: 0.88rem;
    line-height: 1.7;
}
</style>
"""


def _escape_text(value: Any) -> str:
    """HTML-escape a value for safe insertion into st.markdown(unsafe_allow_html=True)."""
    if value is None or value == "":
        return "-"
    return html.escape(str(value))


def _strip_html(text: str) -> str:
    """Remove HTML/Markdown markup and collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text to *max_len* characters, appending '…' if shortened."""
    if not text or len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _normalize_list(value: Any) -> List[Any]:
    """Normalize heterogeneous input (list / str / None) into a clean list."""
    if isinstance(value, list):
        return [item for item in value if item not in (None, "", [])]
    if value in (None, "", []):
        return []
    return [value]


def _component_label(component: Any) -> str:
    """Extract the best display label from a component dict or string."""
    if isinstance(component, dict):
        return str(
            component.get("name")
            or component.get("type")
            or component.get("reference")
            or component.get("value")
            or "未命名元件"
        )
    return str(component)


def _component_summary(components: Any, limit: int = 6) -> str:
    """Generate a short summary string for a component list."""
    labels = [_component_label(item) for item in _normalize_list(components)]
    if not labels:
        return "暂无元件信息"
    preview = labels[:limit]
    suffix = f" 等 {len(labels)} 项" if len(labels) > limit else ""
    return "、".join(preview) + suffix


def _format_circuit_types(circuit: Dict[str, Any]) -> str:
    """Format circuit_type list into a readable slash-separated string."""
    circuit_types = _normalize_list(circuit.get("circuit_type"))
    return " / ".join(str(item) for item in circuit_types) if circuit_types else "通用电路"


def _get_circuit_name(circuit: Dict[str, Any], index: int) -> str:
    """Extract a display name, falling back to '电路 N' if the raw name is invalid JSON."""
    raw_name = str(circuit.get("name") or circuit.get("title") or f"电路 {index + 1}").strip()
    if raw_name.startswith('{"') or "circuit_types" in raw_name:
        return f"电路 {index + 1}"
    return raw_name


def _get_similarity(circuit: Dict[str, Any]) -> float:
    """Extract and normalize the similarity score to the [0, 1] range."""
    score = circuit.get("score", circuit.get("similarity", 0)) or 0
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.0
    return score if score <= 1 else score / 100


def _get_page_value(circuit: Dict[str, Any]) -> Any:
    """Extract page number, handling both 'page' and 'page_number' field names."""
    return circuit.get("page", circuit.get("page_number", "-"))


def _try_page_image_fallback(circuit: Dict[str, Any], documents: List[Dict[str, Any]]) -> str:
    """Attempt to load a page screenshot when circuit image fails.

    Returns a page image URL or empty string if unavailable.
    """
    page = _get_page_value(circuit)
    filename = circuit.get("filename", circuit.get("source", circuit.get("document", "")))
    if not filename or page in ("", None, "-"):
        return ""
    try:
        page_num = int(page)
    except (TypeError, ValueError):
        return ""
    document_id = circuit.get("document_id") or get_document_id_by_filename(filename, documents)
    if document_id:
        return get_page_image_url(document_id, page_num) or ""
    return ""


def _build_search_context(query: str, top_k: int, doc_ids: List[str]) -> Tuple[str, int, Tuple[str, ...]]:
    """Build an immutable context fingerprint for cache validity checks.

    doc_ids are sorted so that order differences do not affect comparison.
    """
    return (query.strip(), top_k, tuple(sorted(doc_ids)))


def _resolve_doc_context(doc_ids: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Resolve document metadata for the currently selected doc IDs.

    Returns:
        (ready_docs, selected_docs, doc_info_text)
    """
    if not doc_ids:
        return [], [], "当前文档: 未选择"

    try:
        documents = get_documents()
        ready_docs = [doc for doc in documents if doc.get("status") == "ready"]
        selected_docs = [
            doc for doc in ready_docs if doc.get("document_id", doc.get("id", "")) in doc_ids
        ]
        if not selected_docs:
            return ready_docs, [], f"当前文档: 已选择 {len(doc_ids)} 份"

        names = [doc.get("filename", "未知文档") for doc in selected_docs]
        if len(names) > 2:
            doc_info = f"当前文档: {names[0]}、{names[1]} 等 {len(names)} 份"
        else:
            doc_info = f"当前文档: {'、'.join(names)}"
        return ready_docs, selected_docs, doc_info
    except Exception:
        return [], [], f"当前文档: 已选择 {len(doc_ids)} 份"


def _resolve_image_url(circuit: Dict[str, Any], documents: List[Dict[str, Any]]) -> str:
    """Resolve the circuit figure URL using a three-tier fallback strategy.

    Priority: image_path > circuit_id > filename+page.
    Returns an empty string if all methods fail.
    """
    image_path = circuit.get("image_path", "")
    if image_path:
        url = get_circuit_image_by_path_url(image_path)
        if url:
            return url

    circuit_id = circuit.get("circuit_id") or circuit.get("id")
    if circuit_id:
        url = get_circuit_image_url(circuit_id)
        if url:
            return url

    filename = circuit.get("filename", circuit.get("source", circuit.get("document", "")))
    page = _get_page_value(circuit)
    if filename and page not in ("", None, "-"):
        document_id = circuit.get("document_id") or get_document_id_by_filename(filename, documents)
        try:
            page_num = int(page)
        except (TypeError, ValueError):
            return ""
        if document_id:
            return get_page_image_url(document_id, page_num) or ""

    return ""


apply_academic_theme()
check_api_connection()
st.markdown(PAGE_CSS, unsafe_allow_html=True)

with st.sidebar:
    render_sidebar_nav()

api_connected = is_api_connected()
doc_ids = st.session_state.get("selected_doc_ids", [])

for key, default in [
    (RESULT_KEY, None),
    (RESULT_CONTEXT_KEY, None),
    (SELECTED_RESULT_KEY, 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

documents, selected_docs, doc_info = _resolve_doc_context(doc_ids)

st.markdown(
    f"""
<div class="circuit-page-header">
    <h2 class="circuit-page-title">🔌 电路检索</h2>
    <span class="circuit-page-meta">{_escape_text(doc_info)}</span>
</div>
""",
    unsafe_allow_html=True,
)

if not api_connected:
    st.error("⚠️ 后端服务未连接，无法进行电路检索。")
    if st.button("🔄 重新连接", type="primary"):
        for k in ("api_connected", "api_health_ts"):
            st.session_state.pop(k, None)
        st.rerun()
    st.stop()

if not doc_ids:
    st.warning("⚠️ 请先在侧边栏选择文档。")
    if st.button("📄 前往上传文档", type="primary"):
        st.switch_page("pages/1_Documents.py")
    st.stop()

with st.container(border=False):
    input_col, btn_col = st.columns([5.5, 1])
    with input_col:
        query = st.text_input(
            "电路检索",
            placeholder=CIRCUIT_TYPE_EXAMPLES,
            key=QUERY_KEY,
        ).strip()
    with btn_col:
        search_button = st.button("🔍 检索", type="primary", use_container_width=True)

current_context = _build_search_context(query, 3, doc_ids)
stored_result = st.session_state.get(RESULT_KEY)
stored_context = st.session_state.get(RESULT_CONTEXT_KEY)
result_is_current = bool(stored_result) and stored_context == current_context

if search_button:
    if not query:
        st.info("请输入要检索的电路类型、功能或元件描述。")
    else:
        with st.spinner("正在检索电路，请稍候..."):
            result = search_circuits(query, top_k=3, doc_ids=doc_ids)

        if result and result.get("success", True) and not result.get("error"):
            st.session_state[RESULT_KEY] = result
            st.session_state[RESULT_CONTEXT_KEY] = current_context
            st.session_state[SELECTED_RESULT_KEY] = 0
            stored_result = result
            stored_context = current_context
            result_is_current = True
        else:
            st.error(f"检索失败：{(result or {}).get('error', '未知错误')}")

if stored_result and not result_is_current and query:
    st.warning("当前关键词或文档范围已变化，请重新点击「检索」以刷新结果。")

if result_is_current:
    result = stored_result
    circuits = result.get("circuits", result.get("results", [])) if isinstance(result, dict) else []
    search_time = float(result.get("search_time", 0) or 0)
    cached = bool(result.get("cached", False))

    if not circuits:
        st.info(f"💡 未找到匹配电路。建议尝试：\n\n"
                f"- **电路类型**：放大电路、滤波器、电源电路、振荡电路、复位电路、启动电路、电平转换、ADC/DAC、去耦电路、比较器\n"
                f"- **器件型号**：NE5532、LM358、TL072、STM32、运放、555定时器\n"
                f"- **应用场景**：典型应用、LED驱动、ADC前端、降压电路、上电复位、看门狗\n\n"
                f"系统会自动识别查询意图并检索相关电路图。")
    else:
        selected_index = int(st.session_state.get(SELECTED_RESULT_KEY, 0) or 0)
        selected_index = max(0, min(selected_index, len(circuits) - 1))
        st.session_state[SELECTED_RESULT_KEY] = selected_index
        selected_circuit = circuits[selected_index]
        selected_name = _get_circuit_name(selected_circuit, selected_index)
        selected_similarity = _get_similarity(selected_circuit)

        current_doc_name = selected_docs[0].get("filename", "未知文档") if selected_docs else "未选择文档"

        left_col, right_col = st.columns([0.8, 1], gap="medium")

        with left_col:
            detail_name = _get_circuit_name(selected_circuit, selected_index)
            detail_types = _format_circuit_types(selected_circuit)
            detail_components = _normalize_list(selected_circuit.get("components"))
            detail_page = _get_page_value(selected_circuit)
            detail_figure = selected_circuit.get("figure_label", "")
            detail_caption = _strip_html(selected_circuit.get("caption") or "")
            detail_description = _strip_html(
                selected_circuit.get("description") or selected_circuit.get("content") or ""
            )
            detail_image_url = _resolve_image_url(selected_circuit, documents)

            st.markdown(
                f"""
<div class="circuit-detail-title">{_escape_text(detail_name)}</div>
<div class="circuit-chip-row">
    <span class="circuit-chip primary">{_escape_text(detail_types)}</span>
    <span class="circuit-chip">P{_escape_text(detail_page)}</span>
</div>
""",
                unsafe_allow_html=True,
            )

            render_similarity_bar(_get_similarity(selected_circuit), label="匹配度")

            st.markdown(
                f"""
<div class="circuit-kv">
    <div class="circuit-kv-label">分析文档</div>
    <div class="circuit-kv-value">{_escape_text(current_doc_name)}</div>
    <div class="circuit-kv-label">图号</div>
    <div class="circuit-kv-value">{_escape_text(detail_figure or '未提供')}</div>
    <div class="circuit-kv-label">元件</div>
    <div class="circuit-kv-value">{_escape_text(_component_summary(detail_components, limit=8))}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            if detail_components:
                chip_html = "".join(
                    f'<span class="circuit-chip">{_escape_text(_component_label(item))}</span>'
                    for item in detail_components[:10]
                )
                st.markdown(f'<div class="circuit-chip-row">{chip_html}</div>', unsafe_allow_html=True)

            evidence_text = detail_caption or detail_description or "后端当前未返回更详细的文字说明。"
            st.markdown(
                f"""
<div class="circuit-evidence-box">
    <div class="circuit-evidence-title">关键信息摘要</div>
    <div class="circuit-evidence-text">{_escape_text(_truncate(evidence_text, 260))}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            selected_index = st.selectbox(
                "切换电路",
                options=list(range(len(circuits))),
                format_func=lambda idx: f"#{idx + 1} {_get_circuit_name(circuits[idx], idx)} ({_get_similarity(circuits[idx]):.0%})",
                key=SELECTED_RESULT_KEY,
            )

        with right_col:
            if detail_image_url:
                _, img_col = st.columns([0.1, 1])
                with img_col:
                    try:
                        st.image(detail_image_url, caption=detail_name, width=730)
                    except Exception:
                        page_fallback = _try_page_image_fallback(selected_circuit, documents)
                        if page_fallback:
                            st.image(page_fallback, caption=f"{detail_name} (页面截图)", width=730)
                        else:
                            st.warning("电路图文件无法加载，原图可能尚未提取或路径已失效。")
            else:
                st.info("当前结果未找到可展示的截图或电路图路径。")
