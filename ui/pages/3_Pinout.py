"""
Pinout Analysis Page — EC-VeriQuery

Provides an interactive interface for analyzing chip pinout definitions extracted
from uploaded electronic component datasheets. Users input a chip model and
package type, the backend retrieves pin information via RAG, and results are
rendered as an interactive SVG pin diagram alongside a pin list table.

Key features:
    - Chip model + package type input with dynamic package selection
    - Interactive SVG pin diagram with click-to-detail popup (HTML+JS via components.html)
    - Pin list table with type labels, functions, and alternate functions
    - Citation tracing: document source and page number with screenshot links
    - Query context caching: skip re-analysis when inputs are unchanged
    - Custom CSS for card-style layout and scrollable pin table

Dependencies:
    - theme.py: Academic-style CSS variables and empty-state component
    - sidebar_nav.py: Sidebar navigation and document selector
    - api_client.py: HTTP REST API client for pinout analysis endpoints
"""

from typing import Any, Dict, List, Optional

import html
import json
import os
import sys

import streamlit as st
import streamlit.components.v1 as components

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.set_page_config(
    page_title="引脚分析 - EC-VeriQuery",
    page_icon="📍",
    layout="wide",
)

try:
    from theme import apply_academic_theme
    from sidebar_nav import render_sidebar_nav
    from api_client import (
        analyze_pinout,
        check_api_connection,
        get_document_id_by_filename,
        get_documents,
        get_page_image_url,
        is_api_connected,
    )
except ImportError as e:
    st.error(f"组件导入失败: {e}")
    st.stop()

apply_academic_theme()
check_api_connection()

PIN_TYPE_LABELS = {
    "power": "电源",
    "ground": "接地",
    "io": "双向IO",
    "bidirectional": "GPIO双向",
    "input": "输入",
    "output": "输出",
    "analog": "模拟",
    "nc": "NC",
    "special": "特殊",
}

PAGE_CSS = """
<style>
[data-testid="stMetricLabel"]  { font-size: 0.75rem !important; color: #64748b !important; margin-bottom: 0 !important; }
[data-testid="stMetricValue"]  { font-size: 1.2rem !important; font-weight: 600 !important; color: #1e293b !important; line-height: 1.2 !important; }
[data-testid="stMetric"]       { padding: 0.4rem 0 0.25rem 0 !important; }
div[data-testid="stHorizontalBlock"] { gap: 0.7rem !important; }
section.main > div { padding-top: 2.5rem !important; }
section.main > div:first-child { padding-top: 2.5rem !important; }
.block-container {
    max-width: 100% !important;
    padding-top: 2.5rem !important;
}
div[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem !important;
}
.pinout-list-dataframe thead th, .pinout-list-dataframe tbody td {
    text-align: center !important;
    font-size: 1rem !important;
}
.pinout-list-dataframe thead th {
    font-weight: 600 !important;
    background-color: #f8fafc !important;
}
.pinout-list-dataframe tbody td {
    padding: 0.5rem 0.75rem !important;
}
.pinout-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 1.05rem;
    table-layout: fixed;
}
.pinout-table thead th {
    text-align: center;
    font-weight: 600;
    background-color: #f8fafc;
    padding: 0.75rem 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}
.pinout-table tbody td {
    text-align: center;
    padding: 0.6rem 0.5rem;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: middle;
}
.pinout-table tbody tr:hover {
    background-color: #f8fafc;
}
.pinout-table tbody tr:last-child td {
    border-bottom: none;
}
.pin-table-scroll {
    max-height: 580px;
    overflow-y: auto;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #fff;
}
.pin-table-scroll::-webkit-scrollbar {
    width: 6px;
}
.pin-table-scroll::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 3px;
}
.pin-table-scroll::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}
.pin-table-scroll::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
.pin-table-scroll .pinout-table thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    background-color: #f8fafc;
}
.stDataFrame thead th, .stDataFrame tbody td {
    text-align: center !important;
    font-size: 1.05rem !important;
}
.stDataFrame thead th {
    font-weight: 600 !important;
    background-color: #f8fafc !important;
}
.stDataFrame tbody td {
    padding: 0.5rem 0.75rem !important;
}
.stDataFrame table thead tr th {
    text-align: center !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
.stDataFrame table tbody tr td {
    text-align: center !important;
    font-size: 1.05rem !important;
}
[data-testid="stDataFrame"] table thead tr th {
    text-align: center !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
[data-testid="stDataFrame"] table tbody tr td {
    text-align: center !important;
    font-size: 1.05rem !important;
}
.pinout-page-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.2rem;
}
.pinout-page-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1e293b;
    line-height: 1.2;
}
.pinout-page-meta {
    font-size: 0.84rem;
    color: #64748b;
    text-align: right;
    vertical-align: bottom;
}
.pinout-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.05rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 0.85rem;
}
.pinout-card-title {
    font-size: 0.98rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.75rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid #f1f5f9;
}
.pinout-chip-info {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}
.pinout-chip-info-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.75rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid #f1f5f9;
}
.pinout-chip-info-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
}
.pinout-chip-label {
    display: block;
    color: #64748b;
    font-size: 0.85rem;
    margin-bottom: 0.25rem;
}
.pinout-chip-value {
    display: block;
    font-weight: 600;
    color: #1e293b;
    font-size: 1.1rem;
}
</style>
"""


def _normalize_doc_ids(doc_ids: List[Any]) -> List[str]:
    """Normalize document IDs to a sorted list of strings for consistent comparison."""
    return sorted(str(doc_id) for doc_id in (doc_ids or []))


def _build_query_context(chip_name: str, doc_ids: List[Any], requested_package: Optional[str]) -> Dict[str, Any]:
    """Build a query context dict used as a cache key to detect input changes."""
    return {
        "chip_name": chip_name.strip(),
        "doc_ids": _normalize_doc_ids(doc_ids),
        "requested_package": (requested_package or "").strip(),
    }


def _build_package_context(chip_name: str, doc_ids: List[Any]) -> Dict[str, Any]:
    """Build a package-only context (excludes requested_package) for available_packages caching."""
    return {
        "chip_name": chip_name.strip(),
        "doc_ids": _normalize_doc_ids(doc_ids),
    }


def _resolve_doc_context(doc_ids: List[Any]):
    """Resolve document IDs to full document objects and a display string.

    Returns:
        Tuple of (all_documents, selected_docs, display_text).
    """
    if not doc_ids:
        return [], [], "当前文档: 未选择"

    try:
        documents = get_documents()
        ready_docs = [doc for doc in documents if doc.get("status") == "ready"]
        selected_docs = [
            doc for doc in ready_docs
            if str(doc.get("document_id") or doc.get("id")) in _normalize_doc_ids(doc_ids)
        ]
        if selected_docs:
            names = ", ".join(doc.get("filename", "未知文档") for doc in selected_docs)
            return documents, selected_docs, f"当前文档: {names}"
        return documents, [], f"当前文档: {len(doc_ids)} 份文档"
    except Exception:
        return [], [], f"当前文档: {len(doc_ids)} 份文档"


def _pin_type_label(pin_type: str) -> str:
    """Convert a backend pin type identifier to a Chinese display label."""
    return PIN_TYPE_LABELS.get((pin_type or "").lower(), (pin_type or "未标注").upper())


def _build_pin_rows(pins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw pin data from the backend into standardized display rows.

    Handles field name variations (pin/pin_number, name/pin_name, pin_type/type),
    formats lists and dicts into display strings, and sorts by pin number
    (numeric pins first, then alphanumeric).
    """
    rows: List[Dict[str, Any]] = []
    for pin in pins or []:
        functions = pin.get("functions") or []
        alternate_functions = pin.get("alternate_functions") or []
        electrical = pin.get("electrical") or {}
        row = {
            "引脚号": pin.get("pin", pin.get("pin_number", "")),
            "引脚名称": pin.get("name", pin.get("pin_name", "")),
            "类型": _pin_type_label(pin.get("pin_type", pin.get("type", ""))),
            "功能": ", ".join(str(item) for item in functions) if functions else "-",
            "电气信息": ", ".join(f"{k}: {v}" for k, v in electrical.items()) if electrical else "-",
            "描述": pin.get("description", "") or "-",
            "pin_type_raw": (pin.get("pin_type", pin.get("type", "")) or "").lower(),
            "functions_raw": functions,
            "alternate_raw": alternate_functions,
            "electrical_raw": electrical,
            "raw": pin,
        }
        rows.append(row)

    def _pin_sort_key(item: Dict[str, Any]) -> tuple:
        try:
            return (0, int(item["引脚号"]), "")
        except Exception:
            return (1, 2**31, str(item["引脚号"]))

    rows.sort(key=_pin_sort_key)
    return rows


def _render_svg(svg_content: str, pin_rows: List[Dict[str, Any]]):
    """Render an interactive SVG pin diagram with click-to-detail popup.

    Embeds a full HTML page via components.html() to enable JavaScript event
    handling for pin click popups. Pin data is serialized as JSON and injected
    into the JS context for the popup to display.
    """
    card_css = "background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.04);"

    pin_data_json = json.dumps(
        {str(row["引脚号"]): row for row in pin_rows},
        ensure_ascii=False,
        default=str,
    )

    svg_page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  html,body{{width:100%;height:100%;overflow:hidden;background:transparent;}}
  #card{{
    width:100%;height:100%;
    {card_css}
    display:flex;
    align-items:center;
    justify-content:center;
    padding:14px;
  }}
  #card svg{{
    display:block;
    width:100% !important;
    height:100% !important;
    max-width:100%;
    max-height:100%;
    overflow:visible;
  }}
  .pin-detail{{
    display:none;
    position:fixed;
    top:50%;left:50%;
    transform:translate(-50%,-50%);
    background:#fff;
    border:1px solid #e2e8f0;
    border-radius:12px;
    padding:1.1rem 1.3rem;
    box-shadow:0 8px 30px rgba(0,0,0,.12);
    z-index:9999;
    min-width:260px;
    max-width:360px;
  }}
  .pin-detail.active{{display:block;}}
  .pin-detail-title{{font-size:1rem;font-weight:600;color:#1e293b;margin-bottom:0.6rem;padding-bottom:0.4rem;border-bottom:1px solid #f1f5f9;}}
  .pin-detail-kv{{display:grid;grid-template-columns:80px 1fr;row-gap:0.35rem;column-gap:0.6rem;font-size:0.88rem;}}
  .pin-detail-label{{color:#64748b;}}
  .pin-detail-value{{color:#1e293b;font-weight:500;word-break:break-word;}}
  .pin-detail-close{{position:absolute;top:8px;right:12px;cursor:pointer;color:#94a3b8;font-size:1.1rem;}}
  .pin-detail-close:hover{{color:#475569;}}
  .pin-overlay{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.15);z-index:9998;}}
  .pin-overlay.active{{display:block;}}
</style>
</head>
<body>
<div id="card">{svg_content}</div>
<div class="pin-overlay" id="pinOverlay"></div>
<div class="pin-detail" id="pinDetail">
  <span class="pin-detail-close" id="pinDetailClose">&times;</span>
  <div class="pin-detail-title" id="pinDetailTitle">引脚详情</div>
  <div class="pin-detail-kv" id="pinDetailContent"></div>
</div>
<script>
(function(){{
  const svg = document.querySelector('#card svg');
  if (!svg) return;
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.style.width = '100%';
  svg.style.height = '100%';

  var pinMap = {pin_data_json};
  var detail = document.getElementById('pinDetail');
  var overlay = document.getElementById('pinOverlay');
  var title = document.getElementById('pinDetailTitle');
  var content = document.getElementById('pinDetailContent');
  var closeBtn = document.getElementById('pinDetailClose');

  function showDetail(pinNum){{
    var row = pinMap[pinNum];
    if(!row) return;
    title.textContent = '引脚 #' + pinNum + ' 详情';
    var html = '';
    html += '<div class="pin-detail-label">名称</div><div class="pin-detail-value">' + (row['引脚名称']||'-') + '</div>';
    html += '<div class="pin-detail-label">类型</div><div class="pin-detail-value">' + (row['类型']||'-') + '</div>';
    html += '<div class="pin-detail-label">功能</div><div class="pin-detail-value">' + (row['功能']||'-') + '</div>';
    var desc = row['描述'] || '-';
    if(desc && desc !== '-') html += '<div class="pin-detail-label">描述</div><div class="pin-detail-value">' + desc + '</div>';
    content.innerHTML = html;
    detail.classList.add('active');
    overlay.classList.add('active');
  }}

  function hideDetail(){{
    detail.classList.remove('active');
    overlay.classList.remove('active');
  }}

  closeBtn.addEventListener('click', hideDetail);
  overlay.addEventListener('click', hideDetail);

  window.addEventListener('pinClick', function(e){{
    var num = String(e.detail.number);
    showDetail(num);
  }});
}})();
</script>
</body>
</html>"""
    components.html(svg_page, height=620, scrolling=False)


def _render_citations(citations: List[Dict[str, Any]], documents: List[Dict[str, Any]]):
    """Render citation source cards showing document name, page number, and screenshot links."""
    if not citations:
        return

    seen_citations = set()
    unique_citations = []
    for c in citations:
        fn = c.get("file", c.get("document", "未知文档"))
        pg = c.get("page", 1)
        dedup_key = f"{fn}_{pg}"
        if dedup_key not in seen_citations:
            seen_citations.add(dedup_key)
            unique_citations.append(c)

    citation_items = []
    for citation in unique_citations:
        file_name = citation.get("file", citation.get("document", "未知文档"))
        page = citation.get("page", 1)

        document_id = get_document_id_by_filename(file_name, documents) if documents else None
        image_url = get_page_image_url(document_id, page) if document_id else None

        source_text = f"📄 {html.escape(str(file_name))} · 第 {html.escape(str(page))} 页"
        if image_url:
            citation_items.append(f'<div style="font-size:1rem;color:#1e293b;margin-bottom:0.5rem;">{source_text} · <a href="{image_url}" target="_blank">查看截图</a></div>')
        else:
            citation_items.append(f'<div style="font-size:1rem;color:#1e293b;margin-bottom:0.5rem;">{source_text}</div>')

    st.markdown(
        f'''
<div class="pinout-card" style="margin-top:1rem;">
    <div class="pinout-card-title">证据来源</div>
    {"".join(citation_items)}
</div>
''',
        unsafe_allow_html=True,
    )


with st.sidebar:
    render_sidebar_nav()
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="pinout-page-header">'
    '<div class="pinout-page-title">📍 引脚分析</div>'
    '</div>',
    unsafe_allow_html=True,
)

doc_ids = st.session_state.get("selected_doc_ids", [])
_, selected_docs, doc_info = _resolve_doc_context(doc_ids)
st.markdown(f'<div class="pinout-page-meta">{doc_info}</div>', unsafe_allow_html=True)

if not is_api_connected():
    st.warning("⚠️ 后端API未连接，请先启动后端服务。")
    if st.button("🔄 重新连接"):
        check_api_connection()
        st.rerun()
    st.stop()

if not doc_ids:
    st.info("📌 请先在左侧边栏选择文档，再进行引脚分析。")
    if st.button("📄 前往上传文档", type="primary"):
        st.switch_page("pages/1_Documents.py")

else:
    package_context = _build_package_context(st.session_state.get("pinout_chip_name", ""), doc_ids)
    package_options = []
    if st.session_state.get("pinout_package_context") == package_context:
        package_options = st.session_state.get("available_packages", [])

    input_col1, input_col2, input_col3 = st.columns([3.2, 2.4, 1.1])
    with input_col1:
        chip_name_input = st.text_input(
            "芯片型号",
            placeholder="例如：STM32F103、SN74HC04、ATmega328P",
            key="pinout_chip_name",
            label_visibility="collapsed",
        )
    with input_col2:
        if package_options:
            selected_option = st.selectbox(
                "封装类型",
                options=["自动选择"] + package_options,
                key="pinout_package_select",
                label_visibility="collapsed",
            )
            selected_package = selected_option if selected_option != "自动选择" else None
        else:
            package_text = st.text_input(
                "封装类型",
                placeholder="例如：QFP48、BGA、DIP14、SOIC-8",
                key="pinout_package_text",
                label_visibility="collapsed",
            )
            selected_package = package_text.strip() if package_text else None
    with input_col3:
        analyze_button = st.button(
            "🔍 分析",
            type="primary",
            use_container_width=True,
            disabled=not bool(chip_name_input.strip()),
        )

    current_query_context = _build_query_context(chip_name_input, doc_ids, selected_package)
    stored_result = st.session_state.get("pinout_result")
    stored_context = st.session_state.get("pinout_result_context")
    result_is_current = bool(stored_result) and stored_context == current_query_context

    if analyze_button and chip_name_input.strip():
        with st.spinner("正在分析引脚定义，请稍候..."):
            result = analyze_pinout(chip_name_input.strip(), doc_ids, selected_package)
        if result and result.get("success", True) and not result.get("error"):
            st.session_state.pinout_result = result
            st.session_state.pinout_result_context = current_query_context
            st.session_state.pinout_package_context = _build_package_context(chip_name_input, doc_ids)
            st.session_state.available_packages = result.get("available_packages") or []
            st.session_state.show_success = True
            stored_result = result
            stored_context = current_query_context
            result_is_current = True
        else:
            error_message = (result or {}).get("error", "未知错误")
            st.error(f"分析失败：{error_message}")

    if st.session_state.get("show_success", False):
        st.session_state.show_success = False

    if stored_result and not result_is_current and chip_name_input.strip():
        st.warning('当前输入条件已经变化，请重新点击"分析"以生成与当前芯片/封装/文档一致的结果。')

    if result_is_current:
        result = stored_result
        pins = result.get("pins") or []
        svg_content = result.get("svg", "")
        pin_rows = _build_pin_rows(pins)
        citations = result.get("citations") or []

        if not pin_rows and not svg_content:
            st.info("💡 未在文档中找到引脚信息，请尝试其他文档或确认芯片型号。")
        else:
            st.markdown(
                f"""
<div class="pinout-chip-info" style="margin-bottom:1.25rem;">
    <div class="pinout-chip-info-grid" style="text-align:center;">
        <div>
            <div class="pinout-chip-label" style="font-size:1rem;">芯片型号</div>
            <div class="pinout-chip-value" style="font-size:1.5rem;">{html.escape(result.get('chip_name', '未知'))}</div>
        </div>
        <div>
            <div class="pinout-chip-label" style="font-size:1rem;">封装类型</div>
            <div class="pinout-chip-value" style="font-size:1.5rem;">{html.escape(result.get('package', '未知'))}</div>
        </div>
        <div>
            <div class="pinout-chip-label" style="font-size:1rem;">引脚数量</div>
            <div class="pinout-chip-value" style="font-size:1.5rem;">{result.get('pin_count', len(pin_rows))}</div>
        </div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            svg_col, list_col = st.columns([1.3, 1.0], gap="small")
            with svg_col:
                st.markdown('<h3 style="margin:0 0 0.5rem 0;font-size:1.1rem;color:#1e293b;">引脚图</h3>', unsafe_allow_html=True)
                if svg_content:
                    _render_svg(svg_content, pin_rows)
                else:
                    st.info("后端未返回SVG引脚图")

            with list_col:
                st.markdown('<h3 style="margin:0 0 0.5rem 0;font-size:1.1rem;color:#1e293b;">引脚列表</h3>', unsafe_allow_html=True)
                if pin_rows:
                    table_rows = ""
                    for row in pin_rows:
                        table_rows += f'<tr><td>{html.escape(str(row["引脚号"]))}</td><td>{html.escape(str(row["引脚名称"]))}</td><td>{html.escape(str(row["类型"]))}</td><td>{html.escape(str(row["功能"]))}</td><td>{html.escape(str(", ".join(str(item) for item in row["alternate_raw"]) if row["alternate_raw"] else "-"))}</td></tr>'

                    st.markdown(
                        f'''<div class="pin-table-scroll">
<table class="pinout-table"><thead><tr><th>引脚号</th><th>名称</th><th>类型</th><th>功能</th><th>复用功能</th></tr></thead><tbody>{table_rows}</tbody></table>
</div>''',
                        unsafe_allow_html=True,
                    )

            if citations:
                _render_citations(citations, selected_docs)
