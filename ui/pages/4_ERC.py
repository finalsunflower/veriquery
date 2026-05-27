"""
ERC Check Page — EC-VeriQuery

Provides an interactive interface for Electrical Rule Compatibility (ERC) checking
between driver and receiver chips. Users input chip model names, the backend
performs a four-layer ERC analysis via RAG, and results are rendered as metric
bars, parameter comparison tables, and rule cards.

Four-layer ERC architecture:
    Layer 1 - Static stability: voltage level, noise margin, drive capability
    Layer 2 - Signal integrity: critical length, reflection coefficient
    Layer 3 - Topology arbitration: interface protocol, port attribute conflicts
    Layer 4 - Environmental degradation: temperature drift, process degradation

Key features:
    - Driver/receiver chip input with validation
    - Four-layer ERC result visualization with metric bars and rule cards
    - Core electrical parameter comparison table (VOH/VOL/VIH/VIL/IOH/IOL/IIH/IIL)
    - Signal integrity overview with critical length ratio progress bar
    - Topology overview with interface type and supply voltage grid
    - Environmental degradation overview with temperature and lifetime metrics
    - Result caching via session_state to survive page re-renders

Dependencies:
    - theme.py: Academic-style CSS variables and empty-state component
    - sidebar_nav.py: Sidebar navigation and document selector
    - api_client.py: HTTP REST API client for ERC check endpoint
"""

import streamlit as st
import sys
import os
from html import escape
from typing import Any, Dict, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.set_page_config(
    page_title="ERC检查 - EC-VeriQuery",
    page_icon="⚡",
    layout="wide",
)

try:
    from theme import apply_academic_theme
    from sidebar_nav import render_sidebar_nav
    from api_client import erc_check, is_api_connected, check_api_connection, get_documents
except ImportError as e:
    st.error(f"组件导入失败: {e}")
    st.stop()

apply_academic_theme()
check_api_connection()

with st.sidebar:
    render_sidebar_nav()

doc_ids = st.session_state.get("selected_doc_ids", [])
if doc_ids:
    try:
        documents = get_documents()
        ready_docs = [d for d in documents if d.get("status") == "ready"]
        doc_names = ", ".join([
            d.get("filename", "未知") for d in ready_docs
            if d.get("document_id", d.get("id", "")) in doc_ids
        ])
        doc_info = f"当前文档: {doc_names}"
    except Exception:
        doc_info = f"当前文档: {len(doc_ids)} 份文档"
else:
    doc_info = "当前文档: 未选择"

RESULT_KEY = "erc_page_result"
RESULT_CONTEXT_KEY = "erc_page_result_context"

st.markdown("""
<style>
.block-container {
    max-width: 100% !important;
    padding-top: 2.5rem !important;
}
section.main > div { padding-top: 2.5rem !important; }

hr { margin: 0.6rem 0 !important; }
.block-container .stTextInput label { display: none !important; }
.stCheckbox   label { font-size: 0.9rem !important; color: #334155 !important; line-height: 1.5 !important; }
.stCheckbox         { margin-bottom: 0.2rem !important; }
.block-container .stAlert { padding: 0.5rem 0.9rem !important; font-size: 0.85rem !important; }
.block-container .stAlert p { margin: 0 !important; }

.block-container .stButton > button {
    height: 2.8rem !important;
    min-height: 2.8rem !important;
}
.block-container .stButton {
    height: 2.8rem !important;
    min-height: 2.8rem !important;
}
.block-container .stAlert {
    height: 2.8rem !important;
    display: flex !important;
    align-items: center !important;
}

.stTabs [data-baseweb="tab"] {
    font-size: 0.88rem !important;
    padding: 0.65rem 1.3rem !important;
    font-weight: 500 !important;
    min-height: 2.8rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1.3 !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 0 !important; }
.stTabs [data-baseweb="tab-list"] {
    padding: 0.3rem 0.5rem !important;
    min-height: 3.2rem !important;
}

.stTextInput > div > div > input {
    height: 2.8rem !important;
    font-size: 0.92rem !important;
    padding: 0 0.75rem !important;
}

.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.6rem;
    margin-bottom: 0.8rem;
    border-bottom: 2px solid #2563eb;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.ph-left  { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
.ph-title {
    margin: 0 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    border: none !important;
    padding: 0 !important;
    line-height: 1.3 !important;
}
.ph-doc { font-size: 0.8rem; color: #94a3b8; }

.cfg-label {
    font-size: 0.74rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 0.3rem;
    line-height: 1;
}

.metric-bar {
    display: flex;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    background: #fff;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.mbi {
    flex: 1;
    padding: 0.9rem 1rem;
    text-align: center;
    border-right: 1px solid #e2e8f0;
    min-width: 0;
}
.mbi:last-child { border-right: none; }
.mbi-lbl {
    font-size: 0.74rem;
    color: #94a3b8;
    margin-bottom: 0.18rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.mbi-val {
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.15;
}
.mbi-sub { font-size: 0.68rem; color: #cbd5e1; margin-top: 0.08rem; }

.erc-card {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.3rem;
    background: #fff;
}
.ec-err  { border-left: 4px solid #dc2626; background: #fef2f2; }
.ec-warn { border-left: 4px solid #d97706; background: #fffbeb; }
.ec-ok   { border-left: 4px solid #059669; background: #f0fdf4; }
.ec-t    { font-size: 0.92rem; font-weight: 600; color: #1e293b; line-height: 1.4; }
.ec-d    { font-size: 0.83rem; color: #64748b; margin-top: 0.2rem; line-height: 1.5; }

@media (max-width: 1200px) {
    .metric-bar {
        flex-wrap: wrap;
    }
    .mbi {
        min-width: 50%;
        border-right: none;
        border-bottom: 1px solid #e2e8f0;
    }
    .mbi:nth-last-child(-n+2) {
        border-bottom: none;
    }
}

.layer-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #334155;
    margin-top: 0.8rem;
    margin-bottom: 0.3rem;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 0.15rem;
}

.suggestion-list {
    margin-top: 0.5rem;
    padding-left: 1.2rem;
    font-size: 0.88rem;
    color: #475569;
    line-height: 1.7;
}
.suggestion-list li { margin-bottom: 0.15rem; }

.source-badge {
    display: inline-block;
    font-size: 0.68rem;
    padding: 0.08rem 0.4rem;
    border-radius: 4px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.sb-table  { background: #dbeafe; color: #1d4ed8; }
.sb-regex  { background: #fef3c7; color: #92400e; }
.sb-llm    { background: #ede9fe; color: #6d28d9; }
.sb-kg     { background: #d1fae5; color: #065f46; }
.sb-default{ background: #f1f5f9; color: #64748b; }

.erc-param-table { width: 100%; border-collapse: collapse; font-size: 1.3rem; margin-top: 0.5rem; }
.erc-param-table th { background: #f1f5f9; font-size: 1.2rem; font-weight: 700; text-align: center; padding: 0.8rem 0.5rem; border: 1px solid #e2e8f0; }
.erc-param-table td { text-align: center; padding: 0.7rem 0.5rem; border: 1px solid #e2e8f0; font-size: 1.15rem; }
.erc-param-table tr:nth-child(even) { background: #fafbfc; }
.erc-param-table tr:hover { background: #f0f9ff; }

.l1-metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.35rem;
    margin-bottom: 0.5rem;
}
.l1-metric-item {
    background: #f8fafc;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    border-left: 3px solid #3b82f6;
}
.l1-metric-label {
    font-size: 0.7rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
.l1-metric-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
}
</style>
""", unsafe_allow_html=True)


def _format_value(value: Any) -> str:
    """Format a parameter value for display, handling dict/float/str/list types.

    Dict values with {value, unit, source} are formatted with unit conversion
    and adaptive precision. Pure numerics are formatted with trailing-zero removal.
    """
    if value is None or value == "":
        return "N/A"

    if isinstance(value, dict):
        val = value.get("value")
        unit = value.get("unit", "")
        source = value.get("source")

        if val is None:
            return "N/A"

        if isinstance(val, dict):
            val = val.get("value")

        if isinstance(val, (int, float)):
            display_val = float(val)
            display_unit = unit

            if unit == "mA" and abs(display_val) < 0.1:
                display_val = display_val * 1000
                display_unit = "µA"
            elif unit == "µA" and abs(display_val) >= 1000:
                display_val = display_val / 1000
                display_unit = "mA"
            elif unit == "mA" and abs(display_val) >= 1000:
                display_val = display_val / 1000
                display_unit = "A"

            if abs(display_val) >= 100:
                formatted = f"{display_val:.1f}"
            elif abs(display_val) >= 10:
                formatted = f"{display_val:.2f}"
            elif abs(display_val) >= 1:
                formatted = f"{display_val:.3f}"
            else:
                formatted = f"{display_val:.4f}"

            return f"{formatted} {display_unit}"

        return str(val)

    if isinstance(value, (int, float)):
        return f"{float(value):.3f}".rstrip("0").rstrip(".")

    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)

    return str(value)


def _to_float(value: Any) -> Optional[float]:
    """Convert a parameter value (dict/float/str/None) to float.

    Strips common unit suffixes (V, A, cm, %, ns, ohm, Ω) from strings
    before conversion. Returns None for missing or unconvertible values.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        val = value.get("value")
        if isinstance(val, (int, float)):
            return float(val)
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = (
            value.strip()
            .replace("V", "")
            .replace("A", "")
            .replace("cm", "")
            .replace("%", "")
            .replace("ns", "")
            .replace("ohm", "")
            .replace("Ω", "")
        )
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _get_rule_actual_values(layer_result: Dict[str, Any], rule_id: str) -> Dict[str, Any]:
    """Extract actual_values for a specific rule from a layer's ERC result."""
    for rule in layer_result.get("results", []):
        if rule.get("rule_id") == rule_id:
            return rule.get("actual_values", {}) or {}
    return {}


def _get_param_unit(params: Dict[str, Any], param_name: str) -> str:
    """Get the unit string for a parameter, checking both flat and nested formats."""
    unit = params.get(f"{param_name}_unit", "")
    if not unit:
        val = params.get(param_name)
        if isinstance(val, dict):
            unit = val.get("unit", "")
    return unit or ""


def _current_to_ma(value: float, unit: str) -> float:
    """Convert a current value to milliamps (mA) from µA or A."""
    u = (unit or "").lower().strip().replace("μ", "µ")
    if u in ("µa", "ua"):
        return value / 1000.0
    elif u == "a":
        return value * 1000.0
    return value


def _render_metric_bar(items: List[Dict[str, str]]) -> None:
    """Render a horizontal metric bar displaying multiple key indicators."""
    parts = []
    for item in items:
        parts.append(
            f'<div class="mbi">'
            f'<div class="mbi-lbl">{escape(item["label"])}</div>'
            f'<div class="mbi-val" style="color:{item.get("color", "#1e293b")};">{escape(item["value"])}</div>'
            f'<div class="mbi-sub">{escape(item.get("sub", ""))}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="metric-bar">{"".join(parts)}</div>', unsafe_allow_html=True)


def _render_rule_cards(rules: List[Dict[str, Any]], card_margin: str = "0.5rem") -> None:
    """Render ERC rule result cards with color-coded left borders.

    Green (ec-ok) for passed, yellow (ec-warn) for warnings, red (ec-err) for errors.
    """
    if not rules:
        st.info("当前层未返回详细规则。")
        return
    for rule in rules:
        passed = rule.get("passed", False)
        severity = str(rule.get("severity", "info")).lower()
        icon = "✅" if passed else ("⚠️" if severity == "warning" else "❌")
        card_class = "ec-ok" if passed else ("ec-warn" if severity == "warning" else "ec-err")
        title = escape(str(rule.get("rule_name", "未命名规则")))
        message = escape(str(rule.get("message", "")))
        st.markdown(
            f'<div class="erc-card {card_class}" style="margin-bottom: {card_margin};"><div class="ec-t">{icon} {title}</div><div class="ec-d">{message}</div></div>',
            unsafe_allow_html=True,
        )


def _render_list_suggestions(suggestions: List[Any], title: str = "建议") -> None:
    """Render a numbered list of improvement suggestions."""
    if not suggestions:
        return
    items_html = "".join(f"<li>{escape(str(s))}</li>" for s in suggestions)
    st.markdown(
        f'<div style="margin-top:0.8rem;"><p class="layer-title" style="margin-bottom:0.3rem;">{escape(title)}</p>'
        f'<ol class="suggestion-list">{items_html}</ol></div>',
        unsafe_allow_html=True,
    )


def _get_source_badge(source: str) -> str:
    """Return an HTML badge for a parameter data source (table/regex/llm/knowledge_graph)."""
    s = (source or "").lower().strip()
    if s == "table":
        return '<span class="source-badge sb-table">表格</span>'
    elif s == "regex":
        return '<span class="source-badge sb-regex">正则</span>'
    elif s == "llm":
        return '<span class="source-badge sb-llm">LLM</span>'
    elif s in ("knowledge_graph", "kg"):
        return '<span class="source-badge sb-kg">知识图谱</span>'
    return f'<span class="source-badge sb-default">{escape(source)}</span>'


def _render_unified_parameter_table(
    driver_params: Dict[str, Any],
    receiver_params: Dict[str, Any],
    driver_sources: Dict[str, str],
    receiver_sources: Dict[str, str],
) -> None:
    """Render the core electrical parameter comparison table.

    Compares 8 key parameters (VOH, VOL, VIH, VIL, IOH, IOL, IIH, IIL) between
    driver and receiver, with compatibility status based on noise margin and
    drive capability calculations.
    """
    core_params = [
        ("VOH", "输出高电平", "V"),
        ("VOL", "输出低电平", "V"),
        ("VIH", "输入高电平阈值", "V"),
        ("VIL", "输入低电平阈值", "V"),
        ("IOH", "输出高电流", "mA"),
        ("IOL", "输出低电流", "mA"),
        ("IIH", "输入高电流", "µA"),
        ("IIL", "输入低电流", "µA"),
    ]

    voh = _to_float(driver_params.get("VOH"))
    vih = _to_float(receiver_params.get("VIH"))
    vol = _to_float(driver_params.get("VOL"))
    vil = _to_float(receiver_params.get("VIL"))

    ioh = _to_float(driver_params.get("IOH"))
    iih = _to_float(receiver_params.get("IIH"))
    iol = _to_float(driver_params.get("IOL"))
    iil = _to_float(receiver_params.get("IIL"))

    ioh_ma = _current_to_ma(ioh, _get_param_unit(driver_params, "IOH")) if ioh is not None else None
    iih_ma = _current_to_ma(iih, _get_param_unit(receiver_params, "IIH")) if iih is not None else None
    iol_ma = _current_to_ma(iol, _get_param_unit(driver_params, "IOL")) if iol is not None else None
    iil_ma = _current_to_ma(iil, _get_param_unit(receiver_params, "IIL")) if iil is not None else None

    rows = []
    for param, name, unit in core_params:
        driver_val = driver_params.get(param)
        receiver_val = receiver_params.get(param)
        driver_src = driver_sources.get(param, "")
        receiver_src = receiver_sources.get(param, "")

        driver_display = _format_value(driver_val)
        receiver_display = _format_value(receiver_val)

        driver_badge = _get_source_badge(driver_src) if driver_src else "-"
        receiver_badge = _get_source_badge(receiver_src) if receiver_src else "-"

        status = "-"
        if param == "VOH" and voh is not None and vih is not None:
            margin = voh - vih
            status = f"✅ 通过 ({margin:.3f} V)" if margin >= 0 else f"⚠️ 风险 ({margin:.3f} V)"
        elif param == "VOL" and vol is not None and vil is not None:
            margin = vil - vol
            status = f"✅ 通过 ({margin:.3f} V)" if margin >= 0 else f"⚠️ 风险 ({margin:.3f} V)"
        elif param == "IOH" and ioh_ma is not None and iih_ma is not None:
            margin = abs(ioh_ma) - abs(iih_ma)
            status = f"✅ 通过 ({margin:.4f} mA)" if margin >= 0 else f"⚠️ 风险 ({margin:.4f} mA)"
        elif param == "IOL" and iol_ma is not None and iil_ma is not None:
            margin = abs(iol_ma) - abs(iil_ma)
            status = f"✅ 通过 ({margin:.4f} mA)" if margin >= 0 else f"⚠️ 风险 ({margin:.4f} mA)"
        elif param == "VIH" and voh is not None and vih is not None:
            margin = voh - vih
            status = f"参考: VOH余量 {margin:.3f} V"
        elif param == "VIL" and vol is not None and vil is not None:
            margin = vil - vol
            status = f"参考: VOL余量 {margin:.3f} V"
        elif param == "IIH" and ioh_ma is not None and iih_ma is not None:
            margin = abs(ioh_ma) - abs(iih_ma)
            status = f"参考: IOH余量 {margin:.4f} mA"
        elif param == "IIL" and iol_ma is not None and iil_ma is not None:
            margin = abs(iol_ma) - abs(iil_ma)
            status = f"参考: IOL余量 {margin:.4f} mA"

        rows.append({
            "参数": f"{param} ({name})",
            "驱动端值": driver_display,
            "驱动端来源": driver_badge,
            "接收端值": receiver_display,
            "接收端来源": receiver_badge,
            "兼容性判定": status,
        })

    html = """
    <style>
    .erc-param-table { width: 100%; border-collapse: collapse; font-size: 1.3rem; margin-top: 0.5rem; }
    .erc-param-table th { background: #f1f5f9; font-size: 1.2rem; font-weight: 700; text-align: center; padding: 0.8rem 0.5rem; border: 1px solid #e2e8f0; }
    .erc-param-table td { text-align: center; padding: 0.7rem 0.5rem; border: 1px solid #e2e8f0; font-size: 1.15rem; }
    .erc-param-table tr:nth-child(even) { background: #fafbfc; }
    .erc-param-table tr:hover { background: #f0f9ff; }
    </style>
    <table class="erc-param-table">
        <thead><tr>
            <th>参数</th><th>驱动端值</th><th>驱动端来源</th><th>接收端值</th><th>接收端来源</th><th>兼容性判定</th>
        </tr></thead>
        <tbody>
    """
    for r in rows:
        html += f"<tr><td>{r['参数']}</td><td>{r['驱动端值']}</td><td>{r['驱动端来源']}</td><td>{r['接收端值']}</td><td>{r['接收端来源']}</td><td>{r['兼容性判定']}</td></tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


def _render_layer_2_overview(layer_result: Dict[str, Any]) -> None:
    """Render Layer 2 signal integrity overview with critical length, trace length, and reflection coefficient."""
    critical_values = _get_rule_actual_values(layer_result, "ERC-L2-T001")
    reflection_values = _get_rule_actual_values(layer_result, "ERC-L2-R001")

    if not critical_values and not reflection_values:
        st.caption("该层未提供结构化走线/反射参数，仅展示规则结果。")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("临界长度", f"{critical_values.get('critical_length', 'N/A'):.2f} cm" if isinstance(critical_values.get("critical_length"), (int, float)) else "N/A")
    with c2:
        trace_length = critical_values.get("trace_length")
        st.metric("实际走线长度", f"{trace_length:.2f} cm" if isinstance(trace_length, (int, float)) else "未提供")
    with c3:
        reflection = reflection_values.get("reflection_coefficient")
        st.metric("反射系数", f"{reflection:.3f}" if isinstance(reflection, (int, float)) else "未提供")

    trace_length = critical_values.get("trace_length")
    critical_length = critical_values.get("critical_length")
    if isinstance(trace_length, (int, float)) and isinstance(critical_length, (int, float)) and critical_length > 0:
        ratio = min(trace_length / critical_length, 1.0)
        st.progress(ratio, text=f"走线长度占临界长度比例: {trace_length:.2f} / {critical_length:.2f} cm")


def _render_layer_3_overview(layer_result: Dict[str, Any]) -> None:
    """Render Layer 3 topology overview with interface types and supply voltages."""
    interface_values = _get_rule_actual_values(layer_result, "ERC-L3-I001")
    pam_values = _get_rule_actual_values(layer_result, "ERC-L3-P001")

    driver_if = str(interface_values.get("driver_interface", "N/A"))
    receiver_if = str(interface_values.get("receiver_interface", "N/A"))
    dv = pam_values.get("driver_voltage")
    rv = pam_values.get("receiver_voltage")
    driver_v = f"{dv:.2f} V" if isinstance(dv, (int, float)) else "N/A"
    receiver_v = f"{rv:.2f} V" if isinstance(rv, (int, float)) else "N/A"

    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.3rem;font-size:0.8rem;">
  <div style="background:#f8fafc;padding:0.35rem 0.5rem;border-radius:4px;border-left:3px solid #3b82f6;">
    <div style="color:#64748b;font-size:0.7rem;">驱动接口</div>
    <div style="font-weight:600;color:#1e293b;">{driver_if}</div>
  </div>
  <div style="background:#f8fafc;padding:0.35rem 0.5rem;border-radius:4px;border-left:3px solid #3b82f6;">
    <div style="color:#64748b;font-size:0.7rem;">接收接口</div>
    <div style="font-weight:600;color:#1e293b;">{receiver_if}</div>
  </div>
  <div style="background:#f8fafc;padding:0.35rem 0.5rem;border-radius:4px;border-left:3px solid #8b5cf6;">
    <div style="color:#64748b;font-size:0.7rem;">驱动侧电压</div>
    <div style="font-weight:600;color:#1e293b;">{driver_v}</div>
  </div>
  <div style="background:#f8fafc;padding:0.35rem 0.5rem;border-radius:4px;border-left:3px solid #8b5cf6;">
    <div style="color:#64748b;font-size:0.7rem;">接收侧电压</div>
    <div style="font-weight:600;color:#1e293b;">{receiver_v}</div>
  </div>
</div>
""", unsafe_allow_html=True)


def _render_layer_4_overview(layer_result: Dict[str, Any]) -> None:
    """Render Layer 4 environmental degradation overview with temperature, degradation, and lifetime metrics."""
    thermal_values = _get_rule_actual_values(layer_result, "ERC-L4-T001")
    process_values = _get_rule_actual_values(layer_result, "ERC-L4-P001")

    max_temp = thermal_values.get("max_safe_temperature")
    deg_factor = process_values.get("degradation_factor")
    risk_temps = thermal_values.get("risk_temperatures", [])
    lifetime = process_values.get("estimated_lifetime")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("最高安全温度", f"{max_temp:.0f} °C" if isinstance(max_temp, (int, float)) else "N/A")
    with c2:
        st.metric("退化因子", f"{deg_factor:.3f}" if isinstance(deg_factor, (int, float)) else "N/A")
    with c3:
        if isinstance(risk_temps, list) and risk_temps:
            st.metric("风险温度点", f"{len(risk_temps)} 个")
        else:
            st.metric("风险温度点", "无")
    with c4:
        if isinstance(lifetime, (int, float)):
            if lifetime >= 8760:
                st.metric("估计寿命", f"{lifetime / 8760:.1f} 年")
            else:
                st.metric("估计寿命", f"{lifetime:.0f} h")
        else:
            st.metric("估计寿命", "N/A")


def _render_layer1_metrics(erc_result: Dict[str, Any]) -> None:
    """Render Layer 1 key metrics as a 2x2 grid (noise margins and drive margins).

    The backend ERC engine returns voltage/current values per rule:
      ERC-L1-V001 → {VOH, VIH}       (high-level compatibility)
      ERC-L1-V002 → {VOL, VIL}       (low-level compatibility)
      ERC-L1-N001 → {RNI, high_margin, low_margin}  (noise immunity)
      ERC-L1-I001 → {IOH, IIH}       (high drive capability)
      ERC-L1-I002 → {IOL, IIL}       (low drive capability)

    Derived metrics computed here:
      NMH = VOH - VIH  (high noise margin)
      NML = VIL - VOL  (low noise margin)
      drive_margin_high = |IOH| - |IIH|
      drive_margin_low  = IOL - |IIL|
    """
    layer1 = erc_result.get("layer1_result", {}) or {}
    all_results = layer1.get("results", []) or []

    rule_values = {}
    for rule in all_results:
        av = rule.get("actual_values") or {}
        rule_values.update(av)

    nmh = rule_values.get("high_margin")
    nml = rule_values.get("low_margin")
    drive_high = None
    drive_low = None

    if nmh is None:
        voh = rule_values.get("VOH")
        vih = rule_values.get("VIH")
        if isinstance(voh, (int, float)) and isinstance(vih, (int, float)):
            nmh = voh - vih

    if nml is None:
        vil = rule_values.get("VIL")
        vol = rule_values.get("VOL")
        if isinstance(vil, (int, float)) and isinstance(vol, (int, float)):
            nml = vil - vol

    ioh = rule_values.get("IOH")
    iih = rule_values.get("IIH")
    if isinstance(ioh, (int, float)) and isinstance(iih, (int, float)):
        drive_high = abs(ioh) - abs(iih)

    iol = rule_values.get("IOL")
    iil = rule_values.get("IIL")
    if isinstance(iol, (int, float)) and isinstance(iil, (int, float)):
        drive_low = iol - abs(iil)

    items = []
    if isinstance(nmh, (int, float)):
        items.append(("高电平噪声容限 NMH", f"{nmh:.3f} V"))
    if isinstance(nml, (int, float)):
        items.append(("低电平噪声容限 NML", f"{nml:.3f} V"))
    if isinstance(drive_high, (int, float)):
        drive_high_ma = drive_high * 1000
        items.append(("高电平驱动余量", f"{drive_high_ma:.4f} mA"))
    if isinstance(drive_low, (int, float)):
        drive_low_ma = drive_low * 1000
        items.append(("低电平驱动余量", f"{drive_low_ma:.4f} mA"))

    if not items:
        return

    grid_html = '<div class="l1-metric-grid">'
    for label, value in items:
        grid_html += f'<div class="l1-metric-item"><div class="l1-metric-label">{label}</div><div class="l1-metric-value">{value}</div></div>'
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)


def _render_chip_result(result: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Render the full ERC check result with overview tab and layer detail tab."""
    erc_result = result.get("erc_result", {}) or {}
    driver_params = result.get("driver_parameters", {}) or {}
    receiver_params = result.get("receiver_parameters", {}) or {}
    overall_compatible = result.get("overall_compatible", False)
    overall_confidence = float(result.get("overall_confidence", 0.0) or 0.0)
    processing_time = float(result.get("processing_time", 0.0) or 0.0)
    errors = result.get("errors", []) or []
    warnings = result.get("warnings", []) or []
    info_list = result.get("info", []) or []
    total_checks = len(errors) + len(warnings) + len(info_list)
    status_color = "#059669" if overall_compatible else "#dc2626"
    status_text = "✅ 兼容" if overall_compatible else "❌ 不兼容"

    driver_sources = result.get("driver_data_sources", {}) or {}
    receiver_sources = result.get("receiver_data_sources", {}) or {}

    tab1, tab2 = st.tabs(["📊 总览", "🔍 层级详情"])

    with tab1:
        _render_metric_bar(
            [
                {"label": "兼容性", "value": status_text, "color": status_color},
                {"label": "通过项", "value": f"{len(info_list)}/{total_checks or 0}", "color": "#059669"},
                {"label": "问题", "value": f"{len(errors)}错 {len(warnings)}警", "color": "#d97706" if warnings and not errors else status_color},
                {"label": "处理时间", "value": f"{processing_time:.2f} s", "color": "#64748b"},
                {"label": "置信度", "value": f"{overall_confidence:.0%}", "color": "#64748b"},
            ]
        )

        if driver_params or receiver_params:
            st.markdown('<p class="layer-title" style="margin-top:1rem;">核心电气参数</p>', unsafe_allow_html=True)
            _render_unified_parameter_table(driver_params, receiver_params, driver_sources, receiver_sources)

        suggestions = result.get("suggestions", []) or []
        _render_list_suggestions(suggestions)

    with tab2:
        layer1_result = erc_result.get("layer1_result", {}) or {}
        layer2_result = erc_result.get("layer2_result", {}) or {}
        layer3_result = erc_result.get("layer3_result", {}) or {}
        layer4_result = erc_result.get("layer4_result", {}) or {}

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown('<div id="erc-left-column" style="padding-top: 0.8rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.15rem; font-weight: 700; color: #334155; margin-bottom: 0.4rem; border-bottom: 2px solid #3b82f6; padding-bottom: 0.2rem;">Layer 1 静态稳定性</div>', unsafe_allow_html=True)
            passed = layer1_result.get("passed", False)
            severity = str(layer1_result.get("severity", "info")).lower()
            status_text = '✅ 通过' if passed else ('⚠️ 警告' if severity == 'warning' else '❌ 失败')
            st.markdown(f'<div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">状态: {status_text}</div>', unsafe_allow_html=True)
            _render_layer1_metrics(erc_result)
            _render_rule_cards(layer1_result.get("results", []), card_margin="0.3rem")
            st.markdown('</div>', unsafe_allow_html=True)

        with right_col:
            st.markdown('<div id="erc-right-column" style="margin-top: -1rem;">', unsafe_allow_html=True)

            st.markdown('<div class="erc-right-section" style="margin-bottom: 0.4rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.15rem; font-weight: 700; color: #334155; margin-bottom: 0.4rem; border-bottom: 2px solid #3b82f6; padding-bottom: 0.2rem;">Layer 2 信号完整性</div>', unsafe_allow_html=True)
            passed = layer2_result.get("passed", False)
            severity = str(layer2_result.get("severity", "info")).lower()
            status_text = '✅ 通过' if passed else ('⚠️ 警告' if severity == 'warning' else '❌ 失败')
            st.markdown(f'<div style="font-size: 0.7rem; color: #64748b; margin-bottom: 0.08rem;">状态: {status_text}</div>', unsafe_allow_html=True)
            _render_rule_cards(layer2_result.get("results", []), card_margin="0.1rem")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="erc-right-section" style="margin-bottom: 0.4rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.15rem; font-weight: 700; color: #334155; margin-bottom: 0.1rem; border-bottom: 2px solid #3b82f6; padding-bottom: 0.08rem;">Layer 3 拓扑与接口</div>', unsafe_allow_html=True)
            passed = layer3_result.get("passed", False)
            severity = str(layer3_result.get("severity", "info")).lower()
            status_text = '✅ 通过' if passed else ('⚠️ 警告' if severity == 'warning' else '❌ 失败')
            st.markdown(f'<div style="font-size: 0.7rem; color: #64748b; margin-bottom: 0.08rem;">状态: {status_text}</div>', unsafe_allow_html=True)
            _render_rule_cards(layer3_result.get("results", []), card_margin="0.1rem")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="erc-right-section" style="margin-bottom: 0.4rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.15rem; font-weight: 700; color: #334155; margin-bottom: 0.1rem; border-bottom: 2px solid #3b82f6; padding-bottom: 0.08rem;">Layer 4 环境退化</div>', unsafe_allow_html=True)
            passed = layer4_result.get("passed", False)
            severity = str(layer4_result.get("severity", "info")).lower()
            status_text = '✅ 通过' if passed else ('⚠️ 警告' if severity == 'warning' else '❌ 失败')
            st.markdown(f'<div style="font-size: 0.7rem; color: #64748b; margin-bottom: 0.08rem;">状态: {status_text}</div>', unsafe_allow_html=True)
            _render_rule_cards(layer4_result.get("results", []), card_margin="0.1rem")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('''
<style>
[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
}
#erc-right-column {
    margin-top: -1rem !important;
    padding-top: 0 !important;
}
#erc-right-column > div:first-child,
#erc-right-column > .erc-right-section:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
.erc-right-section {
    margin-bottom: 0.3rem !important;
}
[data-testid="stHorizontalBlock"] > div:nth-child(2) {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
.erc-right-section .ec-t {
    font-size: 0.82rem !important;
}
.erc-right-section .ec-d {
    font-size: 0.75rem !important;
}
.erc-right-section .erc-card {
    padding: 0.45rem 0.65rem !important;
}
</style>
''', unsafe_allow_html=True)


def _render_result_area(result: Optional[Dict[str, Any]], context: Dict[str, Any]) -> None:
    """Render the result area entry point — validates result before delegating to _render_chip_result."""
    if not result:
        return

    if not result.get("success"):
        message = result.get("message") or result.get("error") or "检查失败"
        st.error(message)
        return

    _render_chip_result(result, context)


st.markdown(
    f'<div class="page-header">'
    f'<div class="ph-left">'
    f'<span style="font-size:1.4rem;line-height:1;">⚡</span>'
    f'<h1 class="ph-title">ERC检查</h1>'
    f'</div>'
    f'<span class="ph-doc">{doc_info}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

api_connected = is_api_connected()

if not api_connected:
    st.warning("⚠️ 后端API服务未连接，请先启动后端服务")
    cmd = "cd d:\\veriquery && python -m api.main"
    st.code(cmd, language="bash")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("🔄 重新连接", use_container_width=True):
            for k in ["api_checked", "api_check_time", "api_connected", "api_check_ts"]:
                st.session_state.pop(k, None)
            check_api_connection(force_refresh=True)
            st.rerun()
    with c2:
        st.info("💡 如果后端服务已启动，点击「重新连接」刷新状态。")

elif not doc_ids:
    st.warning("⚠️ 请先在侧边栏选择文档")
    if st.button("📄 前往上传文档", type="primary"):
        st.switch_page("pages/1_Documents.py")

else:
    control_row1 = st.columns([1.7, 1.7, 1.2])
    with control_row1[0]:
        st.markdown('<div class="cfg-label">驱动端芯片</div>', unsafe_allow_html=True)
        driver_chip = st.text_input(
            "驱动端芯片",
            placeholder="例如: 74HC04",
            key="driver_chip",
            label_visibility="collapsed",
        )
    with control_row1[1]:
        st.markdown('<div class="cfg-label">接收端芯片</div>', unsafe_allow_html=True)
        receiver_chip = st.text_input(
            "接收端芯片",
            placeholder="例如: SN74HCT04",
            key="receiver_chip",
            label_visibility="collapsed",
        )
    with control_row1[2]:
        st.markdown('<div class="cfg-label">执行</div>', unsafe_allow_html=True)
        ready = bool(driver_chip and receiver_chip)
        check_button = st.button("🔍 开始检查", type="primary", use_container_width=True, disabled=not ready)

    if check_button and driver_chip and receiver_chip:
        status = st.empty()
        try:
            status.info("📊 正在执行芯片间ERC检查，请稍候...")

            result = erc_check(
                driver_chip=driver_chip,
                receiver_chip=receiver_chip,
                document_ids=doc_ids,
            )
            status.success("✅ 检查完成")
            st.session_state[RESULT_KEY] = result
            st.session_state[RESULT_CONTEXT_KEY] = {
                "driver_chip": driver_chip,
                "receiver_chip": receiver_chip,
                "document_ids": list(doc_ids),
            }
        except Exception as ex:
            st.session_state[RESULT_KEY] = {"success": False, "message": f"执行检查时发生错误: {ex}"}
            st.session_state[RESULT_CONTEXT_KEY] = {
                "driver_chip": driver_chip,
                "receiver_chip": receiver_chip,
                "document_ids": list(doc_ids),
            }
        finally:
            status.empty()

    latest_result = st.session_state.get(RESULT_KEY)
    latest_context = st.session_state.get(RESULT_CONTEXT_KEY, {})

    if latest_result:
        _render_result_area(latest_result, latest_context)
