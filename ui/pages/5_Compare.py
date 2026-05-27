"""
Device Parameter Comparison Page — EC-VeriQuery

Provides an interactive interface for multi-device parameter comparison with
a three-layer scoring architecture. Users select two or more chip documents,
the backend performs parameter extraction and scoring via RAG, and results
are rendered as metric cards, device score cards, a difference parameter
matrix, and interactive Plotly charts.

Three-layer scoring architecture:
    Layer 1 - CCM (Context-Condition Mapping): standardizes parameters across
              different test conditions (temperature, voltage) using semiconductor
              physics to enable fair cross-device comparison.
    Layer 2 - Z-A-FoM (Z-number Augmented Figure of Merit): fuses data
              reliability into scoring via Z-number Kang conversion, penalizing
              low-reliability data sources (table > regex > LLM > default).
    Layer 3 - B-SPOTIS (Balanced SPOTIS): robust multi-criteria decision making
              with range normalization, MEREC objective weighting, and ESP distance
              computation, producing final device rankings with confidence intervals.

Key features:
    - Device selection with name input and validation (minimum 2 documents)
    - Overview tab: metric cards, device score cards, difference parameter matrix
    - Dimension tab: Layer 1 CCM chart, Layer 2 reliability chart, radar chart,
      Layer 3 B-SPOTIS decision chart
    - Data source badges reflecting Z-number reliability transparency
    - Result caching via session_state to survive page re-renders

Dependencies:
    - theme.py: Academic-style CSS variables and empty-state component
    - sidebar_nav.py: Sidebar navigation and document selector
    - api_client.py: HTTP REST API client for comparison endpoint
"""

import os
import re
import sys
from html import escape
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.set_page_config(page_title="参数对比 - EC-VeriQuery", page_icon="📊", layout="wide")

try:
    from theme import apply_academic_theme
    from sidebar_nav import render_sidebar_nav
    from api_client import (
        check_api_connection,
        compare_devices_enhanced,
        get_documents,
        is_api_connected,
    )
except ImportError as e:
    st.error(f"组件导入失败: {e}")
    st.stop()

apply_academic_theme()
check_api_connection()

with st.sidebar:
    render_sidebar_nav()

P      = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626"]
RANK_C = {"gold": "#16a34a", "silver": "#b45309", "gray": "#64748b"}
RBAR   = [P[0], P[1], "#94a3b8"]

PARAM_LABELS = {
    "frequency": "频率", "propagation_delay": "传播延迟",
    "supply_voltage": "供电电压", "output_current": "输出电流",
    "power_consumption": "功耗", "temperature_range": "温度范围",
    "package_size": "封装", "ttl_compatible": "TTL兼容",
    "input_current": "输入电流", "quiescent_current": "静态电流",
    "input_voltage_high": "输入高电平", "input_voltage_low": "输入低电平",
    "output_voltage_high": "输出高电平", "output_voltage_low": "输出低电平",
    "VIH": "输入高电平阈值", "VIL": "输入低电平阈值",
    "VOH": "输出高电平", "VOL": "输出低电平",
    "IOH": "输出高电流", "IOL": "输出低电流",
    "IIH": "输入高电流", "IIL": "输入低电流",
    "ICC": "静态电源电流", "IDD": "静态漏电流",
    "tPLH": "上升延迟", "tPHL": "下降延迟",
    "tpd": "传播延迟", "fmax": "最大频率",
    "Frequency": "频率", "Temperature": "温度范围",
    "VCC": "供电电压", "VDD": "供电电压",
}

PARAM_TO_SCORING = {
    "VCC": "supply_voltage", "VDD": "supply_voltage",
    "VOH": "output_voltage_high", "VOL": "output_voltage_low",
    "VIH": "input_voltage_high", "VIL": "input_voltage_low",
    "IOH": "output_current", "IOL": "output_current",
    "IIH": "input_current", "IIL": "input_current",
    "ICC": "quiescent_current", "IDD": "quiescent_current",
    "tPLH": "propagation_delay", "tPHL": "propagation_delay",
    "tpd": "propagation_delay",
    "fmax": "frequency", "Frequency": "frequency",
    "Temperature": "temperature_range",
}


def gc(index: int) -> str:
    """Return device color by index with cyclic reuse for >5 devices."""
    return P[index % len(P)]


st.markdown("""
<style>
.block-container {
    max-width: 100% !important;
    padding: 1.5rem 1rem 0.3rem 1rem !important;
}
section.main > div { padding-top: 1.5rem !important; }

.pg-hdr {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 0.45rem 0; margin-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary);
}
.pg-ttl { font-size: 1.2rem; font-weight: 700; color: var(--text-primary); margin: 0; }
.pg-sub { font-size: 0.78rem; color: var(--text-secondary); }

.mini-metric {
    border: 1px solid var(--border); border-radius: 8px;
    background: var(--bg-card); padding: 0.55rem 0.8rem; min-height: 68px;
}
.mini-metric-label { color: var(--text-secondary); font-size: 0.72rem; margin-bottom: 0.15rem; }
.mini-metric-value { color: var(--text-primary); font-size: 1rem; font-weight: 700; line-height: 1.2; }
.mini-metric-note  { color: var(--text-muted); font-size: 0.68rem; margin-top: 0.2rem; }

.stl {
    font-size: 0.95rem; font-weight: 700; color: #334155;
    margin: 0.4rem 0 0.2rem; padding-bottom: 0.2rem;
    border-bottom: 1.5px solid #e2e8f0;
}

.device-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem; }
.device-name { font-size:1.15rem; font-weight:700; color:var(--text-primary); }
.device-score { font-size:1.15rem; font-weight:700; }
.device-reliability {
    display:inline-flex; align-items:center; padding:0.18rem 0.5rem;
    border-radius:999px; font-size:0.85rem; font-weight:600; margin-bottom:0.3rem;
}
.device-reliability.high { background:#dcfce7; color:#166534; }
.device-reliability.mid  { background:#fef3c7; color:#92400e; }
.device-reliability.low  { background:#fee2e2; color:#991b1b; }
.sbar { height:6px; background:#e2e8f0; border-radius:999px; overflow:hidden; margin:0.25rem 0 0.5rem; }
.sbar-f { height:100%; border-radius:999px; }
.tag-pros, .tag-cons {
    display:inline-block; padding:0.15rem 0.5rem; border-radius:999px;
    font-size:0.88rem; font-weight:500; margin:0.2rem 0.25rem 0 0;
}
.tag-pros { background:#dcfce7; color:#166534; }
.tag-cons { background:#fee2e2; color:#991b1b; }

.stTabs [data-baseweb="tab"] {
    font-size:0.84rem !important; padding:0.4rem 0.75rem !important; font-weight:500 !important;
}

.compare-param-table {
    width:100%; border-collapse:collapse; font-size:1.05rem; margin-top:0.35rem;
}
.compare-param-table th {
    background:#f1f5f9; font-size:1.03rem; font-weight:700;
    text-align:center; padding:0.38rem 0.3rem;
    border:1px solid #e2e8f0; color:#1e293b;
}
.compare-param-table td {
    text-align:center; padding:0.32rem 0.3rem;
    border:1px solid #e2e8f0; font-size:1.01rem; color:#334155;
}
.compare-param-table tr:nth-child(even) { background:#fafbfc; }
.compare-param-table tr:hover { background:#f0f9ff; }
.source-badge { font-size:0.92rem; color:#64748b; }

.arch-note {
    padding:0.55rem 0.85rem; background:#f8fafc; border-radius:7px;
    border:1px solid #e2e8f0; font-size:0.78rem; color:#64748b;
    line-height:1.8; margin-top:-0.2rem;
}
.arch-note strong { color:#334155; }

.block-container .stButton > button { border-radius:7px !important; font-weight:500 !important; }
</style>
""", unsafe_allow_html=True)


def extract_name(filename: str) -> str:
    """Extract chip model name from a PDF filename.

    Strips the .pdf suffix and matches against prioritized regex patterns
    for common chip families (TI SN74/54, LM, TPS, OPA, TMS320, ST STM32,
    STM8, TLV, MSP430). Falls back to a generic alphanumeric pattern.
    Returns the uppercase match or the stripped filename if no match is found.
    """
    if not filename:
        return ""
    name = filename.replace(".pdf", "").replace(".PDF", "")
    patterns = [
        r"\b(SN(?:74|54)[A-Z]+\d*)\b",
        r"\b(LM\d+[A-Z]?\d*)\b",
        r"\b(TPS\d+[A-Z]?\d*)\b",
        r"\b(OPA\d+[A-Z]?\d*)\b",
        r"\b(TMS320[A-Z0-9]+)\b",
        r"\b(STM32[A-Z0-9]+)\b",
        r"\b(STM8[A-Z0-9]+)\b",
        r"\b(TLV\d+[A-Z]?\d*)\b",
        r"\b(MSP430[A-Z0-9]+)\b",
        r"\b([A-Z]{1,4}[\d]{2,6}[A-Z]*[\d]*[A-Z]*)\b",
    ]
    for pat in patterns:
        matches = re.findall(pat, name, re.IGNORECASE)
        if matches:
            filtered = [m for m in matches if m not in ["分析","应用","设计","原理","功能","特性","性能"]]
            if filtered:
                return filtered[0].upper()
    return name


def get_docs(ids: List[str]) -> Dict[str, Dict[str, str]]:
    """Fetch document info (filename + extracted device name) by document IDs.

    Uses session_state cache for the document list to avoid redundant API calls.
    Silently degrades on failure, returning an empty dict.
    """
    docs_map: Dict[str, Dict[str, str]] = {}
    try:
        cached = st.session_state.get("cached_documents")
        if cached is None:
            cached = get_documents()
            st.session_state.cached_documents = cached
        fetched = {}
        for doc in cached:
            did = doc.get("document_id", doc.get("id", ""))
            if did:
                fn = doc.get("filename", "")
                fetched[did] = {"filename": fn, "device_name": extract_name(fn)}
        for did in ids:
            if did in fetched:
                docs_map[did] = fetched[did]
    except Exception:
        pass
    return docs_map


def stars(score: float) -> str:
    """Convert a 0-100 score to a 1-5 star display string."""
    return "⭐" * min(5, max(1, round(score / 20)))


def render_metric_tile(label: str, value: str, note: str) -> None:
    """Render a compact metric card with label, value, and note using custom HTML."""
    st.markdown(f"""
    <div class="mini-metric">
      <div class="mini-metric-label">{escape(label)}</div>
      <div class="mini-metric-value">{escape(value)}</div>
      <div class="mini-metric-note">{escape(note)}</div>
    </div>""", unsafe_allow_html=True)


def _get_source_badge(source: str) -> str:
    """Convert a data source identifier to a labeled badge string."""
    return {
        "table":"📄 表格","datasheet_table":"📄 表格","regex":"🔍 正则",
        "llm":"🤖 LLM","llm_inference":"🤖 LLM","knowledge_graph":"🧠 知识图谱",
        "datasheet":"📄 数据手册","retrieval":"📚 检索","default_value":"⚙️ 默认",
    }.get(source, "📦 其他")


def build_comparison_dataframe(matrix: Dict[str, Any]) -> tuple:
    """Convert the backend comparison matrix JSON into a Pandas DataFrame and source data.

    Returns:
        (df, sources_data) tuple:
        - df: DataFrame with columns ["参数", device1, device2, ..., "差异"]
        - sources_data: dict mapping parameter name to per-device source identifiers
    """
    chips      = matrix.get("chips", []) or []
    parameters = matrix.get("parameters", []) or []
    data       = matrix.get("data", {}) or {}
    rows, sources_data = [], {}
    for param in parameters:
        row: Dict[str, str] = {"参数": PARAM_LABELS.get(param, param)}
        param_sources: Dict[str, str] = {}
        rendered = []
        for chip in chips:
            payload = data.get(chip, {}).get(param, {}) or {}
            dv = payload.get("display")
            src = payload.get("source", "")
            if dv:
                display = dv
            else:
                v = payload.get("value") or "N/A"
                u = payload.get("unit") or ""
                display = f"{v} {u}".strip() if v != "N/A" else "N/A"
            row[chip] = display
            param_sources[chip] = src
            rendered.append(display)
        comparable = [v for v in rendered if v not in ("", None)]
        has_na = any(v in ("N/A", "") or v is None for v in rendered)
        row["差异"] = "是" if (len(set(comparable)) > 1) or (has_na and len(comparable) >= 1) else ""
        rows.append(row)
        sources_data[param] = param_sources
    if not rows:
        return pd.DataFrame(), sources_data
    df = pd.DataFrame(rows)
    return df[["参数"] + chips + ["差异"]], sources_data


def get_dimension_dataset(scoring: Dict[str, Any], radar: Dict[str, Any]):
    """Extract dimension dataset from scoring result and radar data.

    Radar data takes priority over scoring dimension_scores when available.

    Returns:
        (names, dims, vals) triple of device names, dimension labels, and score lists.
    """
    if radar and radar.get("dimensions") and radar.get("devices"):
        names = [d.get("name", f"设备{i+1}") for i,d in enumerate(radar["devices"])]
        dims  = list(radar["dimensions"])
        vals  = [[float(v) if v else 0.0 for v in d.get("values",[])] for d in radar["devices"]]
        return names, dims, vals
    devices = scoring.get("devices", []) if scoring else []
    if not devices:
        return [], [], []
    dim_map = {"performance":"性能","power":"功耗","reliability":"可靠性","usability":"易用性"}
    names = [d.get("device_name", f"设备{i+1}") for i,d in enumerate(devices)]
    dims  = list(dim_map.values())
    vals  = [[float(d.get("dimension_scores",{}).get(k,0.0)) for k in dim_map] for d in devices]
    return names, dims, vals


def render_device_card(device: Dict[str, Any], index: int) -> None:
    """Render a device score card with rank, score, reliability, advantages, and concerns."""
    icons  = ["🥇","🥈","🥉"]
    levels = ["gold","silver","gray"]
    name   = device.get("device_name","未知")
    score  = float(device.get("overall_score",0.0))
    rel    = float(device.get("reliability_score",0.0))
    adv    = device.get("advantages",[]) or []
    dis    = device.get("disadvantages",[]) or []
    rk     = levels[index] if index < 3 else "gray"
    ri     = icons[index] if index < 3 else f"#{index+1}"
    bc     = RBAR[index] if index < 3 else "#94a3b8"
    if rel >= 0.85:   rcls,rtxt = "high","高可靠度"
    elif rel >= 0.70: rcls,rtxt = "mid", "中可靠度"
    else:             rcls,rtxt = "low", "低可靠度"

    def tr(t):
        """Replace English parameter names in text with Chinese labels."""
        if not t:
            return t
        result = t
        for en, cn in PARAM_LABELS.items():
            result = result.replace(en, cn)
        return result

    with st.container(border=True):
        st.markdown(f"""
        <div class="device-head">
          <div class="device-name">{escape(ri)} {escape(name)}</div>
          <div class="device-score" style="color:{RANK_C[rk]};">{score:.1f}分 {escape(stars(score))}</div>
        </div>
        <div class="device-reliability {rcls}">{rtxt} {rel*100:.0f}%</div>
        <div class="sbar"><div class="sbar-f" style="width:{min(score,100):.1f}%;background:{bc};"></div></div>
        """, unsafe_allow_html=True)
        lc, rc = st.columns(2)
        with lc:
            st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#166534;margin-bottom:0.3rem;">✓ 优势</p>', unsafe_allow_html=True)
            for item in adv[:3]:
                st.markdown(f'<span class="tag-pros">✓ {escape(tr(str(item)))}</span>', unsafe_allow_html=True)
            if not adv: st.caption("暂无明显优势")
        with rc:
            st.markdown('<p style="font-size:0.95rem;font-weight:700;color:#991b1b;margin-bottom:0.3rem;">⚠ 关注点</p>', unsafe_allow_html=True)
            for item in dis[:3]:
                st.markdown(f'<span class="tag-cons">! {escape(tr(str(item)))}</span>', unsafe_allow_html=True)
            if not dis: st.caption("暂无明显短板")


TF   = dict(size=14, color="#475569")
TF_Y = dict(size=14, color="#334155", family="Arial")
LF   = dict(size=14, color="#64748b")
LEG  = dict(size=14, color="#334155")
ANNO = dict(size=14, color="#5a6a80")


def _anno(text: str) -> dict:
    """Create a Plotly top-center annotation dict for chart titles."""
    return dict(text=text, xref="paper", yref="paper",
                x=0.5, y=1.055, showarrow=False,
                font=ANNO, xanchor="center", yanchor="bottom")


def _base(h, l=8, r=14, t=55, b=38) -> dict:
    """Create a Plotly base layout dict with white background and compact margins."""
    return dict(height=h, margin=dict(l=l, r=r, t=t, b=b),
                plot_bgcolor="white", paper_bgcolor="white")


def _legend_h(y=-0.26) -> dict:
    """Create a horizontal legend dict positioned at the bottom of the chart."""
    return dict(orientation="h", x=0.5, y=y, xanchor="center", yanchor="top",
                font=LEG, bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#e2e8f0", borderwidth=1)


def chart_ccm_standardization(comparison_matrix: Dict[str, Any], height=320):
    """Layer 1 CCM test-condition standardization comparison chart.

    Identifies parameters with value differences or test-condition differences
    across devices. Renders a line chart (rank comparison) when >=3 differing
    parameters exist, otherwise a grouped bar chart.
    """
    if not comparison_matrix:
        st.info("暂无标准化数据"); return
    chips = comparison_matrix.get("chips", [])
    parameters = comparison_matrix.get("parameters", [])
    data = comparison_matrix.get("data", {})
    if not chips or not parameters:
        st.info("暂无标准化数据"); return

    def parse_value(display_str: str) -> float:
        if not display_str or display_str == "N/A":
            return -float('inf')
        import re
        match = re.search(r'[-+]?\d*\.?\d+', str(display_str).replace(',', ''))
        return float(match.group()) if match else -float('inf')

    def _conditions_differ(conditions_list: list) -> bool:
        non_empty = [c for c in conditions_list if c]
        if len(non_empty) <= 1:
            return False
        return len(set(str(sorted(c.items())) for c in non_empty if isinstance(c, dict))) > 1

    diff_params = []
    for param in parameters:
        label = PARAM_LABELS.get(param, param)
        values = [data.get(c, {}).get(param, {}).get("display", "N/A") for c in chips]
        conditions = [data.get(c, {}).get(param, {}).get("test_conditions", {}) for c in chips]
        cond_strs = [data.get(c, {}).get(param, {}).get("condition", "") for c in chips]
        
        non_na_values = [v for v in values if v and v != "N/A"]
        val_diff_multi = len(set(non_na_values)) > 1
        val_diff_missing = len(non_na_values) >= 1 and len(non_na_values) < len(values)
        val_diff = val_diff_multi or val_diff_missing
        
        cond_diff = _conditions_differ(conditions)
        if val_diff or cond_diff:
            diff_params.append({
                "param": label,
                "values": values,
                "raw_values": [parse_value(v) for v in values],
                "cond_diff": cond_diff,
                "cond_strs": cond_strs,
            })

    if not diff_params:
        st.success("🎉 所有参数在相同测试条件下，无需标准化调整！"); return

    n_total = len(parameters)
    n_cond_diff = sum(1 for p in diff_params if p["cond_diff"])

    show = diff_params[:12]
    n_chips = len(chips)
    n_show = len(show)

    fig = go.Figure()

    for i, chip in enumerate(chips):
        x_vals, y_vals, text_vals = [], [], []
        offset = (i - (n_chips - 1) / 2) * 0.25
        for j, p in enumerate(show):
            x_vals.append(j + offset)
            val_str = p["values"][i]
            if p["cond_diff"] and p["cond_strs"][i]:
                val_str = f"{val_str}<br><span style='font-size:9px;color:#f59e0b'>({p['cond_strs'][i]})</span>"
            elif p["values"][i] == "N/A":
                val_str = "<span style='color:#94a3b8'>N/A</span>"
            text_vals.append(val_str)

            raw_vals = p["raw_values"]
            sorted_indices = sorted(range(len(raw_vals)), key=lambda idx: raw_vals[idx], reverse=True)
            rank = sorted_indices.index(i) + 1
            y_vals.append(rank)

        txt_pos = "top left" if i % 2 == 0 else "top right"
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers+text",
            name=chip,
            line=dict(color=gc(i), width=2.5),
            marker=dict(size=11),
            text=text_vals,
            textposition=txt_pos,
            textfont=dict(size=11, color="#334155", family="Arial"),
            cliponaxis=False,
            hovertemplate=f"<b>{chip}</b><br>参数: %{{x}}<br>值: %{{text}}<extra></extra>",
        ))

    layout = _base(max(height, 360), l=10, r=14, t=50, b=80)
    layout.update(
        showlegend=True,
        legend=_legend_h(y=-0.40),
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color="#334155"), tickangle=-35,
                   tickmode="array", tickvals=list(range(n_show)),
                   ticktext=[p["param"] for p in show]),
        yaxis=dict(showgrid=False, showticklabels=False,
                   range=[0.5, len(chips) + 0.5], autorange="reversed"),
        annotations=[_anno(
            f"<b>Layer 1: CCM测试条件标准化</b>　"
            f"共 {n_total} 项参数，{len(diff_params)} 项差异（含{n_cond_diff}项测试条件差异）（上方=数值更大）"
        )],
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chart_reliability_heatmap(devices, comparison_matrix, height=340):
    """Layer 2 Z-number parameter reliability assessment chart (grouped bar chart).

    Displays per-parameter reliability percentages for each device with dashed
    reference lines at 85% (high) and 70% (medium) thresholds.
    """
    if not devices or not comparison_matrix:
        st.info("暂无可靠性数据"); return
    chips      = comparison_matrix.get("chips", [])
    parameters = comparison_matrix.get("parameters", [])
    data       = comparison_matrix.get("data", {})
    if not chips or not parameters:
        st.info("暂无可靠性数据"); return

    show_params  = parameters[:14]
    param_labels = [PARAM_LABELS.get(p, p) for p in show_params]

    device_reliabilities = {}
    for d in devices:
        name = d.get("device_name", "")
        device_reliabilities[name] = d.get("parameter_reliabilities", {})

    fig = go.Figure()
    for i, chip in enumerate(chips):
        chip_data = data.get(chip, {})
        chip_rels = device_reliabilities.get(chip, {})
        rels = []
        for param in show_params:
            scoring_name = PARAM_TO_SCORING.get(param)
            r = None
            if scoring_name and scoring_name in chip_rels:
                r = chip_rels[scoring_name]
            if r is None:
                r = chip_data.get(param,{}).get("confidence", 0.5)
            try:   r = max(0.0, min(1.0, float(r) if r is not None else 0.5))
            except: r = 0.5
            rels.append(r * 100)
        fig.add_trace(go.Bar(
            name=chip, x=param_labels, y=rels,
            marker=dict(color=gc(i)),
            hovertemplate=f"<b>{chip}</b><br>参数: %{{x}}<br>可靠度: %{{y:.0f}}%<extra></extra>",
        ))

    n = len(show_params)
    layout = _base(height, l=8, r=10, t=60, b=75)
    layout.update(
        barmode="group", bargap=0.18, bargroupgap=0.06,
        showlegend=True,
        legend=_legend_h(y=-0.30),
        xaxis=dict(showgrid=False, tickfont=dict(size=13,color="#334155"), tickangle=-40),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=TF,
                   title=dict(text="可靠度 (%)", font=LF), range=[0,110]),
        annotations=[_anno(f"<b>Layer 2: Z-number可靠度评估</b>　共 {len(parameters)} 项参数")],
        shapes=[
            dict(type="line", x0=-0.5, y0=85, x1=n-0.5, y1=85,
                 line=dict(color="#22c55e",width=1.5,dash="dash")),
            dict(type="line", x0=-0.5, y0=70, x1=n-0.5, y1=70,
                 line=dict(color="#f59e0b",width=1.5,dash="dash")),
        ],
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chart_enhanced_radar(dim_labels, names, vals_per_dev, height=370):
    """Enhanced radar chart (polar multi-dimension comparison).

    Includes an "ideal target" reference trace at 100% on all axes.
    Device scores are clamped to [0, 100] and polygons are closed for
    correct polar rendering.
    """
    if not dim_labels or not names or not vals_per_dev:
        st.info("暂无雷达图数据"); return

    vals_per_dev_clipped = []
    for vals in vals_per_dev:
        clipped = [min(100.0, max(0.0, float(v) if v is not None else 0.0)) for v in vals]
        vals_per_dev_clipped.append(clipped)

    def rgba(hx, a=0.22):
        """Convert hex color to rgba string with given alpha."""
        h = hx.lstrip("#")
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[100]*(len(dim_labels)+1), theta=list(dim_labels)+[dim_labels[0]],
        fill="toself", name="理想目标",
        line=dict(color="#22c55e",width=1.5,dash="dash"),
        fillcolor="rgba(34,197,94,0.06)",
        hovertemplate="<b>理想目标</b><br>参考点: 100%<extra></extra>",
    ))
    for i,(name,vals) in enumerate(zip(names, vals_per_dev_clipped)):
        cl   = list(dim_labels)+[dim_labels[0]]
        cv   = list(vals)+[vals[0]] if vals else []
        color = gc(i)
        fig.add_trace(go.Scatterpolar(
            r=cv, theta=cl, fill="toself", name=name,
            line=dict(color=color,width=2.5),
            fillcolor=rgba(color,0.25),
            hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r:.1f}}分<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0,100],
                tickfont=dict(size=15,color="#94a3b8"),
                gridcolor="#e2e8f0", linecolor="#cbd5e1",
                tickvals=[25,50,75,100], ticktext=["25","50","75","100"],
            ),
            angularaxis=dict(
                tickfont=dict(size=15,color="#334155"),
                gridcolor="#e2e8f0", linecolor="#cbd5e1", rotation=90,
            ),
        ),
        height=height,
        margin=dict(l=20,r=20,t=85,b=52),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h",x=0.5,y=-0.1,xanchor="center",yanchor="top",
                    font=LEG, bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#e2e8f0",borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chart_comprehensive_decision(devices, height=200):
    """Layer 3 B-SPOTIS comprehensive decision ranking chart (horizontal bar + error bars).

    Error bar length = score × (1 - reliability) × 0.5, reflecting data uncertainty.
    Bar color encodes reliability tier: green (>=85%), yellow (70-85%), red (<70%).
    """
    if not devices:
        st.info("暂无决策数据"); return

    names  = [d.get("device_name",f"设备{i+1}") for i,d in enumerate(devices)]
    scores = [d.get("overall_score",0.0) for d in devices]
    rels   = [d.get("reliability_score",0.8) for d in devices]
    esps   = [d.get("esp_distance",0.5) for d in devices]
    errs   = [s*(1-r)*0.5 for s,r in zip(scores,rels)]
    colors = ["#22c55e" if r>=0.85 else "#f59e0b" if r>=0.70 else "#ef4444" for r in rels]

    fig = go.Figure()
    for idx,(name,score,err,color,rel,esp) in enumerate(
            zip(names,scores,errs,colors,rels,esps)):
        fig.add_trace(go.Bar(
            name=name, y=[name], x=[score], orientation="h",
            marker=dict(color=color,line=dict(color="white",width=1)),
            text=f"{score:.1f}", textposition="inside",
            textfont=dict(size=14,color="white",family="Arial"),
            hovertemplate=(f"<b>{name}</b><br>综合得分: {score:.1f}<br>"
                           f"ESP距离: {esp:.4f}<br>可靠度: {rel*100:.0f}%<extra></extra>"),
            error_x=dict(type="data",array=[err],visible=True,
                         color="rgba(0,0,0,0.3)",thickness=2,width=6),
        ))

    layout = _base(height, l=90, r=18, t=75, b=38)
    layout.update(
        xaxis=dict(range=[0,110], showgrid=True, gridcolor="#f1f5f9",
                   tickfont=TF, title=dict(text="综合得分 (B-SPOTIS)",font=LF)),
        yaxis=dict(tickfont=dict(size=13,color="#334155",family="Arial"),
                   autorange="reversed"),
        showlegend=False,
        annotations=[_anno(
            "<b>Layer 3: B-SPOTIS综合决策</b>　基于Layer 1标准化 + Layer 2可靠度评估"
        )],
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chart_ranking_with_confidence(devices):
    """Overview tab ranking bar chart with error bars.

    Similar to chart_comprehensive_decision but with adaptive height,
    smaller fonts, and no top annotation for a more compact layout.
    """
    if not devices:
        st.info("暂无排名数据"); return
    names  = [d.get("device_name",f"设备{i+1}") for i,d in enumerate(devices)]
    scores = [d.get("overall_score",0.0) for d in devices]
    rels   = [d.get("reliability_score",0.8) for d in devices]
    errs   = [s*(1-r)*0.5 for s,r in zip(scores,rels)]
    colors = ["#22c55e" if r>=0.85 else "#f59e0b" if r>=0.70 else "#ef4444" for r in rels]
    fig = go.Figure()
    for name,score,err,color,rel in zip(names,scores,errs,colors,rels):
        fig.add_trace(go.Bar(
            name=name, y=[name], x=[score], orientation="h",
            marker=dict(color=color,line=dict(color="white",width=1)),
            text=f"{score:.1f}", textposition="inside",
            textfont=dict(size=12,color="white",family="Arial"),
            hovertemplate=f"<b>{name}</b><br>得分: {score:.1f}<br>可靠度: {rel*100:.0f}%<extra></extra>",
            error_x=dict(type="data",array=[err],visible=True,
                         color="rgba(0,0,0,0.3)",thickness=2,width=5),
        ))
    fig.update_layout(
        height=max(150,len(devices)*46+40),
        margin=dict(l=88,r=18,t=14,b=36),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[0,110],showgrid=True,gridcolor="#f1f5f9",
                   tickfont=TF,title=dict(text="综合得分 (B-SPOTIS)",font=LF)),
        yaxis=dict(tickfont=dict(size=12,color="#334155"),autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("误差线反映数据可靠性（Z-number）；绿/橙/红 = 高/中/低可靠度")


doc_ids = st.session_state.get("selected_doc_ids", [])
api_ok  = is_api_connected()

if doc_ids:
    try:
        selected_docs = get_docs(doc_ids)
        current_names = " vs ".join(d["device_name"] or d["filename"] for d in selected_docs.values())
        meta = f"当前: {current_names}"
    except Exception:
        meta = f"{len(doc_ids)} 份文档"
else:
    meta = "未选择文档"

api_state = (
    '<span style="color:#16a34a;font-weight:600;">API已连接</span>' if api_ok
    else '<span style="color:#dc2626;font-weight:600;">API未连接</span>'
)

st.markdown(f"""
<div class="pg-hdr">
  <span class="pg-ttl">📊 参数对比</span>
  <span class="pg-sub">{api_state} &nbsp;|&nbsp; {escape(meta)}</span>
</div>""", unsafe_allow_html=True)

if not api_ok:
    st.warning("⚠️ 后端API未连接：`cd d:\\veriquery && python -m api.main`")
    if st.button("🔄 重新连接"):
        for k in ["api_check_ts","api_connected"]: st.session_state.pop(k,None)
        st.rerun()
    st.stop()

if len(doc_ids) < 2:
    lc, rc = st.columns([4,1])
    with lc: st.warning("⚠️ 请在侧边栏选择至少 2 个文档")
    with rc:
        if st.button("📄 前往上传", type="primary", use_container_width=True):
            st.switch_page("pages/1_Documents.py")
    st.stop()

docs_map   = get_docs(doc_ids)
has_result = "compare_enhanced_result" in st.session_state
doc_items  = list(docs_map.items())
device_names = []

c1, c2, c3 = st.columns([1.7, 1.7, 1.2])
with c1:
    if len(doc_items) >= 1:
        did, info = doc_items[0]
        device_names.append(st.text_input("器件名称 1", value=info["device_name"],
                                          key=f"dn_{did}", label_visibility="collapsed"))
    else:
        device_names.append("")
with c2:
    if len(doc_items) >= 2:
        did, info = doc_items[1]
        device_names.append(st.text_input("器件名称 2", value=info["device_name"],
                                          key=f"dn_{did}", label_visibility="collapsed"))
    else:
        device_names.append("")
with c3:
    ready = len([n for n in device_names if n.strip()]) >= 2
    start_clicked = st.button("🔍 开始对比", type="primary", use_container_width=True, disabled=not ready)

if start_clicked:
    sel = [n.strip() for n in device_names if n.strip()]
    if len(sel) < 2:
        st.warning("至少需要 2 个设备名称")
    else:
        with st.spinner("三层架构评分分析中…"):
            enhanced = compare_devices_enhanced(doc_ids, sel)
        if enhanced and enhanced.get("success", True) and not enhanced.get("error"):
            st.session_state.compare_enhanced_result = enhanced
            st.session_state.compare_devices = sel
            st.rerun()
        else:
            err_msg = "未知错误"
            if isinstance(enhanced, dict):
                err_msg = enhanced.get("error") or enhanced.get("detail") or err_msg
            st.error(f"评分失败: {err_msg}")

if not has_result:
    st.stop()

enhanced          = st.session_state.get("compare_enhanced_result", {})
scoring           = enhanced.get("scoring_result", {}) or {}
radar             = enhanced.get("radar_data", {}) or {}
comparison_matrix = enhanced.get("comparison_matrix", {}) or {}
processing_time   = float(enhanced.get("processing_time", 0.0) or 0.0)
devices           = scoring.get("devices", []) or []
best_device       = devices[0] if devices else {}
runner_up         = devices[1] if len(devices) > 1 else {}
avg_reliability   = (sum(float(d.get("reliability_score",0.0)) for d in devices)/len(devices)
                     if devices else 0.0)
score_gap         = (float(best_device.get("overall_score",0.0)) -
                     float(runner_up.get("overall_score",0.0)) if runner_up else 0.0)

tab_overview, tab_dimensions = st.tabs(["🏆 概览", "📈 维度图"])

with tab_overview:
    m1,m2,m3,m4 = st.columns(4)
    with m1: render_metric_tile("推荐器件",  best_device.get("device_name","-"), "综合评分最高")
    with m2: render_metric_tile("领先差值",  f"{score_gap:.1f} 分" if runner_up else "-", "与第二名得分差")
    with m3: render_metric_tile("平均可靠度", f"{avg_reliability*100:.0f}%", "Z-number可靠性评估")
    with m4: render_metric_tile("分析耗时",  f"{processing_time:.2f}s", "三层架构评分分析")

    st.markdown('<div style="margin:0.4rem 0;"></div>', unsafe_allow_html=True)

    for start in range(0, len(devices), 2):
        cols = st.columns(2)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx < len(devices):
                with col: render_device_card(devices[idx], idx)

    matrix_df, sources_data = build_comparison_dataframe(comparison_matrix)
    if not matrix_df.empty:
        diff_count  = int((matrix_df["差异"]=="是").sum())
        total_count = len(matrix_df)
        display_df  = matrix_df[matrix_df["差异"]=="是"].drop(columns=["差异"])
        st.markdown(
            f'<p style="font-size:0.85rem;font-weight:700;color:#334155;'
            f'margin:0.6rem 0 0.3rem;padding-bottom:0.2rem;border-bottom:1.5px solid #e2e8f0;">'
            f'差异参数矩阵（共 {total_count} 项，差异 {diff_count} 项）</p>',
            unsafe_allow_html=True)
        if display_df.empty:
            st.success("🎉 所有参数均相同，无需对比！")
        else:
            chips = [c for c in display_df.columns if c != "参数"]
            html  = "<table class='compare-param-table'><thead><tr><th>参数</th>"
            for chip in chips: html += f"<th>{chip}</th><th>来源</th>"
            html += "</tr></thead><tbody>"
            for _, row in display_df.iterrows():
                pname = row["参数"]
                pkey  = next((k for k,v in PARAM_LABELS.items() if v==pname), pname)
                html += f"<tr><td>{pname}</td>"
                for chip in chips:
                    val   = row[chip] if row[chip] is not None else "-"
                    badge = _get_source_badge(sources_data.get(pkey,{}).get(chip,""))
                    html += f"<td>{val}</td><td class='source-badge'>{badge}</td>"
                html += "</tr>"
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("暂无参数矩阵数据")

with tab_dimensions:
    names, dimensions, vals = get_dimension_dataset(scoring, radar)
    if not names or not dimensions:
        st.info("暂无维度数据")
    else:
        CCM_H   = 320
        Z_H     = 490
        RADAR_H = 510
        BSPOT_H = 260

        left_col, right_col = st.columns([58, 42])

        with left_col:
            st.markdown('<div class="stl">📋 Layer 1 · CCM 测试条件标准化对比</div>', unsafe_allow_html=True)
            chart_ccm_standardization(comparison_matrix, height=CCM_H)

            st.markdown('<div class="stl">🔥 Layer 2 · Z-number 参数可靠性评估</div>', unsafe_allow_html=True)
            chart_reliability_heatmap(devices, comparison_matrix, height=Z_H)

        with right_col:
            st.markdown('<div class="stl">📡 增强型雷达图（Layer 3 综合展示）</div>', unsafe_allow_html=True)
            chart_enhanced_radar(dimensions, names, vals, height=RADAR_H)

            st.markdown('<div class="stl">📊 Layer 3 · B-SPOTIS 综合决策排名</div>', unsafe_allow_html=True)
            chart_comprehensive_decision(devices, height=BSPOT_H)

    st.markdown("""
    <div class="arch-note">
      <strong>📊 三层架构说明：</strong>&nbsp;&nbsp;
      <span style="color:#3b82f6">● Layer 1 (CCM)</span> 测试条件标准化，确保参数可比性
      &nbsp;｜&nbsp;
      <span style="color:#f59e0b">● Layer 2 (Z-number)</span> 评估数据可靠度（≥85% 高 · 70–85% 中 · &lt;70% 低）
      &nbsp;｜&nbsp;
      <span style="color:#22c55e">● Layer 3 (B-SPOTIS)</span> 多目标优化，ESP距离越小越优；雷达面积越大整体性能越优
    </div>
    """, unsafe_allow_html=True)
