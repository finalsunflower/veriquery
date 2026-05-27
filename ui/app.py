"""
Application Entry Point — EC-VeriQuery

Streamlit multi-page application main page.  Handles page configuration,
global theme injection, sidebar navigation, and a card-grid landing page
with navigation buttons to each of the six feature modules.

Page flow:
    set_page_config → apply_academic_theme → render_sidebar_nav
    → inject landing-page CSS → render nav cards (3×2 grid) → footer

Dependencies:
    - theme.py: Global academic-style CSS
    - sidebar_nav.py: Sidebar navigation and document selector
"""
import streamlit as st

st.set_page_config(
    page_title="EC-VeriQuery - 电子硬件规格书智能问答系统",
    page_icon="🌻",
    layout="wide",
    initial_sidebar_state="expanded",
)

from theme import apply_academic_theme
from sidebar_nav import render_sidebar_nav

apply_academic_theme()

with st.sidebar:
    render_sidebar_nav()

st.markdown("""
<style>
    .main-content {
        margin-top: -3rem;
    }
    .nav-header {
        text-align: center;
        padding: 0.5rem 1rem 0.75rem 1rem;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 0.75rem;
    }
    .nav-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0 0 0.3rem 0;
    }
    .nav-header p {
        font-size: 0.9rem;
        color: #64748b;
        margin: 0;
    }
    .nav-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .nav-card {
        padding: 1.75rem 1.5rem;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        transition: all 0.2s ease;
        cursor: pointer;
        min-height: 160px;
    }
    .nav-card:hover {
        border-color: #2563eb;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
        transform: translateY(-2px);
    }
    .nav-card-icon {
        font-size: 2.25rem;
        margin-bottom: 0.75rem;
    }
    .nav-card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin: 0 0 0.5rem 0;
    }
    .nav-card-desc {
        font-size: 0.875rem;
        color: #64748b;
        margin: 0;
        line-height: 1.5;
    }
    .nav-footer {
        text-align: center;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.78rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.markdown("""
<div class="nav-header">
    <h1>🌻 EC-VeriQuery</h1>
    <p>电子硬件规格书智能问答系统</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-icon">📄</div>
        <div class="nav-card-title">文档管理</div>
        <div class="nav-card-desc">上传、管理电子元器件规格书PDF文档，支持批量处理和状态监控</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📄 进入文档管理", key="goto_doc", use_container_width=True):
        st.switch_page("pages/1_Documents.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-icon">💬</div>
        <div class="nav-card-title">智能问答</div>
        <div class="nav-card-desc">基于RAG技术智能分析规格书内容，快速获取器件参数和技术信息</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("💬 进入智能问答", key="goto_chat", use_container_width=True):
        st.switch_page("pages/2_Chat.py")

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-icon">📍</div>
        <div class="nav-card-title">引脚分析</div>
        <div class="nav-card-desc">自动解析芯片引脚定义，生成引脚对照表和连接建议</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📍 进入引脚分析", key="goto_pinout", use_container_width=True):
        st.switch_page("pages/3_Pinout.py")

with col4:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-icon">⚡</div>
        <div class="nav-card-title">ERC检查</div>
        <div class="nav-card-desc">电气规则检查，验证电源、地、数字逻辑等网络的兼容性</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("⚡ 进入ERC检查", key="goto_erc", use_container_width=True):
        st.switch_page("pages/4_ERC.py")

col5, col6 = st.columns(2)

with col5:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-icon">📊</div>
        <div class="nav-card-title">参数对比</div>
        <div class="nav-card-desc">多器件参数横向对比分析，直观展示不同器件的规格差异</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📊 进入参数对比", key="goto_compare", use_container_width=True):
        st.switch_page("pages/5_Compare.py")

with col6:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-icon">🔌</div>
        <div class="nav-card-title">电路检索</div>
        <div class="nav-card-desc">检索典型应用电路和参考设计，快速找到符合需求的电路方案</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔌 进入电路检索", key="goto_circuit", use_container_width=True):
        st.switch_page("pages/6_Circuit.py")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="nav-footer">
    <p>EC-VeriQuery v1.0 | 电子硬件规格书智能问答系统</p>
</div>
""", unsafe_allow_html=True)
