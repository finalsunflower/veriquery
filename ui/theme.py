"""
Global Theme & Reusable UI Components — EC-VeriQuery

Provides the academic-style CSS theme (``ACADEMIC_THEME``) injected into every
Streamlit page, plus three helper functions:

    - apply_academic_theme()      — inject the global stylesheet
    - render_similarity_bar()     — horizontal progress bar for retrieval scores
    - empty_state()               — centred placeholder for empty-data views

CSS injection is chosen over ``.streamlit/config.toml`` theming because the
project requires full-width layout, custom sidebar structure, and CSS classes
that the limited config.toml format cannot express.

Dependencies:
    - streamlit (st.markdown for CSS/HTML injection)
"""
import streamlit as st

ACADEMIC_THEME = """
<style>
    /* ── Global layout ─────────────────────────────────────────────────── */
    .main {
        max-width: 100%;
        padding: 1.25rem 2rem 1rem 2rem !important;
        margin-top: 0 !important;
        width: 100% !important;
    }

    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1.5rem !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    header {
        padding: 0 !important;
    }

    .main > div:first-child {
        margin-top: 0 !important;
    }

    .stApp {
        width: 100% !important;
        max-width: 100% !important;
    }

    .stApp > div {
        width: 100% !important;
        max-width: 100% !important;
    }

    .main .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* ── Design tokens (CSS variables) ─────────────────────────────────── */
    :root {
        --primary: #2563eb;
        --bg-main: #f8fafc;
        --bg-card: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --border: #e2e8f0;
        --border-light: #f1f5f9;
    }

    /* ── Tag component (BaseWeb override) ──────────────────────────────── */
    [data-baseweb="tag"] {
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        color: #475569 !important;
        border-radius: 6px !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    [data-baseweb="tag"]:hover {
        background-color: #e2e8f0 !important;
        border-color: #cbd5e1 !important;
    }

    [data-baseweb="tag"] svg {
        fill: #64748b !important;
        opacity: 0.6 !important;
    }

    [data-baseweb="tag"]:hover svg {
        opacity: 1 !important;
    }

    /* ── Typography ────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: var(--text-primary);
        background-color: var(--bg-main);
    }

    h1 {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 0.5rem !important;
        margin-top: 0.5rem !important;
        padding-bottom: 0.75rem !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    h2 {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-top: 0.75rem !important;
        margin-bottom: 0.25rem !important;
    }

    /* ── Status badges ─────────────────────────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-success     { background: #dcfce7; color: #166534; }
    .status-warning     { background: #fef3c7; color: #92400e; }
    .status-error       { background: #fee2e2; color: #991b1b; }
    .status-info        { background: #e0f2fe; color: #075985; }
    .status-processing  { background: #f1f5f9; color: #475569; }

    /* ── Buttons ───────────────────────────────────────────────────────── */
    .stButton > button {
        background-color: var(--bg-card);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.15s ease;
        box-sizing: border-box;
    }

    .stButton > button:hover {
        border-color: var(--primary);
        color: var(--primary);
        background-color: #f8fafc;
    }

    .block-container .stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
    }

    .block-container .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }

    .block-container .stButton > button[kind="secondary"] {
        background-color: #f1f5f9 !important;
        color: #64748b !important;
        border: 1px solid #e2e8f0 !important;
    }

    .block-container .stButton > button[kind="secondary"]:hover {
        background-color: #e2e8f0 !important;
        color: #475569 !important;
    }

    /* ── Inputs (unified 2.75rem baseline) ─────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        min-height: 2.75rem !important;
        height: 2.75rem !important;
        line-height: 1.5 !important;
        box-sizing: border-box !important;
    }

    .block-container .stTextInput > div,
    .block-container .stSelectbox > div {
        min-height: 2.75rem !important;
        height: 2.75rem !important;
        display: flex !important;
        align-items: center !important;
    }

    .block-container .stColumns {
        display: flex !important;
        align-items: center !important;
    }

    .block-container .stColumns [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: stretch !important;
        min-height: 2.75rem !important;
    }

    .block-container .stColumns [data-testid="column"] > div {
        display: flex !important;
        align-items: center !important;
        min-height: 2.75rem !important;
        height: 2.75rem !important;
    }

    .block-container .stTextInput,
    .block-container .stSelectbox {
        display: flex !important;
        align-items: center !important;
        min-height: 2.75rem !important;
    }

    /* ── Dataframe ─────────────────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Sidebar ───────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebarContent"] {
        padding-top: 1.0rem !important;
        padding-bottom: 0 !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
        padding-bottom: 0.5rem !important;
        margin-top: 0 !important;
    }
    [data-testid="stSidebarHeader"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebarNav"],
    div[data-testid="stSidebarNav"],
    nav[data-testid="stSidebarNav"],
    [data-testid="stSidebarNavItems"],
    [data-testid="stSidebarNavLink"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        overflow: hidden !important;
        padding: 0 !important;
        margin: 0 !important;
        position: absolute !important;
    }

    section[data-testid="stSidebar"] > div:first-child,
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    section[data-testid="stSidebar"] > div > div > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div,
    section[data-testid="stSidebar"] [data-testid="element-container"],
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stPageLink"],
    section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
        padding: 0.08rem 0.55rem !important;
        margin: 0 !important;
    }
    section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"],
    section[data-testid="stSidebar"] [data-testid="stPageLink"] a {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: 0.6rem 0.72rem !important;
        border-radius: 8px !important;
        font-size: 1.0rem !important;
        font-weight: 400 !important;
        color: #4b5563 !important;
        background: transparent !important;
        border-left: 2px solid transparent !important;
        text-decoration: none !important;
        transition: all 0.12s ease !important;
        line-height: 1.5 !important;
        box-sizing: border-box !important;
    }
    section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover,
    section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
        background: #f8fafc !important;
        color: #1e293b !important;
        border-left-color: #cbd5e1 !important;
        text-decoration: none !important;
    }
    section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"],
    section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
        background: #eff6ff !important;
        color: #2563eb !important;
        border-left-color: #2563eb !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stPageLink"] svg,
    section[data-testid="stSidebar"] [data-testid="stPageLink"] img {
        display: none !important;
    }

    .sb-header {
        padding: 1.2rem 0.9rem 1.0rem 0.9rem;
        border-bottom: 1px solid #e2e8f0;
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 65%);
        margin-top: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .sb-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .sb-brand-icon {
        font-size: 2.0rem;
        line-height: 1;
        flex-shrink: 0;
    }
    .sb-brand-text h3 {
        margin: 0 !important;
        padding: 0 !important;
        font-weight: 700;
        color: #1e293b;
        font-size: 1.5rem;
        line-height: 1.25 !important;
        letter-spacing: -0.01em;
    }
    .sb-brand-text p {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.3 !important;
    }

    .sb-nav-divider {
        height: 1px;
        background: #cbd5e1;
        margin: 1.0rem 0.9rem 0.8rem 0.9rem;
        border-radius: 1px;
    }

    section[data-testid="stSidebar"] [data-testid="stMultiSelect"] {
        padding: 0 0.55rem !important;
        margin-bottom: 0.1rem !important;
        min-height: 2.5rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMultiSelect"] label {
        display: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMultiSelect"] > div {
        gap: 0.25rem !important;
        min-height: 2.5rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        min-height: 2.5rem !important;
        max-height: 6rem !important;
        height: auto !important;
        border-radius: 8px !important;
        border-color: #e2e8f0 !important;
        background: #ffffff !important;
        box-shadow: none !important;
        overflow-y: auto !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover {
        border-color: #cbd5e1 !important;
    }

    .sb-footer {
        padding: 0.8rem 0.9rem 0.9rem 0.9rem;
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
        margin-top: 1.0rem;
    }
    .sb-footer-title {
        display: block;
        font-size: 0.72rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .sb-footer-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.22rem 0;
        font-size: 0.8rem;
    }
    .sb-footer-label {
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.8rem;
    }
    .sb-footer-value         { color: #1e293b; font-weight: 500; }
    .sb-footer-value.online  { color: #059669; }
    .sb-footer-value.offline { color: #dc2626; }

    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        padding: 0.42rem 0.75rem !important;
        margin: 0.1rem 0.55rem 0 0.55rem !important;
        font-size: 0.78rem !important;
        border-radius: 8px !important;
    }

    section[data-testid="stSidebar"] .stButton {
        display: block !important;
        height: auto !important;
        min-height: 0 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        display: inline-flex !important;
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: center !important;
        height: auto !important;
        min-height: 0 !important;
        padding: 0.44rem 0.72rem !important;
        border-radius: 8px !important;
        font-size: 0.855rem !important;
        font-weight: 400 !important;
        color: #475569 !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
        line-height: 1.6 !important;
        transition: background 0.15s ease, color 0.15s ease !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #f1f5f9 !important;
        color: #1e293b !important;
        border-color: #e2e8f0 !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stAlert {
        height: auto !important;
        min-height: 0 !important;
        display: block !important;
        padding: 0.4rem 0.6rem !important;
        font-size: 0.8rem !important;
    }

    /* ── Tabs ──────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        color: var(--text-secondary);
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
    }

    /* ── Similarity progress bar ───────────────────────────────────────── */
    .similarity-bar {
        background: #e2e8f0;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
        width: 100%;
    }

    .similarity-fill {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
</style>
"""


def apply_academic_theme():
    """Inject the global academic-style CSS into the current Streamlit page.

    Must be called at the top of every page file so that the stylesheet is
    present before any UI elements are rendered.
    """
    st.markdown(ACADEMIC_THEME, unsafe_allow_html=True)


def render_similarity_bar(similarity: float, label: str = ""):
    """Render a horizontal progress bar showing a retrieval similarity score.

    Args:
        similarity: Score in [0.0, 1.0] (e.g. cosine similarity from ChromaDB).
        label:      Optional text displayed after the percentage.
    """
    percentage = similarity * 100
    label_html = f'<span style="margin-left: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">{label}</span>' if label else ""

    st.markdown(f"""
    <div style="margin: 0.5rem 0;">
        <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
            <span style="font-size: 0.9rem; font-weight: 500;">{percentage:.1f}%</span>
            {label_html}
        </div>
        <div class="similarity-bar">
            <div class="similarity-fill" style="width: {percentage}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def empty_state(message: str, icon: str = "📭"):
    """Render a centred empty-state placeholder with an icon and message.

    Used when a page or section has no data to display, guiding the user
    towards their next action instead of showing a blank area.

    Args:
        message: Hint text (e.g. "暂无文档，请先上传数据手册PDF").
        icon:    Emoji displayed above the message.
    """
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 1rem; color: var(--text-secondary);">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 1rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)
