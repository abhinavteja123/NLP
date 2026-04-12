"""
IQAS Streamlit Application — Main Entrypoint
==============================================
Multi-page app with sidebar navigation, session state management, and CSS injection.
"""

import sys
from pathlib import Path

# ── Add project root to Python path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.ui.components import load_css, render_sidebar_brand, render_status_badge
from utils.config import APP_TITLE, APP_ICON, APP_LAYOUT


# ── Page Configuration ──
st.set_page_config(
    page_title="IQAS — Intelligent Question Answering System",
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Inject Custom CSS ──
load_css()

# ── Initialize Pipeline in Session State ──
if "pipeline" not in st.session_state:
    from core.pipeline import QAPipeline
    pipeline = QAPipeline()
    # Try to load existing index
    pipeline.load_index()
    st.session_state["pipeline"] = pipeline

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "query_log" not in st.session_state:
    st.session_state["query_log"] = []


# ── Sidebar ──
with st.sidebar:
    render_sidebar_brand()

    st.markdown("---")

    # Navigation
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 0.8rem; text-transform: uppercase; '
        'letter-spacing: 0.1em; margin-bottom: 8px;">Navigation</p>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Go to:",
        options=["📤 Upload", "💬 Q&A", "📊 Analytics"],
        index=0,
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.markdown("---")

    # System Status
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 0.8rem; text-transform: uppercase; '
        'letter-spacing: 0.1em; margin-bottom: 8px;">System Status</p>',
        unsafe_allow_html=True,
    )

    pipeline = st.session_state.get("pipeline")
    is_ready = pipeline.is_ready if pipeline else False

    render_status_badge(is_ready)

    if pipeline:
        stats = pipeline.get_stats()
        st.markdown(
            f'<div style="margin-top: 12px; color: #9AA0A6; font-size: 0.85rem;">'
            f'📚 Documents: <strong style="color: #E8EAED;">{stats["num_documents"]}</strong><br>'
            f'🧩 Chunks: <strong style="color: #E8EAED;">{stats["num_chunks"]}</strong><br>'
            f'🔍 Index: <strong style="color: #E8EAED;">{stats["index_size"]}</strong> vectors'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Session stats
    if st.session_state.get("chat_history"):
        st.markdown("---")
        st.markdown(
            f'<p style="color: #9AA0A6; font-size: 0.85rem;">'
            f'💬 Queries this session: '
            f'<strong style="color: #00B4D8;">{len(st.session_state["chat_history"])}</strong>'
            f'</p>',
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.75rem;">'
        '<p>IQAS v1.0</p>'
        '<p>NLP Project · SRM University AP</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Page Routing ──
if page == "📤 Upload":
    from app.views.upload import render_upload_page
    render_upload_page()

elif page == "💬 Q&A":
    from app.views.qa import render_qa_page
    render_qa_page()

elif page == "📊 Analytics":
    from app.views.analytics import render_analytics_page
    render_analytics_page()
