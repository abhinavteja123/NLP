"""
IQAS Upload Page
=================
Document upload, text preview, NLP stats, chunking, and index building.
"""

import streamlit as st
from pathlib import Path
import tempfile
import shutil
import os
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.components import (
    render_section_header,
    render_metrics_row,
    render_nlp_table,
)
from utils.config import UPLOAD_DIR


def render_upload_page():
    """Render the document upload and indexing page."""

    st.markdown("# 📤 Document Upload & Indexing")
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 1.05rem;">'
        "Upload your documents to build a searchable knowledge base.</p>",
        unsafe_allow_html=True,
    )

    # ── File Upload Zone ──
    render_section_header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, DOCX",
        key="doc_uploader",
    )

    if not uploaded_files:
        st.markdown("""
        <div class="upload-zone">
            <p style="font-size: 2.5rem; margin-bottom: 12px;">📁</p>
            <p style="color: #9AA0A6; font-size: 1rem;">
                Upload PDF, TXT, or DOCX files to get started
            </p>
            <p style="color: #666; font-size: 0.85rem; margin-top: 8px;">
                Your documents will be processed, chunked, and indexed for semantic search
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Option to load sample documents
        st.markdown("---")
        render_section_header("📚 Quick Start with Sample Data")
        sample_path = PROJECT_ROOT / "data" / "sample_docs" / "sample_nlp_notes.txt"

        if sample_path.exists():
            if st.button("📖 Load Sample NLP Notes", key="load_sample"):
                _process_and_index([sample_path], strategy="sentence")
        else:
            st.info("Sample data file not found. Upload your own documents to get started.")

        return

    # ── Save uploaded files ──
    saved_paths = []
    for file in uploaded_files:
        save_path = UPLOAD_DIR / file.name
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(save_path)

    # ── File Preview ──
    render_section_header("📄 Uploaded Files")

    file_data = []
    for path in saved_paths:
        size_kb = path.stat().st_size / 1024
        file_data.append({
            "File": path.name,
            "Format": path.suffix.upper(),
            "Size": f"{size_kb:.1f} KB",
        })

    render_nlp_table(file_data, ["File", "Format", "Size"])

    # ── Text Preview ──
    with st.expander("👁️ Preview Extracted Text", expanded=False):
        pipeline = st.session_state.get("pipeline")
        if pipeline:
            for path in saved_paths:
                docs = pipeline.document_loader.load_any(path)
                for doc in docs[:2]:
                    st.markdown(f"**{doc.filename}** (Page {doc.page_num})")
                    st.text(doc.text[:500] + ("..." if len(doc.text) > 500 else ""))
                    st.markdown("---")

    # ── NLP Preview ──
    with st.expander("🧠 NLP Analysis Preview", expanded=False):
        pipeline = st.session_state.get("pipeline")
        if pipeline:
            for path in saved_paths[:1]:  # Preview first file only
                docs = pipeline.document_loader.load_any(path)
                if docs:
                    text = docs[0].text[:2000]
                    try:
                        from nlp.tokenizer import NLPTokenizer
                        from nlp.ner import NERExtractor

                        tokenizer = NLPTokenizer()
                        ner = NERExtractor()

                        tokens = tokenizer.word_tokenize(text)
                        sentences = tokenizer.sent_tokenize(text)
                        entities = ner.extract(text)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Tokens", len(tokens))
                        with col2:
                            st.metric("Sentences", len(sentences))
                        with col3:
                            st.metric("Entities", len(entities))

                        if entities:
                            st.markdown("**Top Entities:**")
                            entity_data = [
                                {"Entity": e.text, "Type": e.label}
                                for e in entities[:10]
                            ]
                            render_nlp_table(entity_data, ["Entity", "Type"])
                    except Exception as e:
                        st.warning(f"NLP preview error: {e}")

    # ── Chunking Strategy ──
    st.markdown("---")
    render_section_header("⚙️ Chunking Configuration")

    strategy = st.radio(
        "Select chunking strategy:",
        options=["sentence", "paragraph", "fixed"],
        format_func=lambda x: {
            "sentence": "📝 Sentence-Aware (recommended)",
            "paragraph": "📄 Paragraph-Aware",
            "fixed": "🔢 Fixed-Size (512 tokens)",
        }[x],
        index=0,
        key="chunk_strategy",
        horizontal=True,
    )

    # ── Build Index Button ──
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Build Search Index", key="build_index", use_container_width=True):
            _process_and_index(saved_paths, strategy)


def _process_and_index(paths, strategy):
    """Process documents and build the FAISS index."""
    pipeline = st.session_state.get("pipeline")
    if not pipeline:
        st.error("Pipeline not initialized. Please refresh the page.")
        return

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_callback(stage: str, progress: float):
        progress_bar.progress(progress)
        status_text.markdown(
            f'<div style="color: #00B4D8; font-weight: 600;">⏳ {stage}</div>',
            unsafe_allow_html=True,
        )

    try:
        stats = pipeline.ingest_documents(
            paths=[str(p) for p in paths],
            strategy=strategy,
            progress_callback=progress_callback,
        )

        progress_bar.progress(1.0)
        status_text.markdown(
            '<div style="color: #2ECC71; font-weight: 600;">✅ Index built successfully!</div>',
            unsafe_allow_html=True,
        )

        # Show stats
        render_metrics_row([
            {"value": str(stats.num_documents), "label": "Documents"},
            {"value": str(stats.num_chunks), "label": "Chunks"},
            {"value": f"{stats.total_tokens:,}", "label": "Total Tokens"},
            {"value": f"{stats.index_time_seconds}s", "label": "Build Time"},
        ])

        st.success(
            f"✅ Successfully indexed {stats.num_documents} documents → "
            f"{stats.num_chunks} chunks using '{stats.chunking_strategy}' strategy. "
            f"Go to the **Q&A** page to ask questions!"
        )

    except Exception as e:
        progress_bar.progress(0.0)
        status_text.empty()
        st.error(f"❌ Indexing failed: {e}")
        import traceback
        st.code(traceback.format_exc())
