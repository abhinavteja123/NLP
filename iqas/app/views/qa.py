"""
IQAS Q&A Page
==============
Question input, answer display, NLP breakdown, and chat history.
"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.components import (
    render_answer_card,
    render_confidence_badge,
    render_source_chip,
    render_section_header,
    render_nlp_table,
    render_chat_item,
)


def render_qa_page():
    """Render the Question & Answer interface page."""

    st.markdown("# 💬 Ask a Question")
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 1.05rem;">'
        "Ask questions about your uploaded documents — powered by hybrid NLP retrieval.</p>",
        unsafe_allow_html=True,
    )

    pipeline = st.session_state.get("pipeline")

    # Check if index is ready
    if not pipeline or not pipeline.is_ready:
        st.warning(
            "⚠️ No documents have been indexed yet. "
            "Go to the **📤 Upload** page to upload and index documents first."
        )
        return

    # Show system stats
    stats = pipeline.get_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Documents", stats["num_documents"])
    with col2:
        st.metric("🧩 Chunks", stats["num_chunks"])
    with col3:
        st.metric("🔍 Index Size", stats["index_size"])

    st.markdown("---")

    # ── Question Input ──
    question = st.text_input(
        "Your Question:",
        placeholder="e.g., What is tokenization in NLP?",
        key="question_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", key="ask_button", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state["chat_history"] = []
            st.rerun()

    # ── Process Question ──
    if ask_button and question.strip():
        with st.spinner("🧠 Analyzing question and searching..."):
            start_time = time.time()
            answer = pipeline.ask(question.strip())
            elapsed = time.time() - start_time

        # ── Display Answer ──
        render_section_header("📌 Answer")

        # Answer card
        render_answer_card(answer.text)

        # Confidence + Source
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(
                render_confidence_badge(answer.confidence),
                unsafe_allow_html=True,
            )
        with col2:
            render_source_chip(answer.source, answer.page)
        with col3:
            st.markdown(
                f'<div style="color: #9AA0A6; font-size: 0.85rem; padding-top: 6px;">'
                f'⏱️ {elapsed:.2f}s</div>',
                unsafe_allow_html=True,
            )

        # ── Supporting Passage ──
        with st.expander("📖 Supporting Passage", expanded=False):
            st.markdown(
                f'<div style="background: #1A1D29; padding: 16px; border-radius: 8px; '
                f'line-height: 1.7; color: #E8EAED;">{answer.supporting_passage}</div>',
                unsafe_allow_html=True,
            )

        # ── NLP Breakdown ──
        with st.expander("🧠 NLP Breakdown", expanded=False):
            # Question type
            st.markdown(f"**Question Type:** `{answer.question_type}`")

            # POS tags of question
            try:
                from nlp.pos_tagger import POSTagger
                pos_tagger = POSTagger()

                tags = pos_tagger.get_detailed_tags(question)
                pos_data = [
                    {"Word": word, "POS": pos, "Tag": tag}
                    for word, pos, tag in tags[:20]
                ]
                st.markdown("**POS Tags:**")
                render_nlp_table(pos_data, ["Word", "POS", "Tag"])

                # Key entities in answer
                if answer.entities:
                    st.markdown("**Entities Found:**")
                    for ent in answer.entities:
                        st.markdown(f"- `{ent}`")

                # Keywords extracted
                keywords = pos_tagger.get_keywords(question)
                if keywords:
                    st.markdown("**Keywords:** " + ", ".join(f"`{kw}`" for kw in keywords))

            except Exception as e:
                st.warning(f"NLP analysis error: {e}")

            # Retrieval scores
            if answer.retrieval_scores:
                st.markdown("**Top Passage Scores:**")
                score_data = [
                    {
                        "Chunk": s["chunk_id"][:12],
                        "Score": f"{s['score']:.4f}",
                        "Dense": f"{s['dense']:.4f}",
                        "BM25": f"{s['sparse']:.4f}",
                        "Rerank": f"{s['rerank']:.4f}",
                    }
                    for s in answer.retrieval_scores
                ]
                render_nlp_table(score_data, ["Chunk", "Score", "Dense", "BM25", "Rerank"])

        # ── Entity Highlighting ──
        with st.expander("🏷️ Entity Highlights", expanded=False):
            try:
                from nlp.ner import NERExtractor
                ner = NERExtractor()
                highlighted = ner.highlight_entities(answer.text)
                st.markdown(highlighted, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Entity highlighting error: {e}")

        # ── Save to Chat History ──
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        st.session_state["chat_history"].append({
            "question": question,
            "answer": answer.text,
            "confidence": answer.confidence,
            "source": answer.source,
            "page": answer.page,
            "question_type": answer.question_type,
            "timestamp": time.strftime("%H:%M:%S"),
        })

        # Save to analytics
        if "query_log" not in st.session_state:
            st.session_state["query_log"] = []

        st.session_state["query_log"].append({
            "question": question,
            "question_type": answer.question_type,
            "confidence": answer.confidence,
            "source": answer.source,
            "entities": answer.entities,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    elif ask_button and not question.strip():
        st.warning("Please enter a question.")

    # ── Chat History ──
    if st.session_state.get("chat_history"):
        st.markdown("---")
        render_section_header("💬 Chat History")

        # Show last 10 Q&A pairs (newest first)
        history = list(reversed(st.session_state["chat_history"][-10:]))
        for item in history:
            render_chat_item(
                question=item["question"],
                answer=item["answer"],
                confidence=item.get("confidence", 0.0),
            )
