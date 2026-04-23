"""
IQAS Sentiment & Emotion Heatmap Page
=======================================
Interactive emotional analysis with arc visualization, word heatmap,
emotion wheel, and entity-level sentiment tracking.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.components import render_section_header, render_metrics_row


# ── Sample texts for demo ──
SAMPLE_TEXTS = {
    "🎬 Movie Review (Mixed)": (
        "The movie started brilliantly with outstanding performances from the lead actors. "
        "The cinematography was absolutely beautiful and the soundtrack was wonderful. "
        "However, the second half was a complete disaster. The plot became confusing and boring. "
        "The villain was terribly written and felt pathetic. Despite the disappointing ending, "
        "the first half alone makes it worth watching. The director showed incredible talent "
        "but failed to maintain the momentum throughout."
    ),
    "📰 News Article (Neutral-Negative)": (
        "The government announced new regulations for the technology sector today. "
        "Industry leaders expressed concern about the potential impact on innovation. "
        "Several companies warned that the strict rules could damage economic growth. "
        "However, consumer advocacy groups praised the decision as a necessary protection. "
        "The new policies aim to address growing fears about data privacy and security risks. "
        "Critics argue the approach is too aggressive while supporters call it responsible."
    ),
    "🔬 Scientific Text (Positive)": (
        "Natural Language Processing has achieved remarkable progress in recent years. "
        "Transformers have proven to be incredibly effective for various NLP tasks. "
        "The introduction of BERT was a groundbreaking advancement that improved accuracy "
        "across multiple benchmarks. Researchers are optimistic about future developments. "
        "Word embeddings provide powerful representations that capture semantic relationships. "
        "These innovations have successfully enabled many useful real-world applications."
    ),
}


def render_sentiment_page():
    """Render the Sentiment & Emotion Heatmap page."""

    # ── Page Header ──
    st.markdown(
        '<div class="feature-hero">'
        '<div class="feature-hero-icon">🎭</div>'
        '<h1 class="feature-hero-title">Sentiment & Emotion Heatmap</h1>'
        '<p class="feature-hero-subtitle">'
        'Map the emotional landscape of any text — word-by-word coloring, '
        'emotional arc visualization, and entity-level sentiment tracking.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # ── Text Input ──
    st.markdown("---")
    render_section_header("📝 Input Text")

    # Build source options from uploaded documents
    source_options = ["Custom Text"] + list(SAMPLE_TEXTS.keys())
    uploaded_docs = {}

    pipeline = st.session_state.get("pipeline")
    if pipeline and pipeline._is_indexed:
        chunks = pipeline._chunks_data
        if chunks:
            # Group chunks by source document
            for chunk in chunks:
                src = chunk.get("source", "Unknown")
                if src not in uploaded_docs:
                    uploaded_docs[src] = []
                uploaded_docs[src].append(chunk.get("text", ""))

            for doc_name in uploaded_docs:
                source_options.insert(1, f"📚 {doc_name}")

    col1, col2 = st.columns([1, 2])
    with col1:
        sample_choice = st.selectbox(
            "Select text source:",
            source_options,
            key="sentiment_sample",
        )

    if sample_choice.startswith("📚 "):
        # Pull text from uploaded document
        doc_name = sample_choice.replace("📚 ", "")
        doc_chunks = uploaded_docs.get(doc_name, [])
        input_text = " ".join(doc_chunks)

        st.markdown(
            f'<div style="background: rgba(0,180,216,0.08); border: 1px solid rgba(0,180,216,0.2); '
            f'border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;">'
            f'<span style="color: #00B4D8; font-weight: 600;">📚 Using uploaded document:</span> '
            f'<span style="color: #E8EAED;">{doc_name}</span> '
            f'<span style="color: #9AA0A6;">({len(doc_chunks)} chunks, {len(input_text):,} chars)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.text_area(
            "Document text (auto-loaded):",
            value=input_text[:3000] + ("..." if len(input_text) > 3000 else ""),
            height=180,
            key="sentiment_text_display",
            disabled=True,
        )
    elif sample_choice != "Custom Text":
        input_text = SAMPLE_TEXTS[sample_choice]
        st.text_area(
            "Text to analyze:",
            value=input_text,
            height=180,
            key="sentiment_text_display2",
            disabled=True,
        )
    else:
        input_text = st.text_area(
            "Paste your text here:",
            placeholder="Enter or paste any text to analyze its sentiment and emotions...",
            height=180,
            key="sentiment_text_input",
        )

    # ── Analyze Button ──
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_btn = st.button(
            "🎭 Analyze Sentiment",
            key="analyze_sentiment",
            use_container_width=True,
        )

    if not analyze_btn or not input_text or not input_text.strip():
        if analyze_btn:
            st.warning("Please enter some text to analyze.")
        return

    # ── Run Analysis ──
    with st.spinner("🧠 Analyzing sentiment and emotions..."):
        try:
            from nlp.sentiment_analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            result = analyzer.analyze(input_text.strip())
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # ── Summary Metrics ──
    st.markdown("---")
    render_section_header("📊 Sentiment Overview")

    # Overall label with styled card
    polarity = result.overall_polarity
    if polarity > 0.15:
        label_color = "#2ECC71"
        label_emoji = "😊"
        label_bg = "rgba(46, 204, 113, 0.12)"
    elif polarity < -0.15:
        label_color = "#E74C3C"
        label_emoji = "😟"
        label_bg = "rgba(231, 76, 60, 0.12)"
    else:
        label_color = "#F39C12"
        label_emoji = "😐"
        label_bg = "rgba(243, 156, 18, 0.12)"

    st.markdown(
        f'<div style="background: {label_bg}; border: 1px solid {label_color}33; '
        f'border-radius: 16px; padding: 24px 32px; text-align: center; margin-bottom: 20px;">'
        f'<span style="font-size: 3rem;">{label_emoji}</span>'
        f'<h2 style="color: {label_color}; margin: 8px 0 4px 0;">{result.overall_label}</h2>'
        f'<p style="color: #9AA0A6;">Polarity: {polarity:+.3f} · '
        f'Subjectivity: {result.overall_subjectivity:.1%}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    render_metrics_row([
        {"value": f"{polarity:+.3f}", "label": "Polarity Score"},
        {"value": f"{result.overall_subjectivity:.1%}", "label": "Subjectivity"},
        {"value": str(len(result.sentence_sentiments)), "label": "Sentences"},
        {"value": result.emotion_breakdown.dominant, "label": "Dominant Emotion"},
    ])

    # ── Emotional Arc Chart ──
    st.markdown("---")
    render_section_header("📈 Emotional Arc")
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 0.9rem; margin-top: -8px;">'
        'How sentiment flows through your text, sentence by sentence.</p>',
        unsafe_allow_html=True,
    )

    if result.sentence_sentiments:
        try:
            import plotly.graph_objects as go

            sent_indices = [s.index + 1 for s in result.sentence_sentiments]
            sent_polarities = [s.polarity for s in result.sentence_sentiments]
            sent_texts = [s.text[:80] + "..." if len(s.text) > 80 else s.text
                          for s in result.sentence_sentiments]
            sent_emotions = [s.emotion for s in result.sentence_sentiments]

            # Color based on polarity
            colors = []
            for p in sent_polarities:
                if p > 0.1:
                    colors.append(f"rgba(46, 204, 113, {min(0.4 + abs(p), 1.0)})")
                elif p < -0.1:
                    colors.append(f"rgba(231, 76, 60, {min(0.4 + abs(p), 1.0)})")
                else:
                    colors.append("rgba(243, 156, 18, 0.6)")

            fig = go.Figure()

            # Area fill
            fig.add_trace(go.Scatter(
                x=sent_indices, y=sent_polarities,
                fill="tozeroy",
                fillcolor="rgba(0, 180, 216, 0.08)",
                line=dict(color="#00B4D8", width=3, shape="spline"),
                mode="lines+markers",
                marker=dict(size=10, color=colors, line=dict(width=2, color="#1A1D29")),
                hovertemplate=(
                    "<b>Sentence %{x}</b><br>"
                    "Polarity: %{y:.3f}<br>"
                    "%{customdata[0]}<br>"
                    "Emotion: %{customdata[1]}"
                    "<extra></extra>"
                ),
                customdata=list(zip(sent_texts, sent_emotions)),
            ))

            # Threshold lines
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig.add_hline(y=0.15, line_dash="dot", line_color="rgba(46,204,113,0.3)",
                          annotation_text="Positive", annotation_font_color="#2ECC71")
            fig.add_hline(y=-0.15, line_dash="dot", line_color="rgba(231,76,60,0.3)",
                          annotation_text="Negative", annotation_font_color="#E74C3C")

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAED",
                xaxis=dict(
                    title="Sentence #",
                    gridcolor="rgba(255,255,255,0.05)",
                    showline=True, linecolor="rgba(255,255,255,0.1)",
                ),
                yaxis=dict(
                    title="Sentiment Polarity",
                    range=[-0.6, 0.6],
                    gridcolor="rgba(255,255,255,0.05)",
                    showline=True, linecolor="rgba(255,255,255,0.1)",
                    zeroline=False,
                ),
                margin=dict(t=20, b=40, l=50, r=20),
                height=350,
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            import pandas as pd
            df = pd.DataFrame({
                "Sentence #": [s.index + 1 for s in result.sentence_sentiments],
                "Polarity": [s.polarity for s in result.sentence_sentiments],
            })
            st.line_chart(df.set_index("Sentence #"))

    # ── Word-Level Sentiment Heatmap ──
    st.markdown("---")
    render_section_header("🌡️ Word-Level Sentiment Heatmap")
    st.markdown(
        '<p style="color: #9AA0A6; font-size: 0.9rem; margin-top: -8px;">'
        '<span style="color: #2ECC71;">■ Green = Positive</span> · '
        '<span style="color: #B2BEC3;">■ Gray = Neutral</span> · '
        '<span style="color: #E74C3C;">■ Red = Negative</span></p>',
        unsafe_allow_html=True,
    )

    try:
        from nlp.sentiment_analyzer import SentimentAnalyzer
        heatmap_html = analyzer.render_word_heatmap_html(input_text.strip())
        st.markdown(heatmap_html, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Word heatmap error: {e}")

    # ── Emotion Breakdown ──
    st.markdown("---")
    render_section_header("🎨 Emotion Breakdown")

    emotions = result.emotion_breakdown.to_dict()

    col1, col2 = st.columns([1, 1])

    with col1:
        try:
            import plotly.graph_objects as go

            emotion_labels = list(emotions.keys())
            emotion_values = list(emotions.values())
            emotion_colors = ["#2ECC71", "#E74C3C", "#3498DB", "#9B59B6", "#F39C12", "#00B4D8"]

            fig = go.Figure(data=go.Scatterpolar(
                r=emotion_values + [emotion_values[0]],
                theta=emotion_labels + [emotion_labels[0]],
                fill="toself",
                fillcolor="rgba(0, 180, 216, 0.15)",
                line=dict(color="#00B4D8", width=2),
                marker=dict(size=8, color=emotion_colors),
            ))

            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True, range=[0, max(emotion_values) + 0.1],
                        gridcolor="rgba(255,255,255,0.08)",
                        tickfont=dict(color="#9AA0A6"),
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(255,255,255,0.08)",
                        tickfont=dict(color="#E8EAED", size=13),
                    ),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAED",
                margin=dict(t=40, b=40, l=60, r=60),
                height=350,
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            for emo, val in emotions.items():
                st.write(f"{emo}: {val:.2%}")

    with col2:
        # Emotion cards
        emotion_emojis = {
            "Joy": "😊", "Anger": "😡", "Sadness": "😢",
            "Fear": "😨", "Surprise": "😲", "Trust": "🤝",
        }
        emotion_colors_map = {
            "Joy": "#2ECC71", "Anger": "#E74C3C", "Sadness": "#3498DB",
            "Fear": "#9B59B6", "Surprise": "#F39C12", "Trust": "#00B4D8",
        }

        for emo, val in emotions.items():
            emoji = emotion_emojis.get(emo, "❓")
            color = emotion_colors_map.get(emo, "#B2BEC3")
            bar_width = max(val * 100, 2)

            st.markdown(
                f'<div style="display: flex; align-items: center; gap: 12px; '
                f'margin-bottom: 10px; padding: 8px 12px; background: #1A1D29; '
                f'border-radius: 8px;">'
                f'<span style="font-size: 1.4rem; min-width: 32px;">{emoji}</span>'
                f'<div style="flex: 1;">'
                f'<div style="color: #E8EAED; font-size: 0.85rem; font-weight: 600; '
                f'margin-bottom: 4px;">{emo}</div>'
                f'<div style="background: rgba(255,255,255,0.06); border-radius: 6px; '
                f'height: 8px; overflow: hidden;">'
                f'<div style="background: {color}; width: {bar_width}%; height: 100%; '
                f'border-radius: 6px; transition: width 0.5s ease;"></div>'
                f'</div></div>'
                f'<span style="color: {color}; font-weight: 700; font-size: 0.9rem; '
                f'min-width: 45px; text-align: right;">{val:.0%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Most Positive / Negative Sentences ──
    st.markdown("---")
    render_section_header("💬 Key Sentences")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div style="background: rgba(46,204,113,0.08); border-left: 4px solid #2ECC71; '
            f'padding: 16px 20px; border-radius: 0 12px 12px 0;">'
            f'<div style="color: #2ECC71; font-weight: 700; font-size: 0.8rem; '
            f'text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">'
            f'😊 Most Positive</div>'
            f'<div style="color: #E8EAED; line-height: 1.6;">'
            f'{result.most_positive_sentence}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="background: rgba(231,76,60,0.08); border-left: 4px solid #E74C3C; '
            f'padding: 16px 20px; border-radius: 0 12px 12px 0;">'
            f'<div style="color: #E74C3C; font-weight: 700; font-size: 0.8rem; '
            f'text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">'
            f'😟 Most Negative</div>'
            f'<div style="color: #E8EAED; line-height: 1.6;">'
            f'{result.most_negative_sentence}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Entity Sentiments ──
    if result.entity_sentiments:
        st.markdown("---")
        render_section_header("🏷️ Entity Sentiment Tracker")
        st.markdown(
            '<p style="color: #9AA0A6; font-size: 0.9rem; margin-top: -8px;">'
            'How each named entity is talked about in the text.</p>',
            unsafe_allow_html=True,
        )

        for ent_sent in result.entity_sentiments[:10]:
            if ent_sent.label == "Positive":
                dot = "🟢"
                color = "#2ECC71"
            elif ent_sent.label == "Negative":
                dot = "🔴"
                color = "#E74C3C"
            else:
                dot = "⚪"
                color = "#F39C12"

            bar_val = (ent_sent.avg_polarity + 1) / 2 * 100  # -1..1 → 0..100

            st.markdown(
                f'<div style="display: flex; align-items: center; gap: 12px; '
                f'padding: 10px 16px; background: #1A1D29; border-radius: 10px; '
                f'margin-bottom: 6px; border: 1px solid rgba(255,255,255,0.05);">'
                f'<span>{dot}</span>'
                f'<span style="color: #E8EAED; font-weight: 600; min-width: 120px;">'
                f'{ent_sent.entity}</span>'
                f'<span style="color: #9AA0A6; font-size: 0.75rem; min-width: 60px;">'
                f'{ent_sent.entity_type}</span>'
                f'<div style="flex: 1; background: rgba(255,255,255,0.06); '
                f'border-radius: 6px; height: 6px; overflow: hidden;">'
                f'<div style="background: linear-gradient(90deg, #E74C3C, #F39C12, #2ECC71); '
                f'width: 100%; height: 100%; border-radius: 6px; position: relative;">'
                f'<div style="position: absolute; left: {bar_val}%; top: -3px; '
                f'width: 12px; height: 12px; background: {color}; border-radius: 50%; '
                f'border: 2px solid #1A1D29; transform: translateX(-50%);"></div>'
                f'</div></div>'
                f'<span style="color: {color}; font-weight: 700; font-size: 0.85rem; '
                f'min-width: 55px; text-align: right;">{ent_sent.avg_polarity:+.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Sentence Details Table ──
    with st.expander("📋 Detailed Sentence Analysis", expanded=False):
        if result.sentence_sentiments:
            import pandas as pd
            df = pd.DataFrame([
                {
                    "#": s.index + 1,
                    "Sentence": s.text[:80] + ("..." if len(s.text) > 80 else ""),
                    "Polarity": f"{s.polarity:+.3f}",
                    "Subjectivity": f"{s.subjectivity:.1%}",
                    "Emotion": s.emotion,
                    "Words": s.word_count,
                }
                for s in result.sentence_sentiments
            ])
            st.dataframe(df, hide_index=True)
