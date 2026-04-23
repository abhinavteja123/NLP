# 🧠 IQAS — Intelligent Question Answering System

> **NLP Final Year Project | SRM University AP | B.Tech CSE (AI & ML)**

A production-ready Intelligent Question Answering System that answers user questions from uploaded documents (PDFs, textbooks, lecture notes) using classical NLP + semantic search — featuring sentiment analysis, knowledge graph exploration, and hybrid retrieval.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       STREAMLIT FRONTEND                            │
│  ┌──────────┐ ┌────────┐ ┌───────────┐ ┌─────────┐ ┌───────────┐  │
│  │  Upload  │ │  Q&A   │ │ Analytics │ │Sentiment│ │ Knowledge │  │
│  │  Page    │ │  Page  │ │  Page     │ │ Heatmap │ │   Graph   │  │
│  └──────────┘ └────────┘ └───────────┘ └─────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
     │                │               │               │
     ▼                ▼               ▼               ▼
┌──────────┐   ┌───────────────┐  ┌───────────┐  ┌──────────────┐
│ Document │   │  QA Pipeline  │  │ Sentiment │  │  Knowledge   │
│ Ingestion│   │ Orchestrator  │  │ Analyzer  │  │ Graph Builder│
└──────────┘   └───────────────┘  └───────────┘  └──────────────┘
     │                │                                │
     ▼                ├──────────────────────┐         ▼
┌──────────────┐      ▼                      ▼   ┌───────────────┐
│ Preprocessor │  ┌────────────┐     ┌──────────┐│ Dep. Parsing  │
│ (spaCy)      │  │  Question  │     │Retriever ││ Triple Extract│
│ Tokenize     │  │  Analyzer  │     │(FAISS +  ││ Co-occurrence │
│ POS Tag      │  │  (POS/NER) │     │  BM25)   │└───────────────┘
│ NER          │  └────────────┘     └──────────┘
└──────────────┘        │                  │
     │                  ▼                  ▼
     ▼           ┌────────────┐    ┌──────────────┐
┌──────────────┐ │  Keyword   │    │  Top-K       │
│  Chunker     │ │ Extraction │    │  Passages    │
│ (smart split)│ └────────────┘    └──────────────┘
└──────────────┘        │                  │
     │                  └─────────┬────────┘
     ▼                            ▼
┌──────────────┐        ┌──────────────────┐
│  Embedder    │        │ Answer Extractor  │
│ (Sentence-   │        │ (Passage Ranking  │
│  Transformers│        │  + Synthesis)     │
└──────────────┘        └──────────────────┘
     │                            │
     ▼                            ▼
┌──────────────┐        ┌──────────────────┐
│  FAISS Index │        │  Final Answer    │
│  (stored on  │        │  with Source     │
│   disk)      │        │  Citations       │
└──────────────┘        └──────────────────┘
```

---

## ✨ Features

- **📤 Document Upload**: Support for PDF, TXT, and DOCX files
- **🧠 NLP Pipeline**: Tokenization → POS Tagging → NER → Embedding
- **🔍 Hybrid Retrieval**: Dense (FAISS) + Sparse (BM25) + Reciprocal Rank Fusion
- **🎯 Cross-Encoder Re-ranking**: Fine-grained passage scoring with `ms-marco-MiniLM`
- **📝 Smart Chunking**: 3 strategies — Fixed-size, Sentence-aware, Paragraph-aware
- **💡 Question Analysis**: Auto-detect WHO/WHAT/WHEN/WHERE/WHY/HOW/DEFINE
- **📊 Analytics Dashboard**: Question type distribution, confidence trends, entity frequency
- **🎭 Sentiment & Emotion Heatmap**: Sentence-level emotional arc, word-level polarity coloring, 6-emotion radar chart, entity-level sentiment tracking
- **🌐 Knowledge Graph Explorer**: Automatic entity-relationship extraction, interactive network visualization, triple tables, co-occurrence heatmaps
- **🏷️ Entity Highlighting**: Named entities color-coded in answers
- **💾 Persistent Index**: FAISS index saved to disk for instant loading
- **🐳 Docker Ready**: Containerized deployment with Docker Compose

---

## 🛠️ NLP Techniques

| Technique | Library | Stage | Purpose |
|---|---|---|---|
| Tokenization | spaCy | Preprocessing | Split text into words/sentences |
| POS Tagging | spaCy | Preprocessing | Identify nouns, verbs for keywords |
| NER | spaCy | Preprocessing + Answer | Extract named entities |
| Text Cleaning | Python/unicodedata | Preprocessing | Normalize raw text |
| Sentence Splitting | spaCy | Chunking | Preserve sentence boundaries |
| Sentence Embeddings | Sentence-Transformers | Embedding | Dense vector representations |
| Cosine Similarity | numpy/FAISS | Retrieval | Measure semantic similarity |
| BM25 | rank-bm25 | Retrieval | Sparse keyword matching |
| RRF Fusion | Custom | Retrieval | Combine dense + sparse scores |
| Cross-Encoder Reranking | HuggingFace | Reranking | Fine-grained passage scoring |
| Noun Phrase Extraction | spaCy | Answer | Candidate answer spans |
| Confidence Scoring | Custom | Answer | Quality estimation |
| Lexicon-Based Sentiment | Custom Polarity Lexicon | Sentiment | Word/sentence-level polarity scoring |
| POS-Aware Weighting | spaCy | Sentiment | Adjectives/adverbs weighted higher |
| Negation Handling | Custom | Sentiment | Flip polarity for negated words |
| 6-Emotion Classification | Emotion Keyword Lexicon | Sentiment | Joy, Anger, Sadness, Fear, Surprise, Trust |
| Entity Sentiment Tracking | spaCy NER + Lexicon | Sentiment | Per-entity polarity across sentences |
| Dependency Parsing | spaCy | Knowledge Graph | Subject → Verb → Object triple extraction |
| Co-occurrence Analysis | spaCy | Knowledge Graph | Entity pairs in same sentence |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/abhinavteja123/NLP.git
cd NLP

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Run the application
streamlit run app/main.py
```

The app will open at `http://localhost:8501`.

---

## 📖 How to Use

1. **Upload Documents**: Go to the Upload page and upload PDF/TXT/DOCX files
2. **Build Index**: Select a chunking strategy and click "Build Search Index"
3. **Ask Questions**: Go to the Q&A page and ask questions about your documents
4. **View Analytics**: Check the Analytics page for query insights and trends
5. **Analyze Sentiment**: Use the Sentiment page to explore emotional arcs and word-level polarity heatmaps
6. **Explore Knowledge Graph**: Build interactive entity-relationship graphs from your text

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_preprocessor.py -v
pytest tests/test_retriever.py -v
pytest tests/test_pipeline.py -v
```

---

## 🐳 Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t iqas .
docker run -p 8501:8501 -v ./data:/app/data -v ./models:/app/models iqas
```

---

## 📁 Project Structure

```
NLP/
├── app/
│   ├── main.py                  # Streamlit entrypoint
│   ├── ui/
│   │   ├── components.py        # Reusable UI widgets
│   │   └── styles.css           # Custom CSS
│   └── views/
│       ├── upload.py            # Document upload page
│       ├── qa.py                # Q&A interface page
│       ├── analytics.py         # Query analytics page
│       ├── sentiment.py         # Sentiment & Emotion Heatmap page
│       └── knowledge.py         # Knowledge Graph Explorer page
├── core/
│   ├── document_loader.py       # PDF/TXT/DOCX ingestion
│   ├── indexer.py               # FAISS index builder
│   ├── retriever.py             # Semantic search + BM25 hybrid
│   ├── answer_extractor.py      # Passage ranking + answer synthesis
│   └── pipeline.py              # End-to-end QA pipeline
├── nlp/
│   ├── tokenizer.py             # spaCy tokenization
│   ├── pos_tagger.py            # POS tagging + noun phrases
│   ├── ner.py                   # Named entity recognition
│   ├── embedder.py              # Sentence-Transformers embedding
│   ├── similarity.py            # Cosine similarity utils
│   ├── sentiment_analyzer.py    # Lexicon-based sentiment & emotion analysis
│   └── knowledge_graph.py       # Entity-relationship graph builder
├── utils/
│   ├── chunker.py               # Smart text chunking
│   ├── cleaner.py               # Text normalization
│   ├── logger.py                # Logging setup
│   └── config.py                # Global configuration
├── data/
│   ├── uploads/                 # User-uploaded documents
│   ├── processed/               # Processed text
│   └── sample_docs/             # Sample demo data
├── models/
│   ├── faiss_index/             # Saved FAISS indexes
│   └── embeddings_cache/        # Cached embeddings
├── tests/
│   ├── test_preprocessor.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🎭 Sentiment & Emotion Heatmap

The Sentiment page provides multi-level emotional analysis:

- **Overall Polarity & Subjectivity**: Document-wide sentiment score with confidence label
- **Emotional Arc Chart**: Sentence-by-sentence polarity plotted as a spline graph
- **Word-Level Heatmap**: Every word color-coded (green = positive, red = negative, gray = neutral)
- **6-Emotion Radar Chart**: Joy, Anger, Sadness, Fear, Surprise, Trust — normalized breakdown
- **Entity Sentiment Tracker**: Named entities tracked with their average sentiment across mentions
- **Key Sentences**: Most positive and most negative sentences highlighted

Works with custom text, sample texts, or your uploaded documents.

---

## 🌐 Knowledge Graph Explorer

The Knowledge Graph page automatically discovers relationships in text:

- **Triple Extraction**: Subject → Relation → Object triples via dependency parsing
- **Interactive Network Graph**: Plotly-powered entity network with hover details
- **Entity Type Filtering**: Filter nodes by type (PERSON, ORG, GPE, etc.)
- **Entity Frequency Chart**: Horizontal bar chart of most mentioned entities
- **Co-occurrence Heatmap**: Which entities appear together in the same sentences
- **Source Sentences**: View the original text behind each extracted relationship

Powered by spaCy's dependency parser and NER — no external API required.

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Answers are extractive (not generative) — limited to text found in documents
- Single-language support (English only via `en_core_web_sm`)
- CPU-only inference (no GPU acceleration)
- No authentication or multi-user support

### Future Enhancements
- [ ] Add generative answers using LLM (GPT, Llama)
- [ ] Multi-language support with multilingual models
- [ ] GPU acceleration for faster embedding
- [ ] User authentication and document management
- [ ] API endpoint for programmatic access
- [ ] Fine-tuning on domain-specific data
- [ ] Conversational follow-up questions

---

## 🔧 Tools & Versions

| Tool | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| spaCy | 3.8+ | Tokenization, POS, NER, Dependency Parsing |
| Sentence-Transformers | 5.x | Semantic embeddings |
| FAISS-CPU | 1.13+ | Vector index + search |
| rank-bm25 | 0.2.2 | Sparse retrieval |
| PyMuPDF | 1.27+ | PDF text extraction |
| python-docx | 1.2+ | DOCX parsing |
| HuggingFace Transformers | 5.x | Cross-encoder reranking |
| Streamlit | 1.56+ | Web frontend |
| Plotly | 5.19+ | Analytics & graph visualization |

---

*Built for NLP Final Year Project — SRM University AP | B.Tech CSE (AI & ML)*
