# 🧠 IQAS — Intelligent Question Answering System

> **NLP Final Year Project | SRM University AP | B.Tech CSE (AI & ML)**

A production-ready Intelligent Question Answering System that answers user questions from uploaded documents (PDFs, textbooks, lecture notes) using classical NLP + semantic search.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND                     │
│  ┌──────────┐   ┌───────────┐   ┌────────────────────┐  │
│  │  Upload  │   │  Q&A UI   │   │ Analytics Dashboard │  │
│  │  Page    │   │  Page     │   │  Page               │  │
│  └──────────┘   └───────────┘   └────────────────────┘  │
└─────────────────────────────────────────────────────────┘
     │                    │
     ▼                    ▼
┌──────────┐      ┌───────────────┐
│ Document │      │  QA Pipeline  │
│ Ingestion│      │ Orchestrator  │
└──────────┘      └───────────────┘
     │                    │
     ▼                    ├──────────────────────┐
┌──────────────┐          ▼                      ▼
│ Preprocessor │   ┌────────────┐        ┌──────────────┐
│ (spaCy)      │   │  Question  │        │   Retriever  │
│ Tokenize     │   │  Analyzer  │        │  (FAISS +    │
│ POS Tag      │   │  (POS/NER) │        │   BM25)      │
│ NER          │   └────────────┘        └──────────────┘
└──────────────┘          │                      │
     │                    ▼                      ▼
     ▼             ┌────────────┐        ┌──────────────┐
┌──────────────┐   │  Keyword   │        │  Top-K       │
│  Chunker     │   │ Extraction │        │  Passages    │
│ (smart split)│   └────────────┘        └──────────────┘
└──────────────┘          │                      │
     │                    └──────────┬───────────┘
     ▼                               ▼
┌──────────────┐            ┌──────────────────┐
│  Embedder    │            │ Answer Extractor  │
│ (Sentence-   │            │ (Passage Ranking  │
│  Transformers│            │  + Synthesis)     │
└──────────────┘            └──────────────────┘
     │                               │
     ▼                               ▼
┌──────────────┐            ┌──────────────────┐
│  FAISS Index │            │  Final Answer    │
│  (stored on  │            │  with Source     │
│   disk)      │            │  Citations       │
└──────────────┘            └──────────────────┘
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

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone/navigate to the project
cd iqas

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

# In your terminal (Ctrl+C first to stop), then:
cd "c:\Users\ABHINAV TEJA\Downloads\NLP\iqas"
.\venv\Scripts\activate
streamlit run app/main.py

---

## 📁 Project Structure

```
iqas/
├── app/
│   ├── main.py                  # Streamlit entrypoint
│   ├── ui/
│   │   ├── components.py        # Reusable UI widgets
│   │   └── styles.css           # Custom CSS
│   └── pages/
│       ├── upload.py            # Document upload page
│       ├── qa.py                # Q&A interface page
│       └── analytics.py         # Query analytics page
├── core/
│   ├── document_loader.py       # PDF/TXT/DOCX ingestion
│   ├── preprocessor.py          # Tokenization, POS, NER, cleaning
│   ├── indexer.py               # FAISS index builder
│   ├── retriever.py             # Semantic search + BM25 hybrid
│   ├── answer_extractor.py      # Passage ranking + answer synthesis
│   └── pipeline.py              # End-to-end QA pipeline
├── nlp/
│   ├── tokenizer.py             # spaCy tokenization
│   ├── pos_tagger.py            # POS tagging + noun phrases
│   ├── ner.py                   # Named entity recognition
│   ├── embedder.py              # Sentence-Transformers embedding
│   └── similarity.py            # Cosine similarity utils
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
├── Dockerfile
├── docker-compose.yml
└── README.md
```

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
- [ ] Knowledge graph integration
- [ ] Conversational follow-up questions

---

## 🔧 Tools & Versions

| Tool | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| spaCy | 3.7.4 | Tokenization, POS, NER |
| Sentence-Transformers | 2.6.1 | Semantic embeddings |
| FAISS-CPU | 1.7.4 | Vector index + search |
| rank-bm25 | 0.2.2 | Sparse retrieval |
| PyMuPDF | 1.23.26 | PDF text extraction |
| python-docx | 1.1.0 | DOCX parsing |
| HuggingFace Transformers | 4.38.2 | Cross-encoder reranking |
| Streamlit | 1.32.0 | Web frontend |
| Plotly | 5.19.0 | Analytics charts |

---

*Built for NLP Final Year Project — SRM University AP | B.Tech CSE (AI & ML)*
