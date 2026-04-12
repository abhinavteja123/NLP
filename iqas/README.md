<div align="center">

# 🧠 IQAS — Intelligent Question Answering System

**NLP Final Year Project · SRM University AP · B.Tech CSE (AI & ML)**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-blue)](https://github.com/facebookresearch/faiss)
[![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5?logo=spacy&logoColor=white)](https://spacy.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-32%2F32%20Passing-2ECC71)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

*A production-ready system that answers questions from your uploaded documents using classical NLP, semantic search, and hybrid retrieval.*

</div>

---

## 📖 Table of Contents

- [What is IQAS?](#-what-is-iqas)
- [Live Demo](#-live-demo)
- [Architecture](#️-architecture)
- [NLP Pipeline](#-nlp-pipeline)
- [Features](#-features)
- [Quick Start (Local)](#-quick-start-local)
- [Docker Deployment](#-docker-deployment)
- [Project Structure](#-project-structure)
- [Configuration](#️-configuration)
- [Running Tests](#-running-tests)
- [Storage & Data](#-storage--data)
- [NLP Techniques](#-nlp-techniques)
- [Limitations & Roadmap](#️-limitations--roadmap)

---

## 🤔 What is IQAS?

IQAS is an **Intelligent Question Answering System** that reads your documents (PDFs, textbooks, research papers, lecture notes) and answers questions about them. It uses:

- **Classical NLP** (tokenization, POS tagging, NER via spaCy)  
- **Semantic search** (MiniLM sentence embeddings + FAISS vector index)  
- **Keyword search** (BM25 Okapi sparse retrieval)  
- **Hybrid ranking** (Reciprocal Rank Fusion + cross-encoder re-ranking)  

All running **100% offline on CPU** — no GPU, no API keys, no internet needed after setup.

---

## 🎬 Live Demo

> Upload a PDF → Ask a question → Get an answer with source citation

```
📤 Upload any PDF/TXT/DOCX
         ↓
🔍 System builds a searchable vector index
         ↓
💬 Type: "What is tokenization?"
         ↓
📌 Answer: "Tokenization is the process of breaking down text 
           into smaller units called tokens."
           ⚡ Source: nlp_notes.txt · Page 1 · Confidence: 85%
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              STREAMLIT FRONTEND                  │
│  📤 Upload   │   💬 Q&A   │   📊 Analytics      │
└──────────────┴────────────┴─────────────────────┘
        │                    │
        ▼                    ▼
  Document Loader       QA Pipeline Orchestrator
  PDF/TXT/DOCX               │
        │              ┌─────┴──────────────────┐
        ▼              ▼                        ▼
   TextCleaner    Question Analyzer      HybridRetriever
   TextChunker    (POS/NER/Type)              │
        │                             ┌────────┴────────┐
        ▼                             ▼                  ▼
   TextEmbedder                 FAISS Search         BM25 Search
   (MiniLM-L6-v2)               Dense 384-d         Sparse TF-IDF
        │                             └────────┬────────┘
        ▼                                      ▼
   FAISSIndexer                        RRF Fusion
   (FlatIP/IVF)                              │
        │                                    ▼
   index.faiss                     Cross-Encoder Rerank
   metadata.json                   (ms-marco-MiniLM)
                                         │
                                         ▼
                                  AnswerExtractor
                                  text + citation + confidence
```

---

## 🧠 NLP Pipeline

When you upload a document:

```
Raw PDF/TXT/DOCX
   → TextCleaner     (normalize unicode, fix PDF artifacts, remove noise)
   → TextChunker     (split into overlapping passages, 3 strategies)
   → TextEmbedder    (encode each chunk → 384-dim MiniLM vector)
   → FAISSIndexer    (build cosine-similarity index, save to disk)
```

When you ask a question:

```
"What is tokenization?"
   → POSTagger          (detect: DEFINE question, focus: "tokenization")
   → Dense Search       (FAISS inner-product similarity, top-20)
   → Sparse Search      (BM25 Okapi keyword match, top-20)
   → RRF Fusion         (combine rankings: score = Σ 1/(60 + rank))
   → Cross-Encoder      (ms-marco-MiniLM re-ranks top-10 pairs)
   → AnswerExtractor    (sentence selection + NER + confidence scoring)
   → Answer + Citation  (text, source file, page number, confidence %)
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📤 **Multi-format Upload** | PDF, TXT, DOCX (up to 200 MB) |
| 🧩 **Smart Chunking** | 3 strategies: Fixed-size, Sentence-aware, Paragraph-aware |
| 🔢 **Dense Retrieval** | `all-MiniLM-L6-v2` → 384-dim FAISS FlatIP index |
| 🔑 **Sparse Retrieval** | BM25 Okapi for rare keyword matching |
| ⚡ **RRF Fusion** | Merges dense + sparse without score calibration |
| 🎯 **Cross-Encoder Rerank** | `ms-marco-MiniLM-L-6-v2` for fine-grained scoring |
| 🏷️ **Question Type Detection** | WHO / WHAT / WHEN / WHERE / WHY / HOW / DEFINE |
| 🧾 **Source Citations** | Every answer shows filename + page number |
| 📊 **Confidence Scoring** | High / Medium / Low badge with percentage |
| 🔴 **Named Entity Highlighting** | PERSON, ORG, DATE, LOC color-coded in answers |
| 📈 **Analytics Dashboard** | Question type charts, confidence trends, entity frequency |
| 💾 **Persistent Index** | FAISS index saved to disk — loads instantly on restart |
| 🐳 **Docker Ready** | One-command containerised deployment |
| ✅ **Tested** | 32/32 pytest tests passing |

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```powershell
# 1. Navigate to project directory
cd "c:\Users\ABHINAV TEJA\Downloads\NLP\iqas"

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy English model
python -m spacy download en_core_web_sm

# 5. Launch the app
streamlit run app/main.py
```

**App opens at:** `http://localhost:8501`

### First-time usage

1. Go to **📤 Upload** → click **"📖 Load Sample NLP Notes"** to index demo data instantly
2. Switch to **💬 Q&A** → ask any question about the document
3. Check **📊 Analytics** to see your query history and confidence trends

---

## 🐳 Docker Deployment

### Requirements
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Option A — Docker Compose (Recommended)

```bash
# Build and start (first run takes ~5 minutes to download models)
docker-compose up --build

# Run in background
docker-compose up --build -d

# Stop
docker-compose down

# Stop and remove all data volumes (full reset)
docker-compose down -v
```

**App available at:** `http://localhost:8501`

### Option B — Manual Docker

```bash
# Build the image
docker build -t iqas:latest .

# Run with persistent storage
docker run -d \
  --name iqas_app \
  -p 8501:8501 \
  -v iqas_data:/app/data \
  -v iqas_models:/app/models \
  iqas:latest

# View logs
docker logs -f iqas_app

# Stop
docker stop iqas_app && docker rm iqas_app
```

### Docker Notes

| Item | Detail |
|---|---|
| **Base image** | `python:3.10-slim` (~150 MB) |
| **Final image size** | ~3.5 GB (includes PyTorch + all NLP models) |
| **First startup** | ~60s (downloads MiniLM + cross-encoder from HuggingFace) |
| **Subsequent startups** | ~5s (models cached in container layers) |
| **Data persistence** | Uploads and FAISS index stored in Docker named volumes |
| **Health check** | `GET /api/health` every 30s, 60s grace period |

---

## 📁 Project Structure

```
iqas/
│
├── app/                         # Streamlit frontend
│   ├── main.py                  # App entrypoint, routing, session state
│   ├── ui/
│   │   ├── components.py        # Reusable widgets (cards, badges, progress)
│   │   └── styles.css           # Dark theme CSS
│   └── views/
│       ├── upload.py            # Document upload + index builder
│       ├── qa.py                # Question input, answer display, NLP breakdown
│       └── analytics.py         # Plotly charts, CSV export
│
├── core/                        # Pipeline layer
│   ├── document_loader.py       # PDF/TXT/DOCX → text + metadata
│   ├── indexer.py               # FAISS FlatIP/IVF builder + save/load
│   ├── retriever.py             # Dense + BM25 + RRF + cross-encoder
│   ├── answer_extractor.py      # Sentence selection + confidence scoring
│   └── pipeline.py              # End-to-end orchestrator
│
├── nlp/                         # NLP layer
│   ├── tokenizer.py             # spaCy tokenization + lemmatization
│   ├── pos_tagger.py            # POS tags, noun phrases, question detection
│   ├── ner.py                   # Named entity extraction + HTML highlighting
│   ├── embedder.py              # Sentence-Transformers MiniLM encoding
│   └── similarity.py            # Cosine similarity utilities
│
├── utils/                       # Foundation layer
│   ├── config.py                # All paths, models, hyperparameters
│   ├── logger.py                # loguru structured logging
│   ├── cleaner.py               # Text normalization + PDF artifact removal
│   └── chunker.py               # 3-strategy smart chunking
│
├── data/
│   ├── uploads/                 # User-uploaded documents
│   ├── processed/               # Cleaned text (intermediate)
│   └── sample_docs/
│       └── sample_nlp_notes.txt # Built-in demo document
│
├── models/
│   ├── faiss_index/
│   │   ├── index.faiss          # ← VECTOR DATABASE (binary)
│   │   └── metadata.json        # ← CHUNK TEXT + CITATIONS (JSON)
│   └── embeddings_cache/
│       └── embeddings.npy       # Cached numpy embeddings array
│
├── tests/
│   ├── test_preprocessor.py     # 19 NLP unit tests
│   ├── test_retriever.py        # 6 retrieval tests
│   ├── test_pipeline.py         # 7 end-to-end integration tests
│   └── fixtures/
│       └── sample_qa.json       # Test question-answer pairs
│
├── .streamlit/
│   └── config.toml              # fileWatcherType=none (suppresses noise)
│
├── Dockerfile                   # Production container definition
├── docker-compose.yml           # Multi-service orchestration
├── .dockerignore                # Excludes venv, tests, local data
├── requirements.txt             # Python dependencies
├── setup.py                     # Package metadata
├── .env.example                 # Environment variable template
└── README.md                    # This file
```

---

## ⚙️ Configuration

Copy the template and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `IQAS_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `IQAS_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking model |
| `IQAS_CHUNK_SIZE` | `512` | Max tokens per chunk |
| `IQAS_CHUNK_OVERLAP` | `50` | Overlap tokens between chunks |
| `IQAS_TOP_K_RETRIEVE` | `20` | Candidates retrieved before reranking |
| `IQAS_TOP_K_RERANK` | `5` | Final passages after reranking |
| `IQAS_LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG/INFO/WARNING) |

> **For local demo:** No changes needed. Defaults are well-tuned for academic use.

---

## 🧪 Running Tests

```powershell
# Activate venv first
.\venv\Scripts\activate

# Run all 32 tests
pytest tests/ -v

# Run by module
pytest tests/test_preprocessor.py -v   # NLP unit tests (19)
pytest tests/test_retriever.py -v      # Retrieval tests (6)
pytest tests/test_pipeline.py -v       # End-to-end tests (7)

# Run with coverage report
pytest tests/ -v --tb=short
```

**Expected output:**
```
32 passed in ~112s (0:01:52)
```

---

## 💾 Storage & Data

IQAS uses **file-based vector storage** — no database server required.

| File | Location | Contents |
|---|---|---|
| Vector index | `models/faiss_index/index.faiss` | Binary FAISS FlatIP index |
| Chunk metadata | `models/faiss_index/metadata.json` | Text + source + page per vector |
| Embedding cache | `models/embeddings_cache/embeddings.npy` | Cached numpy float32 array |
| App logs | `logs/iqas.log` | Rotating log with loguru |

The index **auto-loads on startup** — documents you indexed previously are available immediately without re-uploading.

---

## 📚 NLP Techniques

| Technique | Library | Purpose |
|---|---|---|
| Tokenization | spaCy `en_core_web_sm` | Split text into words/sentences |
| Lemmatization | spaCy | Normalize word forms (running→run) |
| POS Tagging | spaCy | Identify nouns/verbs for keyword extraction |
| NER | spaCy | Extract PERSON, ORG, DATE, LOC entities |
| Question Type Detection | Custom (spaCy) | WHO/WHAT/WHEN/WHERE/WHY/HOW/DEFINE |
| Sentence Embeddings | `sentence-transformers` | 384-dim semantic vectors |
| Cosine Similarity | FAISS (inner product) | Semantic document-query matching |
| BM25 Okapi | `rank-bm25` | Sparse keyword frequency scoring |
| Reciprocal Rank Fusion | Custom | Combine dense + sparse without calibration |
| Cross-Encoder Reranking | HuggingFace | Joint query-passage relevance scoring |
| Confidence Scoring | Custom formula | Answer quality estimation (0–1) |

---

## ⚠️ Limitations & Roadmap

### Current Limitations
- **Extractive only** — answers come directly from document text (no generative synthesis)
- **English only** — uses `en_core_web_sm` (English spaCy model)
- **CPU only** — no GPU acceleration (inference takes 5–15s per query)
- **Single user** — no authentication or multi-user isolation

### Future Enhancements
- [ ] Generative answers via local LLM (Ollama + Llama 3.2)
- [ ] Multi-language support (multilingual-e5-large-instruct)
- [ ] GPU acceleration (update `device="cpu"` in `embedder.py`)
- [ ] REST API endpoint (FastAPI wrapper around `QAPipeline`)
- [ ] User authentication (Streamlit-Authenticator)
- [ ] Conversational memory (follow-up questions)
- [ ] Fine-tuning on domain-specific corpora

---

## 🔧 Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| spaCy | ≥3.7.0 | Tokenization, POS, NER |
| sentence-transformers | ≥2.6.0 | Semantic embeddings |
| FAISS-CPU | ≥1.7.4 | Vector index + similarity search |
| rank-bm25 | ≥0.2.2 | Sparse keyword retrieval |
| PyMuPDF | ≥1.23.0 | PDF text extraction |
| python-docx | ≥1.1.0 | DOCX parsing |
| HuggingFace Transformers | ≥4.38.0 | Cross-encoder re-ranking |
| Streamlit | ≥1.32.0 | Web frontend |
| Plotly | ≥5.19.0 | Analytics charts |
| loguru | ≥0.7.0 | Structured logging |
| Docker / Compose | latest | Containerised deployment |

---

## 📄 License

MIT License — free for academic and personal use.

---

<div align="center">

*Built for NLP Final Year Project — SRM University AP | B.Tech CSE (AI & ML)*

</div>
