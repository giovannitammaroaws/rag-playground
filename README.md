# RAG Playground

An interactive, fully visual playground for understanding **Retrieval-Augmented Generation (RAG)** — the technique behind most production AI assistants. Every concept is demonstrated live in the browser: no backend, no API keys, no setup.

Inspired by [greeksplayground.com](https://greeksplayground.com/).

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [Sections](#sections)
   - [Playground — Compare Retrieval Methods](#1-playground--compare-retrieval-methods)
   - [Retrieval — Same Query, 4 Rankings](#2-retrieval--same-query-4-rankings)
   - [Embeddings — How Semantic Search Works](#3-embeddings--how-semantic-search-works)
   - [HNSW — How Vector Databases Search Fast](#4-hnsw--how-vector-databases-search-fast)
   - [Evaluation — Precision, Recall and F1](#5-evaluation--precision-recall-and-f1)
   - [Generation — How the LLM Uses Retrieved Chunks](#6-generation--how-the-llm-uses-retrieved-chunks)
3. [The Corpus](#the-corpus)
4. [Tech Stack](#tech-stack)
5. [Run Locally](#run-locally)
6. [Regenerate Embeddings](#regenerate-embeddings)

---

## What is RAG?

Standard LLMs are frozen at their training cutoff and have no access to private or up-to-date knowledge. **RAG** fixes this by splitting the problem in two:

```
User query
    │
    ▼
┌─────────────┐      top-K chunks      ┌─────────────┐
│  Retriever  │ ──────────────────────▶ │     LLM     │ ──▶ Answer
└─────────────┘                         └─────────────┘
  (finds the right                    (reads the chunks,
   pieces of text)                     writes the answer)
```

The quality of the final answer depends almost entirely on the quality of the retrieval step. This playground focuses on **Phase 1 (Retrieval)** and shows exactly how and why different retrieval methods produce different results.

---

## Sections

### 1. Playground — Compare Retrieval Methods

> **Nav:** `Playground`

The main interactive demo. Pick a topic (AI · Climate · Space), set K, and use the **alpha slider** to blend BM25 and Semantic search into a Hybrid retriever.

| Control | What it does |
|---|---|
| **Topic** | Selects the query and ground-truth relevant chunks |
| **K** | How many top results to retrieve (type any number) |
| **Alpha slider** | `α=0` → pure Semantic · `α=1` → pure BM25 · in between → Hybrid |

**Left panel** — the source document broken into 12 chunks. Each chunk shows which methods include it in their top-K (`BM25` / `Sem` / `Hyb` tags).

**Right panel** — three live columns:

| Column | Description |
|---|---|
| **BM25** (reference) | Pure keyword retrieval — fixed at α=1 |
| **HYBRID** (live) | Morphs as you move the slider — P/R/F1 update in real time |
| **Semantic** (reference) | Pure embedding retrieval — fixed at α=0 |

Move the slider from 0 → 1 and watch the HYBRID metrics drop when spam chunks outrank the synonym chunk.

---

### 2. Retrieval — Same Query, 4 Rankings

> **Nav:** `Retrieval`

A side-by-side comparison of all four retrieval methods on the same query. Switch topic with the buttons inside the search bar.

| Method | Formula | Weakness |
|---|---|---|
| **TF-IDF** | `tf(t,d) × log(N/df(t))` | Rewards raw frequency — keyword spam cheats |
| **BM25** | TF-IDF + length normalisation + term saturation | Still keyword-only |
| **Semantic** | Cosine similarity on BGE embeddings | Misses exact-match queries, slower to index |
| **Hybrid** | Reciprocal Rank Fusion of BM25 + Semantic | Best of both worlds |

The corpus is designed so that differences are **visible at K=4**:
- TF-IDF / BM25 rank the **spam** chunk `#1` — it contains all query keywords but in completely the wrong context (roads, archaeology, kitchens).
- Semantic demotes spam to `#4` and promotes the **synonym** chunk to `#3` — same concept, zero shared keywords.

---

### 3. Embeddings — How Semantic Search Works

> **Nav:** `Embeddings`

Two-panel visual explanation of how dense embeddings work.

**Left — Chunking demo**  
Shows a raw passage split into three colour-coded chunks, then visualises each chunk's embedding as a bar chart of 768 dimensions (model: `BAAI/bge-base-en-v1.5`).

**Right — 2D vector space scatter plot**  
All 12 corpus chunks projected to 2D via PCA. Select a topic query to reveal the query vector and nearest-neighbour rings — you can see AI chunks cluster together, away from Climate and Space clusters. This is why semantic search works: *similar meaning → nearby point in vector space*.

---

### 4. HNSW — How Vector Databases Search Fast

> **Nav:** `HNSW`

Brute-force nearest-neighbour search is **O(N)** — too slow at billions of vectors. **HNSW (Hierarchical Navigable Small World)** solves this with a multi-layer graph that achieves **O(log N)** approximate search.

The section includes a **step-by-step animated SVG graph**:

| Layer | Role |
|---|---|
| **Layer 2** (top) | Sparse, long-range connections — fast coarse navigation |
| **Layer 1** | Medium density |
| **Layer 0** (bottom) | Dense, short-range connections — fine-grained search |

Click **Start → Next → Next → Reset** to walk through a live query traversal and see which nodes are visited at each layer. This is exactly what happens inside Weaviate, Pinecone, and Qdrant on every query.

---

### 5. Evaluation — Precision, Recall and F1

> **Nav:** `Metrics`

Interactive metric explorer with topic selector (AI · Climate · Space) and a free K input (type any value).

| Metric | Formula | Meaning |
|---|---|---|
| **Precision@K** | `relevant found / K` | How clean is the result list? |
| **Recall@K** | `relevant found / total relevant` | How many relevant chunks did you miss? |
| **F1** | `2 × P × R / (P + R)` | Harmonic mean — balances precision and recall |

The hit/miss strips show which chunks BM25 and Semantic retrieve at the current K. At **K=3 on the AI topic** the difference is stark: BM25 wastes slot `#1` on the spam chunk (P=67%, R=67%), while Semantic retrieves all three relevant chunks cleanly (P=100%, R=100%).

Key insight: **↑ K → Recall rises, Precision falls.** There is always a tradeoff — choosing K is a product decision.

---

### 6. Generation — How the LLM Uses Retrieved Chunks

> **Nav:** `Generation`

Once retrieval is done, the LLM must synthesise an answer from the top-K chunks. Three common strategies are shown as pipeline diagrams:

| Strategy | LLM calls | Pro | Con |
|---|---|---|---|
| **Stuffing** | 1 | Fast and cheap | Fails if chunks overflow the context window |
| **Map Reduce** | N + 1 | Handles unlimited chunks, map steps run in parallel | LLM never sees all chunks together in one call |
| **Refine** | N | Each step refines and corrects the previous answer | Sequential — cannot parallelise |

---

## The Corpus

The playground uses a **single synthetic document** split into 12 chunks across three topics (AI · Climate · Space). Each topic contains four chunk types, deliberately designed to expose retrieval differences:

| Type | Description | Who finds it |
|---|---|---|
| `normal` | Standard text — clear keywords and clear meaning | All methods |
| `long` | Same keyword density as `normal` but ~3× longer text | TF-IDF penalises it (lower `tf/dl`), BM25 does not (length-normalised) |
| `synonym` | Same meaning as `normal`, zero shared keywords with the query | Semantic only — this is the key semantic advantage |
| `spam` | All query keywords present — but in completely wrong context (roads, archaeology, kitchens, school plays) | TF-IDF + BM25 rank it `#1` (keyword trap), Semantic ignores it |

Embeddings are pre-computed offline with **`BAAI/bge-base-en-v1.5`** via `sentence-transformers` and stored as static JSON. The browser does all similarity maths client-side — no server, no API key required.

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI framework | React 18 + Vite |
| Charts | Recharts |
| Embeddings (offline) | `BAAI/bge-base-en-v1.5` via `sentence-transformers` (Python) |
| Retrieval algorithms | Custom TF-IDF, BM25, Semantic cosine, Hybrid RRF — vanilla JS |
| Vector index viz | Hand-crafted SVG with animated traversal |
| Styling | Plain CSS |

---

## Run Locally

```bash
git clone https://github.com/giovannitammaroaws/rag-playground.git
cd rag-playground
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## Regenerate Embeddings

If you modify the corpus in `src/data/corpus.js`, re-run the embedding script to keep vectors in sync:

```bash
pip install sentence-transformers numpy
python3 generate_corpus_embeddings.py
```

This writes two files:

| File | Contents |
|---|---|
| `src/data/corpus_embeddings.json` | 768-dim L2-normalised vectors for all 12 chunks + 3 query vectors |
| `src/data/corpus_positions_2d.json` | PCA 2D projections used by the scatter plot |

---

*Built by **Giovanni Tammaro** with React + Vite · [GitHub](https://github.com/giovannitammaroaws/rag-playground)*
