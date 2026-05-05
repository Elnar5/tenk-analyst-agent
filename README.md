---
title: 10-K Analyst Agent
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

**🚀 Live Demo:** https://elnar5-tenk-analyst-agent.hf.space

Upload any SEC 10-K PDF, ask questions, get cited answers grounded in the filing.

---

# 10-K Analyst Agent

> Production-grade RAG agent for analyzing SEC 10-K filings.
> Built for analysts and investors who need fast, accurate answers with verified citations.

## Why this exists

Most "chat with PDF" tools collapse on real 10-K filings:

- **Hallucinated facts** — invented revenue numbers, fake risk factors
- **Bad citations** — wrong page numbers, or no citations at all
- **Long-document failure** — 10-Ks are 100-300 pages; context windows overflow
- **No structure awareness** — Risk Factors and MD&A are different beasts

This agent solves all four. Built by an AI engineer who benchmarks LLMs daily —
so the prompts and retrieval are designed around how models actually fail.

## Key features

- **Hallucination guard mode** — every answer must cite a specific section.
  If the model can't ground its answer, it says "Not found in filing" instead of inventing.
- **Hybrid retrieval** — BM25 + vector search + reranking. Outperforms pure vector RAG
  on financial documents (lots of named entities and numbers).
- **Section-aware chunking** — recognizes 10-K structure (Item 1, Item 1A, Item 7, Item 8)
  and chunks accordingly.
- **Page-level citations** — every answer links back to the exact page in the source PDF.

## Demo

[GIF or video — to be added Day 5]

## Architecture

```
                    ┌─────────────────┐
                    │   User Query    │
                    │  (e.g. "iPhone  │
                    │  revenue 2024?")│
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      HYBRID RETRIEVAL        │
              │                              │
              │  ┌────────┐    ┌──────────┐ │
              │  │  BM25  │    │  Vector  │ │
              │  │ (top   │    │  search  │ │
              │  │  20)   │    │  (top 20)│ │
              │  └───┬────┘    └────┬─────┘ │
              │      │              │       │
              │      └──────┬───────┘       │
              │             ▼               │
              │   Reciprocal Rank Fusion    │
              │      → top 5 chunks         │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   PROMPT CONSTRUCTION        │
              │                              │
              │  System: hallucination guard │
              │  Context: 5 chunks + sources │
              │  Question: user query        │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      LLM GENERATION          │
              │                              │
              │  Primary: Groq Llama 3.3 70B │
              │  Fallback: Gemini 2.0 Flash  │
              │  Temperature: 0.1 (factual)  │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   POST-PROCESSING            │
              │                              │
              │  • Extract citations (regex) │
              │  • Detect "Not found"        │
              │  • Build AnswerResult        │
              └──────────────┬───────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Final Answer   │
                    │  + Citations    │
                    │  + Sources      │
                    │  + Grounded?    │
                    └─────────────────┘
```

### Ingestion pipeline (one-time per filing)

```
   PDF File
      │
      ▼
   pdfplumber parse → 121 pages with metadata
      │
      ▼
   Section-aware chunker → detects Item 1A, Item 7, Item 8...
      │
      ▼
   663 chunks with (ticker, page, section, has_tables) metadata
      │
      ▼
   ┌─────────────────────┬──────────────────────┐
   │                     │                      │
   ▼                     ▼                      ▼
 BM25 index       HuggingFace embedder     ChromaDB store
 (in-memory)      (bge-small-en-v1.5)      (persistent)
                        │
                        ▼
                 768-dim vectors
```

## Performance benchmarks

Evaluated on Apple's 10-K filing (FY2024) with 13 test cases across three categories:
grounded questions (answer in filing), not-found questions (answer NOT in filing),
and off-topic queries.

| Metric | Result |
|---|---|
| **Overall pass rate** | **92.3%** (12/13) |
| **Correct behavior** | **100%** (13/13) |
| **Citation precision** | **100%** (13/13) |
| **Hallucination guard accuracy** | **100%** (4/4 not-found correctly refused) |
| **Latency (median)** | **0.69s** |
| **Latency (p95)** | ~14s (longer for "not found" responses) |

### By category

- **Grounded questions:** 7/7 correct behavior — agent answered with exact figures
  and citations (e.g. iPhone revenue $201,183M, R&D $31,370M)
- **Not-found questions:** 4/4 correctly refused — agent said "Not found in the filing"
  for compensation data, real-time prices, and forward-looking budgets that aren't
  in 10-K filings
- **Off-topic queries:** 2/2 correctly redirected — weather, creative writing requests
  triggered the redirect prompt

### Why this matters

Most "chat with PDF" tools hallucinate financial figures or invent citations.
This agent achieves zero hallucinations on the test set because of:

1. **Section-aware chunking** — preserves 10-K structure (Item 1A, Item 7, Item 8)
2. **Hybrid retrieval** — BM25 + vector search outperforms pure vector for
   documents with named entities and numbers
3. **Hallucination guard prompts** — the agent is instructed to refuse when context
   is insufficient, with exact phrasing patterns the model copies reliably

Run benchmarks yourself: `python -m src.evaluation.benchmark`

## Tech stack

- **LLM:** Claude Sonnet 4 (primary), GPT-4o (fallback)
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector store:** ChromaDB (local persistence)
- **BM25:** rank-bm25 for keyword matching
- **Framework:** LangChain + LangGraph for agent orchestration
- **PDF parsing:** pdfplumber

## Setup

```bash
git clone https://github.com/Elnar5/tenk-analyst-agent.git
cd tenk-analyst-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Usage

```python
from src.agent.agent import TenKAnalyst

agent = TenKAnalyst.from_pdf("data/sample_filings/AAPL_10K_2024.pdf")
result = agent.ask("What are the top 3 revenue drivers?")
print(result.answer)
print(result.citations)
```

## Roadmap

- [x] Day 1: Repo setup, scope, sample filings
- [ ] Day 2-3: Core ingestion + retrieval
- [ ] Day 4: Evaluation harness, benchmarks
- [ ] Day 5: README polish, architecture diagram, demo
- [ ] Week 2: MuleRun deployment, LinkedIn launch

## Live deployment

Available on [MuleRun](https://mulerun.com/) — link added after launch.

## About

Built by Elnar Babayev — production AI engineer specializing in LLM evaluation
and RAG systems.

## License

MIT