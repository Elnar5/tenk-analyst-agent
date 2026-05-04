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

[Diagram — to be added Day 5]

## Performance benchmarks

Tested on [N] real 10-K filings:

| Metric | This agent | Baseline RAG (vector only) | Improvement |
|---|---|---|---|
| Answer accuracy | TBD | TBD | TBD |
| Citation precision | TBD | TBD | TBD |
| Hallucination rate | TBD | TBD | TBD |
| Latency (p50) | TBD | TBD | TBD |

[Real benchmarks to be added Day 4]

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