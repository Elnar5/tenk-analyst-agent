"""
Hybrid retrieval combining BM25 (keyword) and vector (semantic) search.

Why hybrid:
- Vector search: great at semantic similarity ("risks" finds "exposures", "threats")
- BM25: great at exact terms (numbers, names, specific phrases)
- Together: outperforms either alone, especially for financial documents
  full of named entities, numbers, and specific terminology

Algorithm:
1. Run both searches independently, get top-K from each
2. Combine using Reciprocal Rank Fusion (RRF)
3. Return top-N final results

Reference for RRF: Cormack et al., 2009 - "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods"
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import TenKVectorStore


@dataclass
class SearchResult:
    """A single result from hybrid search with combined ranking."""
    chunk_id: str
    text: str
    metadata: dict
    rrf_score: float
    vector_rank: Optional[int] = None  # Rank in vector search (lower = better)
    bm25_rank: Optional[int] = None    # Rank in BM25 search
    
    @property
    def section_name(self) -> str:
        return self.metadata.get("section_name", "Unknown")
    
    @property
    def page_number(self) -> int:
        return self.metadata.get("page_number", 0)
    
    @property
    def section_item(self) -> str:
        return self.metadata.get("section_item", "")
    
    @property
    def citation(self) -> str:
        """Human-readable citation string."""
        if self.section_item and self.section_name and self.section_name != "Unknown":
            return f"Item {self.section_item} ({self.section_name}), page {self.page_number}"
        return f"Page {self.page_number}"


def _tokenize(text: str) -> List[str]:
    """
    Simple lowercase whitespace tokenization for BM25.
    
    Note: This is intentionally simple. Production systems often use
    spaCy or NLTK, but for English financial text, simple tokenization
    is competitive and 10x faster.
    """
    return text.lower().split()


class HybridSearcher:
    """
    Combines BM25 keyword search with ChromaDB vector search.
    
    Usage:
        searcher = HybridSearcher(vector_store, all_chunks)
        results = searcher.search("What are revenue risks?", top_k=5)
    """
    
    def __init__(
        self,
        vector_store: TenKVectorStore,
        chunks: List[Chunk],
        rrf_k: int = 60,
    ):
        """
        Args:
            vector_store: Initialized TenKVectorStore (already has embeddings)
            chunks: All chunks (used to build BM25 index)
            rrf_k: RRF constant. Standard value is 60. Higher = less weight on rank.
        """
        self.vector_store = vector_store
        self.chunks = chunks
        self.rrf_k = rrf_k
        
        # Build BM25 index from chunk texts
        print(f"Building BM25 index over {len(chunks)} chunks...")
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Index chunks by chunk_id for fast lookup
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        print("BM25 index ready.")
    
    def _bm25_search(self, query: str, top_k: int) -> List[tuple]:
        """
        Run BM25 search.
        Returns list of (chunk_id, rank, score) tuples.
        """
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices by score (descending)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk_id = self.chunks[idx].chunk_id
            results.append((chunk_id, rank, float(scores[idx])))
        
        return results
    
    def _vector_search(
        self,
        query: str,
        top_k: int,
        ticker_filter: Optional[str] = None,
    ) -> List[tuple]:
        """
        Run vector similarity search.
        Returns list of (chunk_id, rank, distance) tuples.
        """
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            ticker_filter=ticker_filter,
        )
        
        return [
            (r["chunk_id"], rank, r["distance"])
            for rank, r in enumerate(results, start=1)
        ]
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[tuple],
        vector_results: List[tuple],
    ) -> List[SearchResult]:
        """
        Combine two ranked lists using RRF.
        
        RRF formula: score(d) = Σ 1 / (k + rank_i(d))
        
        Where rank_i(d) is the rank of document d in result list i,
        and k is a constant (typically 60).
        """
        # rrf_scores[chunk_id] = total RRF score
        rrf_scores: dict = defaultdict(float)
        bm25_ranks: dict = {}
        vector_ranks: dict = {}
        
        for chunk_id, rank, _ in bm25_results:
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            bm25_ranks[chunk_id] = rank
        
        for chunk_id, rank, _ in vector_results:
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            vector_ranks[chunk_id] = rank
        
        # Sort by combined RRF score, descending
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )
        
        # Build SearchResult objects
        final_results: List[SearchResult] = []
        for chunk_id in sorted_chunk_ids:
            chunk = self.chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            
            metadata = {
                "ticker": chunk.ticker,
                "page_number": chunk.page_number,
                "section_item": chunk.section_item or "",
                "section_name": chunk.section_name or "",
                "has_tables": chunk.has_tables,
            }
            
            final_results.append(SearchResult(
                chunk_id=chunk_id,
                text=chunk.text,
                metadata=metadata,
                rrf_score=rrf_scores[chunk_id],
                vector_rank=vector_ranks.get(chunk_id),
                bm25_rank=bm25_ranks.get(chunk_id),
            ))
        
        return final_results
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int = 20,
        ticker_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining BM25 and vector retrieval.
        
        Args:
            query: User's question
            top_k: Number of final results to return
            candidate_pool_size: How many candidates to fetch from each method
                                 before fusion. Larger = more diverse, slower.
            ticker_filter: Optional ticker (e.g. "AAPL")
        
        Returns:
            Top-K results sorted by RRF score
        """
        # Get candidates from both methods
        bm25_results = self._bm25_search(query, candidate_pool_size)
        vector_results = self._vector_search(
            query,
            candidate_pool_size,
            ticker_filter=ticker_filter,
        )
        
        # Fuse and return top-K
        fused = self._reciprocal_rank_fusion(bm25_results, vector_results)
        return fused[:top_k]


def main():
    """Demo: compare pure vector search vs hybrid search."""
    from src.ingestion.pdf_parser import parse_10k_pdf
    from src.ingestion.chunker import chunk_document
    
    print("Setting up: parse, chunk, embed Apple 10-K...")
    doc = parse_10k_pdf("data/sample_filings/AAPL_10K_2024.pdf")
    chunks = chunk_document(doc)
    
    store = TenKVectorStore()
    store.add_chunks(chunks)
    
    searcher = HybridSearcher(vector_store=store, chunks=chunks)
    
    # Test queries that benefit from hybrid (mix of semantic + exact terms)
    test_queries = [
        "What are the main risks Apple faces?",          # Pure semantic
        "How much was iPhone revenue in 2024?",          # Mix (semantic + numbers)
        "What did Tim Cook earn?",                       # Exact name lookup
        "Explain Apple's research and development spending",  # Semantic but specific
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print('=' * 60)
        
        results = searcher.search(query, top_k=3, candidate_pool_size=20)
        
        for i, r in enumerate(results, 1):
            bm25_rank_str = f"#{r.bm25_rank}" if r.bm25_rank else "—"
            vec_rank_str = f"#{r.vector_rank}" if r.vector_rank else "—"
            
            print(f"\n  Result {i}: {r.citation}")
            print(f"  RRF score: {r.rrf_score:.4f} | BM25: {bm25_rank_str} | Vector: {vec_rank_str}")
            print(f"  Preview: {r.text[:200]}...")


if __name__ == "__main__":
    main()