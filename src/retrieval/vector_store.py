"""
Vector store for 10-K analyst agent.

Embeds chunks using a local HuggingFace model (bge-small-en-v1.5) and
stores them in ChromaDB for similarity search.

Why local embeddings:
- No API costs (free, unlike OpenAI)
- Works offline after first download
- bge-small-en-v1.5 is a top-performing small model on MTEB benchmark
- Privacy: financial data never leaves the local machine

Why ChromaDB:
- Lightweight, no separate server needed
- Persistent: embeddings stay between runs
- Filter support: query by ticker, section, page
"""

import os
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.chunker import Chunk


class TenKVectorStore:
    """
    Manages embeddings and similarity search for 10-K filings.
    
    Usage:
        store = TenKVectorStore()
        store.add_chunks(chunks)  # one-time ingestion
        results = store.search("What are the main risks?", top_k=5)
    """
    
    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        collection_name: str = "tenk_filings",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model (downloads ~130MB on first run)
        print(f"Loading embedding model: {embedding_model}")
        print("(First run downloads ~130MB, subsequent runs use cache)")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},  # Use "mps" on Mac M-series for speedup
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )
        
        print(f"Vector store ready. Current count: {self.collection.count()}")
    
    def add_chunks(self, chunks: List[Chunk], batch_size: int = 50) -> None:
        """
        Embed and store a list of chunks.
        
        Skips chunks that are already in the store (by chunk_id).
        Embeds in batches to avoid memory issues.
        """
        if not chunks:
            return
        
        # Check which chunks are already stored
        existing_ids = set()
        try:
            existing = self.collection.get(ids=[c.chunk_id for c in chunks])
            existing_ids = set(existing["ids"])
        except Exception:
            pass  # Empty collection, nothing exists yet
        
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        
        if not new_chunks:
            print(f"All {len(chunks)} chunks already in store. Skipping.")
            return
        
        print(f"Adding {len(new_chunks)} new chunks to vector store...")
        
        # Process in batches
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [
                {
                    "ticker": c.ticker,
                    "page_number": c.page_number,
                    "section_item": c.section_item or "",
                    "section_name": c.section_name or "",
                    "has_tables": c.has_tables,
                    "char_count": c.char_count,
                }
                for c in batch
            ]
            
            # Embed the texts
            vectors = self.embeddings.embed_documents(texts)
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )
            
            print(f"  Batch {i // batch_size + 1}: {len(batch)} chunks added")
        
        print(f"Done. Total chunks in store: {self.collection.count()}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Find the most similar chunks to a query.
        
        Args:
            query: Natural language question
            top_k: Number of results to return
            ticker_filter: Optional ticker to filter by (e.g. "AAPL")
            section_filter: Optional Item number filter (e.g. "1A" for Risk Factors)
        
        Returns:
            List of dicts with text, metadata, and similarity score
        """
        # Embed the query
        query_vector = self.embeddings.embed_query(query)
        
        # Build metadata filter
        where_clause = {}
        if ticker_filter:
            where_clause["ticker"] = ticker_filter
        if section_filter:
            where_clause["section_item"] = section_filter
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_clause if where_clause else None,
        )
        
        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })
        
        return formatted
    
    def get_stats(self) -> dict:
        """Quick stats about the store."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
        }


def main():
    """Quick test: ingest Apple 10-K and run sample queries."""
    from src.ingestion.pdf_parser import parse_10k_pdf
    from src.ingestion.chunker import chunk_document
    
    # 1. Parse and chunk Apple 10-K
    print("=" * 60)
    print("Step 1: Parse and chunk Apple 10-K")
    print("=" * 60)
    doc = parse_10k_pdf("data/sample_filings/AAPL_10K_2024.pdf")
    chunks = chunk_document(doc, chunk_size=800, chunk_overlap=150)
    print(f"Created {len(chunks)} chunks")
    
    # 2. Add to vector store
    print("\n" + "=" * 60)
    print("Step 2: Add to vector store")
    print("=" * 60)
    store = TenKVectorStore()
    store.add_chunks(chunks)
    print(f"\nStore stats: {store.get_stats()}")
    
    # 3. Run sample queries
    print("\n" + "=" * 60)
    print("Step 3: Sample queries")
    print("=" * 60)
    
    queries = [
        "What are the main risks Apple faces?",
        "How does Apple generate revenue?",
        "What did Apple say about iPhone sales growth?",
    ]
    
    for query in queries:
        print(f"\n--- Query: {query} ---")
        results = store.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            section = r["metadata"].get("section_name") or "Unknown"
            page = r["metadata"].get("page_number")
            distance = r["distance"]
            preview = r["text"][:200].replace("\n", " ")
            print(f"\n  Result {i}: {section}, page {page} (distance: {distance:.3f})")
            print(f"  Preview: {preview}...")


if __name__ == "__main__":
    main()