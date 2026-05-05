"""
10-K Analyst Agent — main orchestration.

Combines hybrid retrieval with LLM generation, enforcing the
hallucination guard discipline defined in prompts.py.

Architecture:

    User question
         │
         ▼
    HybridSearcher (BM25 + vector + RRF)  → top-K chunks
         │
         ▼
    Format context with citations
         │
         ▼
    Groq Llama 3.3 70B (primary)
    Google Gemini 2.0 Flash (fallback)
         │
         ▼
    Parse citations from answer
         │
         ▼
    AnswerResult (answer + citations + sources)
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.prompts import (
    SYSTEM_PROMPT,
    QA_PROMPT_TEMPLATE,
    format_context_chunks,
)
from src.ingestion.chunker import Chunk
from src.retrieval.hybrid_search import HybridSearcher, SearchResult
from src.retrieval.vector_store import TenKVectorStore

import os

def _load_env_manual(path: str = ".env") -> None:
    """Manually parse .env to avoid dotenv encoding issues."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_manual()


@dataclass
class AnswerResult:
    """Output of the agent for a single question."""
    question: str
    answer: str
    citations: List[str] = field(default_factory=list)  # e.g. ["Item 7, p.26"]
    sources: List[SearchResult] = field(default_factory=list)
    is_grounded: bool = True  # False if "Not found" response
    model_used: str = ""
    
    def __str__(self) -> str:
        lines = [
            f"Q: {self.question}",
            f"A: {self.answer}",
        ]
        if self.citations:
            lines.append(f"\nCitations found: {', '.join(self.citations)}")
        lines.append(f"Model: {self.model_used}")
        return "\n".join(lines)


# Citation pattern matches:
#   [Item 7 — MD&A, p.26]
#   [Item 1A, p.12]
#   [p.34]
CITATION_PATTERN = re.compile(
    r"\[\s*(?:Item\s+\d+[A-Z]?\s*[—\-:,\s]*[A-Za-z&\s]*?,?\s*)?p\.?\s*(\d+)\s*\]",
    re.IGNORECASE,
)


def extract_citations(answer: str) -> List[str]:
    """Pull all citation strings from the answer."""
    # Find all citation-shaped substrings
    cites = re.findall(r"\[[^\]]+\]", answer)
    # Filter: must mention page or Item
    return [c for c in cites if "p." in c.lower() or "item" in c.lower()]


def is_not_found_response(answer: str) -> bool:
    """Check if the model declined to answer (hallucination guard worked)."""
    lowered = answer.lower()
    triggers = [
        "not found in the filing",
        "not found in the 10-k",
        "the filing does not contain",
        "the retrieved excerpts do not",
        "no relevant information",
    ]
    return any(t in lowered for t in triggers)


class TenKAnalyst:
    """
    The full agent: retrieval + LLM + citation parsing.
    
    Usage:
        # Build (one-time, slow)
        agent = TenKAnalyst.from_pdf("data/sample_filings/AAPL_10K_2024.pdf")
        
        # Ask (fast)
        result = agent.ask("What was iPhone revenue in 2024?")
        print(result.answer)
    """
    
    def __init__(
        self,
        searcher: HybridSearcher,
        primary_model: str = "llama-3.3-70b-versatile",
        fallback_model: str = "gemini-2.0-flash",
        top_k: int = 5,
        candidate_pool_size: int = 20,
    ):
        self.searcher = searcher
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        
        # Primary LLM: Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        self.primary_llm = ChatGroq(
            api_key=groq_key,
            model=primary_model,
            temperature=0.1,  # Low temperature: facts, not creativity
            max_tokens=2048,
        )
        self.primary_model_name = primary_model
        
        # Fallback LLM: Gemini
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            self.fallback_llm = ChatGoogleGenerativeAI(
                google_api_key=google_key,
                model=fallback_model,
                temperature=0.1,
                max_tokens=2048,
            )
            self.fallback_model_name = fallback_model
        else:
            self.fallback_llm = None
            self.fallback_model_name = None
    
    @classmethod
    def from_pdf(cls, pdf_path: str, **kwargs) -> "TenKAnalyst":
        """
        Convenience constructor: parse, chunk, embed, build searcher.
        """
        from src.ingestion.pdf_parser import parse_10k_pdf
        from src.ingestion.chunker import chunk_document
        
        print(f"Building agent from: {pdf_path}")
        doc = parse_10k_pdf(pdf_path)
        chunks = chunk_document(doc)
        
        store = TenKVectorStore()
        store.add_chunks(chunks)
        
        searcher = HybridSearcher(vector_store=store, chunks=chunks)
        return cls(searcher=searcher, **kwargs)
    
    def _call_llm(self, system_msg: str, user_msg: str) -> tuple:
        """
        Try primary LLM, fall back to secondary on failure.
        Returns (answer_text, model_name_used).
        """
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ]
        
        try:
            response = self.primary_llm.invoke(messages)
            return response.content, self.primary_model_name
        except Exception as e:
            print(f"⚠️  Primary LLM failed: {type(e).__name__}: {e}")
            if self.fallback_llm:
                print(f"   Falling back to {self.fallback_model_name}...")
                response = self.fallback_llm.invoke(messages)
                return response.content, self.fallback_model_name
            raise
    
    def ask(
        self,
        question: str,
        ticker_filter: Optional[str] = None,
        verbose: bool = False,
    ) -> AnswerResult:
        """
        Answer a question about the 10-K filing.
        
        Args:
            question: Natural-language question
            ticker_filter: Optionally restrict search to one ticker
            verbose: Print intermediate steps
        
        Returns:
            AnswerResult with answer, citations, sources
        """
        if verbose:
            print(f"\nQuestion: {question}")
            print("Step 1: Hybrid retrieval...")
        
        # 1. Retrieve relevant chunks
        results = self.searcher.search(
            query=question,
            top_k=self.top_k,
            candidate_pool_size=self.candidate_pool_size,
            ticker_filter=ticker_filter,
        )
        
        if not results:
            return AnswerResult(
                question=question,
                answer="Not found in the filing. No relevant content was retrieved.",
                is_grounded=False,
                model_used="(no retrieval)",
            )
        
        if verbose:
            print(f"  Retrieved {len(results)} chunks")
            for r in results:
                print(f"    - {r.citation}")
        
        # 2. Format context for the LLM
        context = format_context_chunks(results)
        user_msg = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )
        
        if verbose:
            print(f"\nStep 2: Calling LLM...")
        
        # 3. Generate answer
        answer, model_used = self._call_llm(SYSTEM_PROMPT, user_msg)
        
        # 4. Parse citations and check grounding
        citations = extract_citations(answer)
        not_found = is_not_found_response(answer)
        
        if verbose:
            print(f"  Model: {model_used}")
            print(f"  Citations found: {len(citations)}")
            print(f"  Grounded: {not not_found}")
        
        return AnswerResult(
            question=question,
            answer=answer,
            citations=citations,
            sources=results,
            is_grounded=not not_found,
            model_used=model_used,
        )


def main():
    """Quick demo with several question types."""
    agent = TenKAnalyst.from_pdf("data/sample_filings/AAPL_10K_2024.pdf")
    
    print("\n" + "=" * 70)
    print("10-K Analyst Agent — Demo")
    print("=" * 70)
    
    test_questions = [
        # Should answer with exact figure
        "What was Apple's iPhone revenue in fiscal 2024?",
        # Should answer with R&D number
        "How much did Apple spend on research and development in 2024?",
        # Should trigger hallucination guard — answer not in 10-K
        "How much did Tim Cook earn in 2024?",
        # Off-topic
        "What's the weather in Cupertino?",
    ]
    
    for q in test_questions:
        print(f"\n{'─' * 70}")
        result = agent.ask(q, verbose=True)
        print(f"\n📊 Answer:\n{result.answer}")
        print(f"\n🔗 Citations: {result.citations or '(none)'}")
        print(f"✓ Grounded: {result.is_grounded}")


if __name__ == "__main__":
    main()