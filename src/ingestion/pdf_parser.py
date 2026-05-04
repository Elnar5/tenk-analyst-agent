"""
PDF Parser for 10-K filings.

Extracts text from SEC 10-K PDF documents while preserving page-level
metadata. This is the foundation for citation-grounded retrieval.

Why this matters:
- Standard pypdf.extract_text() loses page boundaries
- 10-Ks have tables, footnotes, multi-column layouts
- Citations require knowing the exact source page
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber


@dataclass
class PageContent:
    """A single page extracted from a 10-K filing."""
    page_number: int
    text: str
    char_count: int
    has_tables: bool
    is_likely_empty: bool  # Cover pages, blank pages, etc.


@dataclass
class ParsedDocument:
    """A complete 10-K filing parsed into structured pages."""
    source_path: str
    ticker: str  # Extracted from filename, e.g. "AAPL"
    total_pages: int
    pages: List[PageContent]
    
    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)
    
    @property
    def non_empty_pages(self) -> List[PageContent]:
        return [p for p in self.pages if not p.is_likely_empty]


def parse_10k_pdf(pdf_path: str) -> ParsedDocument:
    """
    Parse a 10-K PDF file into structured page-level content.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        ParsedDocument with page-level metadata preserved
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Extract ticker from filename (e.g., "AAPL_10K_2024.pdf" -> "AAPL")
    ticker = path.stem.split("_")[0].upper()
    
    pages: List[PageContent] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Extract text — pdfplumber preserves layout better than pypdf
            text = page.extract_text() or ""
            
            # Detect tables (10-Ks are full of them)
            tables = page.find_tables()
            has_tables = len(tables) > 0
            
            # Heuristic: pages with very little text are likely covers/blanks
            char_count = len(text.strip())
            is_likely_empty = char_count < 100
            
            pages.append(PageContent(
                page_number=i,
                text=text,
                char_count=char_count,
                has_tables=has_tables,
                is_likely_empty=is_likely_empty,
            ))
    
    return ParsedDocument(
        source_path=str(path.absolute()),
        ticker=ticker,
        total_pages=len(pages),
        pages=pages,
    )


def main():
    """Quick test: parse Apple 10-K and show stats."""
    pdf_path = "data/sample_filings/AAPL_10K_2024.pdf"
    
    print(f"Parsing: {pdf_path}")
    doc = parse_10k_pdf(pdf_path)
    
    print(f"\n=== Document Stats ===")
    print(f"Ticker: {doc.ticker}")
    print(f"Total pages: {doc.total_pages}")
    print(f"Non-empty pages: {len(doc.non_empty_pages)}")
    print(f"Total characters: {doc.total_chars:,}")
    print(f"Pages with tables: {sum(1 for p in doc.pages if p.has_tables)}")
    
    print(f"\n=== First 500 chars of page 5 ===")
    if len(doc.pages) >= 5:
        print(doc.pages[4].text[:500])
    
    print(f"\n=== Pages 1-3 metadata ===")
    for p in doc.pages[:3]:
        print(f"Page {p.page_number}: {p.char_count} chars, "
              f"tables={p.has_tables}, empty={p.is_likely_empty}")


if __name__ == "__main__":
    main()