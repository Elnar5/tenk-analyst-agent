"""
Section-aware chunker for 10-K filings.

Standard RAG chunkers split text every N characters, breaking sentences
and tables mid-content. This chunker:

1. Detects 10-K section structure (Item 1, Item 1A, Item 7, etc.)
2. Splits at sentence boundaries within sections
3. Preserves section + page metadata for citations
4. Handles overlapping chunks for context continuity

Why this matters for citations:
- Every chunk knows which section it came from
- Citations can say "Item 1A — Risk Factors, page 23" instead of "page 23"
- Models trained on 10-Ks expect this structure
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.pdf_parser import ParsedDocument, PageContent


# Standard 10-K section patterns
# Match: "ITEM 1.", "Item 1A.", "ITEM 7. MANAGEMENT'S DISCUSSION..."
SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:ITEM|Item)\s+(\d+[A-Z]?)\.?\s*[\.\—\-:]*\s*([A-Z][^\n]{0,150})",
    re.MULTILINE,
)

# Map item numbers to canonical section names
SECTION_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity",
    "6": "Reserved",
    "7": "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}


@dataclass
class Chunk:
    """A single chunk of text with full traceable metadata."""
    text: str
    chunk_id: str  # Unique ID, e.g. "AAPL_p23_c2"
    ticker: str
    page_number: int
    section_item: Optional[str] = None  # e.g. "1A"
    section_name: Optional[str] = None  # e.g. "Risk Factors"
    char_count: int = 0
    has_tables: bool = False

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.text)
    
    @property
    def citation(self) -> str:
        """Human-readable citation string."""
        if self.section_item and self.section_name:
            return f"Item {self.section_item} ({self.section_name}), page {self.page_number}"
        return f"Page {self.page_number}"


def detect_section_at_page(page_text: str, current_section: Optional[tuple]) -> Optional[tuple]:
    """
    Check if a section header appears on this page.
    Returns (item_number, section_name) if found, else current_section.
    """
    matches = SECTION_PATTERN.findall(page_text)
    if not matches:
        return current_section
    
    # Take the FIRST section header on the page (others would be subsections or refs)
    item_num, _ = matches[0]
    item_num = item_num.upper()
    
    # Only update if it's a known section
    if item_num in SECTION_NAMES:
        return (item_num, SECTION_NAMES[item_num])
    
    return current_section


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """
    Chunk a parsed 10-K into section-aware chunks.
    
    Args:
        doc: Parsed document from pdf_parser
        chunk_size: Target characters per chunk
        chunk_overlap: Overlap between consecutive chunks (preserves context)
    
    Returns:
        List of Chunk objects with full metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try sentence boundaries first, then word boundaries, then characters
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
    )
    
    chunks: List[Chunk] = []
    current_section: Optional[tuple] = None  # (item_num, section_name)
    
    for page in doc.pages:
        if page.is_likely_empty:
            continue
        
        # Update current section if a new one starts on this page
        current_section = detect_section_at_page(page.text, current_section)
        
        # Split this page's text into chunks
        page_chunks = splitter.split_text(page.text)
        
        for idx, chunk_text in enumerate(page_chunks):
            # Skip chunks that are mostly whitespace or very short
            if len(chunk_text.strip()) < 50:
                continue
            
            chunk = Chunk(
                text=chunk_text.strip(),
                chunk_id=f"{doc.ticker}_p{page.page_number}_c{idx}",
                ticker=doc.ticker,
                page_number=page.page_number,
                section_item=current_section[0] if current_section else None,
                section_name=current_section[1] if current_section else None,
                has_tables=page.has_tables,
            )
            chunks.append(chunk)
    
    return chunks


def main():
    """Quick test: chunk Apple 10-K and show stats."""
    from src.ingestion.pdf_parser import parse_10k_pdf
    
    print("Parsing Apple 10-K...")
    doc = parse_10k_pdf("data/sample_filings/AAPL_10K_2024.pdf")
    
    print(f"Document has {doc.total_pages} pages.")
    print("Chunking...")
    
    chunks = chunk_document(doc, chunk_size=800, chunk_overlap=150)
    
    print(f"\n=== Chunking Stats ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk size: {sum(c.char_count for c in chunks) // len(chunks)} chars")
    
    # Group by section
    section_counts = {}
    for c in chunks:
        key = c.section_name or "Unknown"
        section_counts[key] = section_counts.get(key, 0) + 1
    
    print(f"\n=== Chunks per Section ===")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  {section}: {count} chunks")
    
    print(f"\n=== Sample chunks ===")
    # Show one chunk from Risk Factors (Item 1A) if available
    risk_chunks = [c for c in chunks if c.section_item == "1A"]
    if risk_chunks:
        sample = risk_chunks[0]
        print(f"\nFrom Risk Factors:")
        print(f"  Citation: {sample.citation}")
        print(f"  Chunk ID: {sample.chunk_id}")
        print(f"  Text preview: {sample.text[:300]}...")
    
    # Show one chunk from MD&A (Item 7) if available
    mda_chunks = [c for c in chunks if c.section_item == "7"]
    if mda_chunks:
        sample = mda_chunks[0]
        print(f"\nFrom MD&A:")
        print(f"  Citation: {sample.citation}")
        print(f"  Chunk ID: {sample.chunk_id}")
        print(f"  Text preview: {sample.text[:300]}...")


if __name__ == "__main__":
    main()