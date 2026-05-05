"""
Evaluation harness for the 10-K Analyst Agent.

Measures:
1. Grounding accuracy — does the agent correctly answer / decline?
2. Citation quality — does the agent cite real sections?
3. Hallucination rate — does the agent invent facts?
4. Latency — how fast does it respond?

Test questions are grouped into:
- GROUNDED: answer exists in 10-K, agent should answer
- NOT_FOUND: answer NOT in 10-K, agent should say "Not found"
- OFF_TOPIC: irrelevant question, agent should redirect

Run: python -m src.evaluation.benchmark
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.agent.agent import TenKAnalyst, AnswerResult


@dataclass
class TestCase:
    """A single evaluation test case."""
    question: str
    expected_behavior: str  # "GROUNDED", "NOT_FOUND", "OFF_TOPIC"
    expected_keywords: List[str] = field(default_factory=list)  # Keywords the answer should contain
    expected_section: Optional[str] = None  # E.g. "Item 7" — citation should mention this
    notes: str = ""


@dataclass
class TestResult:
    """Result of running one test case."""
    case: TestCase
    answer: AnswerResult
    latency_seconds: float
    correct_behavior: bool
    has_expected_keywords: bool
    has_expected_section: bool
    
    @property
    def passed(self) -> bool:
        return self.correct_behavior and self.has_expected_keywords and self.has_expected_section


# Apple 10-K test set (FY2024)
APPLE_TEST_CASES: List[TestCase] = [
    # === GROUNDED — answers in the filing ===
    TestCase(
        question="What was Apple's iPhone revenue in fiscal 2024?",
        expected_behavior="GROUNDED",
        expected_keywords=["201,183", "iPhone"],
        expected_section="Item 7",
        notes="Direct table lookup, MD&A section",
    ),
    TestCase(
        question="How much did Apple spend on research and development in 2024?",
        expected_behavior="GROUNDED",
        expected_keywords=["31,370", "research"],
        expected_section="Item 7",
        notes="R&D expense in MD&A",
    ),
    TestCase(
        question="What were Apple's total net sales in 2024?",
        expected_behavior="GROUNDED",
        expected_keywords=["391,035", "net sales"],
        expected_section="Item 7",
        notes="Top-line revenue",
    ),
    TestCase(
        question="What are the main risk factors Apple identifies?",
        expected_behavior="GROUNDED",
        expected_keywords=["risk", "competition"],
        expected_section="Item 1A",
        notes="Risk Factors section",
    ),
    TestCase(
        question="Where is Apple's headquarters located?",
        expected_behavior="GROUNDED",
        expected_keywords=["Cupertino", "California"],
        expected_section=None,  # May appear in multiple sections
        notes="Properties section",
    ),
    TestCase(
        question="What was Mac revenue in 2024 compared to 2023?",
        expected_behavior="GROUNDED",
        expected_keywords=["29,984", "Mac"],
        expected_section="Item 7",
        notes="Product segment comparison",
    ),
    TestCase(
        question="What is Apple's effective tax rate for 2024?",
        expected_behavior="GROUNDED",
        expected_keywords=["tax", "%"],
        expected_section="Item 8",
        notes="Tax disclosure in financials",
    ),
    
    # === NOT_FOUND — answers NOT in 10-K ===
    TestCase(
        question="How much did Tim Cook earn in 2024?",
        expected_behavior="NOT_FOUND",
        expected_keywords=["not found", "filing"],
        expected_section=None,
        notes="Compensation is in DEF 14A, not 10-K",
    ),
    TestCase(
        question="What is Apple's stock price today?",
        expected_behavior="NOT_FOUND",
        expected_keywords=["not found", "filing"],
        expected_section=None,
        notes="Real-time data not in annual filing",
    ),
    TestCase(
        question="Who are Apple's biggest individual shareholders?",
        expected_behavior="NOT_FOUND",
        expected_keywords=["not found", "filing"],
        expected_section=None,
        notes="Beneficial ownership in DEF 14A",
    ),
    TestCase(
        question="What is Apple's marketing budget for 2025?",
        expected_behavior="NOT_FOUND",
        expected_keywords=["not found", "filing"],
        expected_section=None,
        notes="Forward-looking detail not disclosed",
    ),
    
    # === OFF_TOPIC — should redirect ===
    TestCase(
        question="What's the weather in Cupertino?",
        expected_behavior="OFF_TOPIC",
        expected_keywords=["not found", "filing"],
        expected_section=None,
        notes="Off-topic, agent should redirect",
    ),
    TestCase(
        question="Can you write me a poem about technology?",
        expected_behavior="OFF_TOPIC",
        expected_keywords=["not found", "filing"],
        expected_section=None,
        notes="Off-topic creative request",
    ),
]


def evaluate_case(agent: TenKAnalyst, case: TestCase) -> TestResult:
    """Run a single test case and score it."""
    start = time.time()
    answer = agent.ask(case.question, verbose=False)
    latency = time.time() - start
    
    # 1. Did the agent get the behavior right?
    if case.expected_behavior == "GROUNDED":
        correct_behavior = answer.is_grounded
    elif case.expected_behavior in ("NOT_FOUND", "OFF_TOPIC"):
        correct_behavior = not answer.is_grounded
    else:
        correct_behavior = False
    
    # 2. Are expected keywords in the answer?
    answer_lower = answer.answer.lower()
    has_expected_keywords = all(
        kw.lower() in answer_lower for kw in case.expected_keywords
    )
    
    # 3. Is the expected section cited (if specified)?
    if case.expected_section is None:
        has_expected_section = True  # No requirement
    else:
        citations_text = " ".join(answer.citations)
        has_expected_section = case.expected_section.lower() in citations_text.lower()
    
    return TestResult(
        case=case,
        answer=answer,
        latency_seconds=latency,
        correct_behavior=correct_behavior,
        has_expected_keywords=has_expected_keywords,
        has_expected_section=has_expected_section,
    )


def print_result(result: TestResult, idx: int) -> None:
    """Print a single test result in compact form."""
    pass_marker = "✅" if result.passed else "❌"
    behavior_marker = "✓" if result.correct_behavior else "✗"
    keyword_marker = "✓" if result.has_expected_keywords else "✗"
    section_marker = "✓" if result.has_expected_section else "✗"
    
    print(f"\n[{idx}] {pass_marker} {result.case.expected_behavior}: {result.case.question}")
    print(f"    Behavior: {behavior_marker}  Keywords: {keyword_marker}  Section: {section_marker}  "
          f"Latency: {result.latency_seconds:.2f}s")
    
    if not result.passed:
        print(f"    Answer: {result.answer.answer[:200]}...")
        print(f"    Citations: {result.answer.citations}")


def print_summary(results: List[TestResult]) -> None:
    """Print aggregate metrics."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    correct_behavior = sum(1 for r in results if r.correct_behavior)
    has_keywords = sum(1 for r in results if r.has_expected_keywords)
    has_sections = sum(1 for r in results if r.has_expected_section)
    
    avg_latency = sum(r.latency_seconds for r in results) / total
    median_latency = sorted([r.latency_seconds for r in results])[total // 2]
    
    # Per behavior type
    grounded_results = [r for r in results if r.case.expected_behavior == "GROUNDED"]
    notfound_results = [r for r in results if r.case.expected_behavior == "NOT_FOUND"]
    offtopic_results = [r for r in results if r.case.expected_behavior == "OFF_TOPIC"]
    
    grounded_correct = sum(1 for r in grounded_results if r.correct_behavior)
    notfound_correct = sum(1 for r in notfound_results if r.correct_behavior)
    offtopic_correct = sum(1 for r in offtopic_results if r.correct_behavior)
    
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n  Overall pass rate:       {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"  Correct behavior:        {correct_behavior}/{total} ({correct_behavior/total*100:.1f}%)")
    print(f"  Keyword precision:       {has_keywords}/{total} ({has_keywords/total*100:.1f}%)")
    print(f"  Citation precision:      {has_sections}/{total} ({has_sections/total*100:.1f}%)")
    print(f"\n  By category:")
    print(f"    Grounded answers:      {grounded_correct}/{len(grounded_results)} correct")
    print(f"    Not-found refusals:    {notfound_correct}/{len(notfound_results)} correct")
    print(f"    Off-topic redirects:   {offtopic_correct}/{len(offtopic_results)} correct")
    print(f"\n  Latency:")
    print(f"    Mean:                  {avg_latency:.2f}s")
    print(f"    Median:                {median_latency:.2f}s")
    print(f"    Max:                   {max(r.latency_seconds for r in results):.2f}s")


def main():
    print("Building agent for benchmark...")
    agent = TenKAnalyst.from_pdf("data/sample_filings/AAPL_10K_2024.pdf")
    
    print(f"\nRunning {len(APPLE_TEST_CASES)} test cases...")
    print("=" * 70)
    
    results: List[TestResult] = []
    for i, case in enumerate(APPLE_TEST_CASES, 1):
        result = evaluate_case(agent, case)
        results.append(result)
        print_result(result, i)
    
    print_summary(results)


if __name__ == "__main__":
    main()