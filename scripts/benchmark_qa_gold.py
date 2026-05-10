"""Gold-answer QA benchmark.

Evaluates answer quality against a gold-standard question-answer set.
Produces a JSON report with per-question metrics and an overall score.

Usage::
    python scripts/benchmark_qa_gold.py

The benchmark reads ``benchmark_gold_qa.json`` from the project root.
If the file doesn't exist, it creates a template with example questions.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_paper_qa.service import LocalPaperQA


# Default gold questions (templates for users to fill in)
DEFAULT_GOLD_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question": "What is the main contribution of the curriculum learning paper?",
        "expected_keywords": ["curriculum learning", "convergence", "generalization", "data-driven", "dynamical"],
        "expected_citation_count": 3,
    },
    {
        "question": "How does AlphaGo use neural networks and tree search?",
        "expected_keywords": ["neural network", "tree search", "monte carlo", "policy"],
        "expected_citation_count": 3,
    },
    {
        "question": "What are the key findings across all indexed papers?",
        "expected_keywords": ["model", "learning", "training"],
        "expected_citation_count": 2,
    },
]

GOLD_FILE = Path("benchmark_gold_qa.json")
OUTPUT_DIR = Path("benchmark_outputs")
OUTPUT_FILE = OUTPUT_DIR / "qa_gold_benchmark.json"


def load_gold_questions() -> List[Dict[str, Any]]:
    """Load gold questions from file, creating a template if not found."""
    if not GOLD_FILE.exists():
        GOLD_FILE.write_text(json.dumps(DEFAULT_GOLD_QUESTIONS, indent=2))
        print(f"Created template at {GOLD_FILE}. Edit it with your gold answers.")
        return DEFAULT_GOLD_QUESTIONS
    return json.loads(GOLD_FILE.read_text())


def _token_overlap(answer: str, keywords: List[str]) -> float:
    """Calculate token overlap between answer and expected keywords."""
    answer_tokens = set(answer.lower().split())
    if not answer_tokens:
        return 0.0
    hits = sum(1 for kw in keywords if kw.lower() in answer)
    return hits / max(len(keywords), 1)


def evaluate_question(
    qa: LocalPaperQA, question: str, gold: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate a single question against gold standard."""
    start = time.perf_counter()
    result = qa.ask(question)
    elapsed = time.perf_counter() - start

    answer_text = result.answer
    citations = result.citations
    keywords = gold.get("expected_keywords", [])
    expected_citations = gold.get("expected_citation_count", 0)

    overlap = _token_overlap(answer_text, keywords)
    has_citations = len(citations) >= expected_citations if expected_citations > 0 else bool(citations)

    return {
        "question": question,
        "answer_length": len(answer_text),
        "answer_preview": answer_text[:200].replace("\n", " "),
        "num_citations": len(citations),
        "expected_citations": expected_citations,
        "has_expected_citations": has_citations,
        "token_overlap": round(overlap, 3),
        "keywords_found": [kw for kw in keywords if kw.lower() in answer_text.lower()],
        "has_answer": bool(answer_text and answer_text.strip() != "Insufficient evidence."),
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    qa = LocalPaperQA("papers")
    papers = qa.ensure_index()
    chunks = [c for p in papers for c in p.chunks]

    gold_questions = load_gold_questions()
    results = []

    for gold in gold_questions:
        question = gold["question"]
        print(f"Evaluating: {question[:60]}...")
        result = evaluate_question(qa, question, gold)
        results.append(result)

    # Compute summary stats
    avg_overlap = statistics.mean([r["token_overlap"] for r in results]) if results else 0
    avg_citations = statistics.mean([r["num_citations"] for r in results]) if results else 0
    avg_elapsed = statistics.mean([r["elapsed_seconds"] for r in results]) if results else 0
    total_has_answer = sum(1 for r in results if r["has_answer"])

    report = {
        "total_questions": len(gold_questions),
        "total_papers_indexed": len(papers),
        "total_chunks": len(chunks),
        "questions_answered": total_has_answer,
        "avg_token_overlap": round(avg_overlap, 3),
        "avg_citations": round(avg_citations, 2),
        "avg_latency_seconds": round(avg_elapsed, 3),
        "results": results,
    }

    OUTPUT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{'='*60}")
    print(f"Gold QA Benchmark Results")
    print(f"{'='*60}")
    print(f"Questions: {total_has_answer}/{len(gold_questions)} answered")
    print(f"Avg token overlap: {avg_overlap:.3f}")
    print(f"Avg citations: {avg_citations:.1f}")
    print(f"Avg latency: {avg_elapsed:.1f}s")
    print(f"\nFull report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
