"""Evidence-first retrieval benchmark.

This benchmark evaluates whether the system retrieves the expected Papers and
Evidence Spans before answer synthesis. That keeps embedding/retrieval model
comparison separate from chat-model quality.

Usage:
    python scripts/benchmark_retrieval_gold.py
    python scripts/benchmark_retrieval_gold.py --include-answer
    python scripts/benchmark_retrieval_gold.py --cases benchmark_cases.json --run-name openai-fast
    python scripts/benchmark_retrieval_gold.py --cached-only --max-cases 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_paper_qa.service import IndexingError, LocalPaperQA
from local_paper_qa.settings import (
    get_chat_model,
    get_chat_provider,
    get_embedding_dimension,
    get_embedding_model,
    get_embedding_provider,
    get_indexing_profile,
    get_openai_reasoning_effort,
)


DEFAULT_CASES_FILE = Path("benchmark_cases.json")
OUTPUT_DIR = Path("benchmark_outputs")


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark cases file not found: {path}")
    cases = json.loads(path.read_text())
    if not isinstance(cases, list):
        raise ValueError("Benchmark cases file must contain a JSON list.")
    return cases


def evaluate_case(
    qa: LocalPaperQA,
    case: dict[str, Any],
    *,
    k: int,
    include_answer: bool,
) -> dict[str, Any]:
    question = str(case["question"])
    started = time.perf_counter()
    citations = qa.retrieve(question)[:k]
    elapsed = time.perf_counter() - started

    expected_papers = _clean_list(case.get("expected_papers", []))
    expected_evidence = _expected_evidence(case)
    scored = bool(expected_papers or expected_evidence or case.get("expect_no_evidence"))
    expect_no_evidence = bool(case.get("expect_no_evidence", False))

    paper_hits = _paper_hits(citations, expected_papers)
    evidence_hits = _evidence_hits(citations, expected_evidence)
    first_hit_rank = _first_hit_rank(citations, expected_papers, expected_evidence)
    reciprocal_rank = 1 / first_hit_rank if first_hit_rank else 0.0

    if not scored:
        case_passed = False
    elif expect_no_evidence:
        case_passed = len(citations) == 0
    elif expected_evidence:
        case_passed = bool(evidence_hits)
    elif expected_papers:
        case_passed = bool(paper_hits)
    else:
        case_passed = bool(citations)

    result = {
        "id": case.get("id", question[:60]),
        "question_type": case.get("question_type", "unknown"),
        "question": question,
        "scored": scored,
        "passed": case_passed,
        "expect_no_evidence": expect_no_evidence,
        "elapsed_seconds": round(elapsed, 3),
        "citation_count": len(citations),
        "paper_recall": _recall(len(paper_hits), len(expected_papers)) if scored else 0.0,
        "evidence_recall": _recall(len(evidence_hits), len(expected_evidence)) if scored else 0.0,
        "reciprocal_rank": round(reciprocal_rank, 3),
        "paper_hits": paper_hits,
        "evidence_hits": evidence_hits,
        "citations": [_citation_summary(citation, rank) for rank, citation in enumerate(citations, start=1)],
    }

    if include_answer:
        answer_started = time.perf_counter()
        answer = qa.answer_from_evidence(question, citations)
        answer_elapsed = time.perf_counter() - answer_started
        expected_terms = _clean_list(case.get("expected_answer_terms", []))
        result["answer"] = {
            "elapsed_seconds": round(answer_elapsed, 3),
            "length": len(answer),
            "preview": answer[:500].replace("\n", " "),
            "expected_terms_found": [term for term in expected_terms if term in answer.lower()],
            "expected_term_recall": _recall(
                sum(1 for term in expected_terms if term in answer.lower()),
                len(expected_terms),
            ),
        }

    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored_results = [result for result in results if result.get("scored")]
    if not scored_results:
        return {
            "case_count": len(results),
            "scored_case_count": 0,
            "passed": 0,
            "mean_paper_recall": 0,
            "mean_evidence_recall": 0,
            "mean_mrr": 0,
            "mean_latency_seconds": 0,
            "retrieval_score": 0,
        }

    mean_paper_recall = statistics.mean(result["paper_recall"] for result in scored_results)
    mean_evidence_recall = statistics.mean(result["evidence_recall"] for result in scored_results)
    mean_mrr = statistics.mean(result["reciprocal_rank"] for result in scored_results)
    mean_latency = statistics.mean(result["elapsed_seconds"] for result in scored_results)
    retrieval_score = (0.4 * mean_evidence_recall) + (0.3 * mean_paper_recall) + (0.3 * mean_mrr)

    return {
        "case_count": len(results),
        "scored_case_count": len(scored_results),
        "passed": sum(1 for result in scored_results if result["passed"]),
        "mean_paper_recall": round(mean_paper_recall, 3),
        "mean_evidence_recall": round(mean_evidence_recall, 3),
        "mean_mrr": round(mean_mrr, 3),
        "mean_latency_seconds": round(mean_latency, 3),
        "retrieval_score": round(retrieval_score, 3),
    }


def write_report(report: dict[str, Any], run_name: str | None) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    suffix = f"_{_slug(run_name)}" if run_name else ""
    output_path = OUTPUT_DIR / f"retrieval_gold_benchmark{suffix}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-first retrieval benchmark")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_FILE)
    parser.add_argument("--papers-dir", default="papers")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Fail instead of making embedding calls when the current index cache is missing or stale.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Run only the first N benchmark cases. Useful for low-credit smoke runs.",
    )
    parser.add_argument("--include-answer", action="store_true")
    args = parser.parse_args()

    qa = LocalPaperQA(args.papers_dir)
    try:
        if args.cached_only:
            papers = qa._load_index_if_fresh()
            if papers is None:
                raise IndexingError(
                    "No fresh cached index is available. Run once without --cached-only to build embeddings."
                )
        else:
            papers = qa.ensure_index(force=args.force_reindex)
    except IndexingError as exc:
        print(f"Indexing failed: {exc}")
        raise SystemExit(2) from exc

    cases = load_cases(args.cases)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    results = [
        evaluate_case(qa, case, k=args.k, include_answer=args.include_answer)
        for case in cases
    ]
    report = {
        "run_name": args.run_name,
        "model_config": {
            "chat_provider": get_chat_provider(),
            "chat_model": get_chat_model(),
            "openai_reasoning_effort": get_openai_reasoning_effort(),
            "embedding_provider": get_embedding_provider(),
            "embedding_model": get_embedding_model(),
            "embedding_dimension": get_embedding_dimension(),
            "indexing_profile": get_indexing_profile(),
        },
        "corpus": {
            "papers_dir": str(Path(args.papers_dir).resolve()),
            "paper_count": len(papers),
            "span_count": sum(len(paper.chunks) for paper in papers),
        },
        "summary": summarize(results),
        "results": results,
    }
    output_path = write_report(report, args.run_name or None)

    summary = report["summary"]
    print("Evidence Retrieval Benchmark")
    print(f"Cases: {summary['passed']}/{summary['case_count']} passed")
    print(f"Retrieval score: {summary['retrieval_score']}")
    print(f"Mean evidence recall: {summary['mean_evidence_recall']}")
    print(f"Mean paper recall: {summary['mean_paper_recall']}")
    print(f"Mean MRR: {summary['mean_mrr']}")
    print(f"Report: {output_path}")


def _expected_evidence(case: dict[str, Any]) -> list[dict[str, Any]]:
    values = case.get("expected_evidence", [])
    return [
        value
        for value in values
        if isinstance(value, dict) and _is_meaningful_evidence_case(value)
    ]


def _paper_hits(citations: list[Any], expected_papers: list[str]) -> list[str]:
    if not expected_papers:
        return []
    hits: list[str] = []
    for expected in expected_papers:
        for citation in citations:
            if _matches_paper(citation, expected):
                hits.append(expected)
                break
    return hits


def _evidence_hits(citations: list[Any], expected_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for expected in expected_evidence:
        if not _is_meaningful_evidence_case(expected):
            continue
        for rank, citation in enumerate(citations, start=1):
            if _matches_evidence(citation, expected):
                hits.append({"rank": rank, "expected": expected})
                break
    return hits


def _first_hit_rank(
    citations: list[Any],
    expected_papers: list[str],
    expected_evidence: list[dict[str, Any]],
) -> int:
    meaningful_expected = [expected for expected in expected_evidence if _is_meaningful_evidence_case(expected)]
    for rank, citation in enumerate(citations, start=1):
        if any(_matches_evidence(citation, expected) for expected in meaningful_expected):
            return rank
        if any(_matches_paper(citation, expected) for expected in expected_papers):
            return rank
    return 0


def _matches_evidence(citation: Any, expected: dict[str, Any]) -> bool:
    paper = str(expected.get("paper") or expected.get("paper_id") or expected.get("paper_title") or "").lower()
    if paper and not _matches_paper(citation, paper):
        return False

    expected_page = expected.get("page")
    if expected_page is not None and citation.page != expected_page:
        return False

    terms = _clean_list(expected.get("must_contain", []))
    quote = citation.quote.lower()
    return all(term in quote for term in terms)


def _matches_paper(citation: Any, expected: str) -> bool:
    expected = expected.lower()
    return expected in citation.paper_id.lower() or expected in citation.paper_title.lower()


def _is_meaningful_evidence_case(expected: dict[str, Any]) -> bool:
    return bool(
        expected.get("paper")
        or expected.get("paper_id")
        or expected.get("paper_title")
        or expected.get("page") is not None
        or expected.get("must_contain")
    )


def _citation_summary(citation: Any, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "paper_id": citation.paper_id,
        "paper_title": citation.paper_title,
        "page": citation.page,
        "section": citation.section,
        "score": round(citation.score, 4),
        "quote_preview": citation.quote[:350].replace("\n", " "),
    }


def _clean_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(value).lower().strip() for value in values if str(value).strip()]


def _recall(hits: int, total: int) -> float:
    if total == 0:
        return 1.0
    return round(hits / total, 3)


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.lower()).strip("-")


if __name__ == "__main__":
    main()
