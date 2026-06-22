from __future__ import annotations

from local_paper_qa.models import PaperCitation
from scripts import benchmark_retrieval_gold as bench


def test_empty_expected_evidence_is_not_scored():
    case = {
        "expected_evidence": [
            {"paper": "", "page": None, "must_contain": []},
            {"paper": "alpha", "must_contain": ["tree search"]},
        ]
    }

    evidence = bench._expected_evidence(case)

    assert len(evidence) == 1
    assert evidence[0]["paper"] == "alpha"


def test_evidence_match_uses_paper_page_and_terms():
    citation = PaperCitation(
        paper_id="p1",
        paper_title="Mastering the game of Go",
        authors="Silver",
        year="2016",
        page=3,
        section="Methods",
        quote="The system combines policy networks with Monte Carlo tree search.",
    )

    assert bench._matches_evidence(
        citation,
        {"paper": "game of go", "page": 3, "must_contain": ["policy networks", "tree search"]},
    )
    assert not bench._matches_evidence(
        citation,
        {"paper": "game of go", "page": 4, "must_contain": ["policy networks"]},
    )


def test_summary_ignores_unscored_placeholders():
    summary = bench.summarize(
        [
            {"scored": False, "passed": False, "paper_recall": 0, "evidence_recall": 0, "reciprocal_rank": 0, "elapsed_seconds": 1},
            {"scored": True, "passed": True, "paper_recall": 1, "evidence_recall": 0.5, "reciprocal_rank": 1, "elapsed_seconds": 2},
        ]
    )

    assert summary["case_count"] == 2
    assert summary["scored_case_count"] == 1
    assert summary["passed"] == 1
    assert summary["mean_evidence_recall"] == 0.5
