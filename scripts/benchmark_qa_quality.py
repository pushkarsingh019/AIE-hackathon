"""Simple QA quality benchmark.

The benchmark runs a list of question strings against the local QA system and records
basic quality signals:

- answer length (character count)
- number of citations returned
- whether any citations were returned
- the raw answer text (truncated to 200 characters for brevity)

The script writes a JSON report to ``benchmark_outputs/qa_quality_report.json``.

This is a lightweight proxy for quality; a full gold‑answer comparison would require
human‑written reference answers which are outside the scope of the automated
environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import sys
from pathlib import Path

# Ensure the repository root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_paper_qa.service import LocalPaperQA


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_QUESTIONS: List[str] = [
    "What is the main contribution of the paper titled 'The interoceptive origin of reinforcement learning'?",
    "Summarize the key findings of the curriculum learning for data-driven modeling of dynamical systems paper.",
    "How does AlphaGo Zero differ from earlier Go AI approaches?",
]

OUT_DIR = Path("benchmark_outputs")
REPORT_PATH = OUT_DIR / "qa_quality_report.json"
# ---------------------------------------------------------------------------

def evaluate_question(qa: LocalPaperQA, question: str) -> Dict:
    """Run a question through the service and collect simple metrics.

    Returns a dictionary suitable for JSON serialization.
    """
    result = qa.ask(question)
    answer_text = result.answer
    citations = result.citations
    return {
        "question": question,
        "answer_length": len(answer_text),
        "answer_preview": answer_text[:200].replace("\n", " "),
        "num_citations": len(citations),
        "has_citations": bool(citations),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    qa = LocalPaperQA("papers")
    # Ensure index is up‑to‑date
    qa.ensure_index()
    questions = DEFAULT_QUESTIONS
    report = {
        "questions_evaluated": len(questions),
        "results": [evaluate_question(qa, q) for q in questions],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
