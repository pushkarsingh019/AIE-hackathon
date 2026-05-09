from __future__ import annotations

import argparse
import json

from local_paper_qa.service import LocalPaperQA


def main() -> None:
    parser = argparse.ArgumentParser(description="Local paper QA")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--papers-dir", default="papers")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    qa = LocalPaperQA(args.papers_dir)
    if args.reindex:
        papers = qa.ensure_index(force=True)
        print(f"Indexed {len(papers)} papers and {sum(len(p.chunks) for p in papers)} chunks.")
        return

    if not args.question:
        print("Add PDFs to papers/, then ask a question or run --reindex.")
        return

    result = qa.ask(args.question)
    print(json.dumps(result.model_dump(), indent=2) if args.json else result.answer)


if __name__ == "__main__":
    main()
