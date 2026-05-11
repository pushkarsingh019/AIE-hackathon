from __future__ import annotations

import os
from pathlib import Path


WELCOME = r"""
   ____          __     _____                 ____                 
  / __ \____ ___/ /__  / ___/____  ____ _    / __ \__  _____  _____
 / / / / __ `/ / _ \/ /\__ \/ __ \/ __ `/   / / / / / / / _ \/ ___/
/ /_/ / /_/ / /  __/ /___/ / /_/ / /_/ /   / /_/ / /_/ /  __/ /    
\____/\__,_/_/\___/\____/\____/\__,_/    \____/\__,_/\___/_/     

  local-paper-qa  ::  CONFIGURE MODE  ::  no-AI deterministic setup bot
"""


def _stable_toml(pairs: list[tuple[str, object]]) -> str:
    # Stable output helps people diff configs and keeps setup deterministic.
    def fmt(v: object) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v)
        s = s.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{s}"'

    lines = ["# LocalPaperQA Configuration"]
    for k, v in pairs:
        lines.append(f"{k} = {fmt(v)}")
    lines.append("")
    return "\n".join(lines)


def _prompt_url(name: str) -> str:
    while True:
        raw = input(f"{name} (example: http://host:port/v1) > ").strip()
        if raw:
            return raw
        print("Missing value. Provide a URL.")


def _prompt_optional(s: str) -> str | None:
    raw = input(f"{s} (press Enter to skip) > ").strip()
    return raw or None


def main() -> None:
    cwd = Path.cwd()
    toml_path = cwd / "local_paper_qa.toml"

    print(WELCOME)
    print("Mission brief: generate local_paper_qa.toml for this folder.")
    print("Indexing will only consider *.pdf files in the folder configured as papers_dir.")
    print()

    if toml_path.exists():
        ans = input(f"Config exists at {toml_path}. Overwrite? [y/N] > ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Keeping existing config. Exiting.")
            return

    print("Required endpoints:")
    chat_url = _prompt_url("LOCAL_PAPER_QA_CHAT_URL")
    embedding_url = _prompt_url("LOCAL_PAPER_QA_EMBEDDING_URL")

    chat_model = _prompt_optional("LOCAL_PAPER_QA_CHAT_MODEL") or "unsloth/Qwen3.6"
    embedding_model = _prompt_optional("LOCAL_PAPER_QA_EMBEDDING_MODEL") or "unsloth/Qwen3.6"

    # For your desired behavior: index PDFs in *this* folder.
    papers_dir = "."

    # Deterministic defaults that match repo README intent.
    pairs: list[tuple[str, object]] = [
        ("chat_url", chat_url),
        ("embedding_url", embedding_url),
        ("chat_model", chat_model),
        ("embedding_model", embedding_model),
        ("papers_dir", papers_dir),
        ("chunk_size", 150),
        ("chunk_overlap", 0),
        ("max_citations", 8),
        ("reranking_enabled", True),
    ]

    content = _stable_toml(pairs)
    toml_path.write_text(content, encoding="utf-8")

    print("\nWrote:")
    print(f"- {toml_path}")

    print("\nIf you prefer environment variables instead of TOML, export like this:")
    print(f"  export LOCAL_PAPER_QA_CHAT_URL={chat_url}")
    print(f"  export LOCAL_PAPER_QA_EMBEDDING_URL={embedding_url}")
    if chat_model:
        print(f"  export LOCAL_PAPER_QA_CHAT_MODEL={chat_model}")
    if embedding_model:
        print(f"  export LOCAL_PAPER_QA_EMBEDDING_MODEL={embedding_model}")


if __name__ == "__main__":
    main()
