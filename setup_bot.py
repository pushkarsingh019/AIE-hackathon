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


def _prompt_choice(name: str, default: str, choices: set[str]) -> str:
    raw = input(f"{name} [{default}] > ").strip().lower()
    value = raw or default
    return value if value in choices else default


def _upsert_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    rendered = f"{key}={value}"
    replaced = False
    next_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            next_lines.append(rendered)
            replaced = True
        else:
            next_lines.append(line)
    if not replaced:
        next_lines.append(rendered)
    path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def main() -> None:
    cwd = Path.cwd()
    toml_path = cwd / "local_paper_qa.toml"
    env_path = cwd / ".env"

    print(WELCOME)
    print("Mission brief: generate local_paper_qa.toml for this folder.")
    print("Indexing will only consider *.pdf files in the folder configured as papers_dir.")
    print()

    if toml_path.exists():
        ans = input(f"Config exists at {toml_path}. Overwrite? [y/N] > ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Keeping existing config. Exiting.")
            return

    chat_provider = _prompt_choice("Chat provider: openai or local", "openai", {"openai", "local"})
    if chat_provider == "local":
        chat_url = _prompt_url("LOCAL_PAPER_QA_CHAT_URL")
        chat_model = _prompt_optional("LOCAL_PAPER_QA_CHAT_MODEL") or "Gemma4 26B"
    else:
        chat_url = "http://100.67.104.58:8002/v1"
        chat_model = _prompt_optional("LOCAL_PAPER_QA_CHAT_MODEL") or "gpt-5.5"

    embedding_provider = _prompt_choice("Embedding provider: openai or gemini", "openai", {"openai", "gemini"})
    openai_api_key = _prompt_optional("OPENAI_API_KEY") if "openai" in {chat_provider, embedding_provider} else None
    gemini_api_key = _prompt_optional("GEMINI_API_KEY") if embedding_provider == "gemini" else None

    # For your desired behavior: index PDFs in *this* folder.
    papers_dir = "."

    # Deterministic defaults that match repo README intent.
    embedding_model = "text-embedding-3-large" if embedding_provider == "openai" else "gemini-embedding-2"
    embedding_dimension = 3072 if embedding_provider == "openai" else 1536
    pairs: list[tuple[str, object]] = [
        ("chat_provider", chat_provider),
        ("chat_url", chat_url),
        ("chat_model", chat_model),
        ("openai_chat_model", "gpt-5.5"),
        ("openai_reasoning_effort", "medium"),
        ("openai_chat_max_output_tokens", 1200),
        ("embedding_provider", embedding_provider),
        ("embedding_model", embedding_model),
        ("embedding_dimension", embedding_dimension),
        ("embedding_batch_size", 64),
        ("indexing_profile", "fast"),
        ("papers_dir", papers_dir),
        ("chunk_size", 150),
        ("chunk_overlap", 0),
        ("max_citations", 8),
        ("reranking_enabled", True),
    ]
    content = _stable_toml(pairs)
    toml_path.write_text(content, encoding="utf-8")
    if openai_api_key:
        _upsert_env_value(env_path, "OPENAI_API_KEY", openai_api_key)
    if gemini_api_key:
        _upsert_env_value(env_path, "GEMINI_API_KEY", gemini_api_key)

    print("\nWrote:")
    print(f"- {toml_path}")
    if openai_api_key or gemini_api_key:
        print(f"- {env_path} (secret values not echoed)")

    print("\nIf you prefer environment variables instead of TOML, export like this:")
    print(f"  export LOCAL_PAPER_QA_CHAT_PROVIDER={chat_provider}")
    if chat_provider == "local":
        print(f"  export LOCAL_PAPER_QA_CHAT_URL={chat_url}")
    if chat_model:
        print(f"  export LOCAL_PAPER_QA_CHAT_MODEL={chat_model}")
    if "openai" in {chat_provider, embedding_provider}:
        print("  export OPENAI_API_KEY=<your-key>")
    if embedding_provider == "gemini":
        print("  export GEMINI_API_KEY=<your-key>")


if __name__ == "__main__":
    main()
