from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_paper_qa.service import LocalPaperQA


PROMPT = "What are the main findings across these papers?"
MODEL = "unsloth/Qwen3.6"
CHAT_URL = "http://100.67.104.58:8001/v1/chat/completions"
EMBEDDING_URL = "http://100.67.104.58:8003/v1/embeddings"
RUNS = 5
OUT_DIR = Path("benchmark_outputs")


def timed_post(url: str, payload: dict) -> tuple[float, dict]:
    start = time.perf_counter()
    response = httpx.post(url, json=payload, timeout=180)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return elapsed, response.json()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    qa = LocalPaperQA("papers")
    papers = qa.ensure_index()
    chunks = [chunk for paper in papers for chunk in paper.chunks]
    evidence = qa.select_evidence(PROMPT, papers, chunks)

    embed_times = []
    embed_sample = {}
    for _ in range(RUNS):
        elapsed, data = timed_post(EMBEDDING_URL, {"model": MODEL, "input": PROMPT})
        embed_times.append(elapsed)
        embed_sample = data

    prompt = "Answer using only this paper evidence.\n\n" + "\n\n".join(
        f"Paper: {c.paper_title}\nPage: {c.page}\nQuote: {c.quote}" for c in evidence
    ) + f"\n\nQuestion: {PROMPT}"

    chat_times = []
    chat_sample = {}
    for _ in range(RUNS):
        elapsed, data = timed_post(
            CHAT_URL,
            {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 512},
        )
        chat_times.append(elapsed)
        chat_sample = data

    embed_mean = statistics.mean(embed_times)
    chat_mean = statistics.mean(chat_times)
    vector = embed_sample.get("data", [{}])[0].get("embedding", [])
    chat_text = chat_sample.get("choices", [{}])[0].get("message", {}).get("content", "")
    report = {
        "prompt": PROMPT,
        "runs": RUNS,
        "embedding_seconds": embed_times,
        "chat_seconds": chat_times,
        "embedding_mean_seconds": embed_mean,
        "chat_mean_seconds": chat_mean,
        "speedup_chat_over_embedding": chat_mean / embed_mean if embed_mean else None,
        "embedding_sample": {"dimension": len(vector), "first_8_values": vector[:8]},
        "chat_sample": chat_text,
    }
    (OUT_DIR / "local_ai_latency_benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
