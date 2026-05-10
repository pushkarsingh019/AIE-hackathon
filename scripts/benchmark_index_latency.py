"""Benchmark indexing build-time.

Measures the time to index all PDFs in papers/ including:
- PDF parsing (Docling/PyPDF)
- Chunking
- Embedding generation
- Index saving

Usage::
    python scripts/benchmark_index_latency.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_paper_qa.service import LocalPaperQA


def benchmark_index(papers_dir: str = "papers", runs: int = 3) -> dict:
    """Benchmark indexing across multiple runs."""
    results = []
    
    for run in range(runs):
        print(f"Run {run + 1}/{runs}...")
        start = time.perf_counter()
        
        qa = LocalPaperQA(papers_dir)
        papers = qa.ensure_index(force=True)
        chunks = [c for p in papers for c in p.chunks]
        
        total_time = time.perf_counter() - start
        
        # Time a single embedding call
        embed_start = time.perf_counter()
        qa.embed_text("test embedding")
        embed_time = time.perf_counter() - embed_start
        
        results.append({
            "run": run + 1,
            "papers_indexed": len(papers),
            "chunks_indexed": len(chunks),
            "total_seconds": round(total_time, 2),
            "embedding_seconds": round(embed_time, 4),
            "chunks_per_second": round(len(chunks) / total_time, 2) if total_time > 0 else 0,
        })
    
    import statistics
    times = [r["total_seconds"] for r in results]
    return {
        "runs": runs,
        "results": results,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "mean_seconds": round(statistics.mean(times), 2),
    }


def main() -> None:
    out_dir = Path("benchmark_outputs")
    out_dir.mkdir(exist_ok=True)
    
    report = benchmark_index("papers", runs=3)
    
    output_file = out_dir / "index_latency_benchmark.json"
    output_file.write_text(json.dumps(report, indent=2))
    
    print(f"\n{'='*60}")
    print(f"Index Build-Time Benchmark")
    print(f"{'='*60}")
    print(f"Mean time: {report['mean_seconds']:.1f}s")
    print(f"Min time:  {report['min_seconds']:.1f}s")
    print(f"Max time:  {report['max_seconds']:.1f}s")
    print(f"\nFull report: {output_file}")


if __name__ == "__main__":
    main()
