# Branch Change Document: `enhanced-lineage`

This repo already contains multiple experimental branches. Right now, the branch you named **`enhanced-lineage`** does **not** diverge from `origin/main`.

## Branch pointers (important)

- `origin/main` and `enhanced-lineage` point to the same commit: `80bf8c46`
- All substantial changes are on **`sota-indexing`** (HEAD commit: `b400a38`)

So: if you want “all the changes we made while building the enhanced lineage version”, the correct target is `sota-indexing`.

## What changed (the real enhanced version)

### Commits unique to `sota-indexing` (vs `origin/main`)

- `b400a38` Implement Phases 1-14: Docling parser, folder watcher, citation graph, gold QA benchmark, TUI citation navigation, SQLite vector store, TOML config, logging, and tests
- `1a64f26` Update handoff summary with completed tasks and remaining limitations
- `797d703` Add persistent vector store (SQLite+sqlite-vec), improve chunking, and fix config import
- `d2393bc` Improve section detection, chunking, add reranking, and config module
- `0395710` Add Docling fallback parser and QA quality benchmark script

### High-level feature areas added/modified

1. **Enhanced lineage pipeline (Semantic Scholar + Crossref + arXiv)**
   - New academic API clients: `local_paper_qa/academic/*`
   - New lineage service: `local_paper_qa/lineage/enhanced_service.py`
   - `LocalPaperQA.paper_lineage()` now prefers enhanced academic lineage and keeps the legacy Exa lineage as fallback.
   - Enhanced lineage JSON is written under `papers/.enhanced_lineage/`.

2. **Config management via TOML + env overrides**
   - `local_paper_qa/settings.py`
   - `local_paper_qa/config/*`
   - Example config: `local_paper_qa.toml`

3. **PDF parsing upgrade (Docling + PyPDF fallback)**
   - `local_paper_qa/parser.py`

4. **Retrieval improvements**
   - Persistent SQLite vector store: `local_paper_qa/vector_store.py`
   - Updated core indexing/retrieval wiring in `local_paper_qa/service.py`

5. **Citation graph + automation + benchmarking**
   - `local_paper_qa/citation_graph.py`
   - `local_paper_qa/folder_watcher.py`
   - Benchmark scripts + gold artifacts under `scripts/` and `benchmark_gold_qa.json`

6. **TUI updates**
   - `tui.py` updated to match the lineage workflow and keys.

7. **Metadata extraction improvements + tests**
   - New extractor: `local_paper_qa/metadata/enhanced_extractor.py`
   - Tests: `tests/test_core.py`, `tests/test_metadata_extractor.py`

### File list (all changes on `sota-indexing` vs `origin/main`)

Modified (`M`):
- `HANDOFF_SUMMARY.md`
- `README.md`
- `local_paper_qa/citations.py`
- `local_paper_qa/service.py`
- `requirements.txt`
- `tui.py`

Added (`A`):
- `benchmark_gold_qa.json`
- `local_paper_qa.toml`
- `local_paper_qa/academic/__init__.py`
- `local_paper_qa/academic/arxiv.py`
- `local_paper_qa/academic/base.py`
- `local_paper_qa/academic/crossref.py`
- `local_paper_qa/academic/manager.py`
- `local_paper_qa/academic/semantic_scholar.py`
- `local_paper_qa/citation_graph.py`
- `local_paper_qa/config/__init__.py`
- `local_paper_qa/config/manager.py`
- `local_paper_qa/folder_watcher.py`
- `local_paper_qa/lineage/__init__.py`
- `local_paper_qa/lineage/enhanced_service.py`
- `local_paper_qa/logger.py`
- `local_paper_qa/metadata/__init__.py`
- `local_paper_qa/metadata/enhanced_extractor.py`
- `local_paper_qa/parser.py`
- `local_paper_qa/settings.py`
- `local_paper_qa/vector_store.py`
- `papers/.enhanced_lineage/*.json` (multiple lineage artifacts)
- `papers/citation_graph.json`
- `scripts/benchmark_index_latency.py`
- `scripts/benchmark_qa_gold.py`
- `scripts/benchmark_qa_quality.py`
- `tests/test_core.py`
- `tests/test_metadata_extractor.py`

### Stress/sanity checks performed

- `python -m compileall -q local_paper_qa` ✅
- `python -m unittest discover -s tests -q` ✅
- Manual lineage smoke tests confirmed:
  - `LocalPaperQA.paper_lineage()` returns schema-compatible output for the TUI
  - lineage artifacts are written into `papers/.enhanced_lineage/`

## If you truly meant the `enhanced-lineage` branch

Right now it’s effectively the same as `origin/main`. If you want, tell me which commit/behavior you expect on that branch (or I can check the reflog), and I’ll regenerate the document for the correct target.
