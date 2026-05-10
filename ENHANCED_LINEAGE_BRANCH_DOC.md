# Branch Document: `sota-indexing`

This document summarizes the changes introduced on the current branch (vs `origin/main`) so you can quickly understand what was added/modified and where.

## What changed (high level)

1. **Academic paper lineage (new, cleaner path)**
   - Added a multi-source academic lineage pipeline using:
     - Semantic Scholar (Graph API)
     - Crossref (DOI + metadata)
     - arXiv (preprint + metadata)
   - Implemented in `local_paper_qa/academic/*` and `local_paper_qa/lineage/enhanced_service.py`.
   - `LocalPaperQA.paper_lineage()` now prefers this enhanced pipeline and keeps the legacy Exa lineage logic as fallback.

2. **Config system**
   - Introduced TOML-based configuration via `local_paper_qa.toml` and env overrides.
   - Implemented in `local_paper_qa/settings.py` and `local_paper_qa/config/manager.py`.

3. **Better PDF parsing + indexing plumbing**
   - `local_paper_qa/parser.py` was updated to support **Docling** extraction with PyPDF fallback.

4. **Vector store / retrieval improvements**
   - Added a persistent SQLite-based vector store (`local_paper_qa/vector_store.py`).
   - `local_paper_qa/service.py` was updated to use this for faster retrieval.

5. **Citation graph + evaluation + QA benchmarks**
   - Added `local_paper_qa/citation_graph.py`.
   - Added gold QA benchmark artifacts and scripts under `scripts/`.
   - Added tests under `tests/`.

6. **TUI updates**
   - `tui.py` updated to reflect new capabilities while keeping the existing “lineage” workflow.

## File-by-file summary (vs `origin/main`)

### README + docs / handoff
- `README.md` (modified): describes new features (Docling parsing, hybrid retrieval, vector store, lineage exploration, config file/TOML, watcher, citation graph, benchmarks, tests).
- `HANDOFF_SUMMARY.md` (modified): keeps repo context and setup notes.
- `local_paper_qa.toml` (added): example config file for the TOML-based settings.

### Academic clients (new)
- `local_paper_qa/academic/base.py` (added): shared dataclasses + client interface.
- `local_paper_qa/academic/semantic_scholar.py` (added): Semantic Scholar Graph API client (paper lookup + citations/references parsing).
- `local_paper_qa/academic/crossref.py` (added): Crossref DOI lookup + metadata parsing.
- `local_paper_qa/academic/arxiv.py` (added): arXiv query + PDF URL generation.
- `local_paper_qa/academic/manager.py` (added): multi-client manager + fallback strategy + confidence scoring.
- `local_paper_qa/academic/__init__.py` (added)

### Lineage service (new)
- `local_paper_qa/lineage/enhanced_service.py` (added): builds an enhanced lineage report and writes lineage JSON into `papers/.enhanced_lineage/`.
- `local_paper_qa/lineage/__init__.py` (added)

### Config / settings (new/modified)
- `local_paper_qa/config/manager.py` (added): configuration manager abstraction.
- `local_paper_qa/config/__init__.py` (added)
- `local_paper_qa/settings.py` (added): primary config loader (TOML + env + defaults).

### Parsing / metadata extraction (new)
- `local_paper_qa/parser.py` (modified): Docling extraction with PyPDF fallback.
- `local_paper_qa/metadata/enhanced_extractor.py` (added): DOI detection + “best-effort” metadata extraction from PDF text.
- `local_paper_qa/metadata/__init__.py` (added)

### Core service changes
- `local_paper_qa/service.py` (modified):
  - uses the new config system for chat/embedding URLs
  - integrates the new vector store
  - integrates enhanced lineage as the default lineage path (fallback retained)
  - improves PDF metadata extraction by supplementing PDF metadata with `EnhancedMetadataExtractor` when needed.

### Retrieval and indexing
- `local_paper_qa/vector_store.py` (added): SQLite vector store.

### Citation formatting
- `local_paper_qa/citations.py` (modified): updated/extended APA formatting behavior for the new citation flow.

### Citation graph + automation
- `local_paper_qa/citation_graph.py` (added): builds a paper relationship graph.
- `local_paper_qa/folder_watcher.py` (added): auto-reindex on PDF changes.
- `local_paper_qa/logger.py` (added)

### Benchmarks + scripts
- `scripts/benchmark_index_latency.py` (added)
- `scripts/benchmark_qa_gold.py` (added)
- `scripts/benchmark_qa_quality.py` (added)
- `benchmark_gold_qa.json` (added): gold benchmark questions output.

### Tests
- `tests/test_core.py` (added): core behavior tests.
- `tests/test_metadata_extractor.py` (added): metadata extractor test.

### TUI
- `tui.py` (modified): lineage UI strings/behavior aligned with the lineage workflow.

### Dependencies
- `requirements.txt` (modified): includes any new dependencies introduced by Docling/vector store/benchmarks.

## Runtime behavior changes (what you’ll notice)

1. **Lineage lookup output location**
   - New enhanced lineage JSON is written under: `papers/.enhanced_lineage/`.
   - The TUI continues to treat lineage results as JSON-backed lineage data.

2. **Lineage improves coverage**
   - Instead of relying only on Exa, lineage attempts to enrich results using structured academic APIs.

3. **Config file support**
   - You can now set endpoints/models and retrieval parameters in `local_paper_qa.toml`.
   - Environment variables still override TOML.

4. **PDF parsing**
   - Docling is attempted first; PyPDF remains the fallback.

5. **Retrieval speed**
   - Persistent SQLite vector storage reduces repeated embedding work and speeds up retrieval.

## Stress test / sanity checks performed

- `python -m compileall -q local_paper_qa` ✅
- `python -m unittest discover -s tests -q` ✅
- Manual lineage smoke tests for existing local PDFs: confirmed `paper_lineage()` returns the expected schema and produces a `lineage_file`.

## How to run (quick)

- Reindex: `python cli.py --reindex`
- Ask: `python cli.py "<question>"`
- TUI: `python tui.py`
- API: `uvicorn api:app --host 0.0.0.0 --port 5060`

## Notes / caveats

- Academic APIs (Semantic Scholar / arXiv / Crossref) are rate-limited; during stress runs you may see throttling errors in logs.
- Lineage JSON artifacts may be repeatedly generated under `papers/.enhanced_lineage/` depending on parameters and API availability.
