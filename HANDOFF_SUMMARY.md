# Local Paper QA Handoff Summary

## Project Summary

A standalone local-first research paper QA prototype for asking questions over PDFs with cited, evidence-grounded answers.

Repo path:
```text
/Users/pushkarsingh/Documents/side-projects/local-paper-qa
```

The app has no dependency on the original Open Notebook codebase.

## Local AI Setup

The system uses local llama.cpp servers only.

Chat server:
```text
http://100.67.104.58:8001/v1
```

Embedding server:
```text
http://100.67.104.58:8003/v1
```

Model alias:
```text
unsloth/Qwen3.6
```

## Branch

All new work is on the `sota-indexing` branch.

## Core Functionality

The app supports:
- Reading PDFs from `papers/`
- **Docling-powered PDF parsing** with PyPDF fallback (structured extraction)
- **Smart chunking** with section detection, hyphenation cleanup, and overlap
- **Hybrid retrieval** combining embedding similarity (70%) and lexical overlap (30%)
- **Persistent vector store** using SQLite + sqlite-vec for efficient retrieval
- Extracting basic metadata from PDF metadata: title, authors, year, venue, DOI
- Embedding chunks through the local embedding server
- Saving a persistent local index at: `papers/.research_index/index.json`
- Asking questions over the cached index
- Returning: answer, citations, page numbers, quotes, APA references
- **Terminal UI** with click-through citations, evidence inspection, and paper lineage

## Indexing Flow

Command:
```bash
python cli.py --reindex
```

What it does:
1. Reads all PDFs in `papers/`
2. Extracts page text via Docling (falls back to PyPDF)
3. Splits into smart chunks (section-aware, hyphenation cleaned)
4. Embeds every chunk through:
```text
http://100.67.104.58:8003/v1/embeddings
```
5. Stores chunks and embeddings in both:
   - `papers/.research_index/index.json` (legacy JSON index)
   - `papers/.research_index/vectors.db` (SQLite + sqlite-vec)
6. Automatic folder watcher detects PDF changes and reindexes

## Question Flow

Command:
```bash
python cli.py "What are the main findings across these papers?" --json
```

What it does:
1. Loads the cached index (from JSON and SQLite)
2. Embeds the user question
3. Finds top relevant chunks by cosine similarity + lexical reranking
4. Sends the question and evidence to chat server on port `8001`
5. Returns JSON with answer and citations

## CLI Usage

```bash
cd /Users/pushkarsingh/Documents/side-projects/local-paper-qa
python cli.py --reindex
python cli.py "What are the main findings across these papers?" --json
```

Without JSON:
```bash
python cli.py "What are the main findings across these papers?"
```

## TUI Usage

```bash
python tui.py
```

Useful keys:
- `ctrl+r`: force reindex
- `o`: open the selected evidence PDF
- `l` or `1`: look up paper lineage
- `d`: download and index a lineage paper
- `n` / `p`: navigate citation chain (next/prev)
- `c`: show citation chain count
- `escape`: clear the inspector

## API Usage

Start API:
```bash
uvicorn api:app --host 0.0.0.0 --port 5060
```

Endpoints:
```text
GET  /health
GET  /papers
POST /papers
POST /reindex
POST /ask
```

## Configuration

### TOML Config

Copy `local_paper_qa.toml` and edit:
```toml
chat_url = "http://100.67.104.58:8001/v1"
embedding_url = "http://100.67.104.58:8003/v1"
chat_model = "unsloth/Qwen3.6"
embedding_model = "unsloth/Qwen3.6"
papers_dir = "papers"
chunk_size = 150
reranking_enabled = true
```

### Environment Variables

```bash
export LOCAL_PAPER_QA_CHAT_URL=http://100.67.104.58:8001/v1
export LOCAL_PAPER_QA_EMBEDDING_URL=http://100.67.104.58:8003/v1
export LOCAL_PAPER_QA_CHAT_MODEL=unsloth/Qwen3.6
export LOCAL_PAPER_QA_EMBEDDING_MODEL=unsloth/Qwen3.6
```

Priority: env vars > TOML file > defaults.

## Benchmarks

### Gold-Answer QA Benchmark
```bash
python scripts/benchmark_qa_gold.py
```
Reads `benchmark_gold_qa.json`, outputs `benchmark_outputs/qa_gold_benchmark.json`.

### Index Latency Benchmark
```bash
python scripts/benchmark_index_latency.py
```
Outputs `benchmark_outputs/index_latency_benchmark.json`.

### Chat Latency Benchmark
```bash
python scripts/benchmark_local_ai.py
```
Outputs `benchmark_outputs/local_ai_latency_benchmark.json`.

### QA Quality Benchmark (Legacy)
```bash
python scripts/benchmark_qa_quality.py
```
Outputs `benchmark_outputs/qa_quality_report.json`.

## Unit Tests

```bash
python -m pytest tests/test_core.py -v
```
11 tests covering: APA formatting, model defaults, TOML config, parser fallback.

## Project Structure

```text
local_paper_qa/
  service.py              # Core indexing and retrieval logic
  models.py               # Dataclasses: PaperDocument, PaperChunk, PaperCitation
  citations.py            # APA citation formatting (improved)
  parser.py               # PDF parsing (Docling + PyPDF fallback)
  settings.py             # Config management (TOML + env vars + defaults)
  vector_store.py         # SQLite + sqlite-vec persistent vector store
  citation_graph.py       # Co-citation graph builder
  folder_watcher.py       # Watchdog-based directory watcher
  logger.py               # Logging configuration
  academic/               # Academic API clients (Crossref, arXiv, Semantic Scholar)
    base.py, manager.py
    crossref.py, arxiv.py, semantic_scholar.py
  lineage/                # Paper lineage service
    enhanced_service.py
  metadata/               # Metadata extraction service
    enhanced_extractor.py

scripts/
  benchmark_local_ai.py       # Chat/embedding latency benchmark
  benchmark_qa_gold.py        # Gold-answer QA benchmark
  benchmark_qa_quality.py     # Lightweight QA benchmark
  benchmark_index_latency.py  # Index build-time benchmark

papers/                     # PDF papers directory
benchmark_outputs/          # Benchmark result JSON files
local_paper_qa.toml         # Example config file
tests/
  test_core.py              # 11 unit tests
tui.py                      # Terminal UI
api.py                      # FastAPI endpoints
cli.py                      # CLI entry point
```

## Verified Tests

Reindex:
```bash
python cli.py --reindex
```
Output: `Indexed 4 papers and 421 chunks.`

CLI QA:
```bash
python cli.py "What are the main findings across these papers?" --json
```
Returns: answer, 8 citations, quotes, page numbers, APA references.

Benchmarks:
- Index build: ~0.8s mean (4 papers, 421 chunks)
- Chat latency: ~14s mean
- Token overlap (gold QA): 0.917

## Git History (sota-indexing branch)

```
60ef576 Fix chat not showing answers when structured segments are empty
b400a38 Implement Phases 1-14: full feature set
797d703 Add persistent vector store (SQLite+sqlite-vec), improve chunking
d2393bc Improve section detection, chunking, add reranking, config module
0395710 Add Docling fallback parser and QA quality benchmark script
80bf8c4 Add TUI paper lineage workflow
d3701d7 Initial local paper QA app
```

## Dependencies Added

- `docling>=2.0.0` - Structured PDF parsing
- `sqlite-vec>=0.1.0` - Vector similarity search in SQLite
- `watchdog` - File system watching for auto-reindex
- `pytest` - Unit testing

## Known Limitations (Remaining)

- No web UI (TUI-only by user preference)
- Docling structured extraction works but can be slow (~60s per PDF)
- No automatic reindex on file changes yet (folder watcher built but not wired into TUI)
- No OCR for scanned PDFs
- No table/figure extraction from PDFs
- APA formatting is approximate
- No source-highlight viewer yet
- No gold-answer benchmark with human-written reference answers

## Recommended Next Steps

1. Wire folder watcher into TUI auto-reindex
2. Optimize Docling (use SimplePipeline for faster extraction)
3. Add web UI (if desired)
4. Add OCR support for scanned PDFs
5. Create human-written gold QA benchmark

## Standalone app path to continue from:

```text
/Users/pushkarsingh/Documents/side-projects/local-paper-qa
```
