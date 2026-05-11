# Local Paper QA

Local-first scientific PDF question answering with a terminal UI, citation-grounded answers, and paper lineage exploration.

## Features

- **Docling-powered PDF parsing** with PyPDF fallback
- **Smart chunking** with section detection, hyphenation cleanup, and overlap
- **Hybrid retrieval** combining embedding similarity (70%) and lexical overlap (30%)
- **Persistent vector store** using SQLite + sqlite-vec for efficient retrieval
- **Terminal UI** with click-through citations, evidence inspection, and paper lineage
- **FastAPI API** for programmatic access
- **Config file** support via `local_paper_qa.toml` or environment variables
- **Automatic folder watcher** for reindexing on PDF changes
- **Citation graph** builder for mapping paper relationships
- **Gold-answer QA benchmark** for evaluating answer quality
- **Latency benchmarks** for indexing and chat performance
- **Unit tests** for core modules

## Requirements

- Python 3.11+ recommended.
- A local OpenAI-compatible chat server.
- A local OpenAI-compatible embedding server.
- Optional: an Exa API key for lineage lookup.

Default local endpoints:

- Chat: `http://100.67.104.58:8001/v1`
- Embeddings: `http://100.67.104.58:8003/v1`
- Model alias: `unsloth/Qwen3.6`

## Setup

```bash
pip install -r requirements.txt
cp local_paper_qa.toml .  # Optional: edit config
```

Override endpoints via environment:

```bash
export LOCAL_PAPER_QA_CHAT_URL=http://host:port/v1
export LOCAL_PAPER_QA_EMBEDDING_URL=http://host:port/v1
export LOCAL_PAPER_QA_CHAT_MODEL=your-model
```

Or create `local_paper_qa.toml`:

```toml
chat_url = "http://100.67.104.58:8001/v1"
embedding_url = "http://100.67.104.58:8003/v1"
chat_model = "unsloth/Qwen3.6"
embedding_model = "unsloth/Qwen3.6"
papers_dir = "papers"
chunk_size = 150
reranking_enabled = true
```

## CLI

```bash
python cli.py --reindex
python cli.py "What are the main findings across these papers?" --json
```

### Folder-based setup
Run the setup bot in the folder that contains your PDFs:

```bash
python setup_bot.py
```

It writes `local_paper_qa.toml` (with `papers_dir = "."`), so the CLI indexes only `*.pdf` in that folder and caches the index under `./.research_index/`.

After it finishes, you can run:

```bash
python cli.py "Your question here"
```

If you add or replace PDFs and want to force a rebuild:

```bash
python cli.py --reindex
```

Notes:
- Indexing is non-recursive (only `*.pdf` in the current directory).
- Cached indexes are reused when PDFs and the embedding configuration match.

## TUI

```bash
python tui.py
```

Useful keys:

- `ctrl+r`: force reindex
- `o`: open the selected evidence PDF
- `l` or `1`: look up paper lineage
- `d`: download and index a lineage paper
- `n` / `p`: navigate citation chain
- `c`: show citation chain count
- `escape`: clear the inspector

## API

```bash
uvicorn api:app --host 0.0.0.0 --port 5060
```

## Production Setup
On GitHub (and in general), do not commit generated artifacts. This repo keeps runtime outputs locally in:

- `papers/.research_index/` (SQLite vector store + index cache)
- `papers/.enhanced_lineage/` (enhanced lineage JSON)
- `papers/citation_graph.json` (citation graph)

These are ignored by git and regenerated as needed.

### Required environment variables
- Set your local OpenAI-compatible servers for chat and embeddings:
  - `LOCAL_PAPER_QA_CHAT_URL`
  - `LOCAL_PAPER_QA_EMBEDDING_URL`
  - `LOCAL_PAPER_QA_CHAT_MODEL` (optional if defaults are OK)
  - `LOCAL_PAPER_QA_EMBEDDING_MODEL` (optional if defaults are OK)

### Optional keys
- `EXA_API_KEY` enables legacy Exa-based lineage lookups.

Endpoints:

- `GET /health`
- `GET /papers`
- `POST /papers`
- `POST /reindex`
- `POST /ask`

## Benchmarks

### QA Quality Benchmark

```bash
python scripts/benchmark_qa_gold.py
```

Creates `benchmark_gold_qa.json` with gold questions. Results saved to `benchmark_outputs/qa_gold_benchmark.json`.

### Index Latency Benchmark

```bash
python scripts/benchmark_index_latency.py
```

### Chat Latency Benchmark

```bash
python scripts/benchmark_local_ai.py
```

### QA Quality Benchmark (Legacy)

```bash
python scripts/benchmark_qa_quality.py
```

## Project Structure

```
local_paper_qa/
  service.py          # Core indexing and retrieval logic
  models.py           # Dataclasses for papers, chunks, citations
  citations.py        # APA citation formatting
  parser.py           # PDF parsing (Docling + PyPDF)
  settings.py         # Config management (TOML + env vars)
  vector_store.py     # SQLite + sqlite-vec vector store
  citation_graph.py   # Citation graph builder
  folder_watcher.py   # Automatic PDF folder watcher
  web_ui.py           # Web UI (commented out - TUI only)
  logger.py           # Logging configuration
  academic/           # Academic API clients (Crossref, arXiv, Semantic Scholar)
  lineage/            # Paper lineage service
  metadata/           # Metadata extraction service

scripts/
  benchmark_local_ai.py       # Chat/embedding latency benchmark
  benchmark_qa_gold.py        # Gold-answer QA benchmark
  benchmark_qa_quality.py     # Lightweight QA benchmark
  benchmark_index_latency.py  # Index build-time benchmark

papers/                     # PDF papers directory
benchmark_outputs/          # Benchmark result JSON files
local_paper_qa.toml         # Configuration file
```

## Repository Hygiene

- `.env` is ignored and must not be committed.
- Generated lineage reports, downloaded PDFs, benchmark outputs, and local indexes are ignored.
- Two sample PDFs are tracked so the project can run immediately after clone.

## Tests

```bash
python -m pytest tests/ -v
```
