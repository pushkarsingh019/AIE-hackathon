# Local Paper QA

Local-first scientific PDF question answering with a terminal UI, citation-grounded answers, and paper lineage exploration.

## Features

- **Docling-powered PDF parsing** with PyPDF fallback
- **Smart chunking** with section detection, hyphenation cleanup, and overlap
- **Hybrid retrieval** combining hosted embedding similarity with lexical overlap
- **Persistent Extracted Corpus** using SQLite for spans, retrieval representations, and embeddings
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
- An OpenAI API key for default chat and retrieval embeddings (`OPENAI_API_KEY`).
- Optional: a local OpenAI-compatible chat server if you switch chat away from OpenAI.
- Optional: a Gemini API key if you switch retrieval embeddings back to Gemini.
- Optional: an Exa API key for lineage lookup.

Default model configuration:

- Chat provider: OpenAI Responses API
- Chat model: `gpt-5.5`
- Chat reasoning effort: `medium`
- Embeddings: `text-embedding-3-large`

## Setup

```bash
pip install -r requirements.txt
cp local_paper_qa.toml .  # Optional: edit config
```

Override chat and embedding settings via environment:

```bash
export LOCAL_PAPER_QA_CHAT_URL=http://host:port/v1
export LOCAL_PAPER_QA_CHAT_MODEL=your-model
export OPENAI_API_KEY=your-key
```

Or create `local_paper_qa.toml`:

```toml
chat_provider = "openai"
chat_url = "http://100.67.104.58:8002/v1"
chat_model = "gpt-5.5"
openai_chat_model = "gpt-5.5"
openai_reasoning_effort = "medium"
openai_chat_max_output_tokens = 1200
embedding_provider = "openai"
embedding_model = "text-embedding-3-large"
embedding_dimension = 3072
indexing_profile = "fast"  # Cheaper smoke runs; use "deep" for richer indexing.
multimodal_provider = "openai"
multimodal_model = "gpt-5.5"
openai_vision_detail = "low"
openai_vision_max_output_tokens = 500
papers_dir = "papers"
chunk_size = 150
reranking_enabled = true
```

To use Gemini embeddings instead, set `embedding_provider = "gemini"`,
`embedding_model = "gemini-embedding-2"`, `embedding_dimension = 1536`, and provide
`GEMINI_API_KEY`.

To use a local OpenAI-compatible chat server instead:

```toml
chat_provider = "local"
chat_url = "http://100.67.104.58:8002/v1"
chat_model = "Gemma4 26B"
```

Figure/page image indexing is opt-in because it can spend vision tokens:

```toml
indexing_profile = "deep_figures"
figure_indexing = "auto"
multimodal_provider = "openai"
multimodal_model = "gpt-5.5"
openai_vision_detail = "low"
```

`fast` and `deep` remain text-only indexing profiles.

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

### Agent prompt (copy/paste)
Use the following prompt if you want another automation agent to get this running in a new folder:

```text
You are in a folder that contains research PDF files. Your goal is to set up local-paper-qa for this folder.

Steps:
1. Ensure Python dependencies are installed: pip install -r requirements.txt
2. Run the deterministic setup bot to generate local_paper_qa.toml: python setup_bot.py
   - Provide LOCAL_PAPER_QA_CHAT_URL and optionally GEMINI_API_KEY when prompted.
   - Confirm overwrite if the config already exists.
3. Verify there are PDF files (*.pdf) in this folder.
4. Build/load the index and answer once:
   - python cli.py "What are the main findings across these papers?"
5. If indexing looks stale after changing PDFs, force rebuild:
   - python cli.py --reindex "What are the main findings across these papers?"

Constraints/behavior:
- Indexing is non-recursive; only *.pdf in the current directory are included.
- The cache is stored under ./.research_index/ and is reused when PDFs + embedding config match.
```

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
- Set OpenAI access:
  - `OPENAI_API_KEY`
- Configure OpenAI chat if needed:
  - `LOCAL_PAPER_QA_CHAT_PROVIDER` (optional; defaults to `openai`)
  - `LOCAL_PAPER_QA_CHAT_MODEL` (optional; defaults to `gpt-5.5`)
  - `LOCAL_PAPER_QA_OPENAI_REASONING_EFFORT` (optional; defaults to `medium`)
- Configure OpenAI embeddings if needed:
  - `LOCAL_PAPER_QA_EMBEDDING_PROVIDER` (optional; defaults to `openai`)
  - `LOCAL_PAPER_QA_EMBEDDING_MODEL` (optional; defaults to `text-embedding-3-large`)
  - `LOCAL_PAPER_QA_OPENAI_EMBEDDING_MODEL` (optional; fallback model when `embedding_model` is not OpenAI)

### Optional keys
- `GEMINI_API_KEY` enables Gemini embeddings if configured.
- `EXA_API_KEY` enables legacy Exa-based lineage lookups.

Endpoints:

- `GET /health`
- `GET /papers`
- `POST /papers`
- `POST /reindex`
- `POST /ask`

## Benchmarks

### Evidence Retrieval Benchmark

```bash
python scripts/benchmark_retrieval_gold.py --run-name openai-fast
python scripts/benchmark_retrieval_gold.py --cached-only --max-cases 3 --run-name smoke
```

Edit `benchmark_cases.json` with expected papers and evidence terms. Results are saved to `benchmark_outputs/retrieval_gold_benchmark*.json`. This is the benchmark to use first when comparing embedding and retrieval setups.

Cost controls:

- By default this benchmark evaluates retrieval only; it does not synthesize answers unless `--include-answer` is passed.
- Use `--cached-only` to fail instead of making fresh embedding calls.
- Use `--max-cases N` for low-credit smoke runs.

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
