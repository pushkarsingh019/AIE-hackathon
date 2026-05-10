# Local Paper QA Handoff Summary

## Project Summary

We extracted a standalone local-first research paper QA prototype from the larger Open Notebook repo into:

```text
/Users/pushkarsingh/Documents/side-projects/local-paper-qa
```

The goal is a small, local-only "paper LM" app for asking questions over PDFs with cited, evidence-grounded answers.

## Local AI Setup

The system uses the user's local llama.cpp servers only.

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

The embedding server on port `8003` was verified to support OpenAI-compatible embeddings after being started with:

```bash
--embeddings --pooling mean
```

Verified request:

```bash
curl -s -X POST http://100.67.104.58:8003/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/Qwen3.6","input":"test embedding"}'
```

It returns a 2048-dimensional embedding vector.

## Standalone Repo Contents

Repo path:

```text
/Users/pushkarsingh/Documents/side-projects/local-paper-qa
```

Important files:

```text
local_paper_qa/
  __init__.py
  models.py
  citations.py
  service.py

cli.py
api.py
requirements.txt
README.md
scripts/benchmark_local_ai.py
papers/
```

The app has no dependency on the original Open Notebook codebase.

## Core Functionality

The app supports:

- Reading PDFs from `papers/`
- Extracting text with `pypdf`
- Chunking paper text into paragraph-like chunks
- Extracting basic metadata from PDF metadata:
  - title
  - authors
  - year
  - venue
  - DOI
- Embedding chunks through the local embedding server
- Saving a persistent local index at:

```text
papers/.research_index/index.json
```

- Asking questions over the cached index
- Embedding the question
- Retrieving relevant chunks via cosine similarity
- Sending top evidence chunks to the local chat server
- Returning:
  - answer
  - citations
  - page numbers
  - quotes
  - APA references

## Indexing Flow

Command:

```bash
python cli.py --reindex
```

What it does:

1. Reads all PDFs in `papers/`
2. Extracts page text
3. Splits into chunks
4. Sends every chunk to:

```text
http://100.67.104.58:8003/v1/embeddings
```

5. Stores chunks and embeddings in:

```text
papers/.research_index/index.json
```

This means PDFs are not re-embedded on every question. Reindex is only needed when PDFs change.

## Question Flow

Command:

```bash
python cli.py "What are the main findings across these papers?" --json
```

What it does:

1. Loads the cached index
2. Embeds the user question through port `8003`
3. Finds the top relevant chunks by cosine similarity
4. Sends the question and evidence to chat server on port `8001`
5. Returns JSON with answer and citations

## CLI Usage

From standalone repo:

```bash
cd /Users/pushkarsingh/Documents/side-projects/local-paper-qa
conda activate open-notebook-research
python cli.py --reindex
python cli.py "What are the main findings across these papers?" --json
```

Without JSON:

```bash
python cli.py "What are the main findings across these papers?"
```

## API Usage

Start API:

```bash
cd /Users/pushkarsingh/Documents/side-projects/local-paper-qa
conda activate open-notebook-research
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

Example:

```bash
curl -X POST http://localhost:5060/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main findings across these papers?"}'
```

## Environment Variables

Defaults are hardcoded in `local_paper_qa/service.py`, but can be overridden:

```bash
export LOCAL_PAPER_QA_CHAT_URL=http://100.67.104.58:8001/v1
export LOCAL_PAPER_QA_EMBEDDING_URL=http://100.67.104.58:8003/v1
export LOCAL_PAPER_QA_CHAT_MODEL=unsloth/Qwen3.6
export LOCAL_PAPER_QA_EMBEDDING_MODEL=unsloth/Qwen3.6
```

## Verified Tests

The standalone app was tested with two PDFs copied into `papers/`.

Reindex test:

```bash
conda run -n open-notebook-research python cli.py --reindex
```

Output:

```text
Indexed 2 papers and 124 chunks.
```

CLI QA test:

```bash
conda run -n open-notebook-research python cli.py "What are the main findings across these papers?" --json
```

Worked and returned:

- answer
- 8 citations
- quotes
- page numbers
- APA references

FastAPI test using `TestClient`:

```text
health 200 {'status': 'healthy'}
papers 200 2
ask 200
citations 8
```

Benchmark test:

```bash
conda run -n open-notebook-research python scripts/benchmark_local_ai.py
```

Latest standalone benchmark:

```text
Embedding mean: 0.369s
Chat mean:      11.336s
Chat/embedding: ~30.7x slower
```

This confirms the architecture is right: embed/search is fast, chat is slower, so we should keep PDF chunks indexed and only send selected evidence to chat.

## Current Limitations

This is a working MVP, not a finished product.

Known limitations (remaining):

- APA formatting is approximate.
- No web UI yet.
- No folder watcher yet.
- No automatic reindex on file changes yet.
- No source-highlight viewer yet.
- No table/figure extraction.
- No OCR for scanned PDFs.
- Docling structured extraction not yet working (pipeline init issues).
- No gold-answer QA comparison benchmark (current benchmark is lightweight).

## Recommended Next Features

Next agent should likely work in this order:

1. Fix Docling pipeline integration (StandardPdfPipeline requires specific options).
2. Add automatic folder watcher for `papers/`.
3. Improve APA metadata parsing.
4. Add a simple web UI:
   - upload PDFs
   - list papers
   - reindex
   - ask question
   - show answer
   - show evidence cards
5. Add click-through citation view showing paper/page/quote.
6. Create gold-answer benchmark with human-written reference answers.

## Completed Tasks

The following tasks have been completed on the `sota-indexing` branch:

- **Docling parser**: Added `local_paper_qa/parser.py` with Docling integration and PyPDF fallback.
- **QA quality benchmark**: Added `scripts/benchmark_qa_quality.py` that produces JSON reports.
- **Better section detection**: Improved `_section_heading` to handle numbered sections, all-caps headings, and more academic section names.
- **Chunking improvements**: Better hyphenation handling, larger paragraph threshold (150 words), lower minimum chunk size (20 words).
- **Reranking**: Hybrid reranking combining embedding similarity (70%) and lexical overlap (30%).
- **Config module**: Added `local_paper_qa/settings.py` for all configuration via environment variables.
- **Vector store**: Added `local_paper_qa/vector_store.py` using SQLite + sqlite-vec for persistent vector storage.

## Original Repo Changes

Some prototype code was also added inside the original Open Notebook repo, but the standalone app is the important clean version.

Standalone app path to continue from:

```text
/Users/pushkarsingh/Documents/side-projects/local-paper-qa
```
