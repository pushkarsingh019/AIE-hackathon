# Local Paper QA

Local-first scientific PDF question answering with a terminal UI, citation-grounded answers, and Exa-powered paper lineage exploration.

## Features

- Indexes PDFs from `papers/` into local chunks and embeddings.
- Answers questions with references, claim highlighting, and evidence inspection.
- Opens evidence PDFs from the TUI.
- Shows a terminal lineage flowchart for selected papers using Exa.
- Downloads a lineage/demo paper into `papers/` and reindexes it for follow-up questions.
- Exposes a small FastAPI API for paper listing, upload, reindexing, and QA.

## Requirements

- Python 3.11+ recommended.
- A local OpenAI-compatible chat server.
- A local OpenAI-compatible embedding server.
- Optional: an Exa API key for lineage lookup.

Default local endpoints:

- Chat: `http://100.67.104.58:8001/v1`
- Embeddings: `http://100.67.104.58:8003/v1`
- Model alias: `unsloth/Qwen3.6`

No cloud model APIs are required for normal QA. Exa is only used when you trigger paper lineage lookup.

## Setup

```bash
conda activate open-notebook-research
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` if you want lineage lookup:

```bash
EXA_API_KEY=your_exa_key_here
```

You can also override local model endpoints:

```bash
export LOCAL_PAPER_QA_CHAT_URL=http://host:port/v1
export LOCAL_PAPER_QA_EMBEDDING_URL=http://host:port/v1
export LOCAL_PAPER_QA_CHAT_MODEL=your-model
export LOCAL_PAPER_QA_EMBEDDING_MODEL=your-model
```

## CLI

```bash
python cli.py --reindex
python cli.py "What are the main findings across these papers?" --json
```

## TUI

```bash
python tui.py
```

Useful keys:

- `ctrl+r`: force reindex.
- `o`: open the selected evidence PDF.
- `l` or `1`: look up paper lineage for the selected project paper.
- `d`: after lineage finishes, download the first available lineage paper into `papers/` and reindex.
- `escape`: clear the inspector.

Demo flow:

1. Start the TUI with `python tui.py`.
2. Select a paper in `Papers In Project`.
3. Press `l` or `1` to show the lineage flowchart.
4. Press `d` to download and index a lineage/demo paper.
5. Ask a new question; retrieval now searches across the expanded paper set.

## API

```bash
uvicorn api:app --host 0.0.0.0 --port 5060
```

Endpoints:

- `GET /papers`
- `POST /papers`
- `POST /reindex`
- `POST /ask`

## Repository Hygiene

- `.env` is ignored and must not be committed.
- Generated lineage reports, downloaded PDFs, benchmark outputs, and local indexes are ignored.
- Two sample PDFs are tracked so the project can run immediately after clone.
