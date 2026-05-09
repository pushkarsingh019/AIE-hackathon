# Local Paper QA

Standalone local-first scientific PDF question answering.

## What It Uses

- PDFs from `papers/`
- Local chat server: `http://100.67.104.58:8001/v1`
- Local embedding server: `http://100.67.104.58:8003/v1`
- Model alias: `unsloth/Qwen3.6`

No cloud model APIs and no downloaded embedding models are used.

## Quick Start

```bash
conda activate open-notebook-research
pip install -r requirements.txt
python cli.py --reindex
python cli.py "What are the main findings across these papers?" --json
```

## API

```bash
uvicorn api:app --host 0.0.0.0 --port 5060
```

Endpoints:

- `GET /papers`
- `POST /papers`
- `POST /reindex`
- `POST /ask`
