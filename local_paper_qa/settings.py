"""Configuration for LocalPaperQA.

All settings can be overridden via environment variables:
- LOCAL_PAPER_QA_CHAT_URL
- LOCAL_PAPER_QA_EMBEDDING_URL
- LOCAL_PAPER_QA_CHAT_MODEL
- LOCAL_PAPER_QA_EMBEDDING_MODEL
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_CHAT_URL = "http://100.67.104.58:8001/v1"
_DEFAULT_EMBEDDING_URL = "http://100.67.104.58:8003/v1"
_DEFAULT_CHAT_MODEL = "unsloth/Qwen3.6"
_DEFAULT_EMBEDDING_MODEL = "unsloth/Qwen3.6"


def get_chat_url() -> str:
    return os.environ.get("LOCAL_PAPER_QA_CHAT_URL", _DEFAULT_CHAT_URL)


def get_embedding_url() -> str:
    return os.environ.get("LOCAL_PAPER_QA_EMBEDDING_URL", _DEFAULT_EMBEDDING_URL)


def get_chat_model() -> str:
    return os.environ.get("LOCAL_PAPER_QA_CHAT_MODEL", _DEFAULT_CHAT_MODEL)


def get_embedding_model() -> str:
    return os.environ.get("LOCAL_PAPER_QA_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
