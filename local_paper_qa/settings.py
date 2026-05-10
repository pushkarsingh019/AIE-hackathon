"""Configuration management for LocalPaperQA.

Loads settings from:
1. ``local_paper_qa.toml`` (if present in the working directory)
2. Environment variables (prefixed with ``LOCAL_PAPER_QA_``)
3. Default values

Supported TOML keys:
    chat_url, embedding_url, chat_model, embedding_model, papers_dir,
    chunk_size, chunk_overlap, max_citations, reranking_enabled
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "chat_url": "http://100.67.104.58:8001/v1",
    "embedding_url": "http://100.67.104.58:8003/v1",
    "chat_model": "unsloth/Qwen3.6",
    "embedding_model": "unsloth/Qwen3.6",
    "papers_dir": "papers",
    "chunk_size": 150,
    "chunk_overlap": 0,
    "max_citations": 8,
    "reranking_enabled": True,
}

_ENV_PREFIX = "LOCAL_PAPER_QA_"
_ENV_MAP = {
    "chat_url": f"{_ENV_PREFIX}CHAT_URL",
    "embedding_url": f"{_ENV_PREFIX}EMBEDDING_URL",
    "chat_model": f"{_ENV_PREFIX}CHAT_MODEL",
    "embedding_model": f"{_ENV_PREFIX}EMBEDDING_MODEL",
    "papers_dir": f"{_ENV_PREFIX}PAPERS_DIR",
    "chunk_size": f"{_ENV_PREFIX}CHUNK_SIZE",
    "chunk_overlap": f"{_ENV_PREFIX}CHUNK_OVERLAP",
    "max_citations": f"{_ENV_PREFIX}MAX_CITATIONS",
    "reranking_enabled": f"{_ENV_PREFIX}RERANKING_ENABLED",
}

# ---------------------------------------------------------------------------
# TOML parser (stdlib-free: reads basic TOML manually)
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> Dict[str, Any]:
    """Minimal TOML loader for flat key=value files."""
    result: Dict[str, Any] = {}
    if not path.exists():
        return result
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Try to coerce types
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                result[key] = value
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_cache: Dict[str, Any] | None = None


def _resolve_config() -> Dict[str, Any]:
    """Merge TOML, env vars, and defaults."""
    config = dict(DEFAULTS)
    # 1) TOML file
    toml_path = Path("local_paper_qa.toml")
    toml_config = _load_toml(toml_path)
    config.update(toml_config)
    # 2) Environment variables
    for key, env_var in _ENV_MAP.items():
        env_value = os.environ.get(env_var)
        if env_value is not None and env_value != "":
            # Coerce booleans
            if isinstance(config.get(key), bool):
                config[key] = env_value.lower() in ("true", "1", "yes")
            elif isinstance(config.get(key), int):
                try:
                    config[key] = int(env_value)
                except ValueError:
                    pass
            elif isinstance(config.get(key), float):
                try:
                    config[key] = float(env_value)
                except ValueError:
                    pass
            else:
                config[key] = env_value
    return config


def get_config() -> Dict[str, Any]:
    """Get the resolved configuration (memoised)."""
    global _cache
    if _cache is None:
        _cache = _resolve_config()
    return _cache


def get_chat_url() -> str:
    return get_config().get("chat_url", DEFAULTS["chat_url"])


def get_embedding_url() -> str:
    return get_config().get("embedding_url", DEFAULTS["embedding_url"])


def get_chat_model() -> str:
    return get_config().get("chat_model", DEFAULTS["chat_model"])


def get_embedding_model() -> str:
    return get_config().get("embedding_model", DEFAULTS["embedding_model"])


def get_papers_dir() -> str:
    return get_config().get("papers_dir", DEFAULTS["papers_dir"])


def get_chunk_size() -> int:
    return int(get_config().get("chunk_size", DEFAULTS["chunk_size"]))


def get_chunk_overlap() -> int:
    return int(get_config().get("chunk_overlap", DEFAULTS["chunk_overlap"]))


def get_max_citations() -> int:
    return int(get_config().get("max_citations", DEFAULTS["max_citations"]))


def is_reranking_enabled() -> bool:
    return bool(get_config().get("reranking_enabled", DEFAULTS["reranking_enabled"]))
