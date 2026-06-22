"""Configuration management for LocalPaperQA.

Loads settings from:
1. ``local_paper_qa.toml`` (if present in the working directory)
2. ``.env`` (if present in the working directory)
3. Environment variables (prefixed with ``LOCAL_PAPER_QA_``)
4. Default values

Supported TOML keys:
    chat_url, embedding_url, chat_model, embedding_model, papers_dir,
    chunk_size, chunk_overlap, max_citations, reranking_enabled

Nested TOML sections are flattened into the same config namespace. For example,
``[embedding] provider = "openai"`` becomes ``embedding_provider``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "chat_provider": "openai",
    "chat_url": "http://100.67.104.58:8002/v1",
    "embedding_url": "http://100.67.104.58:8003/v1",
    "chat_model": "gpt-5.5",
    "embedding_model": "text-embedding-3-large",
    "embedding_provider": "openai",
    "embedding_dimension": 3072,
    "embedding_batch_size": 64,
    "indexing_quality": "best_available",
    "indexing_profile": "deep",
    "gemini_api_key": "",
    "gemini_embedding_model": "gemini-embedding-2",
    "openai_api_key": "",
    "openai_chat_model": "gpt-5.5",
    "openai_reasoning_effort": "medium",
    "openai_chat_max_output_tokens": 1200,
    "openai_embedding_model": "text-embedding-3-large",
    "openai_vision_model": "gpt-5.5",
    "openai_vision_detail": "low",
    "openai_vision_max_output_tokens": 500,
    "multimodal_provider": "openai",
    "multimodal_model": "gpt-5.5",
    "figure_indexing": "auto",
    "figure_indexing_max_candidates": 0,
    "papers_dir": "papers",
    "chunk_size": 150,
    "chunk_overlap": 0,
    "max_citations": 8,
    "reranking_enabled": True,
}

_ENV_PREFIX = "LOCAL_PAPER_QA_"
_ENV_MAP = {
    "chat_provider": f"{_ENV_PREFIX}CHAT_PROVIDER",
    "chat_url": f"{_ENV_PREFIX}CHAT_URL",
    "embedding_url": f"{_ENV_PREFIX}EMBEDDING_URL",
    "chat_model": f"{_ENV_PREFIX}CHAT_MODEL",
    "embedding_model": f"{_ENV_PREFIX}EMBEDDING_MODEL",
    "embedding_provider": f"{_ENV_PREFIX}EMBEDDING_PROVIDER",
    "embedding_dimension": f"{_ENV_PREFIX}EMBEDDING_DIMENSION",
    "embedding_batch_size": f"{_ENV_PREFIX}EMBEDDING_BATCH_SIZE",
    "indexing_quality": f"{_ENV_PREFIX}INDEXING_QUALITY",
    "indexing_profile": f"{_ENV_PREFIX}INDEXING_PROFILE",
    "gemini_api_key": "GEMINI_API_KEY",
    "gemini_embedding_model": f"{_ENV_PREFIX}GEMINI_EMBEDDING_MODEL",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_chat_model": f"{_ENV_PREFIX}OPENAI_CHAT_MODEL",
    "openai_reasoning_effort": f"{_ENV_PREFIX}OPENAI_REASONING_EFFORT",
    "openai_chat_max_output_tokens": f"{_ENV_PREFIX}OPENAI_CHAT_MAX_OUTPUT_TOKENS",
    "openai_embedding_model": f"{_ENV_PREFIX}OPENAI_EMBEDDING_MODEL",
    "openai_vision_model": f"{_ENV_PREFIX}OPENAI_VISION_MODEL",
    "openai_vision_detail": f"{_ENV_PREFIX}OPENAI_VISION_DETAIL",
    "openai_vision_max_output_tokens": f"{_ENV_PREFIX}OPENAI_VISION_MAX_OUTPUT_TOKENS",
    "multimodal_provider": f"{_ENV_PREFIX}MULTIMODAL_PROVIDER",
    "multimodal_model": f"{_ENV_PREFIX}MULTIMODAL_MODEL",
    "figure_indexing": f"{_ENV_PREFIX}FIGURE_INDEXING",
    "figure_indexing_max_candidates": f"{_ENV_PREFIX}FIGURE_INDEXING_MAX_CANDIDATES",
    "papers_dir": f"{_ENV_PREFIX}PAPERS_DIR",
    "chunk_size": f"{_ENV_PREFIX}CHUNK_SIZE",
    "chunk_overlap": f"{_ENV_PREFIX}CHUNK_OVERLAP",
    "max_citations": f"{_ENV_PREFIX}MAX_CITATIONS",
    "reranking_enabled": f"{_ENV_PREFIX}RERANKING_ENABLED",
}

# ---------------------------------------------------------------------------
# TOML parser
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> Dict[str, Any]:
    """Load flat or sectioned TOML into the app's flat config namespace."""
    if not path.exists():
        return {}
    if tomllib is None:
        return _load_flat_toml(path)
    data = tomllib.loads(path.read_text())
    return _flatten_config(data)


def _load_flat_toml(path: Path) -> Dict[str, Any]:
    """Fallback loader for simple key=value files."""
    result: Dict[str, Any] = {}
    section = ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[_flatten_key(section, key.strip())] = _coerce_scalar(value.strip().strip('"').strip("'"))
    return result


def _flatten_config(data: dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                result[_flatten_key(key, child_key)] = child_value
        else:
            result[key] = value
    return result


def _flatten_key(section: str, key: str) -> str:
    if not section:
        return key
    if section == "chat" and key in {"provider", "url", "model"}:
        return f"chat_{key}"
    if section == "embedding" and key in {"provider", "model", "dimension", "batch_size"}:
        return f"embedding_{key}"
    if section == "indexing" and key in {"quality", "profile"}:
        return f"indexing_{key}"
    if section == "indexing" and key == "figure_indexing":
        return "figure_indexing"
    if section == "indexing" and key in {"figure_max_candidates", "figure_indexing_max_candidates"}:
        return "figure_indexing_max_candidates"
    if section == "multimodal" and key in {"provider", "model"}:
        return f"multimodal_{key}"
    if section == "openai" and key in {
        "api_key",
        "chat_model",
        "reasoning_effort",
        "chat_max_output_tokens",
        "embedding_model",
        "vision_model",
        "vision_detail",
        "vision_max_output_tokens",
    }:
        return f"openai_{key}"
    if section == "gemini" and key in {"api_key", "embedding_model"}:
        return f"gemini_{key}"
    return f"{section}_{key}"


def _coerce_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _load_dotenv(path: Path) -> Dict[str, Any]:
    """Load supported environment-style keys from a local .env file."""
    if not path.exists():
        return {}

    env_to_key = {env_var: key for key, env_var in _ENV_MAP.items()}
    result: Dict[str, Any] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        env_name, _, raw_value = stripped.partition("=")
        key = env_to_key.get(env_name.strip())
        if key is None:
            continue
        result[key] = _coerce_scalar(raw_value.strip().strip('"').strip("'"))
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
    # 2) Local .env file
    config.update(_load_dotenv(Path(".env")))
    # 3) Environment variables
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


def get_chat_provider() -> str:
    return get_config().get("chat_provider", DEFAULTS["chat_provider"])


def get_embedding_url() -> str:
    return get_config().get("embedding_url", DEFAULTS["embedding_url"])


def get_chat_model() -> str:
    return get_config().get("chat_model", DEFAULTS["chat_model"])


def get_embedding_model() -> str:
    return get_config().get("embedding_model", DEFAULTS["embedding_model"])


def get_embedding_provider() -> str:
    return get_config().get("embedding_provider", DEFAULTS["embedding_provider"])


def get_embedding_dimension() -> int:
    return int(get_config().get("embedding_dimension", DEFAULTS["embedding_dimension"]))


def get_embedding_batch_size() -> int:
    return int(get_config().get("embedding_batch_size", DEFAULTS["embedding_batch_size"]))


def get_indexing_quality() -> str:
    return get_config().get("indexing_quality", DEFAULTS["indexing_quality"])


def get_indexing_profile() -> str:
    return get_config().get("indexing_profile", DEFAULTS["indexing_profile"])


def get_gemini_api_key() -> str:
    return get_config().get("gemini_api_key", DEFAULTS["gemini_api_key"])


def get_gemini_embedding_model() -> str:
    return get_config().get("gemini_embedding_model", DEFAULTS["gemini_embedding_model"])


def get_openai_api_key() -> str:
    return get_config().get("openai_api_key", DEFAULTS["openai_api_key"])


def get_openai_chat_model() -> str:
    return get_config().get("openai_chat_model", DEFAULTS["openai_chat_model"])


def get_openai_reasoning_effort() -> str:
    return get_config().get("openai_reasoning_effort", DEFAULTS["openai_reasoning_effort"])


def get_openai_chat_max_output_tokens() -> int:
    return int(get_config().get("openai_chat_max_output_tokens", DEFAULTS["openai_chat_max_output_tokens"]))


def get_openai_embedding_model() -> str:
    return get_config().get("openai_embedding_model", DEFAULTS["openai_embedding_model"])


def get_openai_vision_model() -> str:
    return get_config().get("openai_vision_model", DEFAULTS["openai_vision_model"])


def get_openai_vision_detail() -> str:
    return get_config().get("openai_vision_detail", DEFAULTS["openai_vision_detail"])


def get_openai_vision_max_output_tokens() -> int:
    return int(get_config().get("openai_vision_max_output_tokens", DEFAULTS["openai_vision_max_output_tokens"]))


def get_multimodal_provider() -> str:
    return get_config().get("multimodal_provider", DEFAULTS["multimodal_provider"])


def get_multimodal_model() -> str:
    return get_config().get("multimodal_model", DEFAULTS["multimodal_model"])


def get_figure_indexing() -> str:
    return get_config().get("figure_indexing", DEFAULTS["figure_indexing"])


def get_figure_indexing_max_candidates() -> int:
    return int(get_config().get("figure_indexing_max_candidates", DEFAULTS["figure_indexing_max_candidates"]))


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
