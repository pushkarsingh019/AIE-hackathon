from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from local_paper_qa.settings import (
    get_chat_model,
    get_embedding_batch_size,
    get_embedding_dimension,
    get_embedding_model,
    get_embedding_provider,
    get_gemini_api_key,
    get_gemini_embedding_model,
    get_indexing_profile,
    get_multimodal_model,
    get_multimodal_provider,
    get_openai_api_key,
    get_openai_chat_max_output_tokens,
    get_openai_chat_model,
    get_openai_embedding_model,
    get_openai_reasoning_effort,
    get_openai_vision_detail,
    get_openai_vision_max_output_tokens,
    get_openai_vision_model,
)


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot return usable vectors."""


class VisionError(RuntimeError):
    """Raised when a vision provider cannot describe an image."""


class ChatError(RuntimeError):
    """Raised when a chat provider cannot return a usable answer."""


@dataclass(frozen=True)
class EmbeddingProviderInfo:
    provider: str
    model: str
    dimension: int
    profile: str


class EmbeddingProvider(Protocol):
    info: EmbeddingProviderInfo

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class ChatClientInfo:
    provider: str
    model: str
    reasoning_effort: str


class ChatClient(Protocol):
    info: ChatClientInfo

    def complete(self, prompt: str, *, max_output_tokens: int | None = None) -> str:
        ...


class GeminiEmbeddingProvider:
    def __init__(self, api_key: str, model: str, dimension: int, batch_size: int, profile: str):
        if not api_key:
            raise EmbeddingError("Gemini embedding provider requires GEMINI_API_KEY or gemini_api_key config.")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on optional environment state
            raise EmbeddingError("Install google-genai to use Gemini embeddings.") from exc

        self.client = genai.Client(api_key=api_key)
        self.batch_size = max(1, batch_size)
        self.info = EmbeddingProviderInfo(
            provider="gemini",
            model=model,
            dimension=dimension,
            profile=profile,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            for text in texts[start : start + self.batch_size]:
                results.append(self._embed(f"title: none | text: {text.strip()}"))
        return results

    def embed_query(self, text: str) -> list[float]:
        text = text.strip()
        if not text:
            return []
        return self._embed(f"task: question answering | query: {text}")

    def _embed(self, content: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.info.model,
            contents=content,
            config={"output_dimensionality": self.info.dimension},
        )
        embeddings = getattr(result, "embeddings", None) or []
        if not embeddings:
            raise EmbeddingError("Gemini embedding response did not include embeddings.")
        values = getattr(embeddings[0], "values", None)
        if values is None:
            raise EmbeddingError("Gemini embedding response did not include values.")
        embedding = [float(value) for value in values]
        if not embedding:
            raise EmbeddingError("Gemini embedding response was empty.")
        return embedding


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str, model: str, dimension: int, batch_size: int, profile: str):
        if not api_key:
            raise EmbeddingError("OpenAI embedding provider requires OPENAI_API_KEY or openai_api_key config.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional environment state
            raise EmbeddingError("Install openai to use OpenAI embeddings.") from exc

        self.client = OpenAI(api_key=api_key)
        self.batch_size = max(1, batch_size)
        self.info = EmbeddingProviderInfo(
            provider="openai",
            model=model,
            dimension=dimension,
            profile=profile,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts if text.strip()]
        if len(cleaned) != len(texts):
            raise EmbeddingError("OpenAI embedding inputs must be non-empty strings.")

        results: list[list[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.info.model,
                input=batch,
                dimensions=self.info.dimension,
                encoding_format="float",
            )
            results.extend(_openai_embedding_values(response))
        return results

    def embed_query(self, text: str) -> list[float]:
        text = text.strip()
        if not text:
            return []
        response = self.client.embeddings.create(
            model=self.info.model,
            input=[text],
            dimensions=self.info.dimension,
            encoding_format="float",
        )
        values = _openai_embedding_values(response)
        return values[0] if values else []


class OpenAIVisionClient:
    """Small Responses API wrapper for describing paper figures or pages."""

    def __init__(self, api_key: str, model: str, detail: str, max_output_tokens: int):
        if not api_key:
            raise VisionError("OpenAI vision provider requires OPENAI_API_KEY or openai_api_key config.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional environment state
            raise VisionError("Install openai to use OpenAI vision models.") from exc

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.detail = detail
        self.max_output_tokens = max(64, max_output_tokens)

    def describe_image(self, image_path: str | Path, prompt: str) -> str:
        path = Path(image_path)
        image_url = f"data:{_image_mime_type(path)};base64,{_base64_file(path)}"
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt.strip()},
                        {"type": "input_image", "image_url": image_url, "detail": self.detail},
                    ],
                }
            ],
            max_output_tokens=self.max_output_tokens,
        )
        text = _response_output_text(response)
        if not text:
            raise VisionError("OpenAI vision response did not include text output.")
        return text


class OpenAIChatClient:
    def __init__(self, api_key: str, model: str, reasoning_effort: str, max_output_tokens: int):
        if not api_key:
            raise ChatError("OpenAI chat provider requires OPENAI_API_KEY or openai_api_key config.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional environment state
            raise ChatError("Install openai to use OpenAI chat models.") from exc

        self.client = OpenAI(api_key=api_key)
        self.max_output_tokens = max(64, max_output_tokens)
        self.info = ChatClientInfo(
            provider="openai",
            model=model,
            reasoning_effort=reasoning_effort,
        )

    def complete(self, prompt: str, *, max_output_tokens: int | None = None) -> str:
        prompt = prompt.strip()
        if not prompt:
            return ""
        response = self.client.responses.create(
            model=self.info.model,
            reasoning={"effort": self.info.reasoning_effort},
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=max_output_tokens or self.max_output_tokens,
        )
        text = _response_output_text(response)
        if not text:
            raise ChatError("OpenAI chat response did not include text output.")
        return text


def create_chat_client() -> ChatClient:
    model = get_chat_model().strip()
    if not model.startswith("gpt-"):
        model = get_openai_chat_model()
    return OpenAIChatClient(
        api_key=get_openai_api_key().strip(),
        model=model,
        reasoning_effort=get_openai_reasoning_effort().strip() or "medium",
        max_output_tokens=get_openai_chat_max_output_tokens(),
    )


def create_embedding_provider() -> EmbeddingProvider:
    provider = get_embedding_provider().strip().lower()
    dimension = get_embedding_dimension()
    batch_size = get_embedding_batch_size()
    profile = get_indexing_profile()
    gemini_key = get_gemini_api_key().strip()
    openai_key = get_openai_api_key().strip()

    if provider in {"openai", "auto"}:
        return OpenAIEmbeddingProvider(
            api_key=openai_key,
            model=_openai_embedding_model_name(),
            dimension=dimension,
            batch_size=batch_size,
            profile=profile,
        )
    if provider in {"gemini", "google"}:
        return GeminiEmbeddingProvider(
            api_key=gemini_key,
            model=_gemini_model_name(),
            dimension=dimension,
            batch_size=batch_size,
            profile=profile,
        )
    raise EmbeddingError(
        f"Unsupported embedding provider '{provider}'. Local embedding providers are disabled; use OpenAI or Gemini."
    )


def create_vision_client() -> OpenAIVisionClient:
    provider = get_multimodal_provider().strip().lower()
    if provider not in {"openai", "auto"}:
        raise VisionError(
            f"Unsupported multimodal provider '{provider}'. Use OpenAI for figure/page image understanding."
        )
    model = get_multimodal_model().strip() or get_openai_vision_model()
    return OpenAIVisionClient(
        api_key=get_openai_api_key().strip(),
        model=model,
        detail=get_openai_vision_detail().strip() or "low",
        max_output_tokens=get_openai_vision_max_output_tokens(),
    )


def _gemini_model_name() -> str:
    configured = get_embedding_model().strip()
    if configured.startswith("gemini-"):
        return configured
    return get_gemini_embedding_model()


def _openai_embedding_model_name() -> str:
    configured = get_embedding_model().strip()
    if configured.startswith("text-embedding-"):
        return configured
    return get_openai_embedding_model()


def _openai_embedding_values(response: object) -> list[list[float]]:
    data = getattr(response, "data", None) or []
    values: list[list[float]] = []
    for item in data:
        embedding = getattr(item, "embedding", None)
        if embedding is None and isinstance(item, dict):
            embedding = item.get("embedding")
        if not embedding:
            raise EmbeddingError("OpenAI embedding response included an empty embedding.")
        values.append([float(value) for value in embedding])
    if not values:
        raise EmbeddingError("OpenAI embedding response did not include embeddings.")
    return values


def _response_output_text(response: object) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str):
        return direct.strip()
    if isinstance(response, dict):
        direct = response.get("output_text")
        if isinstance(direct, str):
            return direct.strip()
    return ""


def _base64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"
