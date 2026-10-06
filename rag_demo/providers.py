import os
from typing import Protocol

from openai import OpenAI

from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.models import Usage


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]: ...


class OpenAIProvider:
    def __init__(self, settings: Settings) -> None:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key == "replace-with-your-key":
            raise AppError("Set OPENAI_API_KEY in your environment or local .env, then retry.")
        self.settings = settings
        self.client = OpenAI(api_key=key, timeout=settings.timeout_seconds, max_retries=1)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]:
        vectors: list[list[float]] = []
        usage = Usage()
        try:
            for start in range(0, len(texts), 32):
                batch = texts[start : start + 32]
                # UTF-8 bytes conservatively bound byte-level tokenizer input tokens.
                if any(len(text.encode("utf-8")) > 8000 for text in batch):
                    raise AppError("Chunks exceed 8000 UTF-8 bytes. Reduce CHUNK_SIZE and retry.")
                response = self.client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=batch,
                    dimensions=self.settings.embedding_dimensions,
                )
                vectors.extend(
                    item.embedding for item in sorted(response.data, key=lambda x: x.index)
                )
                usage.add(Usage(embedding_tokens=response.usage.total_tokens, requests=1))
        except AppError:
            raise
        except Exception:
            raise AppError(
                "OpenAI embedding failed. Check your key, quota, model, and connection."
            ) from None
        return vectors, usage
