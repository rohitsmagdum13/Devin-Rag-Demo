import socket
from pathlib import Path

import pytest

from rag_demo.config import Settings
from rag_demo.models import Usage
from rag_demo.store import VectorStore


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Automated tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]:
        self.calls += 1
        return [[1.0, 0.5, 0.2] for _ in texts], Usage(embedding_tokens=len(texts), requests=1)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        store_path=tmp_path / "chroma", embedding_dimensions=3, chunk_size=50, chunk_overlap=10
    )


@pytest.fixture
def store(settings: Settings) -> VectorStore:
    return VectorStore(settings.store_path, settings.embedding_model, settings.embedding_dimensions)
