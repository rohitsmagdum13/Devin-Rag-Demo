from dataclasses import replace
from unittest.mock import Mock

import pytest

from rag_demo.chunking import chunk_sections
from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.ingestion import ingest
from rag_demo.models import TextSection
from rag_demo.providers import OpenAIProvider
from rag_demo.store import VectorStore
from tests.conftest import FakeEmbedder
from tests.test_extraction import pdf_bytes


def test_chunk_overlap_and_metadata() -> None:
    chunks = chunk_sections(
        [TextSection("abcdefghij", 2), TextSection("klmn", 3)], "hash", "a.pdf", 6, 2
    )
    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij", "klmn"]
    assert [chunk.page for chunk in chunks] == [2, 2, 3]
    assert [chunk.id for chunk in chunks] == ["hash:0", "hash:1", "hash:2"]
    assert all(chunk.document_name == "a.pdf" and chunk.document_id == "hash" for chunk in chunks)


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, -1)])
def test_invalid_chunk_configuration(size: int, overlap: int) -> None:
    with pytest.raises(AppError):
        chunk_sections([], "hash", "name", size, overlap)


def test_duplicate_and_persistence(settings: Settings, store: VectorStore) -> None:
    embedder = FakeEmbedder()
    first = ingest("policy.txt", b"Reimburse within 30 days", settings, store, embedder)
    duplicate = ingest("renamed.txt", b"Reimburse within 30 days", settings, store, embedder)
    assert not first.duplicate and duplicate.duplicate
    assert first.document_id == duplicate.document_id and embedder.calls == 1
    reopened = VectorStore(settings.store_path, settings.embedding_model, 3)
    assert reopened.count() == first.chunk_count
    source = reopened.search([1.0, 0.5, 0.2], 5)[0]
    assert source.document_name == "policy.txt" and source.page is None
    assert "30 days" in source.text and source.id.startswith(first.document_id)


def test_embedding_failure_no_partial_document(settings: Settings, store: VectorStore) -> None:
    provider = Mock()
    provider.embed.side_effect = AppError("Embedding failed")
    with pytest.raises(AppError):
        ingest("policy.txt", b"Some policy", settings, store, provider)
    assert store.count() == 0
    assert not ingest("policy.txt", b"Some policy", settings, store, FakeEmbedder()).duplicate


def test_invalid_vectors_no_partial_document(store: VectorStore) -> None:
    chunks = chunk_sections([TextSection("sample")], "hash", "sample.txt", 10, 0)
    with pytest.raises(AppError, match="dimensions"):
        store.add(chunks, [[1.0, 2.0]])
    assert store.count() == 0


def test_incompatible_index(settings: Settings, store: VectorStore) -> None:
    with pytest.raises(AppError, match="differ"):
        VectorStore(settings.store_path, "different-model", 3)


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AppError, match="OPENAI_API_KEY"):
        OpenAIProvider(Settings())


def test_config_validation() -> None:
    with pytest.raises(AppError):
        replace(Settings(), chunk_overlap=1200)
    with pytest.raises(AppError):
        replace(Settings(), max_retrieval_calls=100)


def test_pdf_metadata_in_store(settings: Settings, store: VectorStore) -> None:
    result = ingest("travel.pdf", pdf_bytes(), settings, store, FakeEmbedder())
    chunks = store.search([1.0, 0.5, 0.2], 100)
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert all(chunk.document_name == "travel.pdf" for chunk in chunks)
    assert all(chunk.document_id == result.document_id for chunk in chunks)


def test_persistence_failure_rolls_back(
    settings: Settings, store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = store.collection.add

    def failing_add(**kwargs: object) -> None:
        original(**kwargs)
        raise RuntimeError("disk simulation")

    monkeypatch.setattr(type(store.collection), "add", lambda self, **kwargs: failing_add(**kwargs))
    with pytest.raises(AppError, match="save"):
        ingest("policy.txt", b"Some policy", settings, store, FakeEmbedder())
    assert store.count() == 0
