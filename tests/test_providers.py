from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.providers import OpenAIProvider


def mocked_provider(monkeypatch: pytest.MonkeyPatch) -> tuple[OpenAIProvider, Mock]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    client = Mock()
    monkeypatch.setattr("rag_demo.providers.OpenAI", Mock(return_value=client))
    return OpenAIProvider(Settings(embedding_dimensions=3)), client


def test_embedding_request_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, client = mocked_provider(monkeypatch)
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=1, embedding=[2.0]), SimpleNamespace(index=0, embedding=[1.0])],
        usage=SimpleNamespace(total_tokens=7),
    )
    vectors, usage = provider.embed(["first", "second"])
    assert vectors == [[1.0], [2.0]] and usage.embedding_tokens == 7 and usage.requests == 1
    assert client.embeddings.create.call_args.kwargs["input"] == ["first", "second"]
    assert client.embeddings.create.call_args.kwargs["dimensions"] == 3


def test_provider_error_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, client = mocked_provider(monkeypatch)
    client.embeddings.create.side_effect = RuntimeError("sensitive-provider-debug-detail")
    with pytest.raises(AppError) as failure:
        provider.embed(["sample"])
    assert "sensitive" not in str(failure.value)


def test_embedding_batching(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, client = mocked_provider(monkeypatch)
    client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=[1.0]) for i in range(len(kwargs["input"]))],
        usage=SimpleNamespace(total_tokens=3),
    )
    vectors, usage = provider.embed(["a"] * 33)
    assert len(vectors) == 33 and client.embeddings.create.call_count == 2
    assert usage.embedding_tokens == 6 and usage.requests == 2


def test_byte_limit_before_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, client = mocked_provider(monkeypatch)
    with pytest.raises(AppError, match="UTF-8"):
        provider.embed(["界" * 3000])
    client.embeddings.create.assert_not_called()
