from unittest.mock import Mock

import pytest

from rag_demo.agent import INSUFFICIENT
from rag_demo.baseline import NaiveBaseline
from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.models import AnswerDraft, Usage
from tests.test_agent import SOURCE, provider_for


def test_baseline_exactly_one_original_question_search(settings: Settings) -> None:
    store = Mock()
    store.search.return_value = [SOURCE]
    provider = provider_for([SOURCE])
    question = "Deadline and equipment allowance?"
    result = NaiveBaseline(settings, store, provider).ask(question)
    assert result.supported and result.retrieval_calls == 1 and result.queries == [question]
    provider.embed.assert_called_once_with([question])
    store.search.assert_called_once_with([1.0, 0.5, 0.2], settings.retrieval_k)
    provider.plan.assert_not_called()
    provider.answer.assert_called_once_with(question, [SOURCE], [])
    assert result.usage.prompt_tokens == 0 and result.usage.embedding_tokens == 4


def test_baseline_empty_search_still_once(settings: Settings) -> None:
    store = Mock()
    store.search.return_value = []
    provider = provider_for([SOURCE])
    result = NaiveBaseline(settings, store, provider).ask("Unknown?")
    assert result.retrieval_calls == 1 and result.answer == INSUFFICIENT
    assert not result.citations and not result.supported
    provider.plan.assert_not_called()
    provider.answer.assert_not_called()
    store.search.assert_called_once()


def test_baseline_uses_identical_citation_validation(settings: Settings) -> None:
    store = Mock()
    store.search.return_value = [SOURCE]
    provider = provider_for([SOURCE])
    provider.answer.return_value = (
        AnswerDraft(supported=True, answer="Invented", citations=[]),
        Usage(),
    )
    assert NaiveBaseline(settings, store, provider).ask("Unknown?").answer == INSUFFICIENT


def test_baseline_invalid_embedding(settings: Settings) -> None:
    provider = Mock()
    provider.embed.return_value = ([], Usage())
    with pytest.raises(AppError, match="Query embedding failed"):
        NaiveBaseline(settings, Mock(), provider).ask("Fact?")
