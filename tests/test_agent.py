from dataclasses import replace
from unittest.mock import Mock

import pytest

from rag_demo.agent import CHAT_HELP, INSUFFICIENT, RetrievalAgent, validated_answer
from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.ingestion import ingest
from rag_demo.models import AnswerDraft, Chunk, CitationDraft, RetrievalPlan, Usage
from rag_demo.store import VectorStore
from tests.conftest import FakeEmbedder

SOURCE = Chunk("id:0", "id", "handbook.txt", "Reimburse within 30 days", 2)


def draft(chunk: Chunk = SOURCE) -> AnswerDraft:
    return AnswerDraft(
        supported=True,
        answer="The deadline is 30 days.",
        citations=[CitationDraft(chunk_id=chunk.id, quote="30 days")],
    )


def provider_for(chunks: list[Chunk]) -> Mock:
    provider = Mock()
    provider.plan.side_effect = [
        (
            RetrievalPlan(mode="retrieve", queries=["reimbursement deadline"]),
            Usage(prompt_tokens=2),
        ),
        (RetrievalPlan(mode="retrieve", queries=[]), Usage(prompt_tokens=3)),
    ]
    provider.embed.return_value = ([[1.0, 0.5, 0.2]], Usage(embedding_tokens=4))
    provider.answer.return_value = (draft(chunks[0]), Usage(completion_tokens=5))
    return provider


def test_rewritten_retrieval_with_real_store(settings: Settings, store: VectorStore) -> None:
    ingest("handbook.txt", SOURCE.text.encode(), settings, store, FakeEmbedder())
    sources = store.search([1.0, 0.5, 0.2], 5)
    provider = provider_for(sources)
    result = RetrievalAgent(settings, store, provider).ask("When do I claim expenses?")
    assert result.supported and "30 days" in result.answer
    assert result.queries == ["reimbursement deadline"] and result.retrieval_calls == 1
    assert result.citations[0].chunk.document_name == "handbook.txt"
    assert result.usage.prompt_tokens == 5 and result.usage.embedding_tokens == 4
    assert result.usage.completion_tokens == 5 and result.latency_seconds >= 0
    provider.embed.assert_called_once_with(["reimbursement deadline"])


def test_no_retrieval(settings: Settings, store: VectorStore) -> None:
    provider = Mock()
    provider.plan.return_value = (RetrievalPlan(mode="chat", queries=[]), Usage())
    result = RetrievalAgent(settings, store, provider).ask("Hello")
    assert result.mode == "chat" and result.answer == CHAT_HELP and result.retrieval_calls == 0
    provider.embed.assert_not_called()
    provider.answer.assert_not_called()


def test_empty_index(settings: Settings, store: VectorStore) -> None:
    provider = provider_for([SOURCE])
    result = RetrievalAgent(settings, store, provider).ask("What is the deadline?")
    assert not result.supported and result.answer == INSUFFICIENT and not result.citations
    provider.embed.assert_not_called()
    provider.answer.assert_not_called()


def test_followup_bounded_deduplicated_and_aggregated(settings: Settings) -> None:
    fake_store = Mock()
    fake_store.count.return_value = 2
    other = replace(SOURCE, id="other:0", document_name="equipment.docx", text="Allowance: $1200")
    fake_store.search.side_effect = [[SOURCE], [other], [SOURCE]]
    provider = provider_for([SOURCE])
    provider.plan.side_effect = [
        (RetrievalPlan(mode="retrieve", queries=["deadline", " DEADLINE "]), Usage()),
        (RetrievalPlan(mode="retrieve", queries=["equipment", "cost", "over-budget"]), Usage()),
    ]
    result = RetrievalAgent(settings, fake_store, provider).ask("Deadline and allowance?")
    assert result.queries == ["deadline", "equipment", "cost"]
    assert result.retrieval_calls == 3 and len(result.retrieved_chunks) == 2
    assert provider.answer.call_args.args[1] == [SOURCE, other]
    assert provider.plan.call_count == 2


def test_planner_overrun_is_bounded(settings: Settings) -> None:
    fake_store = Mock()
    fake_store.count.return_value = 1
    fake_store.search.return_value = [SOURCE]
    provider = provider_for([SOURCE])
    provider.plan.side_effect = None
    provider.plan.return_value = (RetrievalPlan(mode="retrieve", queries=["a", "b", "c"]), Usage())
    result = RetrievalAgent(replace(settings, max_retrieval_calls=1), fake_store, provider).ask(
        "Fact?"
    )
    assert result.retrieval_calls == 1 and provider.plan.call_count == 1


@pytest.mark.parametrize(
    "bad",
    [
        AnswerDraft(supported=False, answer="Invented", citations=[]),
        AnswerDraft(supported=True, answer="Invented", citations=[]),
        AnswerDraft(
            supported=True,
            answer="Invented",
            citations=[CitationDraft(chunk_id="unknown", quote="30 days")],
        ),
        AnswerDraft(
            supported=True,
            answer="Invented",
            citations=[CitationDraft(chunk_id=SOURCE.id, quote="900 days")],
        ),
        AnswerDraft(
            supported=True,
            answer="Invented",
            citations=[CitationDraft(chunk_id=SOURCE.id, quote=" ")],
        ),
    ],
)
def test_invalid_citations_fail_closed(bad: AnswerDraft) -> None:
    assert validated_answer(bad, [SOURCE]) == (INSUFFICIENT, False, [])


def test_citation_source_and_deduplication() -> None:
    answer = draft()
    answer.citations *= 2
    text, supported, citations = validated_answer(answer, [SOURCE])
    assert supported and "30 days" in text and len(citations) == 1
    assert citations[0].chunk.page == 2 and citations[0].quote in citations[0].chunk.text


def test_insufficient_evidence_with_retrieved_context(settings: Settings) -> None:
    fake_store = Mock()
    fake_store.count.return_value = 1
    fake_store.search.return_value = [SOURCE]
    provider = provider_for([SOURCE])
    provider.answer.return_value = (
        AnswerDraft(supported=False, answer="No", citations=[]),
        Usage(),
    )
    result = RetrievalAgent(settings, fake_store, provider).ask("CEO's phone number?")
    assert result.answer == INSUFFICIENT and not result.citations
    assert result.retrieval_calls == 1


def test_history_bounded_and_empty_question(settings: Settings, store: VectorStore) -> None:
    provider = Mock()
    provider.plan.return_value = (RetrievalPlan(mode="chat", queries=[]), Usage())
    agent = RetrievalAgent(settings, store, provider)
    agent.ask("Hello", [("x" * 2000, "y" * 2000)] * 5)
    history = provider.plan.call_args.args[4]
    assert len(history) == 3 and len(history[0][0]) == 1000
    for question in [" ", "x" * 4001]:
        with pytest.raises(AppError):
            agent.ask(question)
