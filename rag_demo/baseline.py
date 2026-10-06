from time import perf_counter

from rag_demo.agent import INSUFFICIENT, validate_question, validated_answer
from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.models import AnswerResult, Usage
from rag_demo.providers import AgentProvider
from rag_demo.store import VectorStore


class NaiveBaseline:
    def __init__(self, settings: Settings, store: VectorStore, provider: AgentProvider) -> None:
        self.settings = settings
        self.store = store
        self.provider = provider

    def ask(self, question: str) -> AnswerResult:
        start = perf_counter()
        question = validate_question(question)
        usage = Usage()
        vectors, embedding_usage = self.provider.embed([question])
        usage.add(embedding_usage)
        if len(vectors) != 1:
            raise AppError("Query embedding failed. Retry your question.")
        chunks = self.store.search(vectors[0], self.settings.retrieval_k)
        if chunks:
            draft, answer_usage = self.provider.answer(question, chunks, [])
            usage.add(answer_usage)
            answer, supported, citations = validated_answer(draft, chunks)
        else:
            answer, supported, citations = INSUFFICIENT, False, []
        return AnswerResult(
            answer,
            supported,
            "retrieve",
            citations,
            chunks,
            [question],
            1,
            perf_counter() - start,
            usage,
        )
