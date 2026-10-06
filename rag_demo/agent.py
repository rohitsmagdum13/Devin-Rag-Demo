import re
from time import perf_counter

from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.models import AnswerDraft, AnswerResult, Chunk, Citation, Usage
from rag_demo.providers import AgentProvider
from rag_demo.store import VectorStore

INSUFFICIENT = "The available documents do not provide sufficient evidence to answer this question."
CHAT_HELP = "Hello! Upload PDF, DOCX, or TXT documents, then ask a question about their contents."


def is_social_message(question: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening)|help)"
            r"(?: there)?[.!?\s]*",
            question,
            flags=re.IGNORECASE,
        )
    )


def validated_answer(draft: AnswerDraft, chunks: list[Chunk]) -> tuple[str, bool, list[Citation]]:
    if not draft.supported or not draft.answer.strip() or not draft.citations:
        return INSUFFICIENT, False, []
    sources = {chunk.id: chunk for chunk in chunks}
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for citation in draft.citations:
        source = sources.get(citation.chunk_id)
        quote = citation.quote.strip()
        if source is None or not quote or quote not in source.text:
            return INSUFFICIENT, False, []
        key = (source.id, quote)
        if key not in seen:
            citations.append(Citation(source, quote))
            seen.add(key)
    return draft.answer.strip(), True, citations


def validate_question(question: str) -> str:
    question = question.strip()
    if not question:
        raise AppError("Enter a question before sending.")
    if len(question) > 4000:
        raise AppError("Question is too long. Use at most 4000 characters.")
    return question


class RetrievalAgent:
    def __init__(self, settings: Settings, store: VectorStore, provider: AgentProvider) -> None:
        self.settings = settings
        self.store = store
        self.provider = provider

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> AnswerResult:
        start = perf_counter()
        question = validate_question(question)
        history = [(q[:1000], a[:1000]) for q, a in (history or [])[-3:]]
        usage = Usage()
        queries: list[str] = []
        sources: dict[str, Chunk] = {}
        plan, planning_usage = self.provider.plan(
            question, [], self.settings.max_retrieval_calls, [], history
        )
        usage.add(planning_usage)
        if plan.mode == "chat" and is_social_message(question):
            return AnswerResult(
                CHAT_HELP, False, "chat", [], [], [], 0, perf_counter() - start, usage
            )
        if not self.store.count():
            return AnswerResult(
                INSUFFICIENT, False, "retrieve", [], [], [], 0, perf_counter() - start, usage
            )

        def retrieve(requested: list[str]) -> None:
            for raw_query in requested:
                query = " ".join(raw_query.split())[:1000]
                if not query or query.casefold() in {q.casefold() for q in queries}:
                    continue
                if len(queries) >= self.settings.max_retrieval_calls:
                    break
                embeddings, embedding_usage = self.provider.embed([query])
                usage.add(embedding_usage)
                if len(embeddings) != 1:
                    raise AppError("Query embedding failed. Retry your question.")
                queries.append(query)
                for chunk in self.store.search(embeddings[0], self.settings.retrieval_k):
                    sources[chunk.id] = chunk

        retrieve(plan.queries if plan.mode == "retrieve" and plan.queries else [question])
        remaining = self.settings.max_retrieval_calls - len(queries)
        if remaining > 0:
            follow_up, planning_usage = self.provider.plan(
                question, list(sources.values()), remaining, queries.copy(), history
            )
            usage.add(planning_usage)
            retrieve(follow_up.queries)
        chunks = list(sources.values())
        citations: list[Citation]
        if not chunks:
            answer, supported, citations = INSUFFICIENT, False, []
        else:
            draft, answer_usage = self.provider.answer(question, chunks, history)
            usage.add(answer_usage)
            answer, supported, citations = validated_answer(draft, chunks)
        return AnswerResult(
            answer,
            supported,
            "retrieve",
            citations,
            chunks,
            queries,
            len(queries),
            perf_counter() - start,
            usage,
        )
