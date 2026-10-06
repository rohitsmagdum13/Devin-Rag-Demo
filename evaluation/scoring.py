import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag_demo.agent import INSUFFICIENT
from rag_demo.models import AnswerResult, Chunk


@dataclass(frozen=True)
class Fact:
    pattern: str
    source: str


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    answerable: bool
    facts: list[Fact]


@dataclass(frozen=True)
class Score:
    correctness: float
    citation_accuracy: float
    unsupported_handling: float | None
    fact_matches: list[bool]
    citation_fact_matches: list[bool]
    valid_citations: int
    total_citations: int


def load_questions() -> list[Question]:
    raw = json.loads(Path(__file__).with_name("questions.json").read_text())
    questions = [Question(**{**item, "facts": [Fact(**f) for f in item["facts"]]}) for item in raw]
    if len({q.id for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    for question in questions:
        if question.answerable != bool(question.facts):
            raise ValueError("Answerable questions must define reference facts")
        for fact in question.facts:
            re.compile(fact.pattern)
    return questions


def score_answer(question: Question, result: AnswerResult, corpus: dict[str, Chunk]) -> Score:
    valid = [
        citation
        for citation in result.citations
        if corpus.get(citation.chunk.id) == citation.chunk
        and citation.quote.strip()
        and citation.quote in corpus[citation.chunk.id].text
    ]
    if not question.answerable:
        refused = (
            not result.supported and not result.citations and result.answer.strip() == INSUFFICIENT
        )
        return Score(
            float(refused),
            float(not result.citations),
            float(refused),
            [],
            [],
            len(valid),
            len(result.citations),
        )
    facts = [
        result.supported and bool(re.search(f.pattern, result.answer, re.IGNORECASE))
        for f in question.facts
    ]
    cited_facts = [
        any(
            citation.chunk.document_name == f.source
            and re.search(f.pattern, citation.quote, re.IGNORECASE)
            for citation in valid
        )
        for f in question.facts
    ]
    validity = len(valid) / len(result.citations) if result.citations else 0.0
    return Score(
        sum(facts) / len(facts),
        validity * sum(cited_facts) / len(cited_facts),
        None,
        facts,
        cited_facts,
        len(valid),
        len(result.citations),
    )
