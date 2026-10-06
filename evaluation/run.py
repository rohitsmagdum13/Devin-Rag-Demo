import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from evaluation.scoring import Question, Score, load_questions, score_answer
from rag_demo.agent import RetrievalAgent
from rag_demo.baseline import NaiveBaseline
from rag_demo.chunking import chunk_sections
from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.extraction import extract
from rag_demo.ingestion import ingest
from rag_demo.models import AnswerResult, Chunk, IngestionResult
from rag_demo.providers import AgentProvider, OpenAIProvider
from rag_demo.store import VectorStore


@dataclass(frozen=True)
class PreparedIndex:
    settings: Settings
    store: VectorStore
    corpus: dict[str, Chunk]
    manifest: dict[str, object]
    fingerprint: str
    ingestion: list[IngestionResult]


@dataclass(frozen=True)
class Trial:
    system: str
    repeat: int
    question: Question
    answer: AnswerResult | None
    score: Score
    latency_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class Summary:
    trials: int
    errors: int
    correctness: float
    citation_accuracy: float
    unsupported_handling: float | None
    median_latency_seconds: float
    mean_retrieval_calls: float | None
    mean_embedding_tokens: float | None
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None


def prepare_index(settings: Settings, root: Path, provider: AgentProvider) -> PreparedIndex:
    files = sorted(Path(__file__).with_name("corpus").glob("*.txt"))
    manifest: dict[str, object] = {
        "files": [
            {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files
        ],
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    settings = replace(settings, store_path=root / fingerprint)
    store = VectorStore(
        settings.store_path, settings.embedding_model, settings.embedding_dimensions
    )
    canonical: dict[str, Chunk] = {}
    ingestion: list[IngestionResult] = []
    expected_documents: list[dict[str, str | int]] = []
    for path in files:
        content = path.read_bytes()
        document_id = hashlib.sha256(content).hexdigest()
        sections = extract(path.name, content, settings.max_upload_mb * 1024 * 1024)
        chunks = chunk_sections(
            sections, document_id, path.name, settings.chunk_size, settings.chunk_overlap
        )
        canonical.update({chunk.id: chunk for chunk in chunks})
        ingestion.append(ingest(path.name, content, settings, store, provider))
        expected_documents.append({"id": document_id, "name": path.name, "chunks": len(chunks)})
    if sorted(store.documents(), key=lambda d: str(d["id"])) != sorted(
        expected_documents, key=lambda d: str(d["id"])
    ):
        raise AppError("Evaluation index is contaminated. Use a new --store-root directory.")
    return PreparedIndex(settings, store, canonical, manifest, fingerprint, ingestion)


def compare(
    settings: Settings,
    store: VectorStore,
    provider: AgentProvider,
    questions: list[Question],
    corpus: dict[str, Chunk],
    repeats: int = 1,
) -> list[Trial]:
    if not questions or not 1 <= repeats <= 10:
        raise AppError("Evaluation needs questions and 1–10 repeats.")
    systems: dict[str, NaiveBaseline | RetrievalAgent] = {
        "baseline": NaiveBaseline(settings, store, provider),
        "agentic": RetrievalAgent(settings, store, provider),
    }
    trials: list[Trial] = []
    for repeat in range(repeats):
        for index, question in enumerate(questions):
            order = (
                ["baseline", "agentic"] if (index + repeat) % 2 == 0 else ["agentic", "baseline"]
            )
            for name in order:
                start = perf_counter()
                try:
                    answer = systems[name].ask(question.question)
                    trials.append(
                        Trial(
                            name,
                            repeat + 1,
                            question,
                            answer,
                            score_answer(question, answer, corpus),
                            answer.latency_seconds,
                        )
                    )
                except AppError as error:
                    trials.append(
                        Trial(
                            name,
                            repeat + 1,
                            question,
                            None,
                            Score(0, 0, None if question.answerable else 0, [], [], 0, 0),
                            perf_counter() - start,
                            str(error),
                        )
                    )
    return trials


def summarize(trials: list[Trial]) -> dict[str, Summary]:
    summary: dict[str, Summary] = {}
    for name in ["baseline", "agentic"]:
        group = [trial for trial in trials if trial.system == name]
        successful = [t.answer for t in group if t.answer is not None]
        unsupported = [
            t.score.unsupported_handling for t in group if t.score.unsupported_handling is not None
        ]
        summary[name] = Summary(
            len(group),
            sum(t.error is not None for t in group),
            mean(t.score.correctness for t in group),
            mean(t.score.citation_accuracy for t in group),
            mean(unsupported) if unsupported else None,
            median(t.latency_seconds for t in group),
            mean(a.retrieval_calls for a in successful) if successful else None,
            mean(a.usage.embedding_tokens for a in successful) if successful else None,
            mean(a.usage.prompt_tokens for a in successful) if successful else None,
            mean(a.usage.completion_tokens for a in successful) if successful else None,
        )
    return summary


def comparison_table(summary: dict[str, Summary]) -> str:
    def number(value: float | None, digits: int = 1) -> str:
        return f"{value:.{digits}f}" if value is not None else "n/a"

    lines = [
        "Deterministic reference checks — no LLM judging",
        "System     Correct  Citations  Unsupported  Median s  Searches  "
        "Embed tok  LLM tok  Errors",
    ]
    for name, row in summary.items():
        unsupported = (
            f"{row.unsupported_handling:.1%}" if row.unsupported_handling is not None else "n/a"
        )
        llm_tokens = (
            row.mean_prompt_tokens + row.mean_completion_tokens
            if row.mean_prompt_tokens is not None and row.mean_completion_tokens is not None
            else None
        )
        lines.append(
            f"{name:<10} {row.correctness:>7.1%}  {row.citation_accuracy:>8.1%}  "
            f"{unsupported:>11}  {row.median_latency_seconds:>8.2f}  "
            f"{number(row.mean_retrieval_calls, 2):>8}  "
            f"{number(row.mean_embedding_tokens):>9}  {number(llm_tokens):>7}  {row.errors:>6}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live, billable fixed-corpus RAG comparison")
    parser.add_argument("--output", type=Path, default=Path("data/evaluation-results.json"))
    parser.add_argument("--store-root", type=Path, default=Path("data/evaluation-indexes"))
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be between 1 and 10")
    try:
        settings = Settings.from_env()
        provider = OpenAIProvider(settings)
        prepared = prepare_index(settings, args.store_root, provider)
        questions = load_questions()
        trials = compare(
            prepared.settings, prepared.store, provider, questions, prepared.corpus, args.repeats
        )
        summary = summarize(trials)
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scoring": {
                "method": "deterministic_regex_reference_checks",
                "llm_judging": "not_used",
                "limitations": "Lexical fact/citation proxies, not entailment or safety proof.",
            },
            "settings": {
                **asdict(prepared.settings),
                "store_path": str(prepared.settings.store_path),
            },
            "corpus_manifest": prepared.manifest,
            "corpus_fingerprint": prepared.fingerprint,
            "questions_sha256": hashlib.sha256(
                json.dumps([asdict(q) for q in questions], sort_keys=True).encode()
            ).hexdigest(),
            "repeats": args.repeats,
            "execution_order": "Alternating per question/repeat; no conversation history.",
            "ingestion": [asdict(result) for result in prepared.ingestion],
            "summary": {name: asdict(row) for name, row in summary.items()},
            "trials": [asdict(trial) for trial in trials],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(comparison_table(summary))
        print(f"Detailed results: {args.output}")
        if any(trial.error for trial in trials):
            parser.exit(1, "Some trials failed; errors are saved in the results.\n")
    except AppError as error:
        parser.exit(1, f"Error: {error}\n")
    except OSError:
        parser.exit(1, "Error: Cannot access corpus, local store, or output. Check permissions.\n")


if __name__ == "__main__":
    main()
