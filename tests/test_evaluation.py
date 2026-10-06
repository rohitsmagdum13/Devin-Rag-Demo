import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from evaluation import run
from evaluation.scoring import Fact, Question, load_questions, score_answer
from rag_demo.agent import INSUFFICIENT
from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.models import AnswerDraft, AnswerResult, Citation, CitationDraft, RetrievalPlan, Usage
from tests.conftest import FakeEmbedder
from tests.test_agent import SOURCE, provider_for


def reference_question() -> Question:
    return Question("deadline", "direct", "Deadline?", True, [Fact(r"30\s+days", "handbook.txt")])


def answer_result() -> AnswerResult:
    return AnswerResult(
        "30 days",
        True,
        "retrieve",
        [Citation(SOURCE, "30 days")],
        [SOURCE],
        ["Deadline?"],
        1,
        0.5,
        Usage(prompt_tokens=10),
    )


def test_fixed_questions_cover_required_categories() -> None:
    questions = load_questions()
    assert len(questions) == 8
    assert {q.category for q in questions} == {"direct", "multipart", "unanswerable"}
    assert all(not q.facts for q in questions if not q.answerable)
    assert all(len(q.facts) > 1 for q in questions if q.category == "multipart")


def test_deterministic_fact_and_reference_citation_scores() -> None:
    question = reference_question()
    result = answer_result()
    score = score_answer(question, result, {SOURCE.id: SOURCE})
    assert score.correctness == score.citation_accuracy == 1
    assert score.unsupported_handling is None and score.valid_citations == 1
    wrong_quote = replace(result, citations=[Citation(SOURCE, "within")])
    score = score_answer(question, wrong_quote, {SOURCE.id: SOURCE})
    assert score.correctness == 1 and score.citation_accuracy == 0
    assert score.valid_citations == 1
    wrong_answer = replace(result, answer="900 days")
    assert score_answer(question, wrong_answer, {SOURCE.id: SOURCE}).correctness == 0


@pytest.mark.parametrize(
    "citation",
    [
        Citation(replace(SOURCE, document_name="different.txt"), "30 days"),
        Citation(replace(SOURCE, page=99), "30 days"),
        Citation(replace(SOURCE, id="invented"), "30 days"),
        Citation(SOURCE, "900 days"),
    ],
)
def test_citations_must_resolve_to_canonical_metadata_and_exact_quote(citation: Citation) -> None:
    score = score_answer(
        reference_question(), replace(answer_result(), citations=[citation]), {SOURCE.id: SOURCE}
    )
    assert score.valid_citations == 0 and score.citation_accuracy == 0


def test_partial_multipart_and_invalid_citation_penalty() -> None:
    question = replace(
        reference_question(),
        category="multipart",
        facts=[Fact(r"30\s+days", "handbook.txt"), Fact(r"75 per day", "travel.txt")],
    )
    result = answer_result()
    score = score_answer(question, result, {SOURCE.id: SOURCE})
    assert score.correctness == score.citation_accuracy == 0.5
    result = replace(result, citations=[*result.citations, Citation(SOURCE, "invalid")])
    assert score_answer(question, result, {SOURCE.id: SOURCE}).citation_accuracy == 0.25


def test_unknown_questions_require_explicit_refusal_without_citations() -> None:
    question = Question("unknown", "unanswerable", "Unknown?", False, [])
    refused = replace(answer_result(), answer=INSUFFICIENT, supported=False, citations=[])
    score = score_answer(question, refused, {SOURCE.id: SOURCE})
    assert score.correctness == score.unsupported_handling == score.citation_accuracy == 1
    for bad in [
        answer_result(),
        replace(refused, answer="Hello!"),
        replace(refused, citations=[Citation(SOURCE, "30 days")]),
    ]:
        assert score_answer(question, bad, {SOURCE.id: SOURCE}).unsupported_handling == 0


def test_corpus_is_stable_isolated_deduplicated_and_rejects_contamination(
    settings: Settings, tmp_path: Path
) -> None:
    provider = FakeEmbedder()
    first = run.prepare_index(settings, tmp_path / "indexes", provider)
    second = run.prepare_index(settings, tmp_path / "indexes", provider)
    assert first.fingerprint == second.fingerprint and first.corpus == second.corpus
    assert provider.calls == 3 and all(r.duplicate for r in second.ingestion)
    assert first.settings.store_path != settings.store_path
    first.store.add([SOURCE], [[1.0, 0.5, 0.2]])
    with pytest.raises(AppError, match="contaminated"):
        run.prepare_index(settings, tmp_path / "indexes", provider)
    changed = run.prepare_index(replace(settings, chunk_size=60), tmp_path / "indexes", provider)
    assert changed.fingerprint != first.fingerprint


def test_both_systems_share_store_generator_settings_and_alternate_order(
    settings: Settings,
) -> None:
    store = Mock()
    store.count.return_value = 1
    store.search.return_value = [SOURCE]
    provider = provider_for([SOURCE])
    provider.plan.side_effect = None
    provider.plan.return_value = (RetrievalPlan(mode="retrieve", queries=["rewritten"]), Usage())
    questions = [reference_question(), replace(reference_question(), id="second")]
    trials = run.compare(settings, store, provider, questions, {SOURCE.id: SOURCE})
    assert [t.system for t in trials] == ["baseline", "agentic", "agentic", "baseline"]
    assert all(t.answer and t.answer.supported for t in trials)
    for trial in trials:
        assert trial.answer is not None
        if trial.system == "baseline":
            assert trial.answer.queries == [trial.question.question]
            assert trial.answer.retrieval_calls == 1
        else:
            assert trial.answer.queries == ["rewritten"]
    assert all(call.args[2] == [] for call in provider.answer.call_args_list)
    summary = run.summarize(trials)
    assert all(row.correctness == 1 and row.trials == 2 for row in summary.values())
    assert "no LLM judging" in run.comparison_table(summary)


def test_trial_errors_are_reported_and_scored_as_failures(settings: Settings) -> None:
    provider = Mock()
    provider.embed.side_effect = AppError("OpenAI unavailable; retry")
    provider.plan.return_value = (RetrievalPlan(mode="retrieve", queries=[]), Usage())
    store = Mock()
    store.count.return_value = 1
    trials = run.compare(settings, store, provider, [reference_question()], {})
    assert all(t.error and t.answer is None and t.score.correctness == 0 for t in trials)
    assert all(row.errors == 1 for row in run.summarize(trials).values())
    assert all(row.mean_retrieval_calls is None for row in run.summarize(trials).values())
    assert "n/a" in run.comparison_table(run.summarize(trials))
    with pytest.raises(AppError):
        run.compare(settings, store, provider, [], {})


def test_cli_saves_detailed_results_without_credentials(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = replace(settings, chunk_size=1200, chunk_overlap=200)
    monkeypatch.setattr(run.Settings, "from_env", lambda: settings)
    provider = Mock()
    provider.embed.side_effect = FakeEmbedder().embed
    provider.plan.return_value = (RetrievalPlan(mode="retrieve", queries=[]), Usage())

    def answer(question: str, chunks: list, history: list) -> tuple[AnswerDraft, Usage]:
        source = next(c for c in chunks if c.document_name == "handbook.txt")
        return AnswerDraft(
            supported=True,
            answer="30 days",
            citations=[CitationDraft(chunk_id=source.id, quote="30 days")],
        ), Usage(prompt_tokens=10)

    provider.answer.side_effect = answer
    monkeypatch.setattr(run, "OpenAIProvider", Mock(return_value=provider))
    monkeypatch.setattr(run, "load_questions", lambda: [reference_question()])
    output = tmp_path / "nested" / "results.json"
    monkeypatch.setattr(
        "sys.argv",
        ["evaluation.run", "--store-root", str(tmp_path / "indexes"), "--output", str(output)],
    )
    run.main()
    report = json.loads(output.read_text())
    assert report["scoring"]["llm_judging"] == "not_used"
    assert len(report["trials"]) == 2 and len(report["corpus_manifest"]["files"]) == 3
    assert report["summary"]["baseline"]["correctness"] == 1
    assert report["summary"]["agentic"]["correctness"] == 1
    assert "OPENAI_API_KEY" not in output.read_text()
    assert "baseline" in capsys.readouterr().out

    provider.embed.side_effect = AppError("OpenAI unavailable; retry")
    with pytest.raises(SystemExit) as failure:
        run.main()
    assert failure.value.code == 1
    report = json.loads(output.read_text())
    assert all(row["errors"] == 1 for row in report["summary"].values())
    assert all(row["mean_retrieval_calls"] is None for row in report["summary"].values())
