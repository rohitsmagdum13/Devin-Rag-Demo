from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from rag_demo.agent import INSUFFICIENT
from rag_demo.config import Settings
from rag_demo.models import AnswerResult, Chunk, Citation, Usage


def test_ui_empty_and_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "ui"))
    monkeypatch.setattr("streamlit.get_option", lambda _: 64)
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    assert app.title[0].value == "Ask your documents"
    assert any("OPENAI_API_KEY" in item.value for item in app.info)
    assert app.button[0].disabled
    assert any("uploader transport limit: 64 MB" in item.value for item in app.caption)
    app.chat_input[0].set_value("What is the deadline?").run()
    assert not app.exception and any("OPENAI_API_KEY" in item.value for item in app.error)


def test_ui_chat_sources_latency_history_and_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Settings, "from_env", lambda: Settings(store_path=tmp_path / "ui"))
    monkeypatch.setattr("rag_demo.providers.OpenAIProvider", Mock())
    chunk = Chunk("hash:0", "hash", "policy.pdf", "Reimburse within 30 days", 2)
    answer = AnswerResult(
        "30 days.",
        True,
        "retrieve",
        [Citation(chunk, "30 days")],
        [chunk],
        ["deadline"],
        1,
        0.25,
        Usage(prompt_tokens=10),
    )
    agent = Mock()
    agent.ask.side_effect = [
        answer,
        replace(answer, answer=INSUFFICIENT, supported=False, citations=[]),
    ]
    monkeypatch.setattr("rag_demo.agent.RetrievalAgent", Mock(return_value=agent))
    app = AppTest.from_file("app.py").run()
    app.chat_input[0].set_value("Deadline?").run()
    assert not app.exception
    assert any("0.25s" in item.value for item in app.caption)
    assert any("Chunk ID: hash:0" == item.value for item in app.caption)
    assert any("Source 1: policy.pdf · page 2" == item.label for item in app.expander)
    assert any("[Source 1](#source-0-" in item.value for item in app.markdown)
    app.chat_input[0].set_value("CEO phone?").run()
    assert len(app.chat_message) == 4
    assert any(INSUFFICIENT == item.value for item in app.text)
    assert agent.ask.call_args.args[1] == [("Deadline?", "30 days.")]
    app.button[1].click().run()
    assert not app.chat_message and not app.exception
