from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from consumer_finance_rag.public_showcase import PublicAnswer, PublicRequestError


class RerunRequested(RuntimeError):
    pass


class SessionState(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class FakeColumn:
    def __init__(self, owner: FakeStreamlit) -> None:
        self._owner = owner

    def button(self, label: str, **kwargs: object) -> bool:
        return self._owner.button(label, **kwargs)


class FakeStreamlit:
    def __init__(self) -> None:
        self.session_state = SessionState()
        self.clicked: set[str] = set()
        self.markdown_values: list[str] = []

    def title(self, value: str) -> None:
        self.markdown_values.append(value)

    def write(self, value: object) -> None:
        self.markdown_values.append(str(value))

    def caption(self, value: str) -> None:
        self.markdown_values.append(value)

    def markdown(self, value: str, **kwargs: object) -> None:
        del kwargs
        self.markdown_values.append(value)

    def warning(self, value: str) -> None:
        self.markdown_values.append(value)

    def error(self, value: str) -> None:
        self.markdown_values.append(value)

    def info(self, value: str) -> None:
        self.markdown_values.append(value)

    def text_area(self, label: str, **kwargs: object) -> str:
        del label, kwargs
        return str(self.session_state.get("public_question", "Reviewed question"))

    def columns(self, count: int) -> list[FakeColumn]:
        return [FakeColumn(self) for _ in range(count)]

    def button(self, label: str, **kwargs: object) -> bool:
        del kwargs
        if label not in self.clicked:
            return False
        self.clicked.remove(label)
        return True

    def spinner(self, value: str) -> nullcontext[None]:
        self.markdown_values.append(value)
        return nullcontext()

    def rerun(self) -> None:
        raise RerunRequested


class FakeService:
    def __init__(self, outcome: PublicAnswer | Exception) -> None:
        self.outcome = outcome
        self.calls = 0

    def answer(self, question: str) -> PublicAnswer:
        assert question == "Reviewed question"
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture(scope="module")
def public_app() -> ModuleType:
    path = Path("app/streamlit_app.py")
    spec = importlib.util.spec_from_file_location("public_streamlit_entrypoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        public_showcase=SimpleNamespace(
            enabled=True,
            inference_enabled=True,
            max_question_characters=500,
        )
    )


def _answer() -> PublicAnswer:
    return PublicAnswer.model_construct(
        answer="Supported answer [C1].",
        citations=(),
        trace=SimpleNamespace(request_id="request-1"),
        models=None,
        token_usage=None,
        quality_decision=None,
    )


def _submit_once(
    public_app: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    service: FakeService,
) -> FakeStreamlit:
    ui = FakeStreamlit()
    ui.session_state.public_question = "Reviewed question"
    ui.clicked.add("Get answer")
    monkeypatch.setattr(public_app, "st", ui)

    with pytest.raises(RerunRequested):
        public_app._ask(_settings(), service, object())
    assert ui.session_state.public_request_state == "RUNNING"
    assert service.calls == 0

    with pytest.raises(RerunRequested):
        public_app._ask(_settings(), service, object())
    return ui


def test_one_click_runs_once_and_completed_result_is_visible_after_rerun(
    public_app: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeService(_answer())
    ui = _submit_once(public_app, monkeypatch, service)
    rendered: list[PublicAnswer] = []
    monkeypatch.setattr(
        public_app,
        "_answer_panel",
        lambda answer, settings: rendered.append(answer),
    )

    public_app._ask(_settings(), service, object())
    public_app._ask(_settings(), service, object())  # ordinary refresh

    assert service.calls == 1
    assert ui.session_state.public_request_state == "COMPLETED"
    assert ui.session_state.public_result_request_id == "request-1"
    assert rendered == [service.outcome, service.outcome]


@pytest.mark.parametrize(
    ("error", "expected_state", "expected_heading"),
    [
        (
            PublicRequestError(
                "insufficient_relevant_evidence",
                "Unable to answer from available sources",
                request_id="request-abstained",
            ),
            "ABSTAINED",
            "### Unable to answer from available sources",
        ),
        (
            PublicRequestError(
                "request_failed",
                "Request could not be completed",
                request_id="request-error",
            ),
            "ERROR",
            "### Request could not be completed",
        ),
    ],
)
def test_one_click_renders_terminal_failure_without_resubmission(
    public_app: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    error: PublicRequestError,
    expected_state: str,
    expected_heading: str,
) -> None:
    service = FakeService(error)
    ui = _submit_once(public_app, monkeypatch, service)

    public_app._ask(_settings(), service, object())

    assert service.calls == 1
    assert ui.session_state.public_request_state == expected_state
    assert expected_heading in ui.markdown_values


def test_running_refresh_does_not_duplicate_inference(
    public_app: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeService(_answer())
    ui = FakeStreamlit()
    ui.session_state.update(
        public_request_state="RUNNING",
        public_submitted_question="Reviewed question",
        public_invocation_started=True,
    )
    monkeypatch.setattr(public_app, "st", ui)

    with pytest.raises(RerunRequested):
        public_app._ask(_settings(), service, object())

    assert service.calls == 0
    assert ui.session_state.public_request_state == "ERROR"


def test_guided_question_and_ask_another_never_invoke_service(
    public_app: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeService(_answer())
    ui = FakeStreamlit()
    ui.clicked.add(public_app.SUGGESTIONS[0])
    monkeypatch.setattr(public_app, "st", ui)

    public_app._question_picker()
    assert ui.session_state.public_question == public_app.SUGGESTIONS[0]
    assert service.calls == 0

    ui.session_state.public_request_state = "ERROR"
    ui.session_state.public_failure = {"code": "request_failed"}
    ui.clicked.add("Ask another question")
    with pytest.raises(RerunRequested):
        public_app._ask(_settings(), service, object())

    assert service.calls == 0
    assert ui.session_state.public_request_state == "IDLE"
