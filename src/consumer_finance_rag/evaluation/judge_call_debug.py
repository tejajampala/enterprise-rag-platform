"""Explicit local-only capture for reviewed-PUBLIC RAGAS judge calls.

This module is deliberately separate from routine evaluation reports and MLflow
traces.  It may retain reviewed PUBLIC content only after an explicit opt-in.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from getpass import getuser
from pathlib import Path
from threading import Lock
from typing import Any, Final, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.costs import ModelInvocationUsage
from consumer_finance_rag.pii import DeterministicPIIDetector

JUDGE_CALL_DEBUG_FORMAT_VERSION: Final[Literal["ragas-judge-call-debug-v1"]] = (
    "ragas-judge-call-debug-v1"
)
DEFAULT_JUDGE_CALL_DEBUG_DIRECTORY = Path(
    "reports/evaluation/private_judge_calls"
)

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "credential",
    "credentials",
    "headers",
    "proxy_authorization",
    "secret",
    "session_token",
    "_hidden_params",
}
_HIDDEN_REASONING_KEYS = {
    "reasoning_content",
    "reasoning_details",
    "thinking",
    "thinking_blocks",
}
_REQUEST_PARAMETER_KEYS = {
    "aws_region_name",
    "max_completion_tokens",
    "max_tokens",
    "model",
    "seed",
    "stop",
    "temperature",
    "tool_choice",
    "top_p",
}


class JudgeCallDebugError(RuntimeError):
    """Raised when private debug capture cannot be authorized or persisted."""


class ApprovedPublicDebugCase(BaseModel):
    """Minimum authorization contract for one content-bearing debug case."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1, repr=False)
    review_status: Literal["reviewed"]
    visibility: Literal["PUBLIC"]
    groups: tuple[str, ...] = ()


class JudgeCallDebugConfig(BaseModel):
    """Bounded local capture configuration; existence means explicit opt-in."""

    model_config = ConfigDict(frozen=True)

    artifact_directory: Path = DEFAULT_JUDGE_CALL_DEBUG_DIRECTORY
    retention_hours: int = Field(default=24, ge=1, le=168)
    maximum_artifact_bytes: int = Field(default=2_000_000, ge=16_384, le=20_000_000)
    maximum_text_characters: int = Field(default=65_536, ge=1_024, le=262_144)
    maximum_collection_items: int = Field(default=128, ge=8, le=1_024)


class CapturedJudgeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: tuple[dict[str, Any], ...]
    structured_output_schema: Any | None = None
    tools: Any | None = None
    parameters: dict[str, Any]
    redaction_applied: bool = True
    truncated: bool = False


class CapturedJudgeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    response_type: str | None = None
    content: Any | None = None
    tool_calls: Any | None = None
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None
    truncated: bool = False


class InstructorProcessingOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    validation_status: Literal["PENDING", "PASSED", "FAILED", "NOT_REACHED"]
    parsed_response: Any | None = None
    error_category: str | None = None
    attempt_number: int | None = Field(default=None, ge=1)
    maximum_attempts: int | None = Field(default=None, ge=1)


class JudgeCallDebugRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    evaluation_trace_id: str | None
    evaluation_run_id: str | None
    evaluation_configuration_id: str
    case_id: str
    metric_name: str
    metric_span_id: str | None
    provider_span_id: str | None
    sequence_within_metric: int = Field(ge=1)
    context_chunk_ids: tuple[str, ...] = ()
    context_correlation: str | None = None
    retry_count: int | None = Field(default=None, ge=0)
    model_id: str
    region: str
    effective_output_token_limit: int | None = Field(default=None, gt=0)
    call_status: Literal["COMPLETED", "FAILED"]
    duration_ms: float | None = Field(default=None, ge=0)
    estimated_cost_usd: str | None = None
    error_category: str | None = None
    request: CapturedJudgeRequest
    response: CapturedJudgeResponse
    instructor: InstructorProcessingOutcome


class JudgeCallDebugArtifact(BaseModel):
    """Content-bearing local artifact; never embed this model in routine reports."""

    model_config = ConfigDict(frozen=True)

    format_version: Literal["ragas-judge-call-debug-v1"] = (
        JUDGE_CALL_DEBUG_FORMAT_VERSION
    )
    artifact_id: str
    created_at: datetime
    expires_at: datetime
    content_policy: Literal["approved_public_private_debug"]
    evaluation_trace_id: str | None
    evaluation_run_id: str | None
    evaluation_configuration_id: str
    model_id: str
    region: str
    calls: tuple[JudgeCallDebugRecord, ...]


class JudgeCallDebugCapture:
    """Evaluation-local recorder with exact PUBLIC authorization and bounded storage."""

    def __init__(
        self,
        config: JudgeCallDebugConfig,
        *,
        cases: Sequence[ApprovedPublicDebugCase],
        evaluation_configuration_id: str,
        evaluation_run_id: str | None,
        model_id: str,
        region: str,
        now: datetime | None = None,
    ) -> None:
        if not cases:
            raise JudgeCallDebugError("judge-call debug capture requires an approved case")
        if any(case.groups for case in cases):
            raise JudgeCallDebugError(
                "judge-call debug capture requires group-free reviewed PUBLIC cases"
            )
        case_ids = [case.case_id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise JudgeCallDebugError("judge-call debug case IDs must be unique")
        created_at = now or datetime.now(UTC)
        if created_at.tzinfo is None:
            raise JudgeCallDebugError("judge-call debug timestamps must be timezone-aware")
        self.config = config
        self.artifact_id = str(uuid4())
        self.created_at = created_at
        self.expires_at = created_at + timedelta(hours=config.retention_hours)
        self.evaluation_configuration_id = evaluation_configuration_id
        self.evaluation_run_id = evaluation_run_id
        self.model_id = model_id
        self.region = region
        self._cases = {case.case_id: case for case in cases}
        self._contexts: dict[str, tuple[tuple[str, str], ...]] = {}
        self._records: OrderedDict[str, JudgeCallDebugRecord] = OrderedDict()
        self._lock = Lock()
        self._detector = DeterministicPIIDetector()
        self._persisted_path: Path | None = None
        self._failure_category: str | None = None

    @property
    def artifact_path(self) -> Path:
        return self.config.artifact_directory / f"{self.artifact_id}.private.json"

    @property
    def persisted_path(self) -> Path | None:
        return self._persisted_path

    @property
    def failure_category(self) -> str | None:
        return self._failure_category

    def bind_case_inputs(
        self,
        *,
        case_id: str,
        question: str,
        contexts: Sequence[str],
        chunk_ids: Sequence[str],
    ) -> None:
        case = self._cases.get(case_id)
        if case is None or case.question != question:
            raise JudgeCallDebugError(
                "judge-call debug input does not match its reviewed PUBLIC case"
            )
        if len(contexts) != len(chunk_ids):
            raise JudgeCallDebugError("judge-call debug context identities are inconsistent")
        self._contexts[case_id] = tuple(zip(contexts, chunk_ids, strict=True))

    def attach_evaluation_run_id(self, evaluation_run_id: str | None) -> None:
        if evaluation_run_id is None:
            return
        if self.evaluation_run_id not in (None, evaluation_run_id):
            raise JudgeCallDebugError(
                "judge-call debug evaluation run correlation cannot be replaced"
            )
        self.evaluation_run_id = evaluation_run_id

    def capture_invocation(
        self,
        *,
        invocation: ModelInvocationUsage,
        request_kwargs: Mapping[str, object],
        response: object,
        evaluation_trace_id: str | None,
        metric_span_id: str | None,
        provider_span_id: str | None,
        sequence_within_metric: int,
    ) -> str:
        case_id = invocation.case_id
        if case_id is None or case_id not in self._cases:
            raise JudgeCallDebugError(
                "judge-call debug invocation is not bound to an approved PUBLIC case"
            )
        call_id = provider_span_id or f"local-{uuid4()}"
        request = self._capture_request(request_kwargs)
        context_chunk_ids = self._match_contexts(case_id, request_kwargs.get("messages"))
        record = JudgeCallDebugRecord(
            call_id=call_id,
            evaluation_trace_id=evaluation_trace_id,
            evaluation_run_id=self.evaluation_run_id,
            evaluation_configuration_id=self.evaluation_configuration_id,
            case_id=case_id,
            metric_name=invocation.metric_name,
            metric_span_id=metric_span_id,
            provider_span_id=provider_span_id,
            sequence_within_metric=sequence_within_metric,
            context_chunk_ids=context_chunk_ids,
            context_correlation=(
                "exact_outgoing_message_match" if context_chunk_ids else None
            ),
            retry_count=invocation.retry_count,
            model_id=invocation.model_id,
            region=invocation.region,
            effective_output_token_limit=invocation.effective_output_token_limit,
            call_status=invocation.status,
            duration_ms=invocation.latency_ms,
            estimated_cost_usd=(
                str(invocation.estimated_cost_usd)
                if invocation.estimated_cost_usd is not None
                else None
            ),
            error_category=invocation.error_category,
            request=request,
            response=self._capture_response(response, invocation),
            instructor=InstructorProcessingOutcome(
                validation_status=(
                    "PENDING" if invocation.status == "COMPLETED" else "NOT_REACHED"
                ),
                error_category=invocation.error_category,
            ),
        )
        with self._lock:
            self._records[call_id] = record
        return call_id

    def record_instructor_result(self, call_id: str, parsed: object) -> None:
        with self._lock:
            record = self._records.get(call_id)
            if record is None:
                return
            parsed_value, _ = self._sanitize(parsed)
            self._records[call_id] = record.model_copy(
                update={
                    "instructor": InstructorProcessingOutcome(
                        validation_status="PASSED",
                        parsed_response=parsed_value,
                    )
                }
            )

    def record_instructor_failure(
        self,
        call_id: str,
        *,
        error_category: str,
        attempt_number: int | None = None,
        maximum_attempts: int | None = None,
    ) -> None:
        with self._lock:
            record = self._records.get(call_id)
            if record is None:
                return
            self._records[call_id] = record.model_copy(
                update={
                    "instructor": InstructorProcessingOutcome(
                        validation_status="FAILED",
                        error_category=error_category[:128],
                        attempt_number=attempt_number,
                        maximum_attempts=maximum_attempts,
                    )
                }
            )

    def persist(self) -> Path | None:
        """Persist atomically with owner-only mode; capture failure never fails judging."""

        try:
            artifact = self.artifact()
            encoded = (artifact.model_dump_json(indent=2) + "\n").encode("utf-8")
            if len(encoded) > self.config.maximum_artifact_bytes:
                raise JudgeCallDebugError("judge-call debug artifact exceeds its size limit")
            directory = self.config.artifact_directory
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            _restrict_local_access(directory, directory=True)
            purge_expired_judge_call_artifacts(directory)
            output = self.artifact_path
            temporary = output.with_suffix(f".tmp-{uuid4()}")
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, output)
                _restrict_local_access(output, directory=False)
            finally:
                if temporary.exists():
                    temporary.unlink()
            self._persisted_path = output
            self._failure_category = None
            return output
        except Exception as error:
            self._failure_category = type(error).__name__
            return None

    def artifact(self) -> JudgeCallDebugArtifact:
        with self._lock:
            calls = tuple(self._records.values())
        trace_ids = {item.evaluation_trace_id for item in calls if item.evaluation_trace_id}
        evaluation_trace_id = next(iter(trace_ids)) if len(trace_ids) == 1 else None
        return JudgeCallDebugArtifact(
            artifact_id=self.artifact_id,
            created_at=self.created_at,
            expires_at=self.expires_at,
            content_policy="approved_public_private_debug",
            evaluation_trace_id=evaluation_trace_id,
            evaluation_run_id=self.evaluation_run_id,
            evaluation_configuration_id=self.evaluation_configuration_id,
            model_id=self.model_id,
            region=self.region,
            calls=calls,
        )

    def _capture_request(self, kwargs: Mapping[str, object]) -> CapturedJudgeRequest:
        messages_value, messages_truncated = self._sanitize(kwargs.get("messages", ()))
        messages = tuple(
            item for item in messages_value if isinstance(item, dict)
        ) if isinstance(messages_value, list | tuple) else ()
        schema_source = kwargs.get("response_format")
        schema, schema_truncated = self._sanitize(schema_source)
        tools, tools_truncated = self._sanitize(kwargs.get("tools"))
        parameters, parameters_truncated = self._sanitize(
            {key: kwargs[key] for key in _REQUEST_PARAMETER_KEYS if key in kwargs}
        )
        return CapturedJudgeRequest(
            messages=messages,
            structured_output_schema=schema,
            tools=tools,
            parameters=parameters if isinstance(parameters, dict) else {},
            truncated=(
                messages_truncated
                or schema_truncated
                or tools_truncated
                or parameters_truncated
            ),
        )

    def _capture_response(
        self,
        response: object,
        invocation: ModelInvocationUsage,
    ) -> CapturedJudgeResponse:
        choices = _read_field(response, "choices")
        first = (
            choices[0]
            if isinstance(choices, Sequence)
            and not isinstance(choices, str | bytes)
            and choices
            else None
        )
        message = _read_field(first, "message") if first is not None else None
        content, content_truncated = self._sanitize(_read_field(message, "content"))
        tool_calls, tools_truncated = self._sanitize(
            _read_field(message, "tool_calls")
        )
        return CapturedJudgeResponse(
            response_type=invocation.response_type,
            content=content,
            tool_calls=tool_calls,
            finish_reason=invocation.finish_reason,
            input_tokens=invocation.input_tokens,
            output_tokens=invocation.output_tokens,
            total_tokens=invocation.total_tokens,
            reasoning_tokens=invocation.reasoning_tokens,
            provider_request_id=invocation.provider_request_id,
            truncated=content_truncated or tools_truncated,
        )

    def _match_contexts(self, case_id: str, messages: object) -> tuple[str, ...]:
        rendered = _render_for_matching(messages)
        matches = [
            chunk_id
            for context, chunk_id in self._contexts.get(case_id, ())
            if context and context in rendered
        ]
        return tuple(dict.fromkeys(matches))

    def _sanitize(self, value: object) -> tuple[Any, bool]:
        truncated = False

        def visit(item: object, depth: int = 0) -> Any:
            nonlocal truncated
            if depth > 20:
                truncated = True
                return "[DEPTH LIMIT]"
            if isinstance(item, BaseModel):
                return visit(item.model_dump(mode="json"), depth + 1)
            if isinstance(item, Mapping):
                result: dict[str, Any] = {}
                for index, (raw_key, child) in enumerate(item.items()):
                    if index >= self.config.maximum_collection_items:
                        truncated = True
                        break
                    key = str(raw_key)
                    lowered = key.casefold()
                    if lowered in _SECRET_KEYS or lowered in _HIDDEN_REASONING_KEYS:
                        continue
                    result[key[:256]] = visit(child, depth + 1)
                return result
            if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
                values = list(item)
                if len(values) > self.config.maximum_collection_items:
                    values = values[: self.config.maximum_collection_items]
                    truncated = True
                return [visit(child, depth + 1) for child in values]
            if isinstance(item, str):
                redacted = self._detector.redact(item)
                if len(redacted) > self.config.maximum_text_characters:
                    truncated = True
                    return redacted[: self.config.maximum_text_characters] + "[TRUNCATED]"
                return redacted
            if isinstance(item, bytes | bytearray):
                truncated = True
                return "[BINARY OMITTED]"
            if item is None or isinstance(item, bool | int | float):
                return item
            if hasattr(item, "model_dump"):
                try:
                    return visit(item.model_dump(mode="json"), depth + 1)
                except Exception:
                    pass
            truncated = True
            return f"[{type(item).__name__} OMITTED]"

        return visit(value), truncated


def load_judge_call_debug_artifact(path: str | Path) -> JudgeCallDebugArtifact:
    return JudgeCallDebugArtifact.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def list_judge_call_debug_artifacts(
    directory: str | Path,
) -> tuple[tuple[Path, JudgeCallDebugArtifact], ...]:
    root = Path(directory)
    if not root.exists():
        return ()
    results: list[tuple[Path, JudgeCallDebugArtifact]] = []
    for path in sorted(root.glob("*.private.json"), reverse=True):
        try:
            results.append((path, load_judge_call_debug_artifact(path)))
        except Exception:
            continue
    return tuple(results)


def purge_expired_judge_call_artifacts(
    directory: str | Path,
    *,
    now: datetime | None = None,
) -> tuple[Path, ...]:
    current = now or datetime.now(UTC)
    removed: list[Path] = []
    for path, artifact in list_judge_call_debug_artifacts(directory):
        if artifact.expires_at <= current:
            delete_judge_call_debug_artifact(path, root=directory)
            removed.append(path)
    return tuple(removed)


def delete_judge_call_debug_artifact(
    path: str | Path,
    *,
    root: str | Path,
) -> None:
    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    if resolved_path.parent != resolved_root or not resolved_path.name.endswith(
        ".private.json"
    ):
        raise JudgeCallDebugError("refusing to delete outside the debug artifact directory")
    resolved_path.unlink(missing_ok=True)


def _read_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _render_for_matching(value: object) -> str:
    try:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return ""


def _restrict_local_access(path: Path, *, directory: bool) -> None:
    mode = (
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
        if directory
        else stat.S_IRUSR | stat.S_IWUSR
    )
    os.chmod(path, mode)
    if os.name != "nt":
        return
    principal = _windows_current_principal()
    permission = f"{principal}:(OI)(CI)F" if directory else f"{principal}:F"
    try:
        subprocess.run(
            [
                "icacls",
                str(path),
                "/inheritance:r",
                "/grant:r",
                permission,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise JudgeCallDebugError(
            "Unable to enforce local owner-only artifact access"
        ) from error


def _windows_current_principal() -> str:
    """Resolve the effective Windows identity rather than trusting USERNAME."""

    try:
        result = subprocess.run(
            ["whoami"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        principal = result.stdout.strip()
        if principal:
            return principal
    except (OSError, subprocess.SubprocessError):
        pass
    return getuser()
