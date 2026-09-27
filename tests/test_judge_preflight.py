from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from botocore.exceptions import ClientError

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.evaluation.judge_preflight import (
    JudgePreflightStatus,
    preflight_bedrock_judge,
    ragas_provider_support,
    resolve_judge_identity,
)

JUDGE_URI = "bedrock:/meta.llama3-70b-instruct-v1:0"
GPT_OSS_URI = "bedrock:/openai.gpt-oss-20b-1:0"


@dataclass
class _Credentials:
    method: str = "test-chain"


class _BedrockClient:
    def __init__(self, *, error_code: str | None = None) -> None:
        self.error_code = error_code
        self.model_ids: list[str] = []

    def get_foundation_model(self, *, modelIdentifier: str) -> dict[str, Any]:
        self.model_ids.append(modelIdentifier)
        if self.error_code:
            raise ClientError(
                {"Error": {"Code": self.error_code, "Message": "safe test error"}},
                "GetFoundationModel",
            )
        return {
            "modelDetails": {
                "modelId": modelIdentifier,
                "modelLifecycle": {"status": "ACTIVE"},
                "inferenceTypesSupported": ["ON_DEMAND"],
            }
        }

    def list_foundation_models(self) -> dict[str, Any]:
        return {"modelSummaries": []}


class _Session:
    region_name: str | None = "ap-southeast-2"

    def __init__(self, client: _BedrockClient, *, credentials: bool = True) -> None:
        self.bedrock = client
        self.credentials = credentials
        self.requested_regions: list[str] = []

    def get_credentials(self) -> _Credentials | None:
        return _Credentials() if self.credentials else None

    def client(self, service_name: str, *, region_name: str) -> _BedrockClient:
        assert service_name == "bedrock"
        self.requested_regions.append(region_name)
        return self.bedrock


def test_identity_distinguishes_uri_litellm_and_raw_aws_model_id() -> None:
    identity = resolve_judge_identity(JUDGE_URI, "us-east-1")

    assert identity.configuration_uri == JUDGE_URI
    assert identity.litellm_model == "bedrock/meta.llama3-70b-instruct-v1:0"
    assert identity.bedrock_model_id == "meta.llama3-70b-instruct-v1:0"
    assert identity.region == "us-east-1"
    assert identity.provider_family == "meta-llama"


def test_gpt_oss_identity_uses_explicit_bedrock_converse_route() -> None:
    identity = resolve_judge_identity(GPT_OSS_URI, "ap-southeast-2")

    assert identity.configuration_uri == GPT_OSS_URI
    assert identity.litellm_model == "bedrock/converse/openai.gpt-oss-20b-1:0"
    assert identity.bedrock_model_id == "openai.gpt-oss-20b-1:0"
    assert identity.region == "ap-southeast-2"
    assert identity.provider_family == "openai-gpt-oss"
    assert ragas_provider_support(identity).status is JudgePreflightStatus.AVAILABLE


def test_gpt_oss_preflight_uses_sydney_and_keeps_mlflow_unavailable() -> None:
    client = _BedrockClient()
    session = _Session(client)
    result = preflight_bedrock_judge(
        MLflowSettings(judge_model=GPT_OSS_URI, judge_region="ap-southeast-2"),
        session_factory=lambda: session,
    )

    assert session.requested_regions == ["ap-southeast-2"]
    assert client.model_ids == ["openai.gpt-oss-20b-1:0"]
    assert result.identity is not None
    assert result.identity.litellm_model == (
        "bedrock/converse/openai.gpt-oss-20b-1:0"
    )
    assert result.ragas.status is JudgePreflightStatus.AVAILABLE
    assert result.mlflow_genai.status is JudgePreflightStatus.PROVIDER_UNSUPPORTED


@pytest.mark.parametrize(
    "uri",
    ["meta.llama3-70b-instruct-v1:0", "bedrock:/bedrock/meta.llama3"],
)
def test_wrong_judge_mapping_is_rejected(uri: str) -> None:
    with pytest.raises(ValueError):
        resolve_judge_identity(uri, "us-east-1")


def test_preflight_uses_judge_region_not_generation_or_session_region() -> None:
    client = _BedrockClient()
    session = _Session(client)
    result = preflight_bedrock_judge(
        MLflowSettings(judge_model=JUDGE_URI, judge_region="us-east-1"),
        session_factory=lambda: session,
    )

    assert session.requested_regions == ["us-east-1"]
    assert client.model_ids == ["meta.llama3-70b-instruct-v1:0"]
    assert result.session_region == "ap-southeast-2"
    assert result.metadata_status is JudgePreflightStatus.AVAILABLE
    assert result.ragas.status is JudgePreflightStatus.AVAILABLE
    assert result.mlflow_genai.status is JudgePreflightStatus.PROVIDER_UNSUPPORTED
    assert result.invocation_permission_verified is False


def test_unsupported_region_and_access_denied_are_distinct() -> None:
    unavailable = preflight_bedrock_judge(
        MLflowSettings(judge_model=JUDGE_URI, judge_region="ap-southeast-2"),
        session_factory=lambda: _Session(_BedrockClient(error_code="ValidationException")),
    )
    denied = preflight_bedrock_judge(
        MLflowSettings(judge_model=JUDGE_URI, judge_region="us-east-1"),
        session_factory=lambda: _Session(_BedrockClient(error_code="AccessDeniedException")),
    )

    assert unavailable.metadata_status is JudgePreflightStatus.MODEL_UNAVAILABLE
    assert denied.metadata_status is JudgePreflightStatus.ACCESS_DENIED
    assert unavailable.ragas.status is JudgePreflightStatus.MODEL_UNAVAILABLE
    assert denied.ragas.status is JudgePreflightStatus.ACCESS_DENIED


def test_missing_credentials_are_reported_without_values() -> None:
    result = preflight_bedrock_judge(
        MLflowSettings(judge_model=JUDGE_URI, judge_region="us-east-1"),
        session_factory=lambda: _Session(_BedrockClient(), credentials=False),
    )

    assert result.metadata_status is JudgePreflightStatus.CREDENTIALS_UNAVAILABLE
    assert result.credentials_resolved is False
    assert "secret" not in result.model_dump_json().casefold()
