from __future__ import annotations

from typing import Any, cast

import pytest
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
from pytest import MonkeyPatch

from consumer_finance_rag.bedrock import (
    BedrockConfigurationError,
    BedrockInvocationError,
    BedrockLLMProvider,
)


class FakeBedrockClient:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.request = kwargs
        return {
            "output": {"message": {"content": [{"text": "Grounded answer [C1]."}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 21, "outputTokens": 7, "totalTokens": 28},
            "ResponseMetadata": {"RequestId": "request-1"},
        }


def test_unconfigured_bedrock_fails_before_creating_or_calling_client() -> None:
    provider = BedrockLLMProvider(
        region="us-east-1",
        model_id=None,
        max_tokens=100,
        temperature=0.1,
        top_p=0.9,
    )
    with pytest.raises(BedrockConfigurationError, match="BEDROCK_GENERATION_MODEL"):
        provider.generate("system", "user")


def test_bedrock_configuration_is_passed_to_converse_without_real_aws() -> None:
    client = FakeBedrockClient()
    provider = BedrockLLMProvider(
        region="ap-southeast-2",
        model_id="vendor.configurable-model",
        max_tokens=321,
        temperature=0.2,
        top_p=0.8,
        client=client,
    )
    result = provider.generate("system prompt", "user prompt")

    assert result.text == "Grounded answer [C1]."
    assert result.model_id == "vendor.configurable-model"
    assert result.response_metadata == {
        "stop_reason": "end_turn",
        "request_id": "request-1",
        "usage": {"inputTokens": 21, "outputTokens": 7, "totalTokens": 28},
    }
    assert client.request is not None
    assert client.request["modelId"] == "vendor.configurable-model"
    assert client.request["inferenceConfig"]["maxTokens"] == 321


def test_runtime_client_has_explicit_timeout_and_retry_configuration(
    monkeypatch: MonkeyPatch,
) -> None:
    client = FakeBedrockClient()
    captured: dict[str, Any] = {}

    def fake_boto3_client(service_name: str, **kwargs: Any) -> FakeBedrockClient:
        captured["service_name"] = service_name
        captured.update(kwargs)
        return client

    monkeypatch.setattr("consumer_finance_rag.bedrock.boto3.client", fake_boto3_client)
    provider = BedrockLLMProvider(
        region="ap-southeast-2",
        model_id="vendor.configurable-model",
        max_tokens=321,
        temperature=0.2,
        top_p=0.8,
        connect_timeout_seconds=4,
        read_timeout_seconds=90,
        retry_mode="adaptive",
        total_max_attempts=4,
        tcp_keepalive=True,
    )

    provider.generate("system prompt", "user prompt")

    assert captured["service_name"] == "bedrock-runtime"
    assert captured["region_name"] == "ap-southeast-2"
    raw_config = captured["config"]
    assert isinstance(raw_config, Config)
    # botocore exposes these values at runtime, but its Config type omits the attributes.
    config = cast(Any, raw_config)
    assert config.connect_timeout == 4
    assert config.read_timeout == 90
    assert config.retries == {"mode": "adaptive", "total_max_attempts": 4}
    assert config.tcp_keepalive is True


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ClientError(
                cast(
                    Any,
                    {
                        "Error": {
                            "Code": "AccessDeniedException",
                            "Message": "model access denied",
                        },
                        "ResponseMetadata": {"RequestId": "denied-request"},
                    },
                ),
                "Converse",
            ),
            "AccessDeniedException.*model access denied.*denied-request",
        ),
        (
            EndpointConnectionError(endpoint_url="https://bedrock-runtime.example.test"),
            "AWS Bedrock client failed.*bedrock-runtime.example.test",
        ),
    ],
)
def test_bedrock_aws_errors_are_actionable_without_real_aws(
    error: Exception, expected: str
) -> None:
    class FailingBedrockClient:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            raise error

    provider = BedrockLLMProvider(
        region="ap-southeast-2",
        model_id="vendor.configurable-model",
        max_tokens=321,
        temperature=0.2,
        top_p=0.8,
        client=FailingBedrockClient(),
    )

    with pytest.raises(BedrockInvocationError, match=expected):
        provider.generate("system prompt", "user prompt")
