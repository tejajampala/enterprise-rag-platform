"""AWS Bedrock generation provider using the model-neutral Converse API."""

from __future__ import annotations

from typing import Any, Literal

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from consumer_finance_rag.providers import LLMResult


class BedrockConfigurationError(RuntimeError):
    """Raised when Bedrock has not been configured for generation."""


class BedrockInvocationError(RuntimeError):
    """Raised when AWS cannot complete a Bedrock generation request."""


class BedrockLLMProvider:
    """Generate through Bedrock without binding the RAG pipeline to a model vendor."""

    def __init__(
        self,
        *,
        region: str,
        model_id: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 120.0,
        retry_mode: Literal["standard", "adaptive"] = "standard",
        total_max_attempts: int = 3,
        tcp_keepalive: bool = True,
        client: Any | None = None,
    ) -> None:
        self._region = region
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._client_config = Config(
            connect_timeout=connect_timeout_seconds,
            read_timeout=read_timeout_seconds,
            retries={
                "mode": retry_mode,
                "total_max_attempts": total_max_attempts,
            },
            tcp_keepalive=tcp_keepalive,
        )
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model_id or "not-configured"

    def _runtime_client(self) -> Any:
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
                config=self._client_config,
            )
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        if not self._model_id:
            raise BedrockConfigurationError(
                "AWS Bedrock generation is not configured. Set BEDROCK_GENERATION_MODEL "
                "to a model ID enabled in your AWS account, and configure standard AWS credentials."
            )
        try:
            response = self._runtime_client().converse(
                modelId=self._model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={
                    "maxTokens": self._max_tokens,
                    "temperature": self._temperature,
                    "topP": self._top_p,
                },
            )
        except ClientError as error:
            aws_error = error.response.get("Error", {})
            code = str(aws_error.get("Code", "UnknownAWSServiceError"))
            message = str(aws_error.get("Message", "AWS returned no error detail"))
            request_id = error.response.get("ResponseMetadata", {}).get("RequestId")
            request_detail = f"; request ID: {request_id}" if request_id else ""
            raise BedrockInvocationError(
                f"AWS Bedrock Converse failed [{code}]: {message}{request_detail}"
            ) from error
        except BotoCoreError as error:
            raise BedrockInvocationError(f"AWS Bedrock client failed: {error}") from error

        message = response.get("output", {}).get("message", {})
        blocks = message.get("content", [])
        text = "\n".join(str(block["text"]) for block in blocks if "text" in block).strip()
        response_metadata: dict[str, Any] = {
            "stop_reason": response.get("stopReason"),
            "request_id": response.get("ResponseMetadata", {}).get("RequestId"),
        }
        usage = response.get("usage")
        if isinstance(usage, dict):
            response_metadata["usage"] = usage
        return LLMResult(
            text=text,
            model_id=self._model_id,
            response_metadata=response_metadata,
        )
