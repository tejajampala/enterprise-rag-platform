"""Read-only validation for the separately configured Bedrock judge."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Protocol, cast

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    CredentialRetrievalError,
    LoginRefreshRequired,
    NoCredentialsError,
)
from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.config import MLflowSettings


class JudgePreflightStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    CREDENTIALS_UNAVAILABLE = "CREDENTIALS_UNAVAILABLE"
    ACCESS_DENIED = "ACCESS_DENIED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_UNSUPPORTED = "PROVIDER_UNSUPPORTED"


class JudgeIdentity(BaseModel):
    """The three deliberately distinct forms of the same configured judge."""

    model_config = ConfigDict(frozen=True)

    configuration_uri: str
    litellm_model: str
    bedrock_model_id: str
    region: str
    provider_family: str
    structured_output_transport: str


class JudgeBackendPreflight(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: JudgePreflightStatus
    reason: str
    supported_metrics: tuple[str, ...] = ()
    unavailable_metrics: dict[str, str] = Field(default_factory=dict)


class JudgePreflightResult(BaseModel):
    """Privacy-safe result; credential values and model content are never retained."""

    model_config = ConfigDict(frozen=True)

    identity: JudgeIdentity | None
    metadata_status: JudgePreflightStatus
    metadata_reason: str
    credentials_resolved: bool
    credential_method: str | None
    session_region: str | None
    invocation_permission_verified: bool = False
    mlflow_genai: JudgeBackendPreflight
    ragas: JudgeBackendPreflight


class _Session(Protocol):
    region_name: str | None

    def get_credentials(self) -> Any: ...

    def client(self, service_name: str, *, region_name: str) -> Any: ...


SessionFactory = Callable[[], _Session]


def resolve_judge_identity(judge_uri: str, judge_region: str) -> JudgeIdentity:
    """Resolve the MLflow URI without ever passing its prefix to AWS."""

    prefix = "bedrock:/"
    if not judge_uri.startswith(prefix):
        raise ValueError("The configured judge must use bedrock:/<model-id>.")
    model_id = judge_uri.removeprefix(prefix).strip()
    if not model_id or model_id.startswith("bedrock/") or model_id.startswith("bedrock:/"):
        raise ValueError("The configured Bedrock judge model ID is invalid.")
    if not judge_region.strip():
        raise ValueError("The configured judge Region is empty.")
    if model_id == "openai.gpt-oss-20b-1:0":
        litellm_model = f"bedrock/converse/{model_id}"
        provider_family = "openai-gpt-oss"
        structured_output_transport = "bedrock-converse-response-format"
    elif model_id == "meta.llama3-70b-instruct-v1:0":
        litellm_model = f"bedrock/{model_id}"
        provider_family = "meta-llama"
        structured_output_transport = "bedrock-native-instructor-json"
    else:
        litellm_model = f"bedrock/{model_id}"
        provider_family = "generic-bedrock"
        structured_output_transport = "unverified"
    return JudgeIdentity(
        configuration_uri=judge_uri,
        litellm_model=litellm_model,
        bedrock_model_id=model_id,
        region=judge_region,
        provider_family=provider_family,
        structured_output_transport=structured_output_transport,
    )


def _mlflow_bedrock_support() -> JudgeBackendPreflight:
    """Detect the installed MLflow Bedrock gateway limitation without invoking a model."""

    try:
        from mlflow.gateway.providers.bedrock import AmazonBedrockProvider

        owns_endpoint = "get_endpoint_url" in AmazonBedrockProvider.__dict__
    except Exception:
        owns_endpoint = False
    if not owns_endpoint:
        reason = (
            "Installed MLflow's AmazonBedrockProvider does not implement get_endpoint_url; "
            "Bedrock judge invocation is unsupported by this MLflow adapter."
        )
        return JudgeBackendPreflight(
            status=JudgePreflightStatus.PROVIDER_UNSUPPORTED,
            reason=reason,
            unavailable_metrics={
                "correctness": "MLFLOW_BEDROCK_ENDPOINT_UNSUPPORTED",
                "relevance_to_query": "MLFLOW_BEDROCK_ENDPOINT_UNSUPPORTED",
                "safety": "MLFLOW_BEDROCK_ENDPOINT_UNSUPPORTED",
                "retrieval_groundedness": "MLFLOW_BUILTIN_GROUNDEDNESS_UNSUPPORTED",
            },
        )
    return JudgeBackendPreflight(
        status=JudgePreflightStatus.AVAILABLE,
        reason=(
            "Installed MLflow exposes the required Bedrock provider endpoint contract. "
            "InvokeModel permission is not verified by metadata preflight."
        ),
        supported_metrics=("correctness", "relevance_to_query", "safety"),
        unavailable_metrics={
            "retrieval_groundedness": "MLFLOW_BUILTIN_GROUNDEDNESS_UNAVAILABLE"
        },
    )


def _unavailable_backend(
    status: JudgePreflightStatus, reason: str
) -> JudgeBackendPreflight:
    return JudgeBackendPreflight(status=status, reason=reason)


def ragas_provider_support(identity: JudgeIdentity) -> JudgeBackendPreflight:
    """Validate the installed local RAGAS/LiteLLM route without model inference."""

    if identity.provider_family == "openai-gpt-oss":
        try:
            from litellm.constants import BEDROCK_CONVERSE_MODELS
            from litellm.llms.bedrock.chat.converse_transformation import (
                AmazonConverseConfig,
            )

            supported = AmazonConverseConfig().get_supported_openai_params(
                identity.bedrock_model_id
            )
        except Exception as error:
            return JudgeBackendPreflight(
                status=JudgePreflightStatus.PROVIDER_UNSUPPORTED,
                reason=(
                    "Installed LiteLLM could not validate GPT-OSS structured output "
                    f"({type(error).__name__})."
                ),
            )
        if (
            identity.bedrock_model_id not in BEDROCK_CONVERSE_MODELS
            or identity.litellm_model
            != f"bedrock/converse/{identity.bedrock_model_id}"
            or not {"max_tokens", "response_format"}.issubset(supported)
        ):
            return JudgeBackendPreflight(
                status=JudgePreflightStatus.PROVIDER_UNSUPPORTED,
                reason=(
                    "Installed LiteLLM does not expose the required GPT-OSS Bedrock "
                    "Converse structured-output contract."
                ),
            )
    return JudgeBackendPreflight(
        status=JudgePreflightStatus.AVAILABLE,
        reason=(
            "RAGAS can use the explicit async LiteLLM Bedrock route and judge Region; "
            "structured-output routing is supported locally, but InvokeModel permission "
            "has not been verified."
        ),
        supported_metrics=(
            "faithfulness",
            "context_precision",
            "context_recall",
            "response_groundedness",
        ),
    )


def preflight_bedrock_judge(
    settings: MLflowSettings,
    *,
    session_factory: SessionFactory | None = None,
) -> JudgePreflightResult:
    """Perform only credential-chain resolution and Bedrock control-plane metadata lookup."""

    judge_region = settings.judge_region
    if not settings.judge_model or not judge_region:
        reason = "Judge model and judge Region must both be configured."
        unavailable = _unavailable_backend(
            JudgePreflightStatus.CONFIGURATION_ERROR, reason
        )
        return JudgePreflightResult(
            identity=None,
            metadata_status=JudgePreflightStatus.CONFIGURATION_ERROR,
            metadata_reason=reason,
            credentials_resolved=False,
            credential_method=None,
            session_region=None,
            mlflow_genai=unavailable,
            ragas=unavailable,
        )
    try:
        identity = resolve_judge_identity(settings.judge_model, judge_region)
    except ValueError as error:
        reason = str(error)
        unavailable = _unavailable_backend(
            JudgePreflightStatus.CONFIGURATION_ERROR, reason
        )
        return JudgePreflightResult(
            identity=None,
            metadata_status=JudgePreflightStatus.CONFIGURATION_ERROR,
            metadata_reason=reason,
            credentials_resolved=False,
            credential_method=None,
            session_region=None,
            mlflow_genai=unavailable,
            ragas=unavailable,
        )

    if session_factory is None:
        import boto3

        factory = cast(SessionFactory, boto3.Session)
    else:
        factory = session_factory
    session = factory()
    credentials = session.get_credentials()
    credential_method = getattr(credentials, "method", None) if credentials else None
    if credentials is None:
        status = JudgePreflightStatus.CREDENTIALS_UNAVAILABLE
        reason = "AWS credentials were not resolved from the established credential chain."
        unavailable = _unavailable_backend(status, reason)
        return JudgePreflightResult(
            identity=identity,
            metadata_status=status,
            metadata_reason=reason,
            credentials_resolved=False,
            credential_method=None,
            session_region=session.region_name,
            mlflow_genai=unavailable,
            ragas=unavailable,
        )

    try:
        client = session.client("bedrock", region_name=identity.region)
        response = client.get_foundation_model(modelIdentifier=identity.bedrock_model_id)
        details = response.get("modelDetails", {})
        lifecycle = details.get("modelLifecycle", {})
        active = lifecycle.get("status") == "ACTIVE"
        on_demand = "ON_DEMAND" in details.get("inferenceTypesSupported", [])
        if not active or not on_demand:
            raise ValueError("The judge model is not ACTIVE with ON_DEMAND inference.")
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", "AWS_ERROR"))
        if code in {"AccessDenied", "AccessDeniedException", "UnauthorizedException"}:
            status = JudgePreflightStatus.ACCESS_DENIED
            reason = "Bedrock model metadata access was denied in the judge Region."
        elif code in {"ValidationException", "ResourceNotFoundException"}:
            try:
                listed = client.list_foundation_models().get("modelSummaries", [])
                found = next(
                    (
                        item
                        for item in listed
                        if item.get("modelId") == identity.bedrock_model_id
                    ),
                    None,
                )
            except ClientError as list_error:
                list_code = str(
                    list_error.response.get("Error", {}).get("Code", "AWS_ERROR")
                )
                if list_code in {
                    "AccessDenied",
                    "AccessDeniedException",
                    "UnauthorizedException",
                }:
                    status = JudgePreflightStatus.ACCESS_DENIED
                    reason = "Bedrock model catalog access was denied in the judge Region."
                else:
                    status = JudgePreflightStatus.PROVIDER_UNAVAILABLE
                    reason = f"Bedrock model catalog lookup failed ({list_code})."
            else:
                if found is None:
                    status = JudgePreflightStatus.MODEL_UNAVAILABLE
                    reason = "The configured Bedrock model is unavailable in the judge Region."
                else:
                    status = JudgePreflightStatus.PROVIDER_UNAVAILABLE
                    reason = (
                        "Bedrock listed the model but GetFoundationModel did not resolve it."
                    )
        else:
            status = JudgePreflightStatus.PROVIDER_UNAVAILABLE
            reason = f"Bedrock model metadata lookup failed ({code})."
    except (NoCredentialsError, LoginRefreshRequired, CredentialRetrievalError):
        status = JudgePreflightStatus.CREDENTIALS_UNAVAILABLE
        reason = (
            "AWS credentials were unavailable or require login refresh during Bedrock "
            "metadata lookup."
        )
    except (BotoCoreError, ValueError) as error:
        status = JudgePreflightStatus.PROVIDER_UNAVAILABLE
        reason = f"Bedrock model metadata lookup failed ({type(error).__name__})."
    else:
        status = JudgePreflightStatus.AVAILABLE
        reason = (
            "The configured model is ACTIVE with ON_DEMAND metadata in the judge Region; "
            "InvokeModel permission has not been verified."
        )

    if status is JudgePreflightStatus.AVAILABLE:
        mlflow = _mlflow_bedrock_support()
        ragas = ragas_provider_support(identity)
    else:
        mlflow = _unavailable_backend(status, reason)
        ragas = _unavailable_backend(status, reason)
    return JudgePreflightResult(
        identity=identity,
        metadata_status=status,
        metadata_reason=reason,
        credentials_resolved=(status is not JudgePreflightStatus.CREDENTIALS_UNAVAILABLE),
        credential_method=(str(credential_method) if credential_method else "resolved"),
        session_region=session.region_name,
        mlflow_genai=mlflow,
        ragas=ragas,
    )
