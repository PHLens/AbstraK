"""Strict YAML manifests for provider endpoints and model checkpoints."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from abstrak.providers.contracts import sha256_json

IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
ENV_PATTERN = r"^[A-Z_][A-Z0-9_]*$"


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetryPolicy(ManifestModel):
    max_attempts: Literal[1] = 1


class TransportPolicy(ManifestModel):
    stream: Literal[False] = False
    candidates: Literal[1] = 1
    allow_fallback: Literal[False] = False
    allow_cache: Literal[False] = False
    drop_unsupported_params: Literal[False] = False


class ProviderManifest(ManifestModel):
    schema_version: Literal["provider.v1"] = "provider.v1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    adapter: Literal["litellm"] = "litellm"
    protocol: Literal["chat_completions"] = "chat_completions"
    litellm_provider: str | None = Field(default=None, min_length=1)
    base_url_env: str | None = Field(default=None, pattern=ENV_PATTERN)
    api_key_env: str = Field(pattern=ENV_PATTERN)
    timeout_seconds: float = Field(default=180, gt=0, le=3600)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    transport: TransportPolicy = Field(default_factory=TransportPolicy)


class GenerationConfig(ManifestModel):
    max_completion_tokens: int = Field(ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    api_seed: int | None = None
    stop: tuple[str, ...] = ()
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None

    @model_validator(mode="after")
    def reject_empty_stop(self) -> GenerationConfig:
        if any(not marker for marker in self.stop):
            raise ValueError("stop markers cannot be empty")
        return self

    def transport_parameters(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {"max_completion_tokens": self.max_completion_tokens}
        optional = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.api_seed,
            "reasoning_effort": self.reasoning_effort,
        }
        parameters.update({key: value for key, value in optional.items() if value is not None})
        if self.stop:
            parameters["stop"] = list(self.stop)
        return parameters


class CapabilityExpectations(ManifestModel):
    usage_reporting: Literal["required", "optional"] = "required"
    returned_model: Literal["required", "optional"] = "required"
    system_messages: Literal["required", "unsupported", "untested"] = "required"


class ModelManifest(ManifestModel):
    schema_version: Literal["model.v1"] = "model.v1"
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    provider: str = Field(pattern=IDENTIFIER_PATTERN)
    api_model: str = Field(min_length=1)
    model_id_policy: Literal["exact", "mutable_alias"]
    expected_returned_model: str | None = Field(default=None, min_length=1)
    interface: Literal["chat_completions"] = "chat_completions"
    allow_live_probe: bool = False
    generation: GenerationConfig
    output_contract: Literal["plain_text", "plain_json"] = "plain_text"
    capabilities: CapabilityExpectations = Field(default_factory=CapabilityExpectations)
    pricing_ref: str | None = None

    @model_validator(mode="after")
    def exact_model_requires_expected_id(self) -> ModelManifest:
        if self.model_id_policy == "exact" and self.expected_returned_model is None:
            raise ValueError("exact model_id_policy requires expected_returned_model")
        if self.allow_live_probe:
            if self.output_contract != "plain_json":
                raise ValueError("the P0.1 live probe requires output_contract: plain_json")
            required_capabilities = {
                "usage_reporting": self.capabilities.usage_reporting,
                "returned_model": self.capabilities.returned_model,
                "system_messages": self.capabilities.system_messages,
            }
            weak = {
                name: value for name, value in required_capabilities.items() if value != "required"
            }
            if weak:
                raise ValueError(
                    "the P0.1 live probe requires usage, returned model, and system messages"
                )
        return self


class PricingManifest(ManifestModel):
    schema_version: Literal["pricing.v1"] = "pricing.v1"
    currency: Literal["USD"] = "USD"
    effective_at: datetime
    retrieved_at: datetime
    source: str = Field(min_length=1)
    per_1m_uncached_input: Decimal | None = Field(default=None, ge=0)
    per_1m_cached_input: Decimal | None = Field(default=None, ge=0)
    per_1m_output: Decimal | None = Field(default=None, ge=0)
    per_1m_reasoning: Decimal | None = Field(default=None, ge=0)
    request_fee: Decimal | None = Field(default=None, ge=0)


class ManifestBundle(ManifestModel):
    provider: ProviderManifest
    model: ModelManifest
    pricing: PricingManifest | None = None

    @model_validator(mode="after")
    def references_match(self) -> ManifestBundle:
        if self.model.provider != self.provider.id:
            raise ValueError(
                f"model provider {self.model.provider!r} does not match {self.provider.id!r}"
            )
        return self


class ManifestLoadError(ValueError):
    pass


class MissingEnvironmentError(ValueError):
    def __init__(self, variables: tuple[str, ...]) -> None:
        super().__init__(f"missing required environment variables: {', '.join(variables)}")
        self.variables = variables


ManifestT = TypeVar("ManifestT", bound=ManifestModel)


def load_manifest(path: str | Path, model_type: type[ManifestT]) -> ManifestT:
    manifest_path = Path(path)
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ManifestLoadError(f"cannot read {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestLoadError(f"{manifest_path} must contain one YAML mapping")
    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        raise ManifestLoadError(f"invalid {manifest_path}: {error}") from error


def manifest_sha256(manifest: ManifestModel) -> str:
    return sha256_json(manifest)


def required_environment(provider: ProviderManifest) -> tuple[str, ...]:
    variables = [provider.api_key_env]
    if provider.base_url_env is not None:
        variables.append(provider.base_url_env)
    return tuple(variables)


def resolve_environment(
    provider: ProviderManifest, environment: Mapping[str, str] | None = None
) -> tuple[str, str | None]:
    values = os.environ if environment is None else environment
    missing = tuple(name for name in required_environment(provider) if not values.get(name))
    if missing:
        raise MissingEnvironmentError(missing)
    api_key = values[provider.api_key_env]
    base_url = values[provider.base_url_env] if provider.base_url_env is not None else None
    return api_key, base_url
