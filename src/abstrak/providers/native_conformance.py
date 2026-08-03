"""Offline dependency conformance for protocol-native provider routes."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from abstrak.providers.native_contracts import (
    EXPECTED_LITELLM_VERSION,
    NATIVE_DEPENDENCY_EVALUATOR_VERSION,
    NativeConformanceCheck,
    NativeDependencyConformance,
    NativeManifestBundle,
    NativeReasoningRecord,
)


def _package_version() -> str:
    try:
        return version("litellm")
    except PackageNotFoundError:
        return "missing"


def _check(name: str, passed: bool, detail: str) -> NativeConformanceCheck:
    return NativeConformanceCheck(
        name=name,
        status="pass" if passed else "fail",
        detail=detail,
    )


def _deepseek_reasoning(
    litellm: Any,
    bundle: NativeManifestBundle,
) -> tuple[NativeReasoningRecord, bool, str]:
    requested: dict[str, Any] = {
        "max_completion_tokens": bundle.model.max_output_tokens,
        "stream": False,
    }
    if bundle.model.temperature is not None:
        requested["temperature"] = bundle.model.temperature
    if bundle.model.top_p is not None:
        requested["top_p"] = bundle.model.top_p
    try:
        resolved = litellm.get_optional_params(
            model=bundle.model.api_model,
            custom_llm_provider="deepseek",
            reasoning_effort="xhigh",
            drop_params=False,
            **requested,
        )
    except Exception as error:
        reasoning = NativeReasoningRecord(
            submitted_parameter="reasoning_effort",
            submitted_value="xhigh",
            effective_mode="unknown",
            fidelity="unknown",
            evidence=f"LiteLLM parameter resolution failed: {type(error).__name__}",
        )
        return reasoning, False, f"parameter rendering failed: {type(error).__name__}: {error}"
    mismatches = tuple(key for key, value in requested.items() if resolved.get(key) != value)
    parameters_preserved = not mismatches
    parameter_detail = (
        "all requested non-reasoning generation parameters were preserved"
        if parameters_preserved
        else "changed or omitted parameters: " + ", ".join(mismatches)
    )
    if resolved.get("reasoning_effort") == "xhigh":
        reasoning = NativeReasoningRecord(
            submitted_parameter="reasoning_effort",
            submitted_value="xhigh",
            effective_mode="literal_xhigh",
            fidelity="literal",
            evidence="LiteLLM DeepSeek optional parameters preserve literal xhigh",
        )
        return reasoning, parameters_preserved, parameter_detail
    if resolved.get("thinking") == {"type": "enabled"}:
        reasoning = NativeReasoningRecord(
            submitted_parameter="reasoning_effort",
            submitted_value="xhigh",
            effective_mode="thinking_enabled",
            fidelity="collapsed",
            evidence="LiteLLM DeepSeek maps every non-none effort to binary thinking-enabled",
        )
        return reasoning, parameters_preserved, parameter_detail
    reasoning = NativeReasoningRecord(
        submitted_parameter="reasoning_effort",
        submitted_value="xhigh",
        effective_mode="unknown",
        fidelity="unknown",
        evidence="LiteLLM DeepSeek effective reasoning parameter is unknown",
    )
    return reasoning, parameters_preserved, parameter_detail


def _openai_responses_reasoning(
    bundle: NativeManifestBundle,
) -> tuple[NativeReasoningRecord, bool, str, bool, str]:
    try:
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_responses_api_config(
            model=bundle.model.api_model,
            provider="openai",
        )
        if config is None:
            raise RuntimeError("OpenAI Responses config is unavailable")
    except Exception as error:
        reasoning = NativeReasoningRecord(
            submitted_parameter="reasoning",
            submitted_value={"effort": "xhigh"},
            effective_mode="unknown",
            fidelity="unknown",
            evidence=f"OpenAI Responses config resolution failed: {type(error).__name__}",
        )
        return reasoning, False, reasoning.evidence, False, reasoning.evidence

    requested: dict[str, Any] = {
        "reasoning": {"effort": "xhigh"},
        "max_output_tokens": bundle.model.max_output_tokens,
        "store": False,
        "truncation": "disabled",
    }
    if bundle.model.temperature is not None:
        requested["temperature"] = bundle.model.temperature
    if bundle.model.top_p is not None:
        requested["top_p"] = bundle.model.top_p
    try:
        resolved = config.map_openai_params(
            requested,
            bundle.model.api_model,
            False,
        )
    except Exception as error:
        reasoning = NativeReasoningRecord(
            submitted_parameter="reasoning",
            submitted_value={"effort": "xhigh"},
            effective_mode="unknown",
            fidelity="unknown",
            evidence=f"OpenAI Responses parameter resolution failed: {type(error).__name__}",
        )
        detail = f"parameter rendering failed: {type(error).__name__}: {error}"
        return reasoning, True, type(config).__name__, False, detail
    mismatches = tuple(key for key, value in requested.items() if resolved.get(key) != value)
    parameters_preserved = not mismatches
    preserved = resolved.get("reasoning") == requested["reasoning"]
    reasoning = NativeReasoningRecord(
        submitted_parameter="reasoning",
        submitted_value={"effort": "xhigh"},
        effective_mode="literal_xhigh" if preserved else "unknown",
        fidelity="literal" if preserved else "unknown",
        evidence=(
            "native OpenAI Responses mapping preserves reasoning.effort=xhigh"
            if preserved
            else "native OpenAI Responses mapping did not preserve literal xhigh"
        ),
    )
    parameter_detail = (
        "all requested Responses generation parameters were preserved"
        if parameters_preserved
        else "changed or omitted parameters: " + ", ".join(mismatches)
    )
    return reasoning, True, type(config).__name__, parameters_preserved, parameter_detail


def evaluate_native_dependency_conformance(
    bundle: NativeManifestBundle,
) -> NativeDependencyConformance:
    """Resolve provider semantics without issuing any network request."""

    import litellm

    installed_version = _package_version()
    checks = [
        _check(
            "litellm_version",
            installed_version == EXPECTED_LITELLM_VERSION,
            f"expected {EXPECTED_LITELLM_VERSION}, found {installed_version}",
        )
    ]

    if bundle.provider.protocol == "chat_completions":
        entrypoint_ready = callable(getattr(litellm, "completion", None))
        checks.append(
            _check(
                "native_chat_entrypoint",
                entrypoint_ready,
                "litellm.completion is available for the DeepSeek Chat route",
            )
        )
        reasoning, parameters_preserved, parameter_detail = _deepseek_reasoning(
            litellm,
            bundle,
        )
    else:
        entrypoint_ready = callable(getattr(litellm, "responses", None))
        checks.append(
            _check(
                "native_responses_entrypoint",
                entrypoint_ready,
                "litellm.responses is available for the OpenAI Responses route",
            )
        )
        (
            reasoning,
            native_config_ready,
            config_detail,
            parameters_preserved,
            parameter_detail,
        ) = _openai_responses_reasoning(
            bundle,
        )
        checks.append(
            _check(
                "native_responses_config",
                native_config_ready,
                config_detail,
            )
        )

    checks.append(
        _check(
            "generation_parameters_preserved",
            parameters_preserved,
            parameter_detail,
        )
    )
    checks.append(
        _check(
            "literal_xhigh_preserved",
            reasoning.fidelity == "literal",
            (
                reasoning.evidence
                if reasoning.fidelity == "literal"
                else f"dependency blocked: {reasoning.evidence}"
            ),
        )
    )
    status = "fail" if any(check.status == "fail" for check in checks) else "pass"
    dependency_ready = status == "pass" and reasoning.fidelity == "literal"
    return NativeDependencyConformance(
        evaluator_version=NATIVE_DEPENDENCY_EVALUATOR_VERSION,
        status=status,
        dependency_ready=dependency_ready,
        study_ready=False,
        study_readiness=(
            "pending_endpoint_conformance"
            if dependency_ready
            else "dependency_blocked"
        ),
        protocol=bundle.provider.protocol,
        provider_manifest_sha256=bundle.provider_sha256,
        model_manifest_sha256=bundle.model_sha256,
        expected_litellm_version=EXPECTED_LITELLM_VERSION,
        observed_litellm_version=installed_version,
        reasoning=reasoning,
        checks=tuple(checks),
    )
