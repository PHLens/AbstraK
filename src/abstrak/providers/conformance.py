"""Offline-evaluable checks and opt-in live provider smoke probes."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from abstrak.providers.artifacts import ProviderArtifactStore
from abstrak.providers.client import ProviderClient
from abstrak.providers.contracts import (
    ChatMessage,
    ConformanceCheck,
    ConformanceReport,
    LogicalRequest,
    MessageRole,
    NormalizedResponse,
    ProviderCallError,
)
from abstrak.providers.manifests import ManifestBundle, manifest_sha256


class ActionProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["finish"]
    nonce: str


class LiveProbeConfigurationError(ValueError):
    pass


def build_probe_request(model_ref: str, nonce: str | None = None) -> tuple[LogicalRequest, str]:
    probe_nonce = nonce or uuid4().hex
    request = LogicalRequest(
        model_ref=model_ref,
        messages=(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "When asked for the finish action, return only a JSON object with "
                    f'action "finish" and nonce "{probe_nonce}". Return no prose or markdown.'
                ),
            ),
            ChatMessage(role=MessageRole.USER, content="Return the finish action now."),
        ),
    )
    return request, probe_nonce


def evaluate_response(
    response: NormalizedResponse,
    bundle: ManifestBundle,
    expected_nonce: str,
    transport_call_count: int,
    source_clean: bool | None,
) -> tuple[ConformanceCheck, ...]:
    checks: list[ConformanceCheck] = [
        ConformanceCheck(
            name="single_transport_invocation",
            status="pass" if transport_call_count == 1 else "fail",
            detail=f"transport object was invoked {transport_call_count} time(s)",
        ),
        ConformanceCheck(
            name="nonempty_text",
            status="pass" if response.text else "fail",
            detail=f"received {len(response.text)} output characters",
        ),
        ConformanceCheck(
            name="finish_reason",
            status=(
                "fail"
                if not response.finish_reason
                else (
                    "pass"
                    if response.finish_reason in {"stop", "length", "content_filter", "tool_calls"}
                    else "warn"
                )
            ),
            detail=response.finish_reason or "provider omitted finish_reason",
        ),
    ]

    usage_required = bundle.model.capabilities.usage_reporting == "required"
    usage_ready = response.usage.provider_reported and response.usage.core_fields_complete
    checks.append(
        ConformanceCheck(
            name="usage_reporting",
            status="pass" if usage_ready else ("fail" if usage_required else "warn"),
            detail=(
                f"input={response.usage.input_tokens}, output={response.usage.output_tokens}, "
                f"total={response.usage.total_tokens}, "
                f"core_complete={response.usage.core_fields_complete}"
            ),
        )
    )
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    total_tokens = response.usage.total_tokens
    arithmetic_ok = not all(
        value is not None for value in (input_tokens, output_tokens, total_tokens)
    )
    if input_tokens is not None and output_tokens is not None and total_tokens is not None:
        arithmetic_ok = total_tokens >= input_tokens + output_tokens
    checks.extend(
        (
            ConformanceCheck(
                name="usage_plausibility",
                status=(
                    "pass"
                    if input_tokens is not None
                    and input_tokens > 0
                    and output_tokens is not None
                    and output_tokens > 0
                    and total_tokens is not None
                    and total_tokens > 0
                    else "fail"
                ),
                detail="a non-empty live probe requires positive input, output, and total tokens",
            ),
            ConformanceCheck(
                name="usage_arithmetic",
                status="pass" if arithmetic_ok else "fail",
                detail="total tokens must not be smaller than input plus output tokens",
            ),
            ConformanceCheck(
                name="cached_usage_consistency",
                status=(
                    "pass"
                    if response.usage.cached_input_tokens is None
                    or input_tokens is None
                    or response.usage.cached_input_tokens <= input_tokens
                    else "fail"
                ),
                detail="cached input tokens, when reported, cannot exceed input tokens",
            ),
            ConformanceCheck(
                name="reasoning_usage_consistency",
                status=(
                    "pass"
                    if response.usage.reasoning_tokens is None
                    or output_tokens is None
                    or response.usage.reasoning_tokens <= output_tokens
                    else "fail"
                ),
                detail="reasoning tokens, when reported, cannot exceed output tokens",
            ),
        )
    )

    returned_required = bundle.model.capabilities.returned_model == "required"
    checks.append(
        ConformanceCheck(
            name="returned_model_present",
            status=(
                "pass" if response.returned_model else ("fail" if returned_required else "warn")
            ),
            detail=response.returned_model or "provider omitted returned model",
        )
    )
    if bundle.model.model_id_policy == "exact":
        expected_model = bundle.model.expected_returned_model
        checks.append(
            ConformanceCheck(
                name="returned_model_exact",
                status="pass" if response.returned_model == expected_model else "fail",
                detail=f"expected {expected_model!r}, received {response.returned_model!r}",
            )
        )
    else:
        if response.returned_model != response.requested_model:
            checks.append(
                ConformanceCheck(
                    name="mutable_alias_observed",
                    status="warn",
                    detail=(
                        f"requested {response.requested_model!r}, "
                        f"provider returned {response.returned_model!r}"
                    ),
                )
            )
        checks.append(
            ConformanceCheck(
                name="pilot_model_identity",
                status="warn",
                detail="mutable aliases may be endpoint-conformant but are not pilot-ready",
            )
        )

    try:
        action = ActionProbe.model_validate_json(response.text)
    except ValidationError as error:
        checks.append(
            ConformanceCheck(
                name="plain_json_action",
                status="fail",
                detail=f"strict action JSON validation failed: {error.errors()[0]['msg']}",
            )
        )
    else:
        checks.extend(
            (
                ConformanceCheck(
                    name="plain_json_action",
                    status="pass",
                    detail="strict finish action JSON is valid",
                ),
                ConformanceCheck(
                    name="system_nonce_fidelity",
                    status="pass" if action.nonce == expected_nonce else "fail",
                    detail=(
                        "nonce supplied only in the system message was preserved"
                        if action.nonce == expected_nonce
                        else f"expected nonce {expected_nonce!r}, received {action.nonce!r}"
                    ),
                ),
            )
        )
    checks.append(
        ConformanceCheck(
            name="source_provenance",
            status="pass" if source_clean else "warn",
            detail=(
                "repository worktree is clean"
                if source_clean
                else "repository worktree is dirty or provenance is unavailable"
            ),
        )
    )
    return tuple(checks)


def run_live_probe(
    client: ProviderClient,
    *,
    artifact_root: str,
    nonce: str | None = None,
) -> tuple[ConformanceReport, ProviderArtifactStore]:
    if not client.bundle.model.allow_live_probe:
        raise LiveProbeConfigurationError(
            "model manifest must set allow_live_probe: true before a live probe"
        )
    if not artifact_root:
        raise LiveProbeConfigurationError("live probes require a non-empty artifact root")

    request, expected_nonce = build_probe_request(client.bundle.model.id, nonce)
    resolved_manifest = client.resolved_manifest_record
    git_state = resolved_manifest["runtime"]["git"]
    source_clean = (
        not git_state["worktree_dirty"] if git_state["worktree_dirty"] is not None else None
    )
    store = ProviderArtifactStore.create(
        artifact_root,
        client.bundle.provider.id,
        client.bundle.model.id,
        secrets=client.artifact_secrets,
    )
    store.write_json("manifest.resolved.json", resolved_manifest)
    store.write_json("request.logical.json", request)
    store.append_event({"event": "request_started", "request_id": request.request_id})

    calls_before = client.transport.call_count
    try:
        response = client.complete(request)
    except ProviderCallError as error:
        transport_call_count = client.transport.call_count - calls_before
        checks = (
            ConformanceCheck(
                name="transport_call",
                status="fail",
                detail=f"{error.record.category.value}: {error.record.sanitized_message}",
            ),
            ConformanceCheck(
                name="single_transport_invocation",
                status="pass" if transport_call_count == 1 else "fail",
                detail=f"transport object was invoked {transport_call_count} time(s)",
            ),
        )
        report = ConformanceReport(
            status="fail",
            transport_ready=False,
            action_protocol_ready=False,
            pilot_ready=False,
            provider_id=client.bundle.provider.id,
            model_id=client.bundle.model.id,
            provider_manifest_sha256=manifest_sha256(client.bundle.provider),
            model_manifest_sha256=manifest_sha256(client.bundle.model),
            checks=checks,
            error=error.record,
        )
        store.write_json("request.transport.json", error.record.sanitized_transport_request)
        store.write_json("response.error.json", error.record)
        store.append_event(
            {
                "event": "request_failed",
                "request_id": request.request_id,
                "category": error.record.category.value,
            }
        )
        store.write_json("result.json", report)
        store.finalize()
        return report, store

    transport_call_count = client.transport.call_count - calls_before
    checks = evaluate_response(
        response,
        client.bundle,
        expected_nonce,
        transport_call_count,
        source_clean,
    )
    failed_checks = {check.name for check in checks if check.status == "fail"}
    action_protocol_ready = not {
        "plain_json_action",
        "system_nonce_fidelity",
    }.intersection(failed_checks)
    transport_ready = not (failed_checks - {"plain_json_action", "system_nonce_fidelity"})
    status = "pass" if transport_ready and action_protocol_ready else "fail"
    pilot_ready = (
        status == "pass" and client.bundle.model.model_id_policy == "exact" and source_clean is True
    )
    report = ConformanceReport(
        status=status,
        transport_ready=transport_ready,
        action_protocol_ready=action_protocol_ready,
        pilot_ready=pilot_ready,
        provider_id=client.bundle.provider.id,
        model_id=client.bundle.model.id,
        provider_manifest_sha256=manifest_sha256(client.bundle.provider),
        model_manifest_sha256=manifest_sha256(client.bundle.model),
        checks=checks,
        response=response,
    )
    store.write_json("request.transport.json", response.sanitized_transport_request)
    store.write_json("response.sdk.json", response.raw_transport_response)
    store.write_json("response.normalized.json", response)
    store.append_event(
        {
            "event": "request_finished",
            "request_id": request.request_id,
            "status": report.status,
            "pilot_ready": report.pilot_ready,
        }
    )
    store.write_json("result.json", report)
    store.finalize()
    return report, store
