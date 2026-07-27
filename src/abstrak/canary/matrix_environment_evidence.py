"""Produce recomputable matrix preflight environment evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from pydantic import Field, ValidationError, model_validator

from abstrak.canary.contracts import IDENTIFIER_PATTERN, SHA256_PATTERN, CanaryModel
from abstrak.canary.matrix_preflight import (
    EnvironmentManifest,
    EnvironmentObservation,
    EnvironmentProbeEvidence,
)
from abstrak.canary.matrix_runner import MatrixWorkerBinding
from abstrak.canary.remote import SshWorkerExecutor, WorkerExecutionError
from abstrak.providers.contracts import sha256_json


class MatrixEnvironmentEvidenceError(ValueError):
    """Raised when a probe is not bound to the frozen expected environment."""


class EnvironmentHealthObservation(CanaryModel):
    """Canonical form of one successful extended worker health response."""

    schema_version: Literal["canary-worker-health.v1"] = "canary-worker-health.v1"
    status: Literal["healthy"] = "healthy"
    device: str = Field(pattern=r"^cuda:[0-9]+$")
    hardware: str = Field(min_length=1)
    compute_capability: tuple[int, int]
    python_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    torch_cuda_version: str = Field(min_length=1)
    triton_version: str = Field(min_length=1)
    tilelang_version: str = Field(min_length=1)
    driver_version: str = Field(min_length=1)
    container_markers: tuple[str, ...]
    non_container_worker: bool
    worker_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    value: float
    compatibility_error: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def health_values_are_self_consistent(self) -> EnvironmentHealthObservation:
        if any(value < 0 for value in self.compute_capability):
            raise ValueError("compute capability components cannot be negative")
        if (
            any(not marker for marker in self.container_markers)
            or len(self.container_markers) != len(set(self.container_markers))
        ):
            raise ValueError("container markers must be unique non-empty strings")
        if self.non_container_worker != (not self.container_markers):
            raise ValueError("container markers disagree with non_container_worker")
        if self.value != 2.0:
            raise ValueError("health probe value must be exactly 2.0")
        return self


class EnvironmentHealthFailure(CanaryModel):
    """Canonical health payload attached to an unsuccessful remote probe."""

    schema_version: Literal["canary-worker-health.v1"] = "canary-worker-health.v1"
    status: Literal["unhealthy", "check_failed"]
    device: str = Field(pattern=r"^cuda:[0-9]+$")
    error: str = Field(min_length=1)


EnvironmentHealth = EnvironmentHealthObservation | EnvironmentHealthFailure


class EnvironmentProbeFailure(CanaryModel):
    """Structured failure raised while obtaining an extended health response."""

    category: str = Field(pattern=IDENTIFIER_PATTERN)
    message: str = Field(min_length=1)


class EnvironmentProbeArtifact(CanaryModel):
    """Raw, hashable input from which environment evidence is derived."""

    schema_version: Literal["abstrak-matrix-environment-probe-artifact.v1"] = (
        "abstrak-matrix-environment-probe-artifact.v1"
    )
    expected_environment_sha256: str = Field(pattern=SHA256_PATTERN)
    controller_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    worker: MatrixWorkerBinding
    health: EnvironmentHealth | None = None
    probe_error: EnvironmentProbeFailure | None = None

    @model_validator(mode="after")
    def successful_probe_has_healthy_payload(self) -> EnvironmentProbeArtifact:
        if self.probe_error is None and not isinstance(
            self.health, EnvironmentHealthObservation
        ):
            raise ValueError("successful environment probe requires extended healthy payload")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class EnvironmentProbeResult(CanaryModel):
    """Artifact plus the evidence and manifest deterministically derived from it."""

    artifact: EnvironmentProbeArtifact
    evidence: EnvironmentProbeEvidence
    environment: EnvironmentManifest


class EnvironmentProbeWorker(Protocol):
    """Narrow worker surface required by the environment producer."""

    matrix_worker_binding: MatrixWorkerBinding | None
    expected_hardware_substring: str | None
    expected_compute_capability: tuple[int, int] | None
    expected_python_version: str | None
    expected_torch_version: str | None
    expected_torch_cuda_version: str | None
    expected_triton_version: str | None
    expected_tilelang_version: str | None
    expected_driver_version: str | None
    expected_non_container_worker: bool | None

    def validate_environment(self, device: str) -> dict[str, object]: ...


def _require_pending(environment: EnvironmentManifest) -> None:
    if environment.status != "pending":
        raise MatrixEnvironmentEvidenceError(
            "environment probe requires one pending expected environment"
        )


def _worker_binding(environment: EnvironmentManifest) -> MatrixWorkerBinding:
    return MatrixWorkerBinding(
        worker_revision=environment.worker_revision,
        transport=environment.transport,
    )


def _expected_worker_values(environment: EnvironmentManifest) -> dict[str, object]:
    major, minor = (int(value) for value in environment.compute_capability.split("."))
    return {
        "expected_hardware_substring": environment.accelerator,
        "expected_compute_capability": (major, minor),
        "expected_python_version": environment.python_version,
        "expected_torch_version": environment.torch_version,
        "expected_torch_cuda_version": environment.cuda_version,
        "expected_triton_version": environment.triton_version,
        "expected_tilelang_version": environment.tilelang_version,
        "expected_driver_version": environment.driver_version,
        "expected_non_container_worker": environment.non_container_worker,
    }


def _require_bound_worker(
    environment: EnvironmentManifest,
    worker: EnvironmentProbeWorker,
) -> None:
    if worker.matrix_worker_binding != _worker_binding(environment):
        raise MatrixEnvironmentEvidenceError(
            "environment probe worker route differs from the expected transport and revision"
        )
    mismatches = tuple(
        name
        for name, expected in _expected_worker_values(environment).items()
        if getattr(worker, name, None) != expected
    )
    if mismatches:
        raise MatrixEnvironmentEvidenceError(
            "environment probe worker expectations differ from the pending manifest: "
            + ", ".join(mismatches)
        )
    if not callable(getattr(worker, "validate_environment", None)):
        raise MatrixEnvironmentEvidenceError(
            "environment probe worker cannot validate its live environment"
        )


def build_environment_probe_worker(
    environment: EnvironmentManifest,
) -> SshWorkerExecutor:
    """Build an SSH health probe from every frozen route and version input."""

    _require_pending(environment)
    transport = environment.transport
    worker = SshWorkerExecutor(
        transport.host,
        port=transport.port,
        python_executable=transport.python_executable,
        pythonpath=transport.pythonpath,
        kernelbench_root=transport.kernelbench_root,
        asset_root=transport.asset_root,
        device=transport.device,
        sandbox_mode=(
            "setpriv" if transport.sandbox == "setpriv-supervised" else "bubblewrap"
        ),
        timeout_seconds=transport.timeout_seconds,
        expected_worker_revision=environment.worker_revision,
        **_expected_worker_values(environment),
    )
    _require_bound_worker(environment, worker)
    return worker


def _health_payload(value: Mapping[str, object]) -> EnvironmentHealth:
    payload = {str(key): item for key, item in value.items()}
    status = payload.get("status")
    if status == "healthy":
        capability = payload.get("compute_capability")
        markers = payload.get("container_markers")
        if isinstance(capability, list):
            payload["compute_capability"] = tuple(capability)
        if isinstance(markers, list):
            payload["container_markers"] = tuple(markers)
        return EnvironmentHealthObservation.model_validate(payload)
    return EnvironmentHealthFailure.model_validate(payload)


def _probe_artifact(
    environment: EnvironmentManifest,
    worker: EnvironmentProbeWorker,
) -> EnvironmentProbeArtifact:
    health: EnvironmentHealth | None = None
    probe_error: EnvironmentProbeFailure | None = None
    try:
        raw_health = worker.validate_environment(environment.transport.device)
        if not isinstance(raw_health, Mapping):
            raise ValueError("environment worker returned a non-object health payload")
        health = _health_payload(raw_health)
    except WorkerExecutionError as error:
        if error.health is not None:
            try:
                health = _health_payload(error.health)
            except (TypeError, ValueError, ValidationError):
                health = None
        probe_error = EnvironmentProbeFailure(
            category=error.category,
            message=str(error),
        )
    except (TypeError, ValueError, ValidationError) as error:
        probe_error = EnvironmentProbeFailure(
            category="invalid_health",
            message=f"{type(error).__name__}: {error}",
        )
    return EnvironmentProbeArtifact(
        expected_environment_sha256=environment.sha256,
        controller_revision=environment.controller_revision,
        worker=_worker_binding(environment),
        health=health,
        probe_error=probe_error,
    )


def _mismatch_reason(
    environment: EnvironmentManifest,
    health: EnvironmentHealthObservation,
) -> str | None:
    major, minor = health.compute_capability
    comparisons = (
        ("device", environment.transport.device, health.device),
        ("accelerator", environment.accelerator, health.hardware),
        ("compute_capability", environment.compute_capability, f"{major}.{minor}"),
        ("python_version", environment.python_version, health.python_version),
        ("tilelang_version", environment.tilelang_version, health.tilelang_version),
        ("triton_version", environment.triton_version, health.triton_version),
        ("torch_version", environment.torch_version, health.torch_version),
        ("cuda_version", environment.cuda_version, health.torch_cuda_version),
        ("driver_version", environment.driver_version, health.driver_version),
        ("worker_revision", environment.worker_revision, health.worker_revision),
        ("non_container_worker", environment.non_container_worker, health.non_container_worker),
        ("container_markers", (), health.container_markers),
        ("probe_value", 2.0, health.value),
        ("compatibility_error", None, health.compatibility_error),
    )
    mismatches = tuple(
        f"{name}: expected {expected!r}, observed {observed!r}"
        for name, expected, observed in comparisons
        if observed != expected
    )
    if not mismatches:
        return None
    return "environment probe mismatch: " + "; ".join(mismatches)


def _invalid_result(
    environment: EnvironmentManifest,
    artifact: EnvironmentProbeArtifact,
    reason: str,
) -> EnvironmentProbeResult:
    evidence = EnvironmentProbeEvidence(
        artifact_sha256=artifact.sha256,
        status="fail",
        failure_reason=reason,
    )
    payload = environment.model_dump(mode="python")
    payload.update(
        status="invalid",
        verification_evidence=evidence,
        invalid_reason=reason,
    )
    invalid = EnvironmentManifest.model_validate(payload)
    return EnvironmentProbeResult(
        artifact=artifact,
        evidence=evidence,
        environment=invalid,
    )


def derive_environment_probe(
    environment: EnvironmentManifest,
    artifact: EnvironmentProbeArtifact,
) -> EnvironmentProbeResult:
    """Recompute one terminal manifest without trusting a stored pass/fail claim."""

    _require_pending(environment)
    if (
        artifact.expected_environment_sha256 != environment.sha256
        or artifact.controller_revision != environment.controller_revision
        or artifact.worker != _worker_binding(environment)
    ):
        raise MatrixEnvironmentEvidenceError(
            "environment probe artifact differs from the expected environment"
        )
    if artifact.probe_error is not None:
        return _invalid_result(
            environment,
            artifact,
            f"environment probe {artifact.probe_error.category}: "
            f"{artifact.probe_error.message}",
        )
    health = artifact.health
    if not isinstance(health, EnvironmentHealthObservation):
        reason = (
            "environment probe returned no healthy observation"
            if health is None
            else f"environment probe {health.status}: {health.error}"
        )
        return _invalid_result(environment, artifact, reason)
    mismatch = _mismatch_reason(environment, health)
    if mismatch is not None:
        return _invalid_result(environment, artifact, mismatch)
    observation = EnvironmentObservation(
        worker_revision=health.worker_revision,
        transport=artifact.worker.transport,
        accelerator=health.hardware,
        compute_capability=f"{health.compute_capability[0]}.{health.compute_capability[1]}",
        python_version=health.python_version,
        tilelang_version=health.tilelang_version,
        triton_version=health.triton_version,
        torch_version=health.torch_version,
        cuda_version=health.torch_cuda_version,
        driver_version=health.driver_version,
    )
    evidence = EnvironmentProbeEvidence(
        artifact_sha256=artifact.sha256,
        status="pass",
        observation=observation,
    )
    payload = environment.model_dump(mode="python")
    payload.update(status="verified", verification_evidence=evidence)
    verified = EnvironmentManifest.model_validate(payload)
    return EnvironmentProbeResult(
        artifact=artifact,
        evidence=evidence,
        environment=verified,
    )


def run_environment_probe(
    environment: EnvironmentManifest,
    worker: EnvironmentProbeWorker,
) -> EnvironmentProbeResult:
    """Run one bound extended health probe and derive its terminal environment."""

    _require_pending(environment)
    _require_bound_worker(environment, worker)
    return derive_environment_probe(environment, _probe_artifact(environment, worker))
