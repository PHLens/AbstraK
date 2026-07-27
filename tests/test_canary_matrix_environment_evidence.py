from __future__ import annotations

from copy import deepcopy

import pytest

from abstrak.canary.matrix_environment_evidence import (
    MatrixEnvironmentEvidenceError,
    build_environment_probe_worker,
    derive_environment_probe,
    run_environment_probe,
)
from abstrak.canary.matrix_preflight import EnvironmentManifest
from abstrak.canary.matrix_runner import MatrixTransportContext, MatrixWorkerBinding
from abstrak.canary.remote import WorkerExecutionError
from abstrak.providers.contracts import sha256_json


def _digest(label: str) -> str:
    return sha256_json({"label": label})


def _transport() -> MatrixTransportContext:
    return MatrixTransportContext(
        host="root@gpu.example",
        port=30554,
        worker_root="/srv/AbstraK",
        python_executable="/srv/venv/bin/python",
        pythonpath="/srv/AbstraK/src",
        kernelbench_root="/srv/KernelBench",
        asset_root="/srv/AbstraK/benchmarks/capability-gate-a100",
        sandbox="setpriv-supervised",
        device="cuda:0",
        timeout_seconds=420.0,
        network_isolated=False,
        filesystem_read_only=False,
    )


def _pending() -> EnvironmentManifest:
    return EnvironmentManifest(
        study_id="matrix-study",
        raw_study_sha256=_digest("raw-study"),
        spec_sha256=_digest("spec"),
        schedule_sha256=_digest("schedule"),
        status="pending",
        controller_revision="1" * 40,
        worker_revision="2" * 40,
        transport=_transport(),
        accelerator="NVIDIA A100-SXM4-80GB",
        compute_capability="8.0",
        python_version="3.10.20",
        tilelang_version="0.1.12",
        triton_version="3.7.1",
        torch_version="2.13.0+cu126",
        cuda_version="12.6",
        driver_version="570.00",
    )


def _health(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "canary-worker-health.v1",
        "status": "healthy",
        "device": "cuda:0",
        "hardware": "NVIDIA A100-SXM4-80GB",
        "compute_capability": [8, 0],
        "python_version": "3.10.20",
        "torch_version": "2.13.0+cu126",
        "torch_cuda_version": "12.6",
        "triton_version": "3.7.1",
        "tilelang_version": "0.1.12",
        "driver_version": "570.00",
        "container_markers": [],
        "non_container_worker": True,
        "worker_revision": "2" * 40,
        "value": 2.0,
    }
    value.update(updates)
    return value


class _Worker:
    def __init__(self, pending: EnvironmentManifest, health: dict[str, object]) -> None:
        major, minor = (int(value) for value in pending.compute_capability.split("."))
        self.matrix_worker_binding = MatrixWorkerBinding(
            worker_revision=pending.worker_revision,
            transport=pending.transport,
        )
        self.expected_hardware_substring = pending.accelerator
        self.expected_compute_capability = (major, minor)
        self.expected_python_version = pending.python_version
        self.expected_torch_version = pending.torch_version
        self.expected_torch_cuda_version = pending.cuda_version
        self.expected_triton_version = pending.triton_version
        self.expected_tilelang_version = pending.tilelang_version
        self.expected_driver_version = pending.driver_version
        self.expected_non_container_worker = pending.non_container_worker
        self.health = health
        self.calls: list[str] = []

    def validate_environment(self, device: str) -> dict[str, object]:
        self.calls.append(device)
        return deepcopy(self.health)


def test_environment_probe_is_hash_bound_and_recomputable() -> None:
    pending = _pending()
    worker = _Worker(pending, _health())

    result = run_environment_probe(pending, worker)

    assert worker.calls == ["cuda:0"]
    assert result.evidence.artifact_sha256 == result.artifact.sha256
    assert result.evidence.status == "pass"
    assert result.environment.status == "verified"
    assert result.environment.verification_evidence == result.evidence
    assert result.evidence.observation is not None
    assert result.evidence.observation.transport == pending.transport
    assert result.evidence.observation.worker_revision == pending.worker_revision
    assert derive_environment_probe(pending, result.artifact) == result


def test_environment_probe_recomputes_version_drift_as_invalid() -> None:
    pending = _pending()
    result = run_environment_probe(
        pending,
        _Worker(pending, _health(triton_version="3.8.0")),
    )

    assert result.evidence.status == "fail"
    assert result.environment.status == "invalid"
    assert result.environment.invalid_reason is not None
    assert "triton_version" in result.environment.invalid_reason
    assert derive_environment_probe(pending, result.artifact) == result


@pytest.mark.parametrize("drift", ["route", "expectation"])
def test_environment_probe_rejects_worker_drift_before_remote_call(drift: str) -> None:
    pending = _pending()
    worker = _Worker(pending, _health())
    if drift == "route":
        worker.matrix_worker_binding = worker.matrix_worker_binding.model_copy(
            update={
                "transport": pending.transport.model_copy(update={"host": "wrong.example"})
            }
        )
    else:
        worker.expected_triton_version = "3.8.0"

    with pytest.raises(MatrixEnvironmentEvidenceError, match="worker"):
        run_environment_probe(pending, worker)

    assert worker.calls == []


def test_environment_probe_records_transport_failure_as_recomputable_evidence() -> None:
    pending = _pending()
    worker = _Worker(pending, _health())

    def fail(_device: str) -> dict[str, object]:
        raise WorkerExecutionError(
            "health_check_failed",
            "SSH health probe timed out",
            health={
                "schema_version": "canary-worker-health.v1",
                "status": "check_failed",
                "device": "cuda:0",
                "error": "TimeoutExpired: health probe timed out",
            },
        )

    worker.validate_environment = fail  # type: ignore[method-assign]

    result = run_environment_probe(pending, worker)

    assert result.artifact.probe_error is not None
    assert result.artifact.probe_error.category == "health_check_failed"
    assert result.evidence.status == "fail"
    assert result.environment.status == "invalid"
    assert derive_environment_probe(pending, result.artifact) == result


def test_environment_probe_requires_complete_extended_health() -> None:
    pending = _pending()
    incomplete = _health()
    incomplete.pop("tilelang_version")

    result = run_environment_probe(pending, _Worker(pending, incomplete))

    assert result.artifact.health is None
    assert result.artifact.probe_error is not None
    assert result.artifact.probe_error.category == "invalid_health"
    assert result.evidence.status == "fail"
    assert derive_environment_probe(pending, result.artifact) == result


def test_environment_artifact_cannot_be_replayed_for_another_expected_route() -> None:
    pending = _pending()
    result = run_environment_probe(pending, _Worker(pending, _health()))
    other = pending.model_copy(
        update={"transport": pending.transport.model_copy(update={"port": 30555})}
    )

    with pytest.raises(MatrixEnvironmentEvidenceError, match="expected environment"):
        derive_environment_probe(other, result.artifact)


def test_environment_probe_worker_binds_every_expected_input() -> None:
    pending = _pending()

    worker = build_environment_probe_worker(pending)

    assert worker.matrix_worker_binding == MatrixWorkerBinding(
        worker_revision=pending.worker_revision,
        transport=pending.transport,
    )
    assert worker.expected_hardware_substring == pending.accelerator
    assert worker.expected_compute_capability == (8, 0)
    assert worker.expected_python_version == pending.python_version
    assert worker.expected_torch_version == pending.torch_version
    assert worker.expected_torch_cuda_version == pending.cuda_version
    assert worker.expected_triton_version == pending.triton_version
    assert worker.expected_tilelang_version == pending.tilelang_version
    assert worker.expected_driver_version == pending.driver_version
    assert worker.expected_non_container_worker is True


def test_environment_probe_worker_rejects_inconsistent_isolation_claims() -> None:
    pending = _pending()
    inconsistent = pending.model_copy(
        update={
            "transport": pending.transport.model_copy(update={"network_isolated": True})
        }
    )

    with pytest.raises(MatrixEnvironmentEvidenceError, match="worker route"):
        build_environment_probe_worker(inconsistent)
