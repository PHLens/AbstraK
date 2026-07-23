from __future__ import annotations

import hashlib
import json
import shlex
import signal
import subprocess
from typing import Any

import pytest

from abstrak.canary.contracts import WorkerJob, WorkerResult
from abstrak.canary.remote import (
    WORKER_MODULE,
    LocalWorkerExecutor,
    SshWorkerExecutor,
    WorkerExecutionError,
)
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import get_task_pack


def _job(*, job_id: str = "remote-job") -> WorkerJob:
    source = "class ModelNew: pass\n"
    task = get_task_pack("row-reduction-scale")
    return WorkerJob(
        job_id=job_id,
        kind="dev",
        task=task,
        target=get_target_stack("triton-a100"),
        case_ids=tuple(case.id for case in task.dev_cases),
        candidate_source=source,
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def _result(job: WorkerJob) -> WorkerResult:
    return WorkerResult(
        job_id=job.job_id,
        job_sha256=job.sha256,
        input_sha256=job.input_sha256,
        candidate_sha256=job.candidate_sha256,
        status="environment_error",
        error="fake GPU environment",
    )


def _oom_result(job: WorkerJob) -> WorkerResult:
    return WorkerResult(
        job_id=job.job_id,
        job_sha256=job.sha256,
        input_sha256=job.input_sha256,
        candidate_sha256=job.candidate_sha256,
        status="runtime_error",
        compiled=True,
        error="RuntimeError: CUDA out of memory",
    )


def _health(
    *,
    status: str = "healthy",
    hardware: str = "Fake A100",
    capability: tuple[int, int] = (8, 0),
    triton_version: str = "3.7.1",
) -> str:
    if status == "healthy":
        payload = {
            "schema_version": "canary-worker-health.v1",
            "status": "healthy",
            "device": "cuda:0",
            "hardware": hardware,
            "compute_capability": list(capability),
            "python_version": "3.10.20",
            "torch_version": "2.13.0+cu126",
            "torch_cuda_version": "12.6",
            "triton_version": triton_version,
            "value": 2.0,
        }
    else:
        payload = {
            "schema_version": "canary-worker-health.v1",
            "status": "unhealthy",
            "device": "cuda:0",
            "error": "GPU is unhealthy",
        }
    return json.dumps(payload)


class FakeProcess:
    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        times_out: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.times_out = times_out
        self.pid = 4321
        self.communications: list[tuple[str | None, float | None]] = []

    def communicate(
        self,
        input: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        self.communications.append((input, timeout))
        if self.times_out and len(self.communications) == 1:
            raise subprocess.TimeoutExpired("worker", timeout)
        return self.stdout, self.stderr


class PopenRecorder:
    def __init__(self, *processes: FakeProcess) -> None:
        self.processes = list(processes)
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> FakeProcess:
        self.calls.append((command, kwargs))
        if not self.processes:
            raise AssertionError("unexpected worker process")
        return self.processes.pop(0)


def test_local_executor_uses_fresh_session_stdin_and_verifies_result(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "must-not-reach-worker")
    monkeypatch.setenv("TEST_BASE_URL", "https://secret.example/token")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/controller-agent.sock")
    job = _job()
    process = FakeProcess(stdout=_result(job).model_dump_json())
    health = FakeProcess(stdout=_health())
    popen = PopenRecorder(process, health)
    executor = LocalWorkerExecutor(
        "/worker/KernelBench",
        asset_root="/worker/assets",
        python_executable="/worker/python",
        popen_factory=popen,
    )

    result = executor.execute(job)

    assert result.metadata["post_job_gpu_health"]["status"] == "healthy"
    command, options = popen.calls[0]
    assert command == [
        "/worker/python",
        "-m",
        WORKER_MODULE,
        "--job",
        "-",
        "--kernelbench-root",
        "/worker/KernelBench",
        "--asset-root",
        "/worker/assets",
        "--device",
        "cuda:0",
    ]
    assert options["start_new_session"] is True
    assert "TEST_API_KEY" not in options["env"]
    assert "TEST_BASE_URL" not in options["env"]
    assert "SSH_AUTH_SOCK" not in options["env"]
    assert json.loads(process.communications[0][0] or "") == job.model_dump(mode="json")
    health_command, health_options = popen.calls[1]
    assert health_command == [
        "/worker/python",
        "-m",
        WORKER_MODULE,
        "--health-check",
        "--device",
        "cuda:0",
    ]
    assert health_options["start_new_session"] is True
    assert health.communications == [(None, 30.0)]


@pytest.mark.parametrize(
    "stdout",
    (
        "not-json",
        "{}\n{}",
    ),
)
def test_executor_rejects_non_single_or_invalid_result_json(stdout: str) -> None:
    executor = LocalWorkerExecutor(
        "/worker/KernelBench",
        popen_factory=PopenRecorder(FakeProcess(stdout=stdout), FakeProcess(stdout=_health())),
    )

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(_job())

    assert captured.value.category == "invalid_output"


def test_executor_rejects_result_bound_to_another_job() -> None:
    job = _job()
    other = _job(job_id="other-job")
    executor = LocalWorkerExecutor(
        "/worker/KernelBench",
        popen_factory=PopenRecorder(
            FakeProcess(stdout=_result(other).model_dump_json()),
            FakeProcess(stdout=_health()),
        ),
    )

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(job)

    assert captured.value.category == "result_mismatch"


def test_nonzero_worker_exit_raises_transport_error() -> None:
    process = FakeProcess(stderr="worker failure: compiler crashed", returncode=3)
    popen = PopenRecorder(process, FakeProcess(stdout=_health()))
    executor = LocalWorkerExecutor(
        "/worker/KernelBench",
        popen_factory=popen,
    )

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(_job())

    assert captured.value.category == "nonzero_exit"
    assert captured.value.returncode == 3
    assert "compiler crashed" in str(captured.value)
    assert captured.value.health is not None
    assert captured.value.health["status"] == "healthy"
    assert executor.quarantined

    with pytest.raises(WorkerExecutionError) as quarantined:
        executor.execute(_job(job_id="later-job"))

    assert quarantined.value.category == "quarantined"
    assert len(popen.calls) == 2


def test_remote_timeout_exit_is_classified_as_timeout() -> None:
    executor = LocalWorkerExecutor(
        "/worker/KernelBench",
        popen_factory=PopenRecorder(
            FakeProcess(returncode=124),
            FakeProcess(stdout=_health()),
        ),
    )

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(_job())

    assert captured.value.category == "timeout"
    assert captured.value.job_scoped
    assert captured.value.health is not None
    assert captured.value.health["status"] == "healthy"


def test_healthy_probe_allows_a_fresh_job_after_candidate_timeout() -> None:
    first = _job(job_id="timed-out-job")
    second = _job(job_id="next-job")
    popen = PopenRecorder(
        FakeProcess(returncode=124),
        FakeProcess(stdout=_health()),
        FakeProcess(stdout=_result(second).model_dump_json()),
        FakeProcess(stdout=_health()),
    )
    executor = LocalWorkerExecutor("/worker/KernelBench", popen_factory=popen)

    with pytest.raises(WorkerExecutionError, match="timeout"):
        executor.execute(first)
    result = executor.execute(second)

    assert result.status == "environment_error"
    assert not executor.quarantined
    assert len(popen.calls) == 4


def test_oom_exit_runs_health_check_and_preserves_it_on_error() -> None:
    popen = PopenRecorder(
        FakeProcess(stderr="CUDA out of memory", returncode=3),
        FakeProcess(stdout=_health()),
    )
    executor = LocalWorkerExecutor("/worker/KernelBench", popen_factory=popen)

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(_job())

    assert captured.value.category == "oom"
    assert captured.value.job_scoped
    assert captured.value.health is not None
    assert captured.value.health["status"] == "healthy"
    assert not executor.quarantined
    assert len(popen.calls) == 2


def test_structured_oom_result_is_recoverable_after_health_check() -> None:
    job = _job()
    popen = PopenRecorder(
        FakeProcess(stdout=_oom_result(job).model_dump_json()),
        FakeProcess(stdout=_health()),
    )
    executor = LocalWorkerExecutor("/worker/KernelBench", popen_factory=popen)

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(job)

    assert captured.value.category == "oom"
    assert captured.value.job_scoped
    assert captured.value.health is not None
    assert captured.value.health["status"] == "healthy"
    assert not executor.quarantined


def test_executor_rejects_non_a100_hardware_when_required() -> None:
    job = _job()
    popen = PopenRecorder(
        FakeProcess(stdout=_result(job).model_dump_json()),
        FakeProcess(stdout=_health(hardware="Tesla V100", capability=(7, 0))),
    )
    executor = LocalWorkerExecutor(
        "/worker/KernelBench",
        popen_factory=popen,
        expected_hardware_substring="A100",
        expected_compute_capability=(8, 0),
    )

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(job)

    assert captured.value.category == "configuration"
    assert "Tesla V100" in str(captured.value)
    assert "(7, 0)" in str(captured.value)
    assert executor.quarantined


def test_executor_rejects_target_version_drift() -> None:
    job = _job()
    popen = PopenRecorder(
        FakeProcess(stdout=_result(job).model_dump_json()),
        FakeProcess(stdout=_health(triton_version="3.6.0")),
    )
    executor = LocalWorkerExecutor(
        "/worker/KernelBench",
        popen_factory=popen,
        expected_triton_version="3.7.1",
    )

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(job)

    assert captured.value.category == "configuration"
    assert "expected Triton '3.7.1'" in str(captured.value)
    assert "found '3.6.0'" in str(captured.value)


def test_timeout_kills_and_reaps_the_fresh_process_group() -> None:
    process = FakeProcess(times_out=True)
    health = FakeProcess(stdout=_health())
    killed: list[tuple[int, int]] = []
    executor = LocalWorkerExecutor(
        "/worker/KernelBench",
        timeout_seconds=12.5,
        popen_factory=PopenRecorder(process, health),
        kill_process_group=lambda pid, sig: killed.append((pid, sig)),
    )

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(_job())

    assert captured.value.category == "timeout"
    assert not captured.value.job_scoped
    assert killed == [(process.pid, signal.SIGKILL)]
    assert len(process.communications) == 2
    payload, timeout = process.communications[0]
    assert json.loads(payload or "") == _job().model_dump(mode="json")
    assert timeout == 12.5
    assert process.communications[1] == (None, None)
    assert health.communications == [(None, 30.0)]
    assert captured.value.health is not None
    assert captured.value.health["status"] == "healthy"
    assert executor.quarantined


def test_unhealthy_post_job_probe_fails_closed_and_quarantines() -> None:
    job = _job()
    popen = PopenRecorder(
        FakeProcess(stdout=_result(job).model_dump_json()),
        FakeProcess(stdout=_health(status="unhealthy"), returncode=1),
    )
    executor = LocalWorkerExecutor("/worker/KernelBench", popen_factory=popen)

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(job)

    assert captured.value.category == "health_unhealthy"
    assert captured.value.health is not None
    assert captured.value.health["status"] == "unhealthy"
    assert executor.quarantined


def test_health_transport_failure_is_structured_and_fails_closed() -> None:
    job = _job()
    popen = PopenRecorder(
        FakeProcess(stdout=_result(job).model_dump_json()),
        FakeProcess(stderr="ssh connection lost", returncode=255),
    )
    executor = LocalWorkerExecutor("/worker/KernelBench", popen_factory=popen)

    with pytest.raises(WorkerExecutionError) as captured:
        executor.execute(job)

    assert captured.value.category == "health_check_failed"
    assert captured.value.health is not None
    assert captured.value.health["status"] == "check_failed"
    assert executor.quarantined


def test_ssh_executor_quotes_each_remote_argument_and_enables_batch_mode(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "must-not-reach-ssh")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/controller-agent.sock")
    job = _job()
    process = FakeProcess(stdout=_result(job).model_dump_json())
    popen = PopenRecorder(process, FakeProcess(stdout=_health()))
    executor = SshWorkerExecutor(
        "gpu.example",
        python_executable="/opt/python 3/bin/python",
        pythonpath="/srv/AbstraK's src",
        kernelbench_root="/srv/Kernel Bench",
        asset_root="/srv/R1 assets",
        popen_factory=popen,
    )

    executor.execute(job)

    command, options = popen.calls[0]
    expected_remote = shlex.join(
        [
            "env",
            "PYTHONPATH=/srv/AbstraK's src",
            "timeout",
            "--signal=TERM",
            "--kill-after=2s",
            "295s",
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--ro-bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/home",
            "--tmpfs",
            "/root",
            "--tmpfs",
            "/run",
            "--tmpfs",
            "/dev/shm",
            "--clearenv",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PATH",
            (
                "/opt/python 3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "--setenv",
            "PYTHONPATH",
            "/srv/AbstraK's src",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--chdir",
            "/tmp",
            "--",
            "/opt/python 3/bin/python",
            "-m",
            WORKER_MODULE,
            "--job",
            "-",
            "--kernelbench-root",
            "/srv/Kernel Bench",
            "--asset-root",
            "/srv/R1 assets",
            "--device",
            "cuda:0",
        ]
    )
    assert command == [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ForwardX11=no",
        "gpu.example",
        expected_remote,
    ]
    assert options["start_new_session"] is True
    assert options["env"]["SSH_AUTH_SOCK"] == "/tmp/controller-agent.sock"
    assert "TEST_API_KEY" not in options["env"]
    health_command, _ = popen.calls[1]
    expected_health = shlex.join(
        [
            "env",
            "PYTHONPATH=/srv/AbstraK's src",
            "timeout",
            "--signal=TERM",
            "--kill-after=2s",
            "25s",
            "/opt/python 3/bin/python",
            "-m",
            WORKER_MODULE,
            "--health-check",
            "--device",
            "cuda:0",
        ]
    )
    assert health_command == [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ForwardX11=no",
        "gpu.example",
        expected_health,
    ]


def test_ssh_sandbox_rebinds_tmp_venv_after_tmpfs() -> None:
    executor = SshWorkerExecutor(
        "gpu.example",
        python_executable="/tmp/abstrak-gpu-venv/bin/python",
        pythonpath="/srv/AbstraK/src",
        kernelbench_root="/srv/KernelBench",
        asset_root="/srv/AbstraK/benchmarks/r1-a100",
    )

    remote = shlex.split(executor._command(_job())[-1])
    tmpfs_index = remote.index("/tmp", remote.index("--tmpfs"))
    bind_index = remote.index("/tmp/abstrak-gpu-venv")

    assert bind_index > tmpfs_index
    assert remote[bind_index - 1] == "--ro-bind"
    assert remote[bind_index : bind_index + 2] == [
        "/tmp/abstrak-gpu-venv",
        "/tmp/abstrak-gpu-venv",
    ]


def test_supervised_ssh_mode_drops_privileges_and_clears_environment() -> None:
    executor = SshWorkerExecutor(
        "gpu.example",
        python_executable="/tmp/abstrak-gpu-venv/bin/python",
        pythonpath="/srv/AbstraK/src",
        kernelbench_root="/srv/KernelBench",
        asset_root="/srv/AbstraK/benchmarks/r1-a100",
        sandbox_mode="setpriv",
    )

    remote = shlex.split(executor._command(_job())[-1])

    assert "bwrap" not in remote
    assert remote[remote.index("setpriv") : remote.index("setpriv") + 6] == [
        "setpriv",
        "--reuid=nobody",
        "--regid=nogroup",
        "--clear-groups",
        "--no-new-privs",
        "--reset-env",
    ]
    assert "--chdir=/tmp" in remote
    assert "PYTHONPATH=/srv/AbstraK/src" in remote


def test_ssh_executor_rejects_paths_hidden_by_sandbox() -> None:
    with pytest.raises(ValueError, match="pythonpath cannot be under"):
        SshWorkerExecutor(
            "gpu.example",
            python_executable="/usr/bin/python3",
            pythonpath="/tmp/AbstraK/src",
            kernelbench_root="/srv/KernelBench",
            asset_root="/srv/AbstraK/benchmarks/r1-a100",
        )
