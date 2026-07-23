"""Fresh-process transports for local and SSH canary workers."""

from __future__ import annotations

import json
import math
import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from abstrak.canary.contracts import WorkerJob, WorkerResult

WORKER_MODULE = "abstrak.canary.worker"
_DIAGNOSTIC_LIMIT = 4000
_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "AUTH",
    "BASE_URL",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "SSH_AUTH_SOCK",
    "TOKEN",
)

FailureCategory = Literal[
    "configuration",
    "spawn",
    "timeout",
    "oom",
    "nonzero_exit",
    "invalid_output",
    "result_mismatch",
    "health_check_failed",
    "health_unhealthy",
    "quarantined",
]
SandboxMode = Literal["bubblewrap", "setpriv"]


class WorkerExecutionError(RuntimeError):
    """Raised when a worker transport cannot return a verified terminal result."""

    def __init__(
        self,
        category: FailureCategory,
        message: str,
        *,
        returncode: int | None = None,
        health: Mapping[str, object] | None = None,
        job_scoped: bool = False,
    ) -> None:
        super().__init__(f"{category}: {message}")
        self.category = category
        self.returncode = returncode
        self.health = None if health is None else dict(health)
        self.job_scoped = job_scoped

    def with_health(self, health: Mapping[str, object]) -> WorkerExecutionError:
        self.health = dict(health)
        return self


class ProcessHandle(Protocol):
    pid: int
    returncode: int | None

    def communicate(
        self,
        input: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]: ...


PopenFactory = Callable[..., ProcessHandle]
KillProcessGroup = Callable[[int, int], Any]


@dataclass(frozen=True)
class _ProcessOutput:
    returncode: int | None
    stdout: str
    stderr: str


def _diagnostic(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "worker emitted no diagnostic"
    return stripped[-_DIAGNOSTIC_LIMIT:]


def _worker_environment(
    values: Mapping[str, str],
    *,
    preserve_ssh_auth_sock: bool,
) -> dict[str, str]:
    return {
        name: value
        for name, value in values.items()
        if (preserve_ssh_auth_sock and name == "SSH_AUTH_SOCK")
        or not any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS)
    }


def _parse_result(stdout: str, job: WorkerJob) -> WorkerResult:
    try:
        result = WorkerResult.model_validate_json(stdout)
    except (ValidationError, ValueError) as error:
        raise WorkerExecutionError(
            "invalid_output",
            f"worker stdout is not one WorkerResult JSON value: {error}",
        ) from error
    try:
        return result.verify_for_job(job)
    except ValueError as error:
        raise WorkerExecutionError("result_mismatch", str(error)) from error


def _oom_diagnostic(result: WorkerResult) -> str | None:
    diagnostics = [result.error or ""]
    diagnostics.extend(case.error or "" for case in result.cases)
    diagnostics.extend(result.static_errors)
    markers = (
        "out of memory",
        "cuda_error_out_of_memory",
        "cublas_status_alloc_failed",
    )
    for diagnostic in diagnostics:
        if any(marker in diagnostic.lower() for marker in markers):
            return _diagnostic(diagnostic)
    return None


def _failed_health(device: str, error: str) -> dict[str, object]:
    return {
        "schema_version": "canary-worker-health.v1",
        "status": "check_failed",
        "device": device,
        "error": error,
    }


def _parse_health(stdout: str, expected_device: str) -> dict[str, object]:
    try:
        value = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"health stdout is not one JSON value: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("health result must be one JSON object")
    if value.get("schema_version") != "canary-worker-health.v1":
        raise ValueError("health result has an unknown schema version")
    if value.get("device") != expected_device:
        raise ValueError("health result device does not match the worker job")
    status = value.get("status")
    if status == "healthy":
        hardware = value.get("hardware")
        health_value = value.get("value")
        if not isinstance(hardware, str) or not hardware:
            raise ValueError("healthy result requires hardware")
        capability = value.get("compute_capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
        ):
            raise ValueError("healthy result requires a two-integer compute capability")
        for field in (
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "triton_version",
        ):
            if not isinstance(value.get(field), str) or not value[field]:
                raise ValueError(f"healthy result requires {field}")
        if (
            isinstance(health_value, bool)
            or not isinstance(health_value, int | float)
            or not math.isfinite(health_value)
            or health_value != 2.0
        ):
            raise ValueError("healthy result requires the expected probe value 2.0")
    elif status == "unhealthy":
        if not isinstance(value.get("error"), str) or not value["error"]:
            raise ValueError("unhealthy result requires an error")
    else:
        raise ValueError("health status must be healthy or unhealthy")
    return {str(key): item for key, item in value.items()}


class _SubprocessExecutor:
    def __init__(
        self,
        *,
        timeout_seconds: float,
        health_timeout_seconds: float,
        popen_factory: PopenFactory,
        kill_process_group: KillProcessGroup,
        environment: Mapping[str, str] | None,
        preserve_ssh_auth_sock: bool,
        expected_hardware_substring: str | None,
        expected_compute_capability: tuple[int, int] | None,
        expected_triton_version: str | None,
    ) -> None:
        if timeout_seconds <= 0 or health_timeout_seconds <= 0:
            raise ValueError("worker and health timeouts must be positive")
        self.timeout_seconds = timeout_seconds
        self.health_timeout_seconds = health_timeout_seconds
        self._popen = popen_factory
        self._kill_process_group = kill_process_group
        source_environment = os.environ if environment is None else environment
        self._environment = _worker_environment(
            source_environment,
            preserve_ssh_auth_sock=preserve_ssh_auth_sock,
        )
        self.expected_hardware_substring = expected_hardware_substring
        self.expected_compute_capability = expected_compute_capability
        self.expected_triton_version = expected_triton_version
        self._quarantined = False
        self._quarantine_error: WorkerExecutionError | None = None
        self.last_health: dict[str, object] | None = None

    @property
    def quarantined(self) -> bool:
        return self._quarantined

    def _command(self, job: WorkerJob) -> Sequence[str]:
        raise NotImplementedError

    def _health_command(self, job: WorkerJob) -> Sequence[str]:
        raise NotImplementedError

    def _communicate(
        self,
        command: Sequence[str],
        *,
        payload: str | None,
        timeout_seconds: float,
        spawn_category: FailureCategory,
        timeout_category: FailureCategory,
    ) -> _ProcessOutput:
        try:
            process = self._popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=self._environment,
            )
        except OSError as error:
            raise WorkerExecutionError(
                spawn_category,
                f"cannot start worker process: {type(error).__name__}: {error}",
            ) from error

        try:
            stdout, stderr = process.communicate(input=payload, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            try:
                self._kill_process_group(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.communicate()
            except Exception:
                pass
            raise WorkerExecutionError(
                timeout_category,
                f"worker exceeded {timeout_seconds:g} seconds",
            ) from error
        return _ProcessOutput(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _check_health(self, job: WorkerJob) -> dict[str, object]:
        try:
            output = self._communicate(
                self._health_command(job),
                payload=None,
                timeout_seconds=self.health_timeout_seconds,
                spawn_category="health_check_failed",
                timeout_category="health_check_failed",
            )
            health = _parse_health(output.stdout, job.device)
        except (ValueError, WorkerExecutionError) as error:
            health = _failed_health(job.device, f"{type(error).__name__}: {error}")
            raise WorkerExecutionError(
                "health_check_failed",
                health["error"],
                health=health,
            ) from error
        if output.returncode is None:
            health = _failed_health(job.device, "health worker did not report an exit status")
            raise WorkerExecutionError(
                "health_check_failed",
                health["error"],
                health=health,
            )
        if health["status"] == "healthy" and output.returncode != 0:
            health = _failed_health(
                job.device,
                f"healthy worker exited with status {output.returncode}: "
                f"{_diagnostic(output.stderr)}",
            )
            raise WorkerExecutionError(
                "health_check_failed",
                health["error"],
                returncode=output.returncode,
                health=health,
            )
        return health

    def _quarantine(self, error: WorkerExecutionError) -> None:
        self._quarantined = True
        self._quarantine_error = error
        self.last_health = None if error.health is None else dict(error.health)

    def _require_compatible_hardware(self, health: dict[str, object]) -> None:
        if health["status"] != "healthy":
            return
        errors: list[str] = []
        hardware = str(health["hardware"])
        capability = tuple(health["compute_capability"])
        if (
            self.expected_hardware_substring is not None
            and self.expected_hardware_substring not in hardware
        ):
            errors.append(
                f"expected hardware containing {self.expected_hardware_substring!r}, "
                f"found {hardware!r}"
            )
        if (
            self.expected_compute_capability is not None
            and capability != self.expected_compute_capability
        ):
            errors.append(
                f"expected compute capability {self.expected_compute_capability}, "
                f"found {capability}"
            )
        if (
            self.expected_triton_version is not None
            and health["triton_version"] != self.expected_triton_version
        ):
            errors.append(
                f"expected Triton {self.expected_triton_version!r}, "
                f"found {health['triton_version']!r}"
            )
        if errors:
            health["compatibility_error"] = "; ".join(errors)
            raise WorkerExecutionError(
                "configuration",
                str(health["compatibility_error"]),
                health=health,
            )

    def execute(self, job: WorkerJob) -> WorkerResult:
        if self._quarantined:
            raise WorkerExecutionError(
                "quarantined",
                "worker executor is quarantined after an earlier failure",
                health=self.last_health,
            ) from self._quarantine_error

        primary_error: WorkerExecutionError | None = None
        result: WorkerResult | None = None
        try:
            output = self._communicate(
                self._command(job),
                payload=f"{job.model_dump_json()}\n",
                timeout_seconds=self.timeout_seconds,
                spawn_category="spawn",
                timeout_category="timeout",
            )
            if output.returncode is None:
                raise WorkerExecutionError(
                    "nonzero_exit",
                    "worker did not report an exit status",
                )
            if output.returncode != 0:
                diagnostic = _diagnostic(output.stderr)
                if output.returncode == 124:
                    category: FailureCategory = "timeout"
                elif "out of memory" in diagnostic.lower():
                    category = "oom"
                else:
                    category = "nonzero_exit"
                raise WorkerExecutionError(
                    category,
                    diagnostic,
                    returncode=output.returncode,
                    job_scoped=category in {"timeout", "oom"},
                )
            result = _parse_result(output.stdout, job)
        except WorkerExecutionError as error:
            primary_error = error

        try:
            health = self._check_health(job)
        except WorkerExecutionError as health_error:
            if primary_error is not None:
                primary_error.with_health(health_error.health or {})
                self._quarantine(primary_error)
                raise primary_error from health_error
            self._quarantine(health_error)
            raise

        self.last_health = health
        try:
            self._require_compatible_hardware(health)
        except WorkerExecutionError as error:
            self._quarantine(error)
            raise
        if health["status"] != "healthy":
            if primary_error is None:
                primary_error = WorkerExecutionError(
                    "health_unhealthy",
                    str(health.get("error", "GPU health probe failed")),
                    health=health,
                )
            else:
                primary_error.with_health(health)
            self._quarantine(primary_error)
            raise primary_error
        if primary_error is not None:
            primary_error.with_health(health)
            recoverable_job_failure = primary_error.job_scoped and primary_error.category in {
                "timeout",
                "oom",
            }
            if not recoverable_job_failure:
                self._quarantine(primary_error)
            raise primary_error

        if result is None:
            error = WorkerExecutionError(
                "invalid_output",
                "worker produced no result",
                health=health,
            )
            self._quarantine(error)
            raise error
        oom = _oom_diagnostic(result)
        if oom is not None:
            error = WorkerExecutionError("oom", oom, health=health, job_scoped=True)
            raise error
        bound_result = result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "post_job_gpu_health": health,
                }
            }
        )
        return bound_result.verify_for_job(job)


class LocalWorkerExecutor(_SubprocessExecutor):
    """Execute every job in a new local worker process."""

    def __init__(
        self,
        kernelbench_root: str | Path,
        *,
        asset_root: str | Path | None = None,
        python_executable: str | Path = sys.executable,
        timeout_seconds: float = 300.0,
        health_timeout_seconds: float = 30.0,
        popen_factory: PopenFactory = subprocess.Popen,
        kill_process_group: KillProcessGroup = os.killpg,
        environment: Mapping[str, str] | None = None,
        expected_hardware_substring: str | None = None,
        expected_compute_capability: tuple[int, int] | None = None,
        expected_triton_version: str | None = None,
    ) -> None:
        super().__init__(
            timeout_seconds=timeout_seconds,
            health_timeout_seconds=health_timeout_seconds,
            popen_factory=popen_factory,
            kill_process_group=kill_process_group,
            environment=environment,
            preserve_ssh_auth_sock=False,
            expected_hardware_substring=expected_hardware_substring,
            expected_compute_capability=expected_compute_capability,
            expected_triton_version=expected_triton_version,
        )
        self.kernelbench_root = str(kernelbench_root)
        self.asset_root = None if asset_root is None else str(asset_root)
        self.python_executable = str(python_executable)

    def _command(self, job: WorkerJob) -> Sequence[str]:
        command = [
            self.python_executable,
            "-m",
            WORKER_MODULE,
            "--job",
            "-",
            "--kernelbench-root",
            self.kernelbench_root,
        ]
        if self.asset_root is not None:
            command.extend(("--asset-root", self.asset_root))
        command.extend(("--device", job.device))
        return command

    def _health_command(self, job: WorkerJob) -> Sequence[str]:
        return [
            self.python_executable,
            "-m",
            WORKER_MODULE,
            "--health-check",
            "--device",
            job.device,
        ]


class SshWorkerExecutor(_SubprocessExecutor):
    """Execute every job through a non-interactive fresh SSH process."""

    def __init__(
        self,
        host: str,
        *,
        python_executable: str,
        pythonpath: str,
        kernelbench_root: str,
        asset_root: str,
        device: str = "cuda:0",
        ssh_executable: str = "ssh",
        remote_timeout_executable: str = "timeout",
        sandbox_executable: str = "bwrap",
        sandbox_mode: SandboxMode = "bubblewrap",
        setpriv_executable: str = "setpriv",
        sandbox_user: str = "nobody",
        sandbox_group: str = "nogroup",
        timeout_seconds: float = 300.0,
        health_timeout_seconds: float = 30.0,
        popen_factory: PopenFactory = subprocess.Popen,
        kill_process_group: KillProcessGroup = os.killpg,
        environment: Mapping[str, str] | None = None,
        expected_hardware_substring: str | None = None,
        expected_compute_capability: tuple[int, int] | None = None,
        expected_triton_version: str | None = None,
    ) -> None:
        super().__init__(
            timeout_seconds=timeout_seconds,
            health_timeout_seconds=health_timeout_seconds,
            popen_factory=popen_factory,
            kill_process_group=kill_process_group,
            environment=environment,
            preserve_ssh_auth_sock=True,
            expected_hardware_substring=expected_hardware_substring,
            expected_compute_capability=expected_compute_capability,
            expected_triton_version=expected_triton_version,
        )
        if not host or host.startswith("-") or any(character.isspace() for character in host):
            raise ValueError("host must be one non-option SSH destination")
        for name, value in {
            "python_executable": python_executable,
            "pythonpath": pythonpath,
            "kernelbench_root": kernelbench_root,
            "asset_root": asset_root,
            "device": device,
            "ssh_executable": ssh_executable,
            "remote_timeout_executable": remote_timeout_executable,
            "sandbox_executable": sandbox_executable,
            "setpriv_executable": setpriv_executable,
            "sandbox_user": sandbox_user,
            "sandbox_group": sandbox_group,
        }.items():
            if not value:
                raise ValueError(f"{name} cannot be empty")
        if sandbox_mode not in {"bubblewrap", "setpriv"}:
            raise ValueError("sandbox_mode must be bubblewrap or setpriv")
        for name, value in {
            "python_executable": python_executable,
            "pythonpath": pythonpath,
            "kernelbench_root": kernelbench_root,
            "asset_root": asset_root,
        }.items():
            path = PurePosixPath(value)
            if not path.is_absolute():
                raise ValueError(f"{name} must be one absolute remote path")
            if name != "python_executable" and any(
                path.is_relative_to(hidden)
                for hidden in (
                    PurePosixPath("/tmp"),
                    PurePosixPath("/home"),
                    PurePosixPath("/root"),
                )
            ):
                raise ValueError(f"{name} cannot be under a sandbox-hidden directory")
        self.host = host
        self.python_executable = python_executable
        self.pythonpath = pythonpath
        self.kernelbench_root = kernelbench_root
        self.asset_root = asset_root
        self.device = device
        self.ssh_executable = ssh_executable
        self.remote_timeout_executable = remote_timeout_executable
        self.sandbox_executable = sandbox_executable
        self.sandbox_mode = sandbox_mode
        self.setpriv_executable = setpriv_executable
        self.sandbox_user = sandbox_user
        self.sandbox_group = sandbox_group

    def _worker_environment_command(self, arguments: Sequence[str]) -> list[str]:
        python_bin = str(PurePosixPath(self.python_executable).parent)
        sandbox_path = ":".join(
            (
                python_bin,
                "/usr/local/cuda/bin",
                "/usr/local/sbin",
                "/usr/local/bin",
                "/usr/sbin",
                "/usr/bin",
                "/sbin",
                "/bin",
            )
        )
        return [
            "env",
            "--chdir=/tmp",
            "HOME=/tmp",
            "TMPDIR=/tmp",
            f"PATH={sandbox_path}",
            f"PYTHONPATH={self.pythonpath}",
            "PYTHONNOUSERSITE=1",
            "PYTHONDONTWRITEBYTECODE=1",
            self.python_executable,
            "-m",
            WORKER_MODULE,
            *arguments,
        ]

    def _setpriv_worker_command(self, arguments: Sequence[str]) -> list[str]:
        return [
            self.setpriv_executable,
            f"--reuid={self.sandbox_user}",
            f"--regid={self.sandbox_group}",
            "--clear-groups",
            "--no-new-privs",
            "--reset-env",
            *self._worker_environment_command(arguments),
        ]

    def _sandboxed_worker_command(self, arguments: Sequence[str]) -> list[str]:
        if self.sandbox_mode == "setpriv":
            return self._setpriv_worker_command(arguments)
        python_path = PurePosixPath(self.python_executable)
        python_bin = str(python_path.parent)
        sandbox_path = ":".join(
            (
                python_bin,
                "/usr/local/cuda/bin",
                "/usr/local/sbin",
                "/usr/local/bin",
                "/usr/sbin",
                "/usr/bin",
                "/sbin",
                "/bin",
            )
        )
        command = [
            self.sandbox_executable,
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
            sandbox_path,
            "--setenv",
            "PYTHONPATH",
            self.pythonpath,
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--chdir",
            "/tmp",
        ]
        if python_path.is_relative_to(PurePosixPath("/tmp")):
            tmp_venv_layout = (
                python_path.parent.name == "bin"
                and python_path.parent.parent.parent == PurePosixPath("/tmp")
            )
            if not tmp_venv_layout:
                raise WorkerExecutionError(
                    "configuration",
                    "a Python under /tmp must use /tmp/<venv>/bin/<python>",
                )
            venv_root = str(python_path.parent.parent)
            command.extend(("--ro-bind", venv_root, venv_root))
        command.extend(("--", self.python_executable, "-m", WORKER_MODULE, *arguments))
        return command

    @staticmethod
    def _remote_timeout(controller_timeout_seconds: float) -> float:
        margin = min(5.0, controller_timeout_seconds / 5.0)
        return controller_timeout_seconds - margin

    def _remote_command(
        self,
        arguments: Sequence[str],
        *,
        controller_timeout_seconds: float,
        sandbox: bool,
    ) -> Sequence[str]:
        remote_timeout = self._remote_timeout(controller_timeout_seconds)
        worker_command = (
            self._sandboxed_worker_command(arguments)
            if sandbox
            else [self.python_executable, "-m", WORKER_MODULE, *arguments]
        )
        remote_arguments = [
            "env",
            f"PYTHONPATH={self.pythonpath}",
            self.remote_timeout_executable,
            "--signal=TERM",
            "--kill-after=2s",
            f"{remote_timeout:g}s",
            *worker_command,
        ]
        return [
            self.ssh_executable,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ForwardX11=no",
            self.host,
            shlex.join(remote_arguments),
        ]

    def _require_device(self, job: WorkerJob) -> None:
        if job.device != self.device:
            raise WorkerExecutionError(
                "configuration",
                f"job device {job.device!r} does not match SSH worker device {self.device!r}",
            )

    def _command(self, job: WorkerJob) -> Sequence[str]:
        self._require_device(job)
        return self._remote_command(
            [
                "--job",
                "-",
                "--kernelbench-root",
                self.kernelbench_root,
                "--asset-root",
                self.asset_root,
                "--device",
                self.device,
            ],
            controller_timeout_seconds=self.timeout_seconds,
            sandbox=True,
        )

    def _health_command(self, job: WorkerJob) -> Sequence[str]:
        self._require_device(job)
        return self._remote_command(
            [
                "--health-check",
                "--device",
                self.device,
            ],
            controller_timeout_seconds=self.health_timeout_seconds,
            sandbox=False,
        )
