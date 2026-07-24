"""JSON-in/JSON-out CLI for one canary GPU worker job."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from abstrak.canary.contracts import WorkerJob, WorkerResult
from abstrak.canary.evaluator import evaluate_job

JobEvaluator = Callable[..., WorkerResult]
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


def load_job_payload(value: str) -> WorkerJob:
    """Parse one strict canonical worker job from JSON text."""

    return WorkerJob.model_validate_json(value)


def run_worker_job(
    job: WorkerJob,
    *,
    kernelbench_root: str | Path,
    device: str | None = None,
    asset_root: str | Path | None = None,
    evaluator: JobEvaluator = evaluate_job,
) -> WorkerResult:
    if device is not None and device != job.device:
        raise ValueError(f"worker device {device!r} does not match job device {job.device!r}")
    selected_device = job.device
    original_environment = dict(os.environ)
    original_directory = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix=f"abstrak-{job.job_id}-") as workspace:
            for name in tuple(os.environ):
                if any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS):
                    os.environ.pop(name, None)
            os.environ.update(
                {
                    "HOME": workspace,
                    "TMPDIR": workspace,
                    "TORCH_EXTENSIONS_DIR": str(Path(workspace) / "torch-extensions"),
                    "TRITON_CACHE_DIR": str(Path(workspace) / "triton-cache"),
                    "XDG_CACHE_HOME": str(Path(workspace) / "xdg-cache"),
                }
            )
            os.chdir(workspace)
            result = evaluator(
                job,
                kernelbench_root,
                device=selected_device,
                asset_root=asset_root,
            )
    finally:
        os.chdir(original_directory)
        os.environ.clear()
        os.environ.update(original_environment)
    return result.verify_for_job(job)


def _read_worker_revision(worker_root: str | Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(worker_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    revision = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("worker checkout did not report a full lowercase Git revision")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(worker_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    ).stdout
    if status:
        raise RuntimeError("worker checkout must be clean")
    return revision


def _read_driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    versions = tuple(dict.fromkeys(line.strip() for line in completed.stdout.splitlines()))
    if len(versions) != 1 or not versions[0]:
        raise RuntimeError("nvidia-smi did not report one consistent driver version")
    return versions[0]


def _container_markers() -> tuple[str, ...]:
    markers: list[str] = []
    for path in (Path("/.dockerenv"), Path("/run/.containerenv")):
        if path.exists():
            markers.append(str(path))
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8").lower()
    except OSError:
        cgroup = ""
    markers.extend(
        marker
        for marker in ("docker", "containerd", "kubepods", "lxc")
        if marker in cgroup
    )
    return tuple(sorted(set(markers)))


def gpu_health(
    device: str,
    *,
    worker_root: str | Path | None = None,
    extended: bool = False,
) -> dict[str, object]:
    """Run a fresh-process allocation and synchronization probe."""

    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        selected = torch.device(device)
        torch.cuda.set_device(selected)
        value = torch.ones(1, device=selected, dtype=torch.float16) + 1
        torch.cuda.synchronize(selected)
        observed = float(value.item())
        if observed != 2.0:
            raise RuntimeError(f"GPU probe returned {observed}, expected 2.0")
        capability = torch.cuda.get_device_capability(selected)
        result: dict[str, object] = {
            "schema_version": "canary-worker-health.v1",
            "status": "healthy",
            "device": device,
            "hardware": str(torch.cuda.get_device_name(selected)),
            "compute_capability": [int(capability[0]), int(capability[1])],
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "triton_version": importlib.metadata.version("triton"),
            "value": observed,
        }
        if extended:
            try:
                result["tilelang_version"] = importlib.metadata.version("tilelang")
            except importlib.metadata.PackageNotFoundError:
                pass
            try:
                result["driver_version"] = _read_driver_version()
            except (OSError, subprocess.SubprocessError):
                pass
            markers = _container_markers()
            result["container_markers"] = list(markers)
            result["non_container_worker"] = not markers
        if worker_root is not None:
            result["worker_revision"] = _read_worker_revision(worker_root)
        return result
    except Exception as error:
        return {
            "schema_version": "canary-worker-health.v1",
            "status": "unhealthy",
            "device": device,
            "error": f"{type(error).__name__}: {error}",
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default="-", help="worker job JSON path, or - for stdin")
    parser.add_argument("--kernelbench-root")
    parser.add_argument("--asset-root")
    parser.add_argument("--device")
    parser.add_argument("--worker-root")
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument("--extended-health", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.health_check:
        with contextlib.redirect_stdout(sys.stderr):
            device = arguments.device or "cuda:0"
            health_options: dict[str, object] = {}
            if arguments.worker_root is not None:
                health_options["worker_root"] = arguments.worker_root
            if arguments.extended_health:
                health_options["extended"] = True
            result = gpu_health(device, **health_options)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return int(result["status"] != "healthy")
    if not arguments.kernelbench_root:
        print("invalid worker job: --kernelbench-root is required", file=sys.stderr)
        return 2
    try:
        if arguments.job == "-":
            payload = sys.stdin.read()
        else:
            payload = Path(arguments.job).read_text(encoding="utf-8")
        job = load_job_payload(payload)
    except (OSError, ValidationError, ValueError) as error:
        print(f"invalid worker job: {error}", file=sys.stderr)
        return 2
    if arguments.device is not None and arguments.device != job.device:
        print(
            f"invalid worker job: --device {arguments.device!r} does not match "
            f"job.device {job.device!r}",
            file=sys.stderr,
        )
        return 2

    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = run_worker_job(
                job,
                kernelbench_root=arguments.kernelbench_root,
                device=job.device,
                asset_root=arguments.asset_root,
            )
    except Exception as error:
        print(f"worker failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 3
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
