"""Inspect whether a host can act as a controller or GPU experiment worker."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CommandCheck:
    name: str
    available: bool
    detail: str


def check_command(name: str, arguments: Sequence[str]) -> CommandCheck:
    """Run a bounded, read-only tool probe without failing the whole report."""
    executable = shutil.which(arguments[0])
    if executable is None:
        return CommandCheck(name=name, available=False, detail="not found")

    try:
        result = subprocess.run(
            [executable, *arguments[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return CommandCheck(name=name, available=False, detail=str(error))

    output = (result.stdout or result.stderr).strip()
    detail = output.splitlines()[0] if output else f"exit code {result.returncode}"
    return CommandCheck(name=name, available=result.returncode == 0, detail=detail)


def collect_report() -> dict[str, object]:
    checks = [
        check_command("git", ["git", "--version"]),
        check_command("uv", ["uv", "--version"]),
        check_command("docker", ["docker", "--version"]),
        check_command("nvcc", ["nvcc", "--version"]),
        check_command(
            "gpu",
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
        ),
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "checks": [asdict(check) for check in checks],
    }


def _render_text(report: dict[str, object]) -> str:
    lines = [f"python: {report['python']}", f"platform: {report['platform']}"]
    for check in report["checks"]:
        marker = "ok" if check["available"] else "missing"
        lines.append(f"{check['name']}: {marker} ({check['detail']})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="return a non-zero status when nvidia-smi cannot inspect a GPU",
    )
    arguments = parser.parse_args(argv)

    report = collect_report()
    print(json.dumps(report, indent=2, sort_keys=True) if arguments.json else _render_text(report))

    gpu = next(check for check in report["checks"] if check["name"] == "gpu")
    return int(arguments.require_gpu and not gpu["available"])


if __name__ == "__main__":
    sys.exit(main())
