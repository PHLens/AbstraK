#!/usr/bin/env python3
"""Freeze, verify, or rehearse the anytime DSL A100 study without live actions."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from abstrak.anytime.freeze import (
    DEFAULT_FREEZE_DIRECTORY,
    DEFAULT_REPOSITORY_ROOT,
    OFFLINE_FREEZE_FILENAME,
    check_anytime_freeze_manifests,
    frozen_request_ceilings,
    load_anytime_offline_freeze,
    write_anytime_freeze_manifests,
)
from abstrak.anytime.freeze_pins import PINNED_OFFLINE_FREEZE_SHA256
from abstrak.anytime.rehearsal import run_anytime_offline_rehearsal
from abstrak.anytime.workloads import KERNELBENCH_REVISION


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    try:
        revision = _git_output(repository, "rev-parse", "HEAD")
        status = _git_output(repository, "status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot inspect repository revision: {error}") from error
    if status:
        raise SystemExit("repository is not clean; refusing clean-revision attestation")
    return revision


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _print_ceilings() -> None:
    for label, trajectories, scientific, operational in frozen_request_ceilings():
        print(f"{label}_planned_trajectories={trajectories}")
        print(f"{label}_scientific_model_call_ceiling={scientific}")
        print(f"{label}_operational_provider_request_ceiling={operational}")
    print("authorization_emitted=false")
    print("formal_authorized=false")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=DEFAULT_REPOSITORY_ROOT,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_FREEZE_DIRECTORY,
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument(
        "--rehearse",
        type=Path,
        metavar="DIRECTORY",
        help="write the full offline synthetic shakeout fixture to a new directory",
    )
    arguments = parser.parse_args()
    if arguments.require_clean and not arguments.check:
        parser.error("--require-clean requires --check")
    if arguments.rehearse is not None and not arguments.check:
        parser.error("--rehearse requires --check against reviewed freeze pins")

    _print_ceilings()
    repository = arguments.repository_root.expanduser().resolve(strict=True)
    output = arguments.output_directory.expanduser().resolve()
    if arguments.check:
        manifest = check_anytime_freeze_manifests(
            output,
            repository_root=repository,
        )
        print("freeze_status=verified")
    else:
        result = write_anytime_freeze_manifests(
            output,
            repository_root=repository,
        )
        manifest = result.manifest
        print("freeze_status=written-unreviewed")
        print(f"formal_raw_sha256={result.formal_raw_sha256}")
        print(f"shakeout_raw_sha256={result.shakeout_raw_sha256}")

    freeze_path = output / OFFLINE_FREEZE_FILENAME
    print(f"offline_freeze_raw_sha256={_raw_sha256(freeze_path)}")
    print(f"offline_freeze_manifest_sha256={manifest.sha256}")
    for dependency in manifest.provider_dependencies:
        print(
            f"provider_dependency_readiness[{dependency.agent_id}]="
            f"{dependency.conformance.study_readiness}"
        )
    print(f"kernelbench_revision={KERNELBENCH_REVISION}")
    print(f"worker_revision_policy={manifest.repository_revision_policy}")
    print("live_worker_revision=pending_m9")
    for blocker in manifest.m9_blockers:
        print(f"m9_blocker={blocker}")

    if arguments.require_clean:
        revision = _require_clean_repository(repository)
        print("repository_clean=true")
        print(f"controller_revision={revision}")
    else:
        print("repository_clean=not-attested")

    if arguments.rehearse is not None:
        pinned = load_anytime_offline_freeze(
            freeze_path,
            expected_sha256=PINNED_OFFLINE_FREEZE_SHA256,
        )
        rehearsal = run_anytime_offline_rehearsal(
            arguments.rehearse,
            pinned_freeze=pinned,
            repository_root=repository,
        )
        print("rehearsal_evidence_scope=offline_synthetic_fixture")
        print(f"rehearsal_directory={rehearsal.directory}")
        print(f"rehearsal_manifest_sha256={rehearsal.manifest.sha256}")
        print(f"rehearsal_attempt_count={rehearsal.receipt.attempt_count}")
        print(
            "rehearsal_scripted_provider_response_count="
            f"{rehearsal.receipt.scripted_provider_response_count}"
        )
        print("rehearsal_live_side_effects=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
