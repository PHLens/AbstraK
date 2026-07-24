"""Derive hash-bound task floors from sealed oracle and baseline gates."""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from abstrak.canary.artifacts import TrajectoryArtifactError, verify_trajectory
from abstrak.canary.contracts import TargetStackSpec, WorkerJob, WorkerResult
from abstrak.canary.gates import GateRecord
from abstrak.canary.matrix_preflight import (
    FORMAL_FLOOR_TIMING,
    AssetManifest,
    BaselineTimingEvidence,
    ExpertCorrectnessEvidence,
    LatencyCeilingDerivation,
    TargetAssetBinding,
    TargetCodegenEvidence,
    TaskAssetBinding,
    TaskFloorRecord,
    VerifiedTaskFloorEvidence,
)
from abstrak.providers.contracts import sha256_json

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INFRASTRUCTURE_FAILURES = {"environment_error", "timeout", "worker_error"}
_METRIC_REL_TOL = 1e-12
_METRIC_ABS_TOL = 1e-12


class MatrixFloorEvidenceError(ValueError):
    """Raised when sealed gates cannot prove a complete task floor."""


def gate_artifact_sha256(record: GateRecord) -> str:
    """Verify one sealed gate and hash its checksum-manifest bytes."""

    directory = Path(record.artifact_directory).expanduser()
    try:
        verify_trajectory(directory)
        persisted = GateRecord.model_validate_json(
            (directory / "gate-record.json").read_text(encoding="utf-8")
        )
        declared_directory = Path(persisted.artifact_directory).expanduser().resolve(strict=True)
        actual_directory = directory.resolve(strict=True)
        if declared_directory != actual_directory:
            raise MatrixFloorEvidenceError(
                "sealed gate artifact_directory does not identify its containing directory"
            )
        if persisted != record:
            raise MatrixFloorEvidenceError("supplied gate record differs from its sealed artifact")
        checksum_bytes = (directory / "sha256sums.txt").read_bytes()
    except MatrixFloorEvidenceError:
        raise
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise MatrixFloorEvidenceError(
            f"sealed gate artifact is invalid: {record.artifact_directory}: {error}"
        ) from error
    return hashlib.sha256(checksum_bytes).hexdigest()


def _validate_targets(
    assets: AssetManifest,
    targets: Iterable[TargetStackSpec],
) -> tuple[TargetStackSpec, ...]:
    target_values = tuple(targets)
    if not target_values:
        raise MatrixFloorEvidenceError("floor derivation requires at least one target")
    target_ids = tuple(target.id for target in target_values)
    if len(target_ids) != len(set(target_ids)):
        raise MatrixFloorEvidenceError("floor targets must have unique IDs")
    expected_ids = tuple(target.target_id for target in assets.targets)
    if target_ids != expected_ids:
        raise MatrixFloorEvidenceError("floor targets do not exactly match the asset manifest")
    for target, binding in zip(target_values, assets.targets, strict=True):
        if (
            sha256_json(target) != binding.target_stack_sha256
            or target.card_sha256 != binding.card_sha256
        ):
            raise MatrixFloorEvidenceError(
                f"target stack differs from the asset manifest: {target.id}"
            )
    return target_values


def _validate_job(
    job: WorkerJob,
    *,
    record: GateRecord,
    task: TaskAssetBinding,
    target: TargetAssetBinding,
) -> None:
    expected_kind = record.kind
    expected_cases = tuple(case.id for case in job.task.sealed_cases)
    if (
        job.kind != expected_kind
        or job.task.id != task.task_id
        or sha256_json(job.task) != task.task_pack_sha256
        or job.target.id != target.target_id
        or sha256_json(job.target) != target.target_stack_sha256
        or job.device != record.summary.device
        or job.candidate_sha256 != record.source_sha256
        or tuple(job.case_ids) != expected_cases
    ):
        raise MatrixFloorEvidenceError(
            f"gate worker job differs from frozen inputs: {record.task_id}/{record.target_id}"
        )


def _validate_summary(
    record: GateRecord,
    *,
    task: TaskAssetBinding,
    target: TargetAssetBinding,
) -> None:
    summary = record.summary
    if summary.timing != FORMAL_FLOOR_TIMING:
        raise MatrixFloorEvidenceError(
            f"floor gate does not use FORMAL_FLOOR_TIMING: "
            f"{record.task_id}/{record.target_id}"
        )
    if (
        summary.task_id != record.task_id
        or summary.target_id != record.target_id
        or summary.candidate_sha256 != record.source_sha256
        or summary.job_kind != record.kind
    ):
        raise MatrixFloorEvidenceError(
            f"gate summary differs from its record: {record.task_id}/{record.target_id}"
        )
    for attempt in summary.attempts:
        for job in attempt.jobs:
            _validate_job(job, record=record, task=task, target=target)
        for job, result in zip(attempt.jobs, attempt.results, strict=False):
            try:
                result.verify_for_job(job)
            except ValueError as error:
                raise MatrixFloorEvidenceError(
                    f"gate worker result is not linked to its job: {record.task_id}/"
                    f"{record.target_id}: {error}"
                ) from error
    _validate_timing_protocol(record)


def _metrics_match(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=_METRIC_REL_TOL,
        abs_tol=_METRIC_ABS_TOL,
    )


def _require_matching_metrics(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    *,
    label: str,
    record: GateRecord,
) -> None:
    if len(actual) != len(expected) or any(
        not _metrics_match(actual_value, expected_value)
        for actual_value, expected_value in zip(actual, expected, strict=True)
    ):
        raise MatrixFloorEvidenceError(
            f"{label} differs from raw timing: {record.task_id}/{record.target_id}"
        )


def _require_passing_timed_result(
    record: GateRecord,
    job: WorkerJob,
    result: WorkerResult,
) -> tuple[float, float]:
    expected_case_ids = tuple(job.case_ids)
    actual_case_ids = tuple(case.case_id for case in result.cases)
    if (
        result.status != "completed"
        or not result.compiled
        or not result.correct
        or result.static_errors
        or len(actual_case_ids) != len(expected_case_ids)
        or set(actual_case_ids) != set(expected_case_ids)
        or any(
            case.status != "pass"
            or not case.correct
            or not case.output_finite
            or not case.inputs_unchanged
            for case in result.cases
        )
        or not result.timing_ms
        or result.timing_cv is None
    ):
        raise MatrixFloorEvidenceError(
            f"timing result is not completed, correct, and case-complete: "
            f"{record.task_id}/{record.target_id}/{job.job_id}"
        )
    process_median = float(statistics.median(result.timing_ms))
    process_mean = statistics.fmean(result.timing_ms)
    process_cv = statistics.pstdev(result.timing_ms) / process_mean
    if not _metrics_match(result.timing_cv, process_cv):
        raise MatrixFloorEvidenceError(
            f"worker timing_cv differs from raw timing: "
            f"{record.task_id}/{record.target_id}/{job.job_id}"
        )
    return process_median, process_cv


def _validate_complete_attempt(record: GateRecord, attempt_index: int) -> None:
    summary = record.summary
    attempt = summary.attempts[attempt_index]
    if (
        len(attempt.jobs) != summary.timing.repetitions
        or len(attempt.results) != summary.timing.repetitions
    ):
        raise MatrixFloorEvidenceError(
            f"complete timing attempt must contain exactly {summary.timing.repetitions} "
            f"processes: {record.task_id}/{record.target_id}"
        )
    metrics = tuple(
        _require_passing_timed_result(record, job, result)
        for job, result in zip(attempt.jobs, attempt.results, strict=True)
    )
    process_medians = tuple(item[0] for item in metrics)
    process_cvs = tuple(item[1] for item in metrics)
    _require_matching_metrics(
        attempt.process_medians_ms,
        process_medians,
        label="attempt process medians",
        record=record,
    )
    _require_matching_metrics(
        attempt.process_cvs,
        process_cvs,
        label="attempt process CVs",
        record=record,
    )
    across_mean = statistics.fmean(process_medians)
    across_process_cv = statistics.pstdev(process_medians) / across_mean
    median_ms = float(statistics.median(process_medians))
    if (
        attempt.across_process_cv is None
        or not _metrics_match(attempt.across_process_cv, across_process_cv)
        or attempt.median_ms is None
        or not _metrics_match(attempt.median_ms, median_ms)
    ):
        raise MatrixFloorEvidenceError(
            f"attempt aggregate differs from raw timing: {record.task_id}/{record.target_id}"
        )
    stable = (
        all(process_cv <= summary.timing.max_cv for process_cv in process_cvs)
        and across_process_cv <= summary.timing.max_cv
    )
    expected_status = "stable" if stable else "unstable"
    if attempt.status != expected_status or attempt.stable != stable or attempt.error is not None:
        raise MatrixFloorEvidenceError(
            f"attempt stability decision differs from raw timing: "
            f"{record.task_id}/{record.target_id}"
        )


def _validate_failed_attempt(record: GateRecord, attempt_index: int) -> None:
    summary = record.summary
    attempt = summary.attempts[attempt_index]
    job_count = len(attempt.jobs)
    result_count = len(attempt.results)
    if job_count > summary.timing.repetitions or result_count not in {
        job_count,
        job_count - 1,
    }:
        raise MatrixFloorEvidenceError(
            f"failed timing attempt has an invalid partial process set: "
            f"{record.task_id}/{record.target_id}"
        )

    terminal_result: WorkerResult | None = None
    successful_count = result_count
    if result_count == job_count:
        terminal_result = attempt.results[-1]
        successful_count -= 1
    metrics = tuple(
        _require_passing_timed_result(record, attempt.jobs[index], attempt.results[index])
        for index in range(successful_count)
    )
    _require_matching_metrics(
        attempt.process_medians_ms,
        tuple(item[0] for item in metrics),
        label="failed-attempt process medians",
        record=record,
    )
    _require_matching_metrics(
        attempt.process_cvs,
        tuple(item[1] for item in metrics),
        label="failed-attempt process CVs",
        record=record,
    )
    if attempt.across_process_cv is not None or attempt.median_ms is not None:
        raise MatrixFloorEvidenceError(
            f"failed timing attempt cannot publish aggregates: "
            f"{record.task_id}/{record.target_id}"
        )
    if terminal_result is None:
        expected_status = "worker_failure"
    else:
        if terminal_result.status == "completed" or terminal_result.correct:
            raise MatrixFloorEvidenceError(
                f"failed timing attempt has no terminal failed result: "
                f"{record.task_id}/{record.target_id}"
            )
        expected_status = (
            "worker_failure"
            if terminal_result.status in _INFRASTRUCTURE_FAILURES
            else "correctness_failure"
        )
    if attempt.status != expected_status or attempt.stable or attempt.error is None:
        raise MatrixFloorEvidenceError(
            f"failed timing status differs from its terminal result: "
            f"{record.task_id}/{record.target_id}"
        )


def _validate_timing_protocol(record: GateRecord) -> None:
    summary = record.summary
    attempts = summary.attempts
    if tuple(attempt.attempt for attempt in attempts) != tuple(range(1, len(attempts) + 1)):
        raise MatrixFloorEvidenceError(
            f"timing attempts are not consecutively numbered: "
            f"{record.task_id}/{record.target_id}"
        )
    if any(
        attempt.status in {"worker_failure", "correctness_failure"}
        for attempt in attempts[:-1]
    ):
        raise MatrixFloorEvidenceError(
            f"failed timing attempt must terminate the protocol: "
            f"{record.task_id}/{record.target_id}"
        )
    if len(attempts) == 1:
        if attempts[0].status == "unstable":
            raise MatrixFloorEvidenceError(
                f"first unstable timing attempt must be retried: "
                f"{record.task_id}/{record.target_id}"
            )
    elif attempts[0].status != "unstable":
        raise MatrixFloorEvidenceError(
            f"second timing attempt requires a first unstable attempt: "
            f"{record.task_id}/{record.target_id}"
        )
    job_ids = tuple(job.job_id for job in summary.jobs)
    if len(job_ids) != len(set(job_ids)):
        raise MatrixFloorEvidenceError(
            f"timing protocol job IDs must be unique: {record.task_id}/{record.target_id}"
        )
    process_timing = summary.timing.model_copy(update={"repetitions": 1})
    if any(job.timing != process_timing for job in summary.jobs):
        raise MatrixFloorEvidenceError(
            f"worker timing differs from its protocol timing: "
            f"{record.task_id}/{record.target_id}"
        )
    for index, attempt in enumerate(attempts):
        if attempt.status in {"stable", "unstable"}:
            _validate_complete_attempt(record, index)
        else:
            _validate_failed_attempt(record, index)

    terminal = attempts[-1]
    if (
        summary.status != terminal.status
        or summary.stable != terminal.stable
        or summary.error != terminal.error
        or (
            terminal.stable
            and (
                summary.median_ms is None
                or terminal.median_ms is None
                or not _metrics_match(summary.median_ms, terminal.median_ms)
            )
        )
        or (not terminal.stable and summary.median_ms is not None)
    ):
        raise MatrixFloorEvidenceError(
            f"timing summary differs from its terminal attempt: "
            f"{record.task_id}/{record.target_id}"
        )


def _require_passing_oracle(record: GateRecord) -> tuple[WorkerResult, ...]:
    summary = record.summary
    if summary.status not in {"stable", "unstable"}:
        raise MatrixFloorEvidenceError(
            f"expert oracle did not complete correctness: {record.task_id}/{record.target_id}"
        )
    if len(summary.results) != len(summary.jobs) or not summary.results:
        raise MatrixFloorEvidenceError(
            f"expert oracle has incomplete worker results: {record.task_id}/{record.target_id}"
        )
    expected_case_ids = tuple(summary.jobs[0].case_ids)
    for job, result in zip(summary.jobs, summary.results, strict=True):
        if tuple(job.case_ids) != expected_case_ids:
            raise MatrixFloorEvidenceError(
                f"expert oracle changed sealed cases: {record.task_id}/{record.target_id}"
            )
        if (
            result.status != "completed"
            or not result.compiled
            or not result.correct
            or result.static_errors
            or tuple(case.case_id for case in result.cases) != expected_case_ids
            or any(
                case.status != "pass"
                or not case.correct
                or not case.output_finite
                or not case.inputs_unchanged
                for case in result.cases
            )
        ):
            raise MatrixFloorEvidenceError(
                f"expert oracle did not pass sealed correctness: "
                f"{record.task_id}/{record.target_id}"
            )
    return summary.results


def _generated_code_sha256(record: GateRecord, results: tuple[WorkerResult, ...]) -> str:
    generated: list[str] = []
    captures: list[str] = []
    sizes: list[int] = []
    for result in results:
        value = result.metadata.get("generated_code_sha256")
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise MatrixFloorEvidenceError(
                f"expert oracle is missing generated code evidence: "
                f"{record.task_id}/{record.target_id}"
            )
        capture = result.metadata.get("generated_code_capture")
        if capture != "tilelang.get_kernel_source.v1":
            raise MatrixFloorEvidenceError(
                f"expert oracle has an invalid generated code capture: "
                f"{record.task_id}/{record.target_id}"
            )
        size = result.metadata.get("generated_code_size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise MatrixFloorEvidenceError(
                f"expert oracle has an invalid generated code size: "
                f"{record.task_id}/{record.target_id}"
            )
        generated.append(value)
        captures.append(capture)
        sizes.append(size)
    if len(set(generated)) != 1 or len(set(captures)) != 1 or len(set(sizes)) != 1:
        raise MatrixFloorEvidenceError(
            f"expert oracle generated code metadata changed across workers: "
            f"{record.task_id}/{record.target_id}"
        )
    return generated[0]


def _expert_correctness(
    record: GateRecord,
    *,
    task: TaskAssetBinding,
    artifact_sha256: str,
) -> ExpertCorrectnessEvidence:
    results = _require_passing_oracle(record)
    return ExpertCorrectnessEvidence(
        artifact_sha256=artifact_sha256,
        task_id=task.task_id,
        task_pack_sha256=task.task_pack_sha256,
        expert_source_sha256=task.expert_source_sha256,
        status="pass",
        compiled=all(result.compiled for result in results),
        all_sealed_cases_passed=all(
            case.status == "pass" for result in results for case in result.cases
        ),
        output_finite=all(
            case.output_finite for result in results for case in result.cases
        ),
        inputs_unchanged=all(
            case.inputs_unchanged for result in results for case in result.cases
        ),
        fallback_free=all(not result.static_errors for result in results),
    )


def _target_codegen(
    record: GateRecord,
    *,
    task: TaskAssetBinding,
    target: TargetAssetBinding,
    artifact_sha256: str,
) -> TargetCodegenEvidence:
    results = _require_passing_oracle(record)
    generated_code_sha256 = _generated_code_sha256(record, results)
    return TargetCodegenEvidence(
        artifact_sha256=artifact_sha256,
        task_id=task.task_id,
        task_pack_sha256=task.task_pack_sha256,
        target_id=target.target_id,
        target_stack_sha256=target.target_stack_sha256,
        expert_source_sha256=task.expert_source_sha256,
        status="pass",
        compiled=all(result.compiled for result in results),
        correct=all(result.correct for result in results),
        fallback_free=all(not result.static_errors for result in results),
        generated_code_sha256=generated_code_sha256,
    )


def _baseline_timing(
    record: GateRecord,
    *,
    variant: str,
    source_sha256: str,
    artifact_sha256: str,
) -> BaselineTimingEvidence:
    summary = record.summary
    process_timing = FORMAL_FLOOR_TIMING.model_copy(update={"repetitions": 1})
    if any(job.timing != process_timing for job in summary.jobs):
        raise MatrixFloorEvidenceError(
            f"baseline worker timing differs from FORMAL_FLOOR_TIMING: "
            f"{record.task_id}/{variant}"
        )
    return BaselineTimingEvidence(
        variant=variant,
        source_sha256=source_sha256,
        artifact_sha256=artifact_sha256,
        timing_summary_sha256=sha256_json(summary),
        status=summary.status,
        median_ms=summary.median_ms if summary.stable else None,
    )


def _expected_gate_keys(
    assets: AssetManifest,
    targets: tuple[TargetStackSpec, ...],
    baseline_target_id: str,
) -> set[tuple[str, str, str, str | None]]:
    return {
        *(
            ("oracle", task.task_id, target.id, None)
            for task in assets.tasks
            for target in targets
        ),
        *(
            ("baseline", task.task_id, baseline_target_id, baseline.variant)
            for task in assets.tasks
            for baseline in task.baselines
        ),
    }


def derive_task_floor_records(
    gates: Iterable[GateRecord],
    assets: AssetManifest,
    targets: Iterable[TargetStackSpec],
    *,
    baseline_target_id: str,
    competitive_factor: float = 1.25,
) -> tuple[TaskFloorRecord, ...]:
    """Derive valid per-task floors from an exact set of sealed gate records."""

    target_values = _validate_targets(assets, targets)
    if baseline_target_id not in {target.id for target in target_values}:
        raise MatrixFloorEvidenceError(
            f"baseline target is not a frozen floor target: {baseline_target_id}"
        )
    target_assets = {target.target_id: target for target in assets.targets}
    task_assets = {task.task_id: task for task in assets.tasks}
    records: dict[tuple[str, str, str, str | None], tuple[GateRecord, str]] = {}
    for record in gates:
        key = (record.kind, record.task_id, record.target_id, record.variant)
        if key in records:
            raise MatrixFloorEvidenceError(f"duplicate gate record: {key}")
        artifact_sha256 = gate_artifact_sha256(record)
        task = task_assets.get(record.task_id)
        target = target_assets.get(record.target_id)
        if task is None or target is None:
            raise MatrixFloorEvidenceError(f"gate references an undeclared asset: {key}")
        if record.kind == "oracle":
            expected_source = task.expert_source_sha256
            if record.variant is not None:
                raise MatrixFloorEvidenceError(
                    f"oracle gate unexpectedly declares a variant: {key}"
                )
        else:
            baseline = next(
                (item for item in task.baselines if item.variant == record.variant),
                None,
            )
            if baseline is None:
                raise MatrixFloorEvidenceError(f"baseline gate has an undeclared variant: {key}")
            expected_source = baseline.source_sha256
        if record.source_sha256 != expected_source:
            raise MatrixFloorEvidenceError(f"gate source differs from the asset manifest: {key}")
        _validate_summary(record, task=task, target=target)
        records[key] = (record, artifact_sha256)

    expected_keys = _expected_gate_keys(assets, target_values, baseline_target_id)
    if set(records) != expected_keys:
        missing = sorted(expected_keys - set(records))
        unexpected = sorted(set(records) - expected_keys)
        raise MatrixFloorEvidenceError(
            f"gate records do not exactly cover the floor; missing={missing}, "
            f"unexpected={unexpected}"
        )

    floors: list[TaskFloorRecord] = []
    try:
        for task in assets.tasks:
            oracle_records = tuple(
                records[("oracle", task.task_id, target.id, None)] for target in target_values
            )
            first_oracle, first_artifact_sha256 = oracle_records[0]
            correctness = _expert_correctness(
                first_oracle,
                task=task,
                artifact_sha256=first_artifact_sha256,
            )
            codegen = tuple(
                _target_codegen(
                    record,
                    task=task,
                    target=target_assets[target.id],
                    artifact_sha256=artifact_sha256,
                )
                for target, (record, artifact_sha256) in zip(
                    target_values, oracle_records, strict=True
                )
            )
            generated_sizes = {
                result.metadata["generated_code_size_bytes"]
                for record, _ in oracle_records
                for result in record.summary.results
            }
            if len(generated_sizes) != 1:
                raise MatrixFloorEvidenceError(
                    f"expert oracle generated code size changed across target validators: "
                    f"{task.task_id}"
                )
            timings = tuple(
                _baseline_timing(
                    records[
                        ("baseline", task.task_id, baseline_target_id, baseline.variant)
                    ][0],
                    variant=baseline.variant,
                    source_sha256=baseline.source_sha256,
                    artifact_sha256=records[
                        ("baseline", task.task_id, baseline_target_id, baseline.variant)
                    ][1],
                )
                for baseline in task.baselines
            )
            stable = tuple(item for item in timings if item.status == "stable")
            if not stable:
                raise MatrixFloorEvidenceError(
                    f"task has no stable baseline timing: {task.task_id}"
                )
            selected = min(stable, key=lambda item: item.median_ms or float("inf"))
            evidence = VerifiedTaskFloorEvidence(
                task_id=task.task_id,
                expert_source_sha256=task.expert_source_sha256,
                expert_correctness=correctness,
                target_codegen=codegen,
                baseline_timings=timings,
                selected_baseline_variant=selected.variant,
                selected_baseline_source_sha256=selected.source_sha256,
                selected_timing_summary_sha256=selected.timing_summary_sha256,
            )
            ceiling = LatencyCeilingDerivation(
                l_star_ms=evidence.l_star_ms,
                competitive_factor=competitive_factor,
                latency_ceiling_ms=evidence.l_star_ms * competitive_factor,
            )
            floors.append(
                TaskFloorRecord(
                    task_id=task.task_id,
                    status="valid",
                    expert_source_sha256=task.expert_source_sha256,
                    verified_evidence=evidence,
                    ceiling=ceiling,
                )
            )
    except ValidationError as error:
        raise MatrixFloorEvidenceError(f"derived floor evidence is invalid: {error}") from error
    return tuple(floors)
