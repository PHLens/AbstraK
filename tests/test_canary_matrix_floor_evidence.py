from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from abstrak.canary.artifacts import TrajectoryStore
from abstrak.canary.contracts import (
    CaseResult,
    InputCaseSpec,
    TargetStackSpec,
    TaskPackSpec,
    TimingSpec,
    WorkerJob,
    WorkerResult,
)
from abstrak.canary.gates import GateRecord
from abstrak.canary.matrix_floor_evidence import (
    MatrixFloorEvidenceError,
    derive_task_floor_records,
    gate_artifact_sha256,
)
from abstrak.canary.matrix_preflight import (
    FORMAL_FLOOR_TIMING,
    AssetManifest,
    BaselineAssetBinding,
    TargetAssetBinding,
    TaskAssetBinding,
)
from abstrak.canary.timing import TimingAttemptSummary, TimingProtocolSummary
from abstrak.providers.contracts import sha256_json


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(label: str) -> str:
    return f'"""{label}."""\n\nclass ModelNew:\n    pass\n'


def _task() -> TaskPackSpec:
    return TaskPackSpec(
        id="floor-task",
        specification="Return x.",
        source_path="tasks/floor_task.py",
        source_sha256=_digest("reference source"),
        dtype="fp16",
        input_shapes=((16,),),
        atol=0.01,
        rtol=0.01,
        fallback_policy="forbid_framework_ops",
        dev_cases=(InputCaseSpec(id="dev-random", kind="random", seed=1),),
        sealed_cases=(
            InputCaseSpec(id="sealed-random", kind="random", seed=2),
            InputCaseSpec(id="sealed-zero", kind="zero", seed=3),
        ),
    )


def _targets() -> tuple[TargetStackSpec, ...]:
    return (
        TargetStackSpec(
            id="tileops-core-a100",
            backend="tilelang",
            version="0.1.12",
            card_path="targets/core.md",
            card_sha256=_digest("core card"),
            adapter="tilelang-capability-core",
        ),
        TargetStackSpec(
            id="tileops-full-a100",
            backend="tilelang",
            version="0.1.12",
            card_path="targets/full.md",
            card_sha256=_digest("full card"),
            adapter="tilelang-capability-full",
        ),
    )


def _completed_summary(
    *,
    task: TaskPackSpec,
    target: TargetStackSpec,
    kind: str,
    source: str,
    variant: str | None,
    latency: float,
    timing: TimingSpec = FORMAL_FLOOR_TIMING,
    generated_hashes: tuple[str, ...] | None = None,
    generated_captures: tuple[object, ...] | None = None,
    generated_sizes: tuple[object, ...] | None = None,
    capture_generated_code: bool = True,
) -> TimingProtocolSummary:
    candidate_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    process_timing = timing.model_copy(update={"repetitions": 1})
    jobs: list[WorkerJob] = []
    results: list[WorkerResult] = []
    case_ids = tuple(case.id for case in task.sealed_cases)
    for repetition in range(1, timing.repetitions + 1):
        suffix = "oracle" if variant is None else variant
        job = WorkerJob(
            job_id=f"{kind}-{task.id}-{target.id}-{suffix}-p{repetition}",
            kind=kind,
            task=task,
            target=target,
            case_ids=case_ids,
            candidate_source=source,
            candidate_sha256=candidate_sha256,
            timing=process_timing,
        )
        generated = (
            generated_hashes[repetition - 1]
            if generated_hashes is not None
            else _digest("shared generated code")
        )
        capture = (
            generated_captures[repetition - 1]
            if generated_captures is not None
            else "tilelang.get_kernel_source.v1"
        )
        size = (
            generated_sizes[repetition - 1]
            if generated_sizes is not None
            else 128
        )
        metadata = {}
        if kind == "oracle" and capture_generated_code:
            metadata = {
                "generated_code_capture": capture,
                "generated_code_sha256": generated,
                "generated_code_size_bytes": size,
            }
        result = WorkerResult(
            job_id=job.job_id,
            job_sha256=job.sha256,
            input_sha256=job.input_sha256,
            candidate_sha256=job.candidate_sha256,
            status="completed",
            compiled=True,
            correct=True,
            cases=tuple(
                CaseResult(
                    case_id=case_id,
                    status="pass",
                    correct=True,
                    max_abs_error=0.0,
                    max_rel_error=0.0,
                    output_finite=True,
                    inputs_unchanged=True,
                )
                for case_id in case_ids
            ),
            timing_ms=tuple(latency for _ in range(process_timing.trial_runs)),
            timing_cv=0.0,
            metadata=metadata,
        )
        jobs.append(job)
        results.append(result)
    attempt = TimingAttemptSummary(
        attempt=1,
        status="stable",
        stable=True,
        jobs=tuple(jobs),
        results=tuple(results),
        process_medians_ms=tuple(latency for _ in jobs),
        process_cvs=tuple(0.0 for _ in jobs),
        across_process_cv=0.0,
        median_ms=latency,
    )
    suffix = "oracle" if variant is None else variant
    return TimingProtocolSummary(
        job_prefix=f"{kind}-{task.id}-{target.id}-{suffix}",
        task_id=task.id,
        target_id=target.id,
        candidate_sha256=candidate_sha256,
        job_kind=kind,
        device="cuda:0",
        timing=timing,
        status="stable",
        stable=True,
        attempts=(attempt,),
        jobs=tuple(jobs),
        results=tuple(results),
        median_ms=latency,
    )


def _failed_summary(
    *,
    task: TaskPackSpec,
    target: TargetStackSpec,
    source: str,
    variant: str,
) -> TimingProtocolSummary:
    candidate_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    job = WorkerJob(
        job_id=f"baseline-{task.id}-{target.id}-{variant}-failure",
        kind="baseline",
        task=task,
        target=target,
        case_ids=tuple(case.id for case in task.sealed_cases),
        candidate_source=source,
        candidate_sha256=candidate_sha256,
        timing=FORMAL_FLOOR_TIMING.model_copy(update={"repetitions": 1}),
    )
    attempt = TimingAttemptSummary(
        attempt=1,
        status="worker_failure",
        stable=False,
        jobs=(job,),
        error="worker unavailable",
    )
    return TimingProtocolSummary(
        job_prefix=f"baseline-{task.id}-{target.id}-{variant}",
        task_id=task.id,
        target_id=target.id,
        candidate_sha256=candidate_sha256,
        job_kind="baseline",
        device="cuda:0",
        timing=FORMAL_FLOOR_TIMING,
        status="worker_failure",
        stable=False,
        attempts=(attempt,),
        jobs=(job,),
        error="worker unavailable",
    )


def _seal_gate(
    root: Path,
    *,
    kind: str,
    task: TaskPackSpec,
    target: TargetStackSpec,
    source: str,
    summary: TimingProtocolSummary,
    variant: str | None = None,
) -> GateRecord:
    suffix = "" if variant is None else f"-{variant}"
    gate_id = f"{kind}-{task.id}-{target.id}{suffix}"
    store = TrajectoryStore.create(root, "floor-study", gate_id)
    record = GateRecord(
        kind=kind,
        task_id=task.id,
        target_id=target.id,
        variant=variant,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        artifact_directory=str(store.run_directory),
        summary=summary,
    )
    store.write_json("gate-record.json", record)
    store.seal()
    return record


@dataclass(frozen=True)
class FloorInputs:
    gates: tuple[GateRecord, ...]
    assets: AssetManifest
    targets: tuple[TargetStackSpec, ...]


def _inputs(
    tmp_path: Path,
    *,
    oracle_hashes: dict[str, tuple[str, ...]] | None = None,
    oracle_captures: dict[str, tuple[object, ...]] | None = None,
    oracle_sizes: dict[str, tuple[object, ...]] | None = None,
    baseline_timing: dict[str, TimingSpec] | None = None,
    failed_baselines: frozenset[str] = frozenset(),
    missing_codegen_targets: frozenset[str] = frozenset(),
) -> FloorInputs:
    task = _task()
    targets = _targets()
    expert_source = _source("expert")
    baseline_sources = {
        "compile": _source("compile"),
        "eager": _source("eager"),
        "vendor": _source("vendor"),
    }
    task_asset = TaskAssetBinding(
        task_id=task.id,
        task_pack_sha256=sha256_json(task),
        reference_source_sha256=task.source_sha256,
        expert_source_sha256=hashlib.sha256(expert_source.encode("utf-8")).hexdigest(),
        baselines=tuple(
            BaselineAssetBinding(
                variant=variant,
                source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )
            for variant, source in baseline_sources.items()
        ),
    )
    assets = AssetManifest(
        study_id="floor-study",
        raw_study_sha256=_digest("raw study"),
        spec_sha256=_digest("study spec"),
        schedule_sha256=_digest("schedule"),
        tasks=(task_asset,),
        targets=tuple(
            TargetAssetBinding(
                target_id=target.id,
                target_stack_sha256=sha256_json(target),
                card_sha256=target.card_sha256,
            )
            for target in targets
        ),
    )
    gates: list[GateRecord] = []
    for target in targets:
        hashes = None if oracle_hashes is None else oracle_hashes[target.id]
        captures = None if oracle_captures is None else oracle_captures[target.id]
        sizes = None if oracle_sizes is None else oracle_sizes[target.id]
        gates.append(
            _seal_gate(
                tmp_path,
                kind="oracle",
                task=task,
                target=target,
                source=expert_source,
                summary=_completed_summary(
                    task=task,
                    target=target,
                    kind="oracle",
                    source=expert_source,
                    variant=None,
                    latency=0.5,
                    generated_hashes=hashes,
                    generated_captures=captures,
                    generated_sizes=sizes,
                    capture_generated_code=target.id not in missing_codegen_targets,
                ),
            )
        )
    latencies = {"compile": 2.0, "eager": 1.0, "vendor": 1.5}
    for variant, source in baseline_sources.items():
        summary = (
            _failed_summary(task=task, target=targets[0], source=source, variant=variant)
            if variant in failed_baselines
            else _completed_summary(
                task=task,
                target=targets[0],
                kind="baseline",
                source=source,
                variant=variant,
                latency=latencies[variant],
                timing=(baseline_timing or {}).get(variant, FORMAL_FLOOR_TIMING),
            )
        )
        gates.append(
            _seal_gate(
                tmp_path,
                kind="baseline",
                task=task,
                target=targets[0],
                source=source,
                summary=summary,
                variant=variant,
            )
        )
    return FloorInputs(gates=tuple(gates), assets=assets, targets=targets)


def _derive(
    inputs: FloorInputs,
    *,
    gates: tuple[GateRecord, ...] | None = None,
    targets: tuple[TargetStackSpec, ...] | None = None,
):
    return derive_task_floor_records(
        inputs.gates if gates is None else gates,
        inputs.assets,
        inputs.targets if targets is None else targets,
        baseline_target_id=inputs.targets[0].id,
    )


def _replace_gate_summary(
    inputs: FloorInputs,
    *,
    gate_index: int,
    summary: TimingProtocolSummary,
    root: Path,
) -> FloorInputs:
    previous = inputs.gates[gate_index]
    replacement = _seal_gate(
        root,
        kind=previous.kind,
        task=summary.jobs[0].task,
        target=summary.jobs[0].target,
        source=summary.jobs[0].candidate_source,
        summary=summary,
        variant=previous.variant,
    )
    gates = list(inputs.gates)
    gates[gate_index] = replacement
    return FloorInputs(gates=tuple(gates), assets=inputs.assets, targets=inputs.targets)


def _clone_attempt(
    attempt: TimingAttemptSummary,
    *,
    attempt_number: int,
    job_suffix: str,
) -> TimingAttemptSummary:
    jobs = tuple(
        job.model_copy(update={"job_id": f"{job.job_id}-{job_suffix}"})
        for job in attempt.jobs
    )
    results = tuple(
        result.model_copy(
            update={
                "job_id": job.job_id,
                "job_sha256": job.sha256,
                "input_sha256": job.input_sha256,
            }
        )
        for job, result in zip(jobs, attempt.results, strict=True)
    )
    return attempt.model_copy(
        update={
            "attempt": attempt_number,
            "jobs": jobs,
            "results": results,
        }
    )


def test_derives_verified_floor_and_fastest_stable_baseline(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    floors = derive_task_floor_records(
        reversed(inputs.gates),
        inputs.assets,
        inputs.targets,
        baseline_target_id=inputs.targets[0].id,
        competitive_factor=1.25,
    )

    assert len(floors) == 1
    floor = floors[0]
    assert floor.status == "valid"
    assert floor.ceiling is not None
    assert floor.ceiling.l_star_ms == 1.0
    assert floor.ceiling.latency_ceiling_ms == 1.25
    assert floor.verified_evidence is not None
    assert floor.verified_evidence.selected_baseline_variant == "eager"
    assert tuple(item.target_id for item in floor.verified_evidence.target_codegen) == tuple(
        target.id for target in inputs.targets
    )
    oracle = next(record for record in inputs.gates if record.kind == "oracle")
    expected_artifact_hash = hashlib.sha256(
        (Path(oracle.artifact_directory) / "sha256sums.txt").read_bytes()
    ).hexdigest()
    assert gate_artifact_sha256(oracle) == expected_artifact_hash


def test_rejects_missing_or_duplicate_gate_coverage(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    with pytest.raises(MatrixFloorEvidenceError, match="do not exactly cover"):
        _derive(inputs, gates=inputs.gates[:-1])
    with pytest.raises(MatrixFloorEvidenceError, match="duplicate gate"):
        _derive(inputs, gates=(*inputs.gates, inputs.gates[-1]))


def test_rejects_tampered_or_unsealed_gate_artifact(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    record = inputs.gates[0]
    checksum = Path(record.artifact_directory) / "sha256sums.txt"
    checksum.chmod(0o600)
    checksum.write_bytes(checksum.read_bytes() + b"tampered\n")

    with pytest.raises(MatrixFloorEvidenceError, match="sealed gate artifact is invalid"):
        gate_artifact_sha256(record)


def test_rejects_supplied_record_that_differs_from_sealed_record(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    record = inputs.gates[0]
    drifted = record.model_copy(update={"source_sha256": _digest("drifted source")})

    with pytest.raises(MatrixFloorEvidenceError, match="differs from its sealed artifact"):
        gate_artifact_sha256(drifted)


def test_rejects_manifest_target_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    drifted = inputs.targets[0].model_copy(update={"version": "0.1.13"})

    with pytest.raises(MatrixFloorEvidenceError, match="target stack differs"):
        _derive(inputs, targets=(drifted, inputs.targets[1]))


def test_rejects_missing_or_inconsistent_generated_code_evidence(tmp_path: Path) -> None:
    shared = _digest("shared generated code")
    inconsistent = _inputs(
        tmp_path / "inconsistent",
        oracle_hashes={
            "tileops-core-a100": (shared, _digest("different code"), shared),
            "tileops-full-a100": (shared, shared, shared),
        },
    )

    with pytest.raises(MatrixFloorEvidenceError, match="changed across workers"):
        _derive(inconsistent)

    missing = _inputs(
        tmp_path / "missing",
        missing_codegen_targets=frozenset({"tileops-core-a100"}),
    )
    with pytest.raises(MatrixFloorEvidenceError, match="missing generated code evidence"):
        _derive(missing)


def test_rejects_nonformal_baseline_timing(tmp_path: Path) -> None:
    nonformal = FORMAL_FLOOR_TIMING.model_copy(update={"warmup_runs": 24})
    inputs = _inputs(tmp_path, baseline_timing={"compile": nonformal})

    with pytest.raises(MatrixFloorEvidenceError, match="does not use FORMAL_FLOOR_TIMING"):
        _derive(inputs)


def test_rejects_nonformal_oracle_timing(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "original")
    gate_index = next(index for index, record in enumerate(inputs.gates) if record.kind == "oracle")
    record = inputs.gates[gate_index]
    nonformal = TimingSpec(warmup_runs=1, trial_runs=1, repetitions=1)
    summary = _completed_summary(
        task=record.summary.jobs[0].task,
        target=record.summary.jobs[0].target,
        kind="oracle",
        source=record.summary.jobs[0].candidate_source,
        variant=None,
        latency=0.5,
        timing=nonformal,
    )
    attacked = _replace_gate_summary(
        inputs,
        gate_index=gate_index,
        summary=summary,
        root=tmp_path / "attacked",
    )

    with pytest.raises(MatrixFloorEvidenceError, match="does not use FORMAL_FLOOR_TIMING"):
        _derive(attacked)


def test_rejects_floor_without_a_stable_baseline(tmp_path: Path) -> None:
    inputs = _inputs(
        tmp_path,
        failed_baselines=frozenset({"compile", "eager", "vendor"}),
    )

    with pytest.raises(MatrixFloorEvidenceError, match="no stable baseline"):
        _derive(inputs)


def test_rejects_cross_target_codegen_drift(tmp_path: Path) -> None:
    core = _digest("core generated code")
    full = _digest("full generated code")
    inputs = _inputs(
        tmp_path,
        oracle_hashes={
            "tileops-core-a100": (core, core, core),
            "tileops-full-a100": (full, full, full),
        },
    )

    with pytest.raises(MatrixFloorEvidenceError, match="identical code"):
        _derive(inputs)


def test_recomputes_worker_and_attempt_metrics_from_raw_timing(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "original")
    gate_index = next(
        index
        for index, record in enumerate(inputs.gates)
        if record.kind == "baseline" and record.variant == "compile"
    )
    summary = inputs.gates[gate_index].summary
    forged_samples = tuple(1.0 if index % 2 == 0 else 3.0 for index in range(200))
    forged_result = summary.results[0].model_copy(update={"timing_ms": forged_samples})
    forged_attempt = summary.attempts[0].model_copy(
        update={"results": (forged_result, *summary.attempts[0].results[1:])}
    )
    forged_summary = summary.model_copy(
        update={
            "attempts": (forged_attempt,),
            "results": (forged_result, *summary.results[1:]),
        }
    )
    attacked = _replace_gate_summary(
        inputs,
        gate_index=gate_index,
        summary=forged_summary,
        root=tmp_path / "attacked",
    )

    with pytest.raises(MatrixFloorEvidenceError, match="timing_cv differs from raw timing"):
        _derive(attacked)


def test_rejects_complete_attempt_with_too_few_processes(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "original")
    gate_index = next(
        index
        for index, record in enumerate(inputs.gates)
        if record.kind == "baseline" and record.variant == "compile"
    )
    summary = inputs.gates[gate_index].summary
    attempt = summary.attempts[0].model_copy(
        update={
            "jobs": summary.attempts[0].jobs[:2],
            "results": summary.attempts[0].results[:2],
            "process_medians_ms": summary.attempts[0].process_medians_ms[:2],
            "process_cvs": summary.attempts[0].process_cvs[:2],
        }
    )
    attacked_summary = summary.model_copy(
        update={
            "attempts": (attempt,),
            "jobs": summary.jobs[:2],
            "results": summary.results[:2],
        }
    )
    attacked = _replace_gate_summary(
        inputs,
        gate_index=gate_index,
        summary=attacked_summary,
        root=tmp_path / "attacked",
    )

    with pytest.raises(MatrixFloorEvidenceError, match="exactly 3 processes"):
        _derive(attacked)


def test_rejects_duplicate_timing_job_ids(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "original")
    gate_index = next(
        index
        for index, record in enumerate(inputs.gates)
        if record.kind == "baseline" and record.variant == "compile"
    )
    summary = inputs.gates[gate_index].summary
    jobs = list(summary.jobs)
    jobs[1] = jobs[1].model_copy(update={"job_id": jobs[0].job_id})
    results = list(summary.results)
    results[1] = results[1].model_copy(
        update={
            "job_id": jobs[1].job_id,
            "job_sha256": jobs[1].sha256,
            "input_sha256": jobs[1].input_sha256,
        }
    )
    attempt = summary.attempts[0].model_copy(
        update={"jobs": tuple(jobs), "results": tuple(results)}
    )
    attacked_summary = summary.model_copy(
        update={
            "attempts": (attempt,),
            "jobs": tuple(jobs),
            "results": tuple(results),
        }
    )
    attacked = _replace_gate_summary(
        inputs,
        gate_index=gate_index,
        summary=attacked_summary,
        root=tmp_path / "attacked",
    )

    with pytest.raises(MatrixFloorEvidenceError, match="job IDs must be unique"):
        _derive(attacked)


def test_rejects_second_attempt_after_stable_first_attempt(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "original")
    gate_index = next(
        index
        for index, record in enumerate(inputs.gates)
        if record.kind == "baseline" and record.variant == "compile"
    )
    summary = inputs.gates[gate_index].summary
    second = _clone_attempt(summary.attempts[0], attempt_number=2, job_suffix="retry")
    attacked_summary = summary.model_copy(
        update={
            "attempts": (summary.attempts[0], second),
            "jobs": (*summary.jobs, *second.jobs),
            "results": (*summary.results, *second.results),
        }
    )
    attacked = _replace_gate_summary(
        inputs,
        gate_index=gate_index,
        summary=attacked_summary,
        root=tmp_path / "attacked",
    )

    with pytest.raises(MatrixFloorEvidenceError, match="requires a first unstable attempt"):
        _derive(attacked)


def test_rejects_nonconsecutive_attempt_number(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "original")
    gate_index = next(
        index
        for index, record in enumerate(inputs.gates)
        if record.kind == "baseline" and record.variant == "compile"
    )
    summary = inputs.gates[gate_index].summary
    attempt = summary.attempts[0].model_copy(update={"attempt": 2})
    attacked_summary = summary.model_copy(update={"attempts": (attempt,)})
    attacked = _replace_gate_summary(
        inputs,
        gate_index=gate_index,
        summary=attacked_summary,
        root=tmp_path / "attacked",
    )

    with pytest.raises(MatrixFloorEvidenceError, match="not consecutively numbered"):
        _derive(attacked)


def test_rejects_attempt_after_terminal_failure(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "original")
    gate_index = next(
        index
        for index, record in enumerate(inputs.gates)
        if record.kind == "baseline" and record.variant == "compile"
    )
    stable_summary = inputs.gates[gate_index].summary
    failed_summary = _failed_summary(
        task=stable_summary.jobs[0].task,
        target=stable_summary.jobs[0].target,
        source=stable_summary.jobs[0].candidate_source,
        variant="compile",
    )
    second = _clone_attempt(stable_summary.attempts[0], attempt_number=2, job_suffix="retry")
    attacked_summary = stable_summary.model_copy(
        update={
            "attempts": (failed_summary.attempts[0], second),
            "jobs": (*failed_summary.jobs, *second.jobs),
            "results": second.results,
        }
    )
    attacked = _replace_gate_summary(
        inputs,
        gate_index=gate_index,
        summary=attacked_summary,
        root=tmp_path / "attacked",
    )

    with pytest.raises(MatrixFloorEvidenceError, match="must terminate the protocol"):
        _derive(attacked)


def test_rejects_codegen_capture_and_size_attacks(tmp_path: Path) -> None:
    valid_capture = "tilelang.get_kernel_source.v1"
    target_ids = tuple(target.id for target in _targets())
    invalid_capture = _inputs(
        tmp_path / "capture",
        oracle_captures={
            target_ids[0]: ("tilelang.other.v1", valid_capture, valid_capture),
            target_ids[1]: (valid_capture, valid_capture, valid_capture),
        },
    )
    with pytest.raises(MatrixFloorEvidenceError, match="invalid generated code capture"):
        _derive(invalid_capture)

    boolean_size = _inputs(
        tmp_path / "boolean-size",
        oracle_sizes={
            target_ids[0]: (True, 128, 128),
            target_ids[1]: (128, 128, 128),
        },
    )
    with pytest.raises(MatrixFloorEvidenceError, match="invalid generated code size"):
        _derive(boolean_size)

    inconsistent_size = _inputs(
        tmp_path / "size-drift",
        oracle_sizes={
            target_ids[0]: (128, 256, 128),
            target_ids[1]: (128, 128, 128),
        },
    )
    with pytest.raises(MatrixFloorEvidenceError, match="metadata changed across workers"):
        _derive(inconsistent_size)

    cross_target_size = _inputs(
        tmp_path / "cross-target-size",
        oracle_sizes={
            target_ids[0]: (128, 128, 128),
            target_ids[1]: (256, 256, 256),
        },
    )
    with pytest.raises(MatrixFloorEvidenceError, match="changed across target validators"):
        _derive(cross_target_size)


def test_requires_explicit_frozen_baseline_target(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    with pytest.raises(MatrixFloorEvidenceError, match="not a frozen floor target"):
        derive_task_floor_records(
            inputs.gates,
            inputs.assets,
            inputs.targets,
            baseline_target_id="undeclared-target",
        )
    with pytest.raises(MatrixFloorEvidenceError, match="do not exactly cover"):
        derive_task_floor_records(
            inputs.gates,
            inputs.assets,
            inputs.targets,
            baseline_target_id=inputs.targets[1].id,
        )
