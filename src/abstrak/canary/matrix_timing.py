"""Hash-bound candidate timing for generic matrix studies."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from abstrak.canary.artifacts import (
    TrajectoryArtifactError,
    TrajectoryStore,
    verify_trajectory,
)
from abstrak.canary.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    CanaryModel,
    TargetStackSpec,
    TaskPackSpec,
    TimingSpec,
)
from abstrak.canary.loop import WorkerExecutor
from abstrak.canary.manifests import PinnedStudySpec
from abstrak.canary.matrix import MatrixCell, MatrixSchedule, build_matrix_schedule
from abstrak.canary.matrix_preflight import FORMAL_FLOOR_TIMING, PreflightBundle
from abstrak.canary.matrix_runner import (
    MatrixPhaseContract,
    MatrixStudyRunError,
    MatrixWorkerBinding,
    load_matrix_phase_contract,
    resolve_terminal_contract_attempts,
)
from abstrak.canary.matrix_study import MatrixCellArtifactIdentity
from abstrak.canary.postprocess_timing import (
    PostprocessTimingError,
    run_or_resume_candidate_timing_artifact,
    verify_qualified_candidate_sources,
)
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import get_task_pack
from abstrak.canary.timing import TimingProtocolSummary
from abstrak.providers.contracts import sha256_json


class MatrixTimingError(RuntimeError):
    """Raised when matrix candidate timing cannot preserve frozen provenance."""


class MatrixCandidateSourceIdentity(CanaryModel):
    """Complete frozen identity of one qualified source selected for timing."""

    schema_version: Literal["abstrak-matrix-candidate-source.v1"] = (
        "abstrak-matrix-candidate-source.v1"
    )
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    raw_study_sha256: str = Field(pattern=SHA256_PATTERN)
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    phase_id: str = Field(pattern=IDENTIFIER_PATTERN)
    phase_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    preflight_receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_context_sha256: str = Field(pattern=SHA256_PATTERN)
    attempt_identity: MatrixCellArtifactIdentity
    source_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_labels: tuple[Literal["first", "final"], ...] = Field(min_length=1)
    candidate_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def identities_are_canonical(self) -> MatrixCandidateSourceIdentity:
        attempt_binding = (
            self.attempt_identity.study_id,
            self.attempt_identity.raw_study_sha256,
            self.attempt_identity.spec_sha256,
            self.attempt_identity.schedule_sha256,
            self.attempt_identity.phase_id,
            self.attempt_identity.execution_context_sha256,
        )
        source_binding = (
            self.study_id,
            self.raw_study_sha256,
            self.spec_sha256,
            self.schedule_sha256,
            self.phase_id,
            self.execution_context_sha256,
        )
        if attempt_binding != source_binding:
            raise ValueError("candidate attempt identity has a different study binding")
        if self.candidate_labels not in {
            ("first",),
            ("final",),
            ("first", "final"),
        }:
            raise ValueError("candidate labels must be ordered unique first/final labels")
        return self

    @property
    def logical_trajectory_id(self) -> str:
        return self.attempt_identity.trajectory_id

    @property
    def attempt_trajectory_id(self) -> str:
        return self.attempt_identity.artifact_trajectory_id

    @property
    def attempt_index(self) -> int:
        return self.attempt_identity.attempt_index

    @property
    def cell(self) -> MatrixCell:
        return self.attempt_identity.cell

    @property
    def cell_identity_sha256(self) -> str:
        return self.attempt_identity.sha256

    @property
    def sha256(self) -> str:
        return sha256_json(self)


@dataclass(frozen=True)
class MatrixQualifiedCandidate:
    """Verified source text and registry objects for one frozen candidate identity."""

    identity: MatrixCandidateSourceIdentity
    source_artifact_directory: Path
    source: str
    task: TaskPackSpec
    target: TargetStackSpec


def matrix_timing_artifact_id(source: MatrixCandidateSourceIdentity) -> str:
    labels = "-".join(source.candidate_labels)
    return f"timing-{source.attempt_trajectory_id}-{labels}"


class MatrixTimingStudyCandidate(CanaryModel):
    """One source and its deterministic child timing artifact ID."""

    source: MatrixCandidateSourceIdentity
    artifact_id: str = Field(pattern=IDENTIFIER_PATTERN)

    @model_validator(mode="after")
    def artifact_id_matches_source(self) -> MatrixTimingStudyCandidate:
        if self.artifact_id != matrix_timing_artifact_id(self.source):
            raise ValueError("matrix timing artifact ID differs from its source")
        return self


class MatrixTimingStudyManifest(CanaryModel):
    """Sealed candidate set and runtime contract written before the first timing job."""

    schema_version: Literal["abstrak-matrix-timing-study-manifest.v1"] = (
        "abstrak-matrix-timing-study-manifest.v1"
    )
    timing_study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    raw_study_sha256: str = Field(pattern=SHA256_PATTERN)
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    schedule_sha256: str = Field(pattern=SHA256_PATTERN)
    phase_id: str = Field(pattern=IDENTIFIER_PATTERN)
    phase_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    preflight_receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_context_sha256: str = Field(pattern=SHA256_PATTERN)
    worker: MatrixWorkerBinding
    device: str = Field(pattern=r"^cuda:[0-9]+$")
    timing: TimingSpec
    candidate_count: int = Field(ge=0)
    candidates: tuple[MatrixTimingStudyCandidate, ...]

    @model_validator(mode="after")
    def candidates_and_runtime_match_study(self) -> MatrixTimingStudyManifest:
        if self.timing != FORMAL_FLOOR_TIMING:
            raise ValueError("matrix candidate timing must use the frozen 25/200/3 protocol")
        if self.worker.transport.device != self.device:
            raise ValueError("matrix timing device differs from the worker binding")
        if self.candidate_count != len(self.candidates):
            raise ValueError("matrix timing candidate count differs from its manifest")
        artifact_ids = tuple(item.artifact_id for item in self.candidates)
        source_hashes = tuple(item.source.sha256 for item in self.candidates)
        if len(artifact_ids) != len(set(artifact_ids)) or len(source_hashes) != len(
            set(source_hashes)
        ):
            raise ValueError("matrix timing candidate identities must be unique")
        binding = (
            self.study_id,
            self.raw_study_sha256,
            self.spec_sha256,
            self.schedule_sha256,
            self.phase_id,
            self.phase_contract_sha256,
            self.preflight_receipt_sha256,
            self.execution_context_sha256,
        )
        if any(
            (
                item.source.study_id,
                item.source.raw_study_sha256,
                item.source.spec_sha256,
                item.source.schedule_sha256,
                item.source.phase_id,
                item.source.phase_contract_sha256,
                item.source.preflight_receipt_sha256,
                item.source.execution_context_sha256,
            )
            != binding
            for item in self.candidates
        ):
            raise ValueError("matrix timing candidates have different study bindings")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class MatrixCandidateTimingRecord(CanaryModel):
    """One sealed formal timing result bound to its matrix source and study manifest."""

    schema_version: Literal["abstrak-matrix-candidate-timing-record.v1"] = (
        "abstrak-matrix-candidate-timing-record.v1"
    )
    source: MatrixCandidateSourceIdentity
    timing_study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    timing_study_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_artifact_directory: str
    artifact_directory: str
    summary: TimingProtocolSummary

    @model_validator(mode="after")
    def summary_matches_source(self) -> MatrixCandidateTimingRecord:
        if (
            self.summary.job_prefix != matrix_timing_artifact_id(self.source)
            or self.summary.task_id != self.source.cell.task_id
            or self.summary.target_id != self.source.cell.target_id
            or self.summary.candidate_sha256 != self.source.candidate_sha256
            or self.summary.job_kind != "sealed"
        ):
            raise ValueError("matrix timing record differs from its candidate source")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _source_artifact_sha256(directory: Path) -> str:
    if directory.is_symlink():
        raise MatrixTimingError("matrix source artifact cannot be a symbolic link")
    try:
        verify_trajectory(directory)
        checksum_bytes = (directory / "sha256sums.txt").read_bytes()
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixTimingError(f"matrix source artifact is invalid: {directory}") from error
    return hashlib.sha256(checksum_bytes).hexdigest()


def _verify_preflight_binding(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    preflight: PreflightBundle,
) -> None:
    binding = (
        preflight.assets.study_id,
        preflight.assets.raw_study_sha256,
        preflight.assets.spec_sha256,
        preflight.assets.schedule_sha256,
    )
    expected = (
        pinned.spec.study_id,
        pinned.sha256,
        pinned.spec.sha256,
        schedule.sha256,
    )
    if binding != expected:
        raise MatrixTimingError("matrix timing preflight differs from the pinned study")


def discover_matrix_qualified_candidates(
    *,
    artifact_root: str | Path,
    pinned: PinnedStudySpec,
    phase_id: str,
    preflight: PreflightBundle,
    schedule: MatrixSchedule | None = None,
) -> tuple[MatrixQualifiedCandidate, ...]:
    """Recover terminal attempts and verify every qualified first/final source."""

    frozen_schedule = schedule or build_matrix_schedule(pinned.spec)
    _verify_preflight_binding(pinned, frozen_schedule, preflight)
    try:
        contract = load_matrix_phase_contract(
            artifact_root,
            pinned,
            phase_id,
            execution_context=preflight.execution_context,
            schedule=frozen_schedule,
        )
        terminal_attempts = resolve_terminal_contract_attempts(contract, artifact_root)
    except MatrixStudyRunError as error:
        raise MatrixTimingError(str(error)) from error

    task_assets = {item.task_id: item for item in preflight.assets.tasks}
    target_assets = {item.target_id: item for item in preflight.assets.targets}
    candidates: list[MatrixQualifiedCandidate] = []
    for planned, attempt in zip(contract.plan.cells, terminal_attempts, strict=True):
        primary = planned.identity
        task_asset = task_assets.get(primary.cell.task_id)
        target_asset = target_assets.get(primary.cell.target_id)
        if (
            task_asset is None
            or target_asset is None
            or primary.task_sha256 != task_asset.task_pack_sha256
            or primary.target_sha256 != target_asset.target_stack_sha256
        ):
            raise MatrixTimingError(
                f"matrix phase identity differs from preflight assets: {primary.trajectory_id}"
            )
        task = get_task_pack(primary.cell.task_id)
        target = get_target_stack(primary.cell.target_id)
        if (
            sha256_json(task) != task_asset.task_pack_sha256
            or sha256_json(target) != target_asset.target_stack_sha256
        ):
            raise MatrixTimingError(
                f"matrix timing registry differs from preflight assets: {primary.trajectory_id}"
            )
        if attempt is None:
            continue
        directory = (
            Path(artifact_root).expanduser()
            / attempt.identity.study_id
            / attempt.identity.artifact_trajectory_id
        )
        source_artifact_sha256 = _source_artifact_sha256(directory)
        try:
            verified_sources = verify_qualified_candidate_sources(
                directory,
                attempt.outcome,
                task=task,
                target=target,
            )
        except PostprocessTimingError as error:
            raise MatrixTimingError(str(error)) from error
        for verified in verified_sources:
            identity = MatrixCandidateSourceIdentity(
                study_id=pinned.spec.study_id,
                raw_study_sha256=pinned.sha256,
                spec_sha256=pinned.spec.sha256,
                schedule_sha256=frozen_schedule.sha256,
                phase_id=phase_id,
                phase_contract_sha256=contract.sha256,
                preflight_receipt_sha256=preflight.receipt.sha256,
                execution_context_sha256=preflight.execution_context.sha256,
                attempt_identity=attempt.identity,
                source_artifact_sha256=source_artifact_sha256,
                candidate_labels=verified.candidate_labels,
                candidate_sha256=verified.source_sha256,
            )
            candidates.append(
                MatrixQualifiedCandidate(
                    identity=identity,
                    source_artifact_directory=directory,
                    source=verified.source,
                    task=task,
                    target=target,
                )
            )
    return tuple(candidates)


def build_matrix_timing_study_manifest(
    pinned: PinnedStudySpec,
    schedule: MatrixSchedule,
    phase_id: str,
    *,
    preflight: PreflightBundle,
    contract: MatrixPhaseContract,
    candidates: tuple[MatrixQualifiedCandidate, ...],
    timing_study_id: str,
) -> MatrixTimingStudyManifest:
    """Freeze the complete candidate set and preflight-authorized timing runtime."""

    _verify_preflight_binding(pinned, schedule, preflight)
    try:
        expected_cells = schedule.cells_for_phase(phase_id)
    except ValueError as error:
        raise MatrixTimingError(f"unknown matrix timing phase: {phase_id}") from error
    contract_binding = (
        contract.plan.study_id,
        contract.plan.raw_study_sha256,
        contract.plan.spec_sha256,
        contract.plan.schedule_sha256,
        contract.plan.phase_id,
        tuple(cell.identity.cell for cell in contract.plan.cells),
    )
    expected_contract_binding = (
        pinned.spec.study_id,
        pinned.sha256,
        pinned.spec.sha256,
        schedule.sha256,
        phase_id,
        expected_cells,
    )
    if (
        contract_binding != expected_contract_binding
        or contract.execution_context != preflight.execution_context
    ):
        raise MatrixTimingError("matrix timing contract differs from preflight execution")
    if preflight.floor.timing != FORMAL_FLOOR_TIMING:
        raise MatrixTimingError("preflight does not authorize the frozen timing protocol")
    gate = pinned.spec.gate
    if gate is not None and gate.metrics.max_timing_cv != preflight.floor.timing.max_cv:
        raise MatrixTimingError("study timing CV differs from the preflight timing protocol")
    sources = tuple(candidate.identity for candidate in candidates)
    if any(source.phase_contract_sha256 != contract.sha256 for source in sources):
        raise MatrixTimingError("matrix timing candidate differs from the phase contract")
    worker = MatrixWorkerBinding(
        worker_revision=preflight.execution_context.worker_revision,
        transport=preflight.execution_context.transport,
    )
    return MatrixTimingStudyManifest(
        timing_study_id=timing_study_id,
        study_id=pinned.spec.study_id,
        raw_study_sha256=pinned.sha256,
        spec_sha256=pinned.spec.sha256,
        schedule_sha256=schedule.sha256,
        phase_id=phase_id,
        phase_contract_sha256=contract.sha256,
        preflight_receipt_sha256=preflight.receipt.sha256,
        execution_context_sha256=preflight.execution_context.sha256,
        worker=worker,
        device=preflight.execution_context.transport.device,
        timing=preflight.floor.timing,
        candidate_count=len(candidates),
        candidates=tuple(
            MatrixTimingStudyCandidate(
                source=source,
                artifact_id=matrix_timing_artifact_id(source),
            )
            for source in sources
        ),
    )


def seal_matrix_timing_study_manifest(
    artifact_root: str | Path,
    manifest: MatrixTimingStudyManifest,
) -> Path:
    """Seal or exactly resume the immutable study-wide candidate timing manifest."""

    directory = (
        Path(artifact_root).expanduser() / manifest.timing_study_id / "study-manifest"
    )
    if directory.exists() or directory.is_symlink():
        if directory.is_symlink():
            raise MatrixTimingError("matrix timing study manifest cannot be a symbolic link")
        try:
            verify_trajectory(directory)
            actual = MatrixTimingStudyManifest.model_validate_json(
                (directory / "run-manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TrajectoryArtifactError) as error:
            raise MatrixTimingError("matrix timing study manifest is invalid") from error
        if actual != manifest:
            raise MatrixTimingError("matrix timing study manifest differs from frozen inputs")
        _verify_matrix_timing_study_directory(artifact_root, manifest)
        return directory
    try:
        store = TrajectoryStore.create(
            artifact_root,
            manifest.timing_study_id,
            "study-manifest",
        )
        store.write_json("run-manifest.json", manifest)
        store.seal()
    except (OSError, TrajectoryArtifactError) as error:
        raise MatrixTimingError("cannot seal matrix timing study manifest") from error
    _verify_matrix_timing_study_directory(artifact_root, manifest)
    return store.run_directory


def _verify_matrix_timing_study_directory(
    artifact_root: str | Path,
    manifest: MatrixTimingStudyManifest,
) -> None:
    root = Path(artifact_root).expanduser() / manifest.timing_study_id
    manifest_directory = root / "study-manifest"
    try:
        verify_trajectory(manifest_directory)
        actual = MatrixTimingStudyManifest.model_validate_json(
            (manifest_directory / "run-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        raise MatrixTimingError("sealed matrix timing study manifest is invalid") from error
    if actual != manifest:
        raise MatrixTimingError("sealed matrix timing study manifest differs from runtime inputs")
    allowed = {"study-manifest"}
    for candidate in manifest.candidates:
        allowed.add(candidate.artifact_id)
        allowed.add(f"{candidate.artifact_id}.incomplete")
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise MatrixTimingError("cannot inspect the matrix timing study directory") from error
    if any(entry.is_symlink() for entry in entries):
        raise MatrixTimingError("matrix timing study cannot contain symbolic links")
    unexpected = sorted(entry.name for entry in entries if entry.name not in allowed)
    if unexpected:
        raise MatrixTimingError(
            f"matrix timing study contains unexpected artifacts: {unexpected}"
        )


MatrixTimingProgress = Callable[[int, int, MatrixCandidateTimingRecord, bool], None]


def run_matrix_candidate_timing(
    worker: WorkerExecutor,
    *,
    artifact_root: str | Path,
    manifest: MatrixTimingStudyManifest,
    candidates: tuple[MatrixQualifiedCandidate, ...],
    progress: MatrixTimingProgress | None = None,
) -> tuple[MatrixCandidateTimingRecord, ...]:
    """Run or exactly resume all candidates in one sealed timing study manifest."""

    if tuple(candidate.identity for candidate in candidates) != tuple(
        item.source for item in manifest.candidates
    ):
        raise MatrixTimingError("runtime candidates differ from the timing study manifest")
    if getattr(worker, "matrix_worker_binding", None) != manifest.worker:
        raise MatrixTimingError("matrix timing worker differs from the study manifest")
    _verify_matrix_timing_study_directory(artifact_root, manifest)
    records: list[MatrixCandidateTimingRecord] = []
    for index, (candidate, manifest_candidate) in enumerate(
        zip(candidates, manifest.candidates, strict=True),
        start=1,
    ):
        timing_id = manifest_candidate.artifact_id
        attempt_identity = candidate.identity.attempt_identity
        if (
            candidate.task.id != attempt_identity.cell.task_id
            or sha256_json(candidate.task) != attempt_identity.task_sha256
            or candidate.target.id != attempt_identity.cell.target_id
            or sha256_json(candidate.target) != attempt_identity.target_sha256
        ):
            raise MatrixTimingError(
                f"matrix timing task or target differs from its source: {timing_id}"
            )
        if (
            _source_artifact_sha256(candidate.source_artifact_directory)
            != candidate.identity.source_artifact_sha256
        ):
            raise MatrixTimingError(
                f"matrix source artifact changed after discovery: {timing_id}"
            )
        final_path = (
            Path(artifact_root).expanduser() / manifest.timing_study_id / timing_id
        )
        expected_manifest = {
            "schema_version": "abstrak-matrix-candidate-timing-manifest.v1",
            "timing_study_id": manifest.timing_study_id,
            "timing_study_manifest_sha256": manifest.sha256,
            "source": candidate.identity,
            "timing": manifest.timing,
            "device": manifest.device,
        }

        def validate_record(
            record: MatrixCandidateTimingRecord,
            expected_source: MatrixCandidateSourceIdentity = candidate.identity,
            expected_path: Path = final_path,
            expected_source_directory: Path = candidate.source_artifact_directory,
            expected_timing_id: str = timing_id,
        ) -> None:
            if (
                record.source != expected_source
                or record.timing_study_id != manifest.timing_study_id
                or record.timing_study_manifest_sha256 != manifest.sha256
                or record.source_artifact_directory
                != str(expected_source_directory)
                or record.artifact_directory != str(expected_path)
                or record.summary.timing != manifest.timing
                or record.summary.device != manifest.device
            ):
                raise MatrixTimingError(
                    f"matrix timing record differs from frozen inputs: {expected_timing_id}"
                )

        def build_record(
            summary: TimingProtocolSummary,
            path: Path,
            source_identity: MatrixCandidateSourceIdentity = candidate.identity,
            source_directory: Path = candidate.source_artifact_directory,
        ) -> MatrixCandidateTimingRecord:
            return MatrixCandidateTimingRecord(
                source=source_identity,
                timing_study_id=manifest.timing_study_id,
                timing_study_manifest_sha256=manifest.sha256,
                source_artifact_directory=str(source_directory),
                artifact_directory=str(path),
                summary=summary,
            )

        try:
            record, resumed = run_or_resume_candidate_timing_artifact(
                worker,
                root=artifact_root,
                timing_study_id=manifest.timing_study_id,
                timing_id=timing_id,
                expected_manifest=expected_manifest,
                task=candidate.task,
                target=candidate.target,
                source=candidate.source,
                source_sha256=candidate.identity.candidate_sha256,
                timing=manifest.timing,
                device=manifest.device,
                record_type=MatrixCandidateTimingRecord,
                build_record=build_record,
                validate_record=validate_record,
            )
        except (PostprocessTimingError, ValueError) as error:
            raise MatrixTimingError(str(error)) from error
        records.append(record)
        if progress is not None:
            progress(index, len(candidates), record, resumed)
    _verify_matrix_timing_study_directory(artifact_root, manifest)
    return tuple(records)
