"""Immutable phase journal and disposable resume indexes for anytime studies."""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from abstrak.anytime.artifacts import (
    _TEMP_PATTERN,
    AnytimeArtifactError,
    AnytimeAttemptAudit,
    AnytimeAttemptIdentity,
    _exclusive_json,
    _file_sha256,
    _fsync_directory,
    _load_model,
    _validate_component,
    artifact_payload_sha256,
    audit_anytime_attempt,
    canonical_artifact_bytes,
)
from abstrak.anytime.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AnytimeLoopPolicy,
    AnytimeModel,
)
from abstrak.providers.contracts import ChatMessage, sha256_json


class AnytimePhaseJournalError(AnytimeArtifactError):
    """Raised when a phase journal or derived resume view is inconsistent."""


class AnytimePhaseJournalHeader(AnytimeModel):
    """External phase anchor and exact expected primary-trajectory population."""

    schema_version: Literal["abstrak-anytime-phase-journal-header.v1"] = (
        "abstrak-anytime-phase-journal-header.v1"
    )
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    phase_id: str = Field(pattern=IDENTIFIER_PATTERN)
    phase_execution_sha256: str = Field(pattern=SHA256_PATTERN)
    expected_trajectory_ids: tuple[str, ...] = Field(min_length=1)
    max_attempts_per_trajectory: Literal[1, 2]

    @field_validator("expected_trajectory_ids")
    @classmethod
    def trajectory_population_is_sorted_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise ValueError("expected trajectory IDs must be sorted and unique")
        if any(_validate_identifier(value) is None for value in values):
            raise ValueError("expected trajectory IDs contain an unsafe identifier")
        return values

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _validate_identifier(value: str) -> Literal[True] | None:
    try:
        _validate_component("trajectory_id", value)
    except AnytimeArtifactError:
        return None
    return True


class AnytimeOperationalTotals(AnytimeModel):
    """Retry-aware phase resource totals folded from sealed attempt tombstones."""

    schema_version: Literal["abstrak-anytime-operational-totals.v1"] = (
        "abstrak-anytime-operational-totals.v1"
    )
    attempts: int = Field(ge=0)
    scientific_calls_consumed: int = Field(ge=0)
    provider_requests_submitted: int = Field(ge=0)
    possibly_charged_requests: int = Field(ge=0)
    known_input_tokens: int = Field(ge=0)
    known_cached_input_tokens: int = Field(ge=0)
    known_output_tokens: int = Field(ge=0)
    known_reasoning_tokens: int = Field(ge=0)
    usage_complete: bool
    compile_attempts: int = Field(ge=0)
    evaluation_attempts: int = Field(ge=0)
    provider_seconds: float = Field(ge=0)
    compile_seconds: float = Field(ge=0)
    evaluation_seconds: float = Field(ge=0)
    gpu_seconds: float = Field(ge=0)
    wall_seconds: float = Field(ge=0)

    @field_validator(
        "provider_seconds",
        "compile_seconds",
        "evaluation_seconds",
        "gpu_seconds",
        "wall_seconds",
    )
    @classmethod
    def elapsed_values_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("operational totals must be finite")
        return 0.0 if value == 0.0 else value


class AnytimePhaseCloseAudit(AnytimeModel):
    """Full phase-close result derived from freshly audited attempt directories."""

    schema_version: Literal["abstrak-anytime-phase-close-audit.v1"] = (
        "abstrak-anytime-phase-close-audit.v1"
    )
    phase_header_sha256: str = Field(pattern=SHA256_PATTERN)
    attempt_count: int = Field(ge=1)
    manifest_set_sha256: str = Field(pattern=SHA256_PATTERN)
    terminal_counts: tuple[tuple[str, int], ...]
    operational_totals: AnytimeOperationalTotals

    @field_validator("terminal_counts")
    @classmethod
    def terminal_counts_are_sorted(cls, values: tuple[tuple[str, int], ...]):
        if values != tuple(sorted(values)) or any(count < 0 for _, count in values):
            raise ValueError("terminal counts must be sorted and nonnegative")
        return values

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimePhaseJournalEntry(AnytimeModel):
    """One exclusively-created file in the phase-wide hash chain."""

    schema_version: Literal["abstrak-anytime-phase-journal-entry.v1"] = (
        "abstrak-anytime-phase-journal-entry.v1"
    )
    phase_header_sha256: str = Field(pattern=SHA256_PATTERN)
    sequence: int = Field(ge=1)
    previous_entry_sha256: str = Field(pattern=SHA256_PATTERN)
    kind: Literal["attempt_finalized", "phase_closed"]
    attempt_identity: AnytimeAttemptIdentity | None = None
    attempt_manifest_file_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    terminal_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    close_audit: AnytimePhaseCloseAudit | None = None

    @model_validator(mode="after")
    def payload_matches_kind(self) -> AnytimePhaseJournalEntry:
        attempt_fields = (
            self.attempt_identity,
            self.attempt_manifest_file_sha256,
            self.terminal_sha256,
        )
        if self.kind == "attempt_finalized":
            if any(value is None for value in attempt_fields) or self.close_audit is not None:
                raise ValueError("attempt-finalized journal entry has incomplete fields")
        elif any(value is not None for value in attempt_fields) or self.close_audit is None:
            raise ValueError("phase-close journal entry has incompatible fields")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimePhaseAudit(AnytimeModel):
    schema_version: Literal["abstrak-anytime-phase-audit.v1"] = "abstrak-anytime-phase-audit.v1"
    header: AnytimePhaseJournalHeader
    journal_head_sha256: str = Field(pattern=SHA256_PATTERN)
    attempts: tuple[AnytimeAttemptAudit, ...]
    closed: bool
    close_audit: AnytimePhaseCloseAudit | None = None


class AnytimeResumeAttempt(AnytimeModel):
    schema_version: Literal["abstrak-anytime-resume-attempt.v1"] = (
        "abstrak-anytime-resume-attempt.v1"
    )
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    attempt_index: int = Field(ge=1, le=2)
    terminal_kind: str
    retry_eligible: bool
    manifest_file_sha256: str = Field(pattern=SHA256_PATTERN)


class AnytimeResumeTrajectory(AnytimeModel):
    schema_version: Literal["abstrak-anytime-resume-trajectory.v1"] = (
        "abstrak-anytime-resume-trajectory.v1"
    )
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    state: Literal[
        "pending_primary",
        "pending_retry",
        "success",
        "scientific_failure",
        "controller_failure",
        "infrastructure_exhausted",
    ]
    attempts: tuple[AnytimeResumeAttempt, ...]


class AnytimeResumeIndex(AnytimeModel):
    """Disposable view rebuilt outside the immutable source tree."""

    schema_version: Literal["abstrak-anytime-resume-index.v1"] = "abstrak-anytime-resume-index.v1"
    phase_header_sha256: str = Field(pattern=SHA256_PATTERN)
    journal_head_sha256: str = Field(pattern=SHA256_PATTERN)
    phase_closed: bool
    trajectories: tuple[AnytimeResumeTrajectory, ...]

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _phase_source(root: str | Path, header: AnytimePhaseJournalHeader) -> Path:
    _validate_component("study_id", header.study_id)
    _validate_component("phase_id", header.phase_id)
    return Path(root).expanduser() / "source" / header.study_id / header.phase_id


def _journal_directory(root: str | Path, header: AnytimePhaseJournalHeader) -> Path:
    return _phase_source(root, header) / "journal"


def _entry_path(journal: Path, sequence: int) -> Path:
    return journal / f"{sequence:06d}.json"


def _load_entries(
    journal: Path,
    header: AnytimePhaseJournalHeader,
) -> tuple[AnytimePhaseJournalEntry, ...]:
    allowed = {"header.json"}
    removed_temp = False
    for artifact in tuple(journal.iterdir()):
        if artifact.is_symlink():
            raise AnytimePhaseJournalError("symbolic links are forbidden in phase journal")
        if artifact.is_dir():
            raise AnytimePhaseJournalError("phase journal cannot contain subdirectories")
        if _TEMP_PATTERN.fullmatch(artifact.name):
            try:
                artifact.unlink()
            except OSError as error:
                raise AnytimePhaseJournalError(
                    "cannot discard unpublished phase-journal temporary file"
                ) from error
            removed_temp = True
            continue
        if artifact.name == "header.json":
            continue
        if not (
            len(artifact.name) == 11
            and artifact.name[:6].isdigit()
            and artifact.name.endswith(".json")
        ):
            raise AnytimePhaseJournalError("phase journal contains an unexpected file")
        allowed.add(artifact.name)
    if removed_temp:
        _fsync_directory(journal)
    paths = sorted(journal.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))
    if {path.name for path in paths} | {"header.json"} != allowed:
        raise AnytimePhaseJournalError("phase journal inventory is incomplete")
    entries = tuple(_load_model(path, AnytimePhaseJournalEntry) for path in paths)
    previous = header.sha256
    for sequence, entry in enumerate(entries, start=1):
        if entry.sequence != sequence:
            raise AnytimePhaseJournalError("phase journal sequence is not contiguous")
        if entry.phase_header_sha256 != header.sha256:
            raise AnytimePhaseJournalError("phase journal entry belongs to another header")
        if entry.previous_entry_sha256 != previous:
            raise AnytimePhaseJournalError("phase journal hash chain is broken")
        if entry.kind == "phase_closed" and sequence != len(entries):
            raise AnytimePhaseJournalError("phase journal continues after close")
        previous = entry.sha256
    return entries


def _fold_operational(attempts: tuple[AnytimeAttemptAudit, ...]) -> AnytimeOperationalTotals:
    snapshots = tuple(attempt.terminal.resource_snapshot for attempt in attempts)
    return AnytimeOperationalTotals(
        attempts=len(attempts),
        scientific_calls_consumed=sum(item.scientific_calls_consumed for item in snapshots),
        provider_requests_submitted=sum(item.provider_requests_submitted for item in snapshots),
        possibly_charged_requests=sum(item.possibly_charged_requests for item in snapshots),
        known_input_tokens=sum(item.known_input_tokens for item in snapshots),
        known_cached_input_tokens=sum(item.known_cached_input_tokens for item in snapshots),
        known_output_tokens=sum(item.known_output_tokens for item in snapshots),
        known_reasoning_tokens=sum(item.known_reasoning_tokens for item in snapshots),
        usage_complete=all(item.usage_complete for item in snapshots),
        compile_attempts=sum(item.compile_attempts for item in snapshots),
        evaluation_attempts=sum(item.evaluation_attempts for item in snapshots),
        provider_seconds=math.fsum(item.provider_seconds for item in snapshots),
        compile_seconds=math.fsum(item.compile_seconds for item in snapshots),
        evaluation_seconds=math.fsum(item.evaluation_seconds for item in snapshots),
        gpu_seconds=math.fsum(item.gpu_seconds for item in snapshots),
        wall_seconds=math.fsum(item.wall_seconds for item in snapshots),
    )


def _build_close_audit(
    header: AnytimePhaseJournalHeader,
    attempts: tuple[AnytimeAttemptAudit, ...],
) -> AnytimePhaseCloseAudit:
    manifest_set = tuple(
        sorted(
            (
                attempt.identity.trajectory_id,
                attempt.identity.infrastructure_attempt_index,
                attempt.manifest_file_sha256,
            )
            for attempt in attempts
        )
    )
    counts = Counter(attempt.terminal.terminal_kind for attempt in attempts)
    return AnytimePhaseCloseAudit(
        phase_header_sha256=header.sha256,
        attempt_count=len(attempts),
        manifest_set_sha256=artifact_payload_sha256(manifest_set),
        terminal_counts=tuple(sorted(counts.items())),
        operational_totals=_fold_operational(attempts),
    )


def _attempt_path(root: str | Path, identity: AnytimeAttemptIdentity) -> Path:
    return (
        Path(root).expanduser()
        / "source"
        / identity.study_id
        / identity.phase_id
        / "attempts"
        / identity.trajectory_id
        / identity.attempt_name
    )


def _validate_attempt_sequence(
    header: AnytimePhaseJournalHeader,
    attempts: tuple[AnytimeAttemptAudit, ...],
) -> None:
    expected = set(header.expected_trajectory_ids)
    grouped: dict[str, list[AnytimeAttemptAudit]] = {}
    for attempt in attempts:
        identity = attempt.identity
        if identity.study_id != header.study_id or identity.phase_id != header.phase_id:
            raise AnytimePhaseJournalError("journaled attempt belongs to another phase")
        if identity.trajectory_id not in expected:
            raise AnytimePhaseJournalError("journaled attempt is outside expected population")
        if identity.infrastructure_attempt_index > header.max_attempts_per_trajectory:
            raise AnytimePhaseJournalError("attempt exceeds the phase infrastructure policy")
        grouped.setdefault(identity.trajectory_id, []).append(attempt)
    for trajectory_id, values in grouped.items():
        indices = tuple(item.identity.infrastructure_attempt_index for item in values)
        if indices not in {(1,), (1, 2)}:
            raise AnytimePhaseJournalError("trajectory attempt sequence is not primary then retry")
        if len(values) == 2:
            primary, retry = values
            if not primary.terminal.retry_eligible:
                raise AnytimePhaseJournalError("retry follows a non-retryable primary attempt")
            if retry.identity.prior_attempt_manifest_sha256 != primary.manifest_file_sha256:
                raise AnytimePhaseJournalError("retry does not bind the primary manifest")
        if values[-1].terminal.retry_eligible:
            raise AnytimePhaseJournalError(
                f"trajectory {trajectory_id} still requires its bounded retry"
            )
    if set(grouped) != expected:
        raise AnytimePhaseJournalError("phase cannot close before every trajectory is terminal")


class AnytimePhaseJournal:
    """Append-only journal whose entries reference already-promoted attempts."""

    def __init__(
        self,
        root: str | Path,
        header: AnytimePhaseJournalHeader,
        *,
        create: bool,
    ) -> None:
        self.root = Path(root).expanduser()
        self.header = AnytimePhaseJournalHeader.model_validate_json(header.model_dump_json())
        self.directory = _journal_directory(self.root, self.header)
        if create:
            self.directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                self.directory.mkdir(mode=0o700)
            except FileExistsError as error:
                raise AnytimePhaseJournalError("phase journal already exists") from error
            _fsync_directory(self.directory.parent)
            path = _exclusive_json(self.directory, "header.json", self.header)
            path.chmod(0o400)
            _fsync_directory(self.directory)
        else:
            if self.directory.is_symlink() or not self.directory.is_dir():
                raise AnytimePhaseJournalError("phase journal is missing or unsafe")
            observed = _load_model(self.directory / "header.json", AnytimePhaseJournalHeader)
            if observed != self.header:
                raise AnytimePhaseJournalError("phase journal header differs from trust anchor")

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        study_id: str,
        phase_id: str,
        phase_execution_sha256: str,
        expected_trajectory_ids: tuple[str, ...],
        max_attempts_per_trajectory: Literal[1, 2],
    ) -> AnytimePhaseJournal:
        return cls(
            root,
            AnytimePhaseJournalHeader(
                study_id=study_id,
                phase_id=phase_id,
                phase_execution_sha256=phase_execution_sha256,
                expected_trajectory_ids=expected_trajectory_ids,
                max_attempts_per_trajectory=max_attempts_per_trajectory,
            ),
            create=True,
        )

    @classmethod
    def open(
        cls,
        root: str | Path,
        header: AnytimePhaseJournalHeader,
    ) -> AnytimePhaseJournal:
        return cls(root, header, create=False)

    def append_attempt(
        self,
        identity: AnytimeAttemptIdentity,
        *,
        loop: AnytimeLoopPolicy,
        base_prompt: tuple[ChatMessage, ...],
    ) -> AnytimePhaseJournalEntry:
        """Audit a final fact first, then append its immutable journal reference."""

        identity = AnytimeAttemptIdentity.model_validate_json(identity.model_dump_json())
        if (
            self.header.max_attempts_per_trajectory
            != loop.infrastructure.max_attempts_per_trajectory
        ):
            raise AnytimePhaseJournalError("phase and loop infrastructure policies differ")
        if identity.study_id != self.header.study_id or identity.phase_id != self.header.phase_id:
            raise AnytimePhaseJournalError("attempt identity belongs to another phase")
        if identity.trajectory_id not in self.header.expected_trajectory_ids:
            raise AnytimePhaseJournalError("attempt is outside the expected trajectory population")
        if identity.infrastructure_attempt_index > self.header.max_attempts_per_trajectory:
            raise AnytimePhaseJournalError("attempt exceeds the phase infrastructure policy")
        entries = _load_entries(self.directory, self.header)
        if entries and entries[-1].kind == "phase_closed":
            raise AnytimePhaseJournalError("cannot append an attempt after phase close")
        attempt = audit_anytime_attempt(
            _attempt_path(self.root, identity),
            loop=loop,
            base_prompt=base_prompt,
        )
        if attempt.identity != identity:
            raise AnytimePhaseJournalError("attempt audit identity differs from requested identity")
        for entry in entries:
            if entry.attempt_identity == identity:
                if (
                    entry.attempt_manifest_file_sha256 == attempt.manifest_file_sha256
                    and entry.terminal_sha256 == attempt.terminal.sha256
                ):
                    return entry
                raise AnytimePhaseJournalError("journal already binds different attempt bytes")
        prior_attempts = tuple(
            audit_anytime_attempt(
                _attempt_path(self.root, entry.attempt_identity),
                loop=loop,
                base_prompt=base_prompt,
            )
            for entry in entries
            if (
                entry.kind == "attempt_finalized"
                and entry.attempt_identity is not None
                and entry.attempt_identity.trajectory_id == identity.trajectory_id
            )
        )
        # This validates retry order while permitting an as-yet-unconsumed
        # retry-eligible primary at the tail.
        same = tuple(
            item for item in prior_attempts if item.identity.trajectory_id == identity.trajectory_id
        )
        if identity.infrastructure_attempt_index == 1 and same:
            raise AnytimePhaseJournalError("primary attempt is already journaled")
        if identity.infrastructure_attempt_index == 2:
            if len(same) != 1 or not same[0].terminal.retry_eligible:
                raise AnytimePhaseJournalError("retry has no retryable journaled primary")
            if identity.prior_attempt_manifest_sha256 != same[0].manifest_file_sha256:
                raise AnytimePhaseJournalError("retry does not bind journaled primary manifest")
        previous = self.header.sha256 if not entries else entries[-1].sha256
        entry = AnytimePhaseJournalEntry(
            phase_header_sha256=self.header.sha256,
            sequence=len(entries) + 1,
            previous_entry_sha256=previous,
            kind="attempt_finalized",
            attempt_identity=identity,
            attempt_manifest_file_sha256=attempt.manifest_file_sha256,
            terminal_sha256=attempt.terminal.sha256,
        )
        path = _exclusive_json(
            self.directory,
            _entry_path(self.directory, entry.sequence).name,
            entry,
        )
        path.chmod(0o400)
        _fsync_directory(self.directory)
        _increment_anytime_resume_index(
            self.root,
            self.header,
            entry=entry,
            attempt=attempt,
            loop=loop,
            base_prompt=base_prompt,
        )
        return entry

    def close(
        self,
        *,
        loop: AnytimeLoopPolicy,
        base_prompt: tuple[ChatMessage, ...],
    ) -> AnytimePhaseJournalEntry:
        """Freshly audit all source attempts before writing the terminal journal entry."""

        audit = audit_anytime_phase(
            self.root,
            self.header,
            loop=loop,
            base_prompt=base_prompt,
            require_closed=False,
        )
        if audit.closed:
            entries = _load_entries(self.directory, self.header)
            return entries[-1]
        _validate_attempt_sequence(self.header, audit.attempts)
        incomplete = tuple(
            _phase_source(self.root, self.header).glob("attempts/*/attempt-[0-9][0-9].incomplete")
        )
        if incomplete:
            raise AnytimePhaseJournalError("phase has unresolved incomplete attempts")
        close_audit = _build_close_audit(self.header, audit.attempts)
        entries = _load_entries(self.directory, self.header)
        entry = AnytimePhaseJournalEntry(
            phase_header_sha256=self.header.sha256,
            sequence=len(entries) + 1,
            previous_entry_sha256=(self.header.sha256 if not entries else entries[-1].sha256),
            kind="phase_closed",
            close_audit=close_audit,
        )
        path = _exclusive_json(
            self.directory,
            _entry_path(self.directory, entry.sequence).name,
            entry,
        )
        path.chmod(0o400)
        _fsync_directory(self.directory)
        self.directory.chmod(0o500)
        refresh_anytime_resume_index(
            self.root,
            self.header,
            loop=loop,
            base_prompt=base_prompt,
        )
        return entry


def _source_final_directories(root: Path, header: AnytimePhaseJournalHeader) -> set[Path]:
    attempts_root = _phase_source(root, header) / "attempts"
    if not attempts_root.exists():
        return set()
    finals: set[Path] = set()
    for trajectory in attempts_root.iterdir():
        if trajectory.is_symlink() or not trajectory.is_dir():
            raise AnytimePhaseJournalError("phase contains an unsafe trajectory path")
        _validate_component("trajectory_id", trajectory.name)
        for candidate in trajectory.iterdir():
            if candidate.is_symlink() or not candidate.is_dir():
                raise AnytimePhaseJournalError("phase contains an unsafe attempt path")
            if candidate.name.endswith(".incomplete"):
                base = candidate.name.removesuffix(".incomplete")
                if not (base.startswith("attempt-") and len(base) == 10 and base[-2:].isdigit()):
                    raise AnytimePhaseJournalError(
                        "phase contains an unexpected incomplete directory"
                    )
                continue
            if not (
                candidate.name.startswith("attempt-")
                and len(candidate.name) == 10
                and candidate.name[-2:].isdigit()
            ):
                raise AnytimePhaseJournalError("phase contains an unexpected attempt directory")
            finals.add(candidate)
    return finals


def audit_anytime_phase(
    root: str | Path,
    header: AnytimePhaseJournalHeader,
    *,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
    require_closed: bool = True,
) -> AnytimePhaseAudit:
    """Replay the journal and freshly audit every referenced final directory."""

    if header.max_attempts_per_trajectory != loop.infrastructure.max_attempts_per_trajectory:
        raise AnytimePhaseJournalError("phase and loop infrastructure policies differ")
    journal = _journal_directory(root, header)
    observed_header = _load_model(journal / "header.json", AnytimePhaseJournalHeader)
    if observed_header != header:
        raise AnytimePhaseJournalError("phase journal header differs from trust anchor")
    entries = _load_entries(journal, header)
    closed = bool(entries and entries[-1].kind == "phase_closed")
    if require_closed and not closed:
        raise AnytimePhaseJournalError("phase journal is not closed")
    attempt_entries = tuple(entry for entry in entries if entry.kind == "attempt_finalized")
    attempts: list[AnytimeAttemptAudit] = []
    for entry in attempt_entries:
        assert entry.attempt_identity is not None
        audit = audit_anytime_attempt(
            _attempt_path(root, entry.attempt_identity),
            loop=loop,
            base_prompt=base_prompt,
        )
        if (
            audit.manifest_file_sha256 != entry.attempt_manifest_file_sha256
            or audit.terminal.sha256 != entry.terminal_sha256
        ):
            raise AnytimePhaseJournalError("journal entry differs from fresh attempt audit")
        attempts.append(audit)
    journaled_paths = {_attempt_path(root, item.identity) for item in attempts}
    if _source_final_directories(Path(root), header) != journaled_paths:
        raise AnytimePhaseJournalError("phase has final attempts missing from its journal")
    close_audit = entries[-1].close_audit if closed else None
    if closed:
        _validate_attempt_sequence(header, tuple(attempts))
        expected = _build_close_audit(header, tuple(attempts))
        if close_audit != expected:
            raise AnytimePhaseJournalError("phase-close audit differs from fresh source audit")
    head = header.sha256 if not entries else entries[-1].sha256
    return AnytimePhaseAudit(
        header=header,
        journal_head_sha256=head,
        attempts=tuple(attempts),
        closed=closed,
        close_audit=close_audit,
    )


def _trajectory_state(
    attempts: tuple[AnytimeAttemptAudit, ...],
) -> str:
    if not attempts:
        return "pending_primary"
    terminal = attempts[-1].terminal
    if terminal.retry_eligible:
        return "pending_retry"
    if terminal.terminal_kind == "success":
        return "success"
    if terminal.terminal_kind == "scientific_failure":
        return "scientific_failure"
    if terminal.terminal_kind == "controller_failure":
        return "controller_failure"
    return "infrastructure_exhausted"


def build_anytime_resume_index(audit: AnytimePhaseAudit) -> AnytimeResumeIndex:
    """Build a disposable phase view solely from a verified source audit."""

    by_trajectory: dict[str, list[AnytimeAttemptAudit]] = {
        trajectory_id: [] for trajectory_id in audit.header.expected_trajectory_ids
    }
    for attempt in audit.attempts:
        by_trajectory[attempt.identity.trajectory_id].append(attempt)
    trajectories = tuple(
        AnytimeResumeTrajectory(
            trajectory_id=trajectory_id,
            state=_trajectory_state(tuple(attempts)),
            attempts=tuple(
                AnytimeResumeAttempt(
                    trajectory_id=trajectory_id,
                    attempt_index=item.identity.infrastructure_attempt_index,
                    terminal_kind=item.terminal.terminal_kind,
                    retry_eligible=item.terminal.retry_eligible,
                    manifest_file_sha256=item.manifest_file_sha256,
                )
                for item in attempts
            ),
        )
        for trajectory_id, attempts in sorted(by_trajectory.items())
    )
    return AnytimeResumeIndex(
        phase_header_sha256=audit.header.sha256,
        journal_head_sha256=audit.journal_head_sha256,
        phase_closed=audit.closed,
        trajectories=trajectories,
    )


def _resume_index_path(root: str | Path, header: AnytimePhaseJournalHeader) -> Path:
    return (
        Path(root).expanduser()
        / "derived"
        / header.study_id
        / header.phase_id
        / "resume-index.json"
    )


def _increment_anytime_resume_index(
    root: str | Path,
    header: AnytimePhaseJournalHeader,
    *,
    entry: AnytimePhaseJournalEntry,
    attempt: AnytimeAttemptAudit,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
) -> Path:
    """Advance the disposable index by one freshly audited immutable attempt."""

    path = _resume_index_path(root, header)
    try:
        if path.is_symlink() or not path.is_file():
            raise AnytimePhaseJournalError("derived resume index is absent")
        current = AnytimeResumeIndex.model_validate_json(path.read_bytes())
        if current.phase_header_sha256 != header.sha256:
            raise AnytimePhaseJournalError("derived resume index belongs to another phase")
        if current.phase_closed or current.journal_head_sha256 != entry.previous_entry_sha256:
            raise AnytimePhaseJournalError("derived resume index is stale")
        trajectories = list(current.trajectories)
        position = next(
            index
            for index, trajectory in enumerate(trajectories)
            if trajectory.trajectory_id == attempt.identity.trajectory_id
        )
        previous = trajectories[position]
        resume_attempt = AnytimeResumeAttempt(
            trajectory_id=attempt.identity.trajectory_id,
            attempt_index=attempt.identity.infrastructure_attempt_index,
            terminal_kind=attempt.terminal.terminal_kind,
            retry_eligible=attempt.terminal.retry_eligible,
            manifest_file_sha256=attempt.manifest_file_sha256,
        )
        next_attempts = (*previous.attempts, resume_attempt)
        trajectories[position] = AnytimeResumeTrajectory(
            trajectory_id=previous.trajectory_id,
            state=_trajectory_state(tuple(item for item in (attempt,))),
            attempts=next_attempts,
        )
        updated = AnytimeResumeIndex(
            phase_header_sha256=header.sha256,
            journal_head_sha256=entry.sha256,
            phase_closed=False,
            trajectories=tuple(trajectories),
        )
        _replace_derived_json(path, updated)
        return path
    except (AnytimePhaseJournalError, StopIteration, ValueError):
        # Derived state is never authoritative.  Missing, stale, or malformed
        # indexes are discarded logically and rebuilt from fresh source audits.
        return refresh_anytime_resume_index(
            root,
            header,
            loop=loop,
            base_prompt=base_prompt,
        )


def _replace_derived_json(path: Path, value: AnytimeModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    content = canonical_artifact_bytes(value)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise AnytimePhaseJournalError("derived index write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    path.chmod(0o600)
    _fsync_directory(path.parent)


def refresh_anytime_resume_index(
    root: str | Path,
    header: AnytimePhaseJournalHeader,
    *,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
) -> Path:
    """Atomically replace a derived index without touching source artifacts."""

    audit = audit_anytime_phase(
        root,
        header,
        loop=loop,
        base_prompt=base_prompt,
        require_closed=False,
    )
    index = build_anytime_resume_index(audit)
    path = _resume_index_path(root, header)
    source = _phase_source(root, header).resolve()
    resolved_parent = path.parent.resolve()
    if resolved_parent == source or source in resolved_parent.parents:
        raise AnytimePhaseJournalError("derived index path overlaps immutable source artifacts")
    _replace_derived_json(path, index)
    if _file_sha256(path) != hashlib.sha256(canonical_artifact_bytes(index)).hexdigest():
        raise AnytimePhaseJournalError("derived index failed its post-write checksum")
    return path
