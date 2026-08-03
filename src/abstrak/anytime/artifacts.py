"""Crash-safe immutable artifacts for offline anytime trajectory attempts.

The store in this module is deliberately transport- and worker-free.  A caller
must persist a dispatch intent *before* it invokes a provider, then persist the
observed provider fact, worker fact, derived ledger record, and checkpoint in
that order.  Recovery never submits a request: an intent without a durable
terminal provider fact becomes a conservative ambiguous-submission record.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from abstrak.anytime.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AnytimeLoopPolicy,
    AnytimeModel,
    AnytimeResourceSnapshot,
)
from abstrak.anytime.ledger import (
    AnytimeCandidateOutcome,
    AnytimeCheckpointRecord,
    AnytimeLedgerHeader,
    AnytimePreparedTurn,
    AnytimeProviderAmbiguousSubmission,
    AnytimeProviderObservation,
    AnytimeProviderSubmittedError,
    AnytimeProviderSuccess,
    AnytimeTokenUsage,
    AnytimeTurnRecord,
    append_anytime_turn,
    rebuild_anytime_checkpoints,
    verify_anytime_ledger,
)
from abstrak.providers.contracts import ChatMessage, sha256_json
from abstrak.providers.native_contracts import (
    NativeNormalizedError,
    NativeNormalizedResponse,
)


class AnytimeArtifactError(RuntimeError):
    """Raised when an attempt artifact is unsafe, incomplete, or inconsistent."""


class AnytimeInjectedCrash(RuntimeError):
    """Test-only crash signal raised by an injected fault callback."""


ArtifactFaultPoint = Literal[
    "after_attempt_create",
    "after_dispatch_intent",
    "provider_terminal_before_publish",
    "after_provider_terminal",
    "worker_evaluation_before_publish",
    "after_worker_evaluation",
    "after_turn_record",
    "checkpoint_before_publish",
    "after_checkpoint",
    "seal_before_manifest_publish",
    "after_manifest_publish",
    "after_atomic_promotion",
]
FaultInjector = Callable[[ArtifactFaultPoint], None]

_SAFE_COMPONENT = re.compile(IDENTIFIER_PATTERN)
_TEMP_PATTERN = re.compile(r"^\..+\.[0-9a-f]{32}\.tmp$")


def _payload(value: Any) -> Any:
    if isinstance(value, AnytimeModel):
        return _payload(value.model_dump(mode="json"))
    if hasattr(value, "model_dump"):
        return _payload(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_payload(item) for item in value]
    return value


def _is_negative_zero(value: Any) -> bool:
    return isinstance(value, float) and value == 0.0 and math.copysign(1.0, value) < 0


def _contains_negative_zero(value: Any) -> bool:
    if _is_negative_zero(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_negative_zero(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_negative_zero(item) for item in value)
    return False


def _canonicalize_signed_zero(value: Any) -> Any:
    if _is_negative_zero(value):
        return 0.0
    if isinstance(value, Mapping):
        return {str(key): _canonicalize_signed_zero(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_signed_zero(item) for item in value]
    return value


def _canonical_nonnegative_float(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _canonicalized_model(value: Any, model: type[Any]) -> Any:
    payload = _canonicalize_signed_zero(value.model_dump(mode="json"))
    return model.model_validate_json(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def canonical_artifact_bytes(value: Any) -> bytes:
    """Render the exact UTF-8 bytes hashed and written for JSON artifacts."""

    try:
        rendered = json.dumps(
            _payload(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise AnytimeArtifactError("artifact payload is not finite canonical JSON") from error
    return f"{rendered}\n".encode()


def artifact_payload_sha256(value: Any) -> str:
    """Return the digest a canonical JSON artifact will have on disk."""

    return hashlib.sha256(canonical_artifact_bytes(value)).hexdigest()


def _finite_nonnegative(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return 0.0 if value == 0.0 else value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_component(label: str, value: str) -> None:
    if _SAFE_COMPONENT.fullmatch(value) is None:
        raise AnytimeArtifactError(f"{label} must be one safe path component")


def _resolve_inside(root: Path, relative: str | Path) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise AnytimeArtifactError(f"unsafe artifact path: {relative_path}")
    destination = root / relative_path
    cursor = root
    if cursor.is_symlink():
        raise AnytimeArtifactError("artifact root cannot be a symbolic link")
    for component in relative_path.parent.parts:
        cursor /= component
        if cursor.is_symlink():
            raise AnytimeArtifactError("symbolic links are forbidden in artifact paths")
    resolved_root = root.resolve()
    resolved_parent = destination.parent.resolve()
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        raise AnytimeArtifactError(f"artifact path escapes attempt: {relative_path}")
    return destination


def _mkdir_chain(directory: Path, stop: Path) -> None:
    missing: list[Path] = []
    cursor = directory
    while cursor != stop and not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if cursor != stop and stop not in cursor.parents:
        raise AnytimeArtifactError("artifact directory escapes its source root")
    for path in reversed(missing):
        path.mkdir(mode=0o700)
        _fsync_directory(path.parent)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically promote without permitting a race to replace final bytes."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:  # pragma: no cover - Linux A100 workers provide renameat2
        if destination.exists() or destination.is_symlink():
            raise AnytimeArtifactError("final attempt path already exists")
        os.rename(source, destination)
        return
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise AnytimeArtifactError("final attempt path already exists")
    if error_number in {errno.ENOSYS, errno.EINVAL}:  # pragma: no cover - portability fallback
        if destination.exists() or destination.is_symlink():
            raise AnytimeArtifactError("final attempt path already exists")
        os.rename(source, destination)
        return
    raise AnytimeArtifactError(f"atomic attempt promotion failed: {os.strerror(error_number)}")


def _exclusive_bytes(
    root: Path,
    relative: str | Path,
    content: bytes,
    *,
    secrets: tuple[bytes, ...] = (),
    fault: FaultInjector | None = None,
    before_publish: ArtifactFaultPoint | None = None,
) -> Path:
    if any(secret and secret in content for secret in secrets):
        raise AnytimeArtifactError("refusing to persist credential material")
    destination = _resolve_inside(root, relative)
    _mkdir_chain(destination.parent, root)
    if destination.exists() or destination.is_symlink():
        raise AnytimeArtifactError(f"immutable artifact already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    published = False
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise AnytimeArtifactError("artifact write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if fault is not None and before_publish is not None:
            fault(before_publish)
        # A hard-link publication is atomic and, unlike replace(), cannot
        # overwrite a previously published immutable fact.
        os.link(temporary, destination)
        published = True
        destination.chmod(0o600)
        temporary.unlink()
        _fsync_directory(destination.parent)
        return destination
    except Exception:
        # A deliberately injected crash leaves the fsynced temp file behind,
        # matching process-death semantics.  Recovery removes only files that
        # match the private temp naming scheme and never treats them as facts.
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if published and temporary.exists():
            temporary.unlink()


def _exclusive_json(
    root: Path,
    relative: str | Path,
    value: Any,
    *,
    secrets: tuple[bytes, ...] = (),
    fault: FaultInjector | None = None,
    before_publish: ArtifactFaultPoint | None = None,
) -> Path:
    return _exclusive_bytes(
        root,
        relative,
        canonical_artifact_bytes(value),
        secrets=secrets,
        fault=fault,
        before_publish=before_publish,
    )


class AnytimeAttemptIdentity(AnytimeModel):
    """Stable path and execution identity for one whole-trajectory attempt."""

    schema_version: Literal["abstrak-anytime-attempt-identity.v1"] = (
        "abstrak-anytime-attempt-identity.v1"
    )
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    phase_id: str = Field(pattern=IDENTIFIER_PATTERN)
    trajectory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    infrastructure_attempt_index: int = Field(ge=1, le=2)
    trajectory_execution_sha256: str = Field(pattern=SHA256_PATTERN)
    prior_attempt_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def retry_link_matches_attempt(self) -> AnytimeAttemptIdentity:
        if (self.infrastructure_attempt_index == 1) != (self.prior_attempt_manifest_sha256 is None):
            raise ValueError("only a retry attempt may bind a prior attempt manifest")
        return self

    @property
    def attempt_name(self) -> str:
        return f"attempt-{self.infrastructure_attempt_index:02d}"

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeAttemptHeader(AnytimeModel):
    """The first durable fact in an attempt staging directory."""

    schema_version: Literal["abstrak-anytime-attempt-header.v1"] = (
        "abstrak-anytime-attempt-header.v1"
    )
    identity: AnytimeAttemptIdentity
    ledger_header: AnytimeLedgerHeader

    @model_validator(mode="after")
    def headers_are_bound(self) -> AnytimeAttemptHeader:
        identity = self.identity
        ledger = self.ledger_header
        if ledger.trajectory_id != identity.trajectory_id:
            raise ValueError("attempt and ledger trajectory IDs differ")
        if ledger.infrastructure_attempt_index != identity.infrastructure_attempt_index:
            raise ValueError("attempt and ledger attempt indices differ")
        if ledger.trajectory_execution_sha256 != identity.trajectory_execution_sha256:
            raise ValueError("attempt and ledger execution hashes differ")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeDispatchIntent(AnytimeModel):
    """Fsynced proof that the caller is about to cross the transport boundary."""

    schema_version: Literal["abstrak-anytime-dispatch-intent.v1"] = (
        "abstrak-anytime-dispatch-intent.v1"
    )
    ledger_header_sha256: str = Field(pattern=SHA256_PATTERN)
    prepared: AnytimePreparedTurn
    logical_request_sha256: str = Field(pattern=SHA256_PATTERN)
    request_submitted: Literal[False] = False
    possibly_charged: Literal[False] = False

    @model_validator(mode="after")
    def prepared_request_is_bound(self) -> AnytimeDispatchIntent:
        if self.prepared.header_sha256 != self.ledger_header_sha256:
            raise ValueError("dispatch intent belongs to another attempt header")
        if sha256_json(self.prepared.logical_request) != self.logical_request_sha256:
            raise ValueError("dispatch intent request hash differs from prepared request")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimePersistedProviderObservation(AnytimeModel):
    """Durable provider terminal fact published after one dispatch intent."""

    schema_version: Literal["abstrak-anytime-persisted-provider-observation.v1"] = (
        "abstrak-anytime-persisted-provider-observation.v1"
    )
    dispatch_intent_sha256: str = Field(pattern=SHA256_PATTERN)
    observation: AnytimeProviderObservation
    observed_wall_seconds: float = Field(ge=0)

    @field_validator("observed_wall_seconds")
    @classmethod
    def wall_time_is_finite(cls, value: float) -> float:
        return _finite_nonnegative(value, "provider observation wall time")

    @model_validator(mode="after")
    def elapsed_time_covers_provider(self) -> AnytimePersistedProviderObservation:
        if self.observed_wall_seconds < self.observation.provider_seconds:
            raise ValueError("provider observation wall time is below provider time")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeWorkerEvaluationArtifact(AnytimeModel):
    """Strict worker/evaluator source fact from which candidate state is derived."""

    schema_version: Literal["abstrak-anytime-worker-evaluation.v1"] = (
        "abstrak-anytime-worker-evaluation.v1"
    )
    provider_observation_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluator_id: str = Field(pattern=IDENTIFIER_PATTERN)
    evaluator_execution_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate: AnytimeCandidateOutcome
    observed_wall_seconds: float = Field(ge=0)
    qualification_artifact_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )

    @field_validator("observed_wall_seconds")
    @classmethod
    def wall_time_is_finite(cls, value: float) -> float:
        return _finite_nonnegative(value, "worker observation wall time")

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimePersistedEvaluation(AnytimeModel):
    """Redundant derived view over one strict worker/evaluator source fact."""

    schema_version: Literal["abstrak-anytime-persisted-evaluation.v1"] = (
        "abstrak-anytime-persisted-evaluation.v1"
    )
    provider_observation_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluator_id: str = Field(pattern=IDENTIFIER_PATTERN)
    evaluator_execution_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate: AnytimeCandidateOutcome
    worker_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    observed_wall_seconds: float = Field(ge=0)
    qualification_artifact_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )

    @field_validator("observed_wall_seconds")
    @classmethod
    def wall_time_is_finite(cls, value: float) -> float:
        return _finite_nonnegative(value, "persisted evaluation wall time")

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _persisted_evaluation_from_worker(
    worker: AnytimeWorkerEvaluationArtifact,
    worker_artifact_sha256: str,
) -> AnytimePersistedEvaluation:
    return AnytimePersistedEvaluation(
        provider_observation_sha256=worker.provider_observation_sha256,
        evaluator_id=worker.evaluator_id,
        evaluator_execution_sha256=worker.evaluator_execution_sha256,
        candidate=worker.candidate,
        worker_artifact_sha256=worker_artifact_sha256,
        observed_wall_seconds=worker.observed_wall_seconds,
        qualification_artifact_sha256=worker.qualification_artifact_sha256,
    )


AttemptTerminalKind = Literal[
    "success",
    "scientific_failure",
    "infrastructure_failure",
    "controller_failure",
]


class AnytimeAttemptTerminal(AnytimeModel):
    """Immutable tombstone that classifies every sealed attempt."""

    schema_version: Literal["abstrak-anytime-attempt-terminal.v1"] = (
        "abstrak-anytime-attempt-terminal.v1"
    )
    attempt_header_sha256: str = Field(pattern=SHA256_PATTERN)
    terminal_kind: AttemptTerminalKind
    reason: str = Field(pattern=IDENTIFIER_PATTERN)
    request_submitted: bool
    possibly_charged: bool
    retry_eligible: bool
    ledger_records: int = Field(ge=0, le=12)
    ledger_head_sha256: str = Field(pattern=SHA256_PATTERN)
    resource_snapshot: AnytimeResourceSnapshot

    @model_validator(mode="after")
    def terminal_flags_are_safe(self) -> AnytimeAttemptTerminal:
        if self.possibly_charged and not self.request_submitted:
            raise ValueError("an unsubmitted terminal request cannot be possibly charged")
        if self.terminal_kind in {"success", "scientific_failure"}:
            if not self.request_submitted:
                raise ValueError("scientific terminal states require a submitted request")
        elif self.request_submitted or self.possibly_charged:
            raise ValueError("local failure terminal states must be unsubmitted")
        if self.retry_eligible and (
            self.terminal_kind != "infrastructure_failure"
            or self.request_submitted
            or self.possibly_charged
        ):
            raise ValueError("only an unsubmitted infrastructure failure may be retried")
        if self.terminal_kind != "infrastructure_failure" and self.retry_eligible:
            raise ValueError("scientific/controller terminal states cannot be retried")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeArtifactChecksum(AnytimeModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0)


class AnytimeAttemptManifest(AnytimeModel):
    """Exact inventory of every durable fact except the manifest itself."""

    schema_version: Literal["abstrak-anytime-attempt-manifest.v1"] = (
        "abstrak-anytime-attempt-manifest.v1"
    )
    attempt_header_sha256: str = Field(pattern=SHA256_PATTERN)
    terminal_sha256: str = Field(pattern=SHA256_PATTERN)
    files: tuple[AnytimeArtifactChecksum, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def inventory_is_sorted_and_unique(self) -> AnytimeAttemptManifest:
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("attempt manifest paths must be sorted and unique")
        if "manifest.json" in paths:
            raise ValueError("attempt manifest cannot inventory itself")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeAttemptAudit(AnytimeModel):
    schema_version: Literal["abstrak-anytime-attempt-audit.v1"] = "abstrak-anytime-attempt-audit.v1"
    identity: AnytimeAttemptIdentity
    directory: str
    manifest_file_sha256: str = Field(pattern=SHA256_PATTERN)
    terminal: AnytimeAttemptTerminal
    ledger_records: int = Field(ge=0, le=12)


class AnytimeRecoveryDecision(AnytimeModel):
    """Offline recovery result; the transport action is structurally forbidden."""

    schema_version: Literal["abstrak-anytime-recovery-decision.v1"] = (
        "abstrak-anytime-recovery-decision.v1"
    )
    action: Literal["missing", "resume_worker_evaluation", "finalized", "complete"]
    directory: str | None = None
    scientific_call_index: int | None = Field(default=None, ge=1, le=12)
    provider_replay_allowed: Literal[False] = False
    request_submitted: bool
    possibly_charged: bool


def anytime_attempt_directory(
    root: str | Path,
    identity: AnytimeAttemptIdentity,
    *,
    incomplete: bool,
) -> Path:
    """Resolve the independent anytime-v1 source path for an attempt."""

    for label, value in (
        ("study_id", identity.study_id),
        ("phase_id", identity.phase_id),
        ("trajectory_id", identity.trajectory_id),
    ):
        _validate_component(label, value)
    name = identity.attempt_name + (".incomplete" if incomplete else "")
    return (
        Path(root).expanduser()
        / "source"
        / identity.study_id
        / identity.phase_id
        / "attempts"
        / identity.trajectory_id
        / name
    )


def _load_model(path: Path, model: type[AnytimeModel]) -> Any:
    if path.is_symlink() or not path.is_file():
        raise AnytimeArtifactError(f"required artifact is missing or unsafe: {path}")
    try:
        content = path.read_bytes()
        decoded = json.loads(content)
        if _contains_negative_zero(decoded):
            raise AnytimeArtifactError("artifact contains noncanonical signed zero")
        return model.model_validate_json(content)
    except Exception as error:
        raise AnytimeArtifactError(f"invalid artifact: {path}") from error


def _turn_directory(call_index: int) -> Path:
    if not 1 <= call_index <= 12:
        raise AnytimeArtifactError("scientific call index must be in [1, 12]")
    return Path("turns") / f"{call_index:04d}"


def _validate_native_request_binding(
    intent: AnytimeDispatchIntent,
    request_id: str,
    logical_request_sha256: str,
) -> None:
    if request_id != intent.prepared.logical_request.request_id:
        raise AnytimeArtifactError("native provider fact has the wrong request ID")
    if logical_request_sha256 != intent.logical_request_sha256:
        raise AnytimeArtifactError("native provider fact belongs to another logical request")


def _anytime_usage_from_native(usage: Any) -> AnytimeTokenUsage:
    return AnytimeTokenUsage(
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )


def _success_observation_from_native(
    response: NativeNormalizedResponse,
    artifact_sha256: str,
) -> AnytimeProviderSuccess:
    if response.transport_request_sha256 != sha256_json(response.sanitized_transport_request):
        raise AnytimeArtifactError("native response transport-request hash is invalid")
    if response.transport_response_sha256 != sha256_json(response.raw_transport_response):
        raise AnytimeArtifactError("native response transport-response hash is invalid")
    if response.finished_at_utc < response.started_at_utc:
        raise AnytimeArtifactError("native response timestamps are reversed")
    return AnytimeProviderSuccess(
        logical_request_sha256=response.logical_request_sha256,
        response_artifact_sha256=artifact_sha256,
        usage=_anytime_usage_from_native(response.usage),
        provider_seconds=response.elapsed_ms / 1000.0,
    )


def _submitted_error_observation_from_native(
    error: NativeNormalizedError,
    artifact_sha256: str,
) -> AnytimeProviderSubmittedError:
    if not error.request_submitted:
        raise AnytimeArtifactError("unsubmitted native errors cannot consume a ledger call")
    if error.failed_at_utc < error.started_at_utc:
        raise AnytimeArtifactError("native error timestamps are reversed")
    return AnytimeProviderSubmittedError(
        logical_request_sha256=error.logical_request_sha256,
        error_artifact_sha256=artifact_sha256,
        category=error.category,
        possibly_charged=error.possibly_charged,
        partial_usage=(
            None if error.partial_usage is None else _anytime_usage_from_native(error.partial_usage)
        ),
        provider_seconds=error.elapsed_ms / 1000.0,
    )


class AnytimeAttemptWriter:
    """Append-only writer for one not-yet-promoted attempt directory."""

    def __init__(
        self,
        root: str | Path,
        header: AnytimeAttemptHeader,
        *,
        secrets: tuple[str, ...] = (),
        fault: FaultInjector | None = None,
        create: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self.header = AnytimeAttemptHeader.model_validate_json(header.model_dump_json())
        self.staging = anytime_attempt_directory(self.root, header.identity, incomplete=True)
        self.final = anytime_attempt_directory(self.root, header.identity, incomplete=False)
        self._secret_bytes = tuple(secret.encode() for secret in secrets if secret)
        self._fault = fault
        if create:
            if self.final.exists() or self.final.is_symlink():
                raise AnytimeArtifactError("sealed attempt already exists")
            _mkdir_chain(self.staging.parent, self.root)
            try:
                self.staging.mkdir(mode=0o700)
            except FileExistsError as error:
                raise AnytimeArtifactError("attempt staging already exists") from error
            _fsync_directory(self.staging.parent)
            _exclusive_json(
                self.staging,
                "header.json",
                self.header,
                secrets=self._secret_bytes,
            )
            self._inject("after_attempt_create")
        else:
            if self.staging.is_symlink() or not self.staging.is_dir():
                raise AnytimeArtifactError("attempt staging directory is missing or unsafe")
            observed = _load_model(self.staging / "header.json", AnytimeAttemptHeader)
            if observed != self.header:
                raise AnytimeArtifactError("attempt staging header differs from trust anchor")

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        identity: AnytimeAttemptIdentity,
        ledger_header: AnytimeLedgerHeader,
        secrets: tuple[str, ...] = (),
        fault: FaultInjector | None = None,
    ) -> AnytimeAttemptWriter:
        return cls(
            root,
            AnytimeAttemptHeader(identity=identity, ledger_header=ledger_header),
            secrets=secrets,
            fault=fault,
        )

    def _inject(self, point: ArtifactFaultPoint) -> None:
        if self._fault is not None:
            self._fault(point)

    def _write_json(
        self,
        relative: str | Path,
        value: Any,
        *,
        before_publish: ArtifactFaultPoint | None = None,
    ) -> Path:
        return _exclusive_json(
            self.staging,
            relative,
            value,
            secrets=self._secret_bytes,
            fault=self._fault,
            before_publish=before_publish,
        )

    def _ensure_json_fact(self, relative: str | Path, value: Any) -> Path:
        """Publish a fact or accept only an identical crash-surviving fact."""

        destination = _resolve_inside(self.staging, relative)
        expected = canonical_artifact_bytes(value)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file():
                raise AnytimeArtifactError("existing immutable fact is unsafe")
            if destination.read_bytes() != expected:
                raise AnytimeArtifactError("existing immutable fact differs from resumed value")
            return destination
        return self._write_json(relative, value)

    def persist_dispatch_intent(self, prepared: AnytimePreparedTurn) -> AnytimeDispatchIntent:
        """Publish the mandatory pre-transport intent and fsync its directory."""

        if prepared.header_sha256 != self.header.ledger_header.sha256:
            raise AnytimeArtifactError("prepared turn belongs to another ledger header")
        intent = AnytimeDispatchIntent(
            ledger_header_sha256=self.header.ledger_header.sha256,
            prepared=prepared,
            logical_request_sha256=sha256_json(prepared.logical_request),
        )
        directory = _turn_directory(prepared.scientific_call_index)
        self._write_json(directory / "dispatch-intent.json", intent)
        self._inject("after_dispatch_intent")
        return intent

    def _persist_provider_observation(
        self,
        *,
        scientific_call_index: int,
        observation: AnytimeProviderObservation,
        observed_wall_seconds: float,
    ) -> AnytimePersistedProviderObservation:
        """Publish an observation already derived from a durable native fact."""

        directory = _turn_directory(scientific_call_index)
        intent = _load_model(
            self.staging / directory / "dispatch-intent.json",
            AnytimeDispatchIntent,
        )
        if observation.logical_request_sha256 != intent.logical_request_sha256:
            raise AnytimeArtifactError("provider observation belongs to another dispatch")
        persisted = AnytimePersistedProviderObservation(
            dispatch_intent_sha256=intent.sha256,
            observation=observation,
            observed_wall_seconds=observed_wall_seconds,
        )
        self._write_json(
            directory / "provider-observation.json",
            persisted,
        )
        self._inject("after_provider_terminal")
        return persisted

    def persist_native_provider_response(
        self,
        *,
        scientific_call_index: int,
        response: NativeNormalizedResponse,
        observed_wall_seconds: float | None = None,
    ) -> AnytimePersistedProviderObservation:
        """Revalidate an M2 response artifact and derive every M3 observation field."""

        response = NativeNormalizedResponse.model_validate_json(response.model_dump_json())
        directory = _turn_directory(scientific_call_index)
        intent = _load_model(
            self.staging / directory / "dispatch-intent.json",
            AnytimeDispatchIntent,
        )
        _validate_native_request_binding(
            intent,
            response.request_id,
            response.logical_request_sha256,
        )
        response_path = self._write_json(
            directory / "provider-native-response.json",
            response,
            before_publish="provider_terminal_before_publish",
        )
        observation = _success_observation_from_native(response, _file_sha256(response_path))
        return self._persist_provider_observation(
            scientific_call_index=scientific_call_index,
            observation=observation,
            observed_wall_seconds=(
                observation.provider_seconds
                if observed_wall_seconds is None
                else observed_wall_seconds
            ),
        )

    def persist_native_provider_error(
        self,
        *,
        scientific_call_index: int,
        error: NativeNormalizedError,
        observed_wall_seconds: float | None = None,
    ) -> AnytimePersistedProviderObservation | None:
        """Persist an M2 error; unsubmitted errors remain outside the M3 ledger."""

        error = NativeNormalizedError.model_validate_json(error.model_dump_json())
        if error.failed_at_utc < error.started_at_utc:
            raise AnytimeArtifactError("native error timestamps are reversed")
        directory = _turn_directory(scientific_call_index)
        intent = _load_model(
            self.staging / directory / "dispatch-intent.json",
            AnytimeDispatchIntent,
        )
        _validate_native_request_binding(intent, error.request_id, error.logical_request_sha256)
        error_path = self._write_json(
            directory / "provider-native-error.json",
            error,
            before_publish="provider_terminal_before_publish",
        )
        if not error.request_submitted:
            self._inject("after_provider_terminal")
            return None
        observation = _submitted_error_observation_from_native(error, _file_sha256(error_path))
        return self._persist_provider_observation(
            scientific_call_index=scientific_call_index,
            observation=observation,
            observed_wall_seconds=(
                observation.provider_seconds
                if observed_wall_seconds is None
                else observed_wall_seconds
            ),
        )

    def persist_worker_evaluation(
        self,
        *,
        scientific_call_index: int,
        worker_artifact: AnytimeWorkerEvaluationArtifact,
    ) -> AnytimePersistedEvaluation:
        """Publish a typed worker fact and derive its redundant ledger view."""

        directory = _turn_directory(scientific_call_index)
        provider = _load_model(
            self.staging / directory / "provider-observation.json",
            AnytimePersistedProviderObservation,
        )
        if not isinstance(provider.observation, AnytimeProviderSuccess):
            raise AnytimeArtifactError("only provider success may have a candidate evaluation")
        worker = AnytimeWorkerEvaluationArtifact.model_validate_json(
            worker_artifact.model_dump_json()
        )
        if worker.provider_observation_sha256 != provider.sha256:
            raise AnytimeArtifactError("worker result belongs to another provider observation")
        if worker.observed_wall_seconds < provider.observed_wall_seconds:
            raise AnytimeArtifactError("worker wall time is below the provider observation")
        worker_path = self._ensure_json_fact(directory / "worker-result.json", worker)
        evaluation = _persisted_evaluation_from_worker(worker, _file_sha256(worker_path))
        self._write_json(
            directory / "evaluation.json",
            evaluation,
            before_publish="worker_evaluation_before_publish",
        )
        self._inject("after_worker_evaluation")
        return evaluation

    def persist_derived_turn(
        self,
        *,
        loop: AnytimeLoopPolicy,
        base_prompt: tuple[ChatMessage, ...],
        scientific_call_index: int,
    ) -> AnytimeTurnRecord:
        """Derive and publish one ledger record from already durable local facts."""

        records, checkpoints = _load_prefix(self.staging)
        if scientific_call_index != len(records) + 1:
            raise AnytimeArtifactError("turn records must be committed contiguously")
        directory = _turn_directory(scientific_call_index)
        persisted = _load_model(
            self.staging / directory / "provider-observation.json",
            AnytimePersistedProviderObservation,
        )
        candidate: AnytimeCandidateOutcome | None = None
        wall_seconds = persisted.observed_wall_seconds
        if isinstance(persisted.observation, AnytimeProviderSuccess):
            evaluation = _load_model(
                self.staging / directory / "evaluation.json",
                AnytimePersistedEvaluation,
            )
            worker_path = self.staging / directory / "worker-result.json"
            worker = _load_model(worker_path, AnytimeWorkerEvaluationArtifact)
            expected_evaluation = _persisted_evaluation_from_worker(
                worker,
                _file_sha256(worker_path),
            )
            if evaluation != expected_evaluation:
                raise AnytimeArtifactError("evaluation is not derived from worker-result bytes")
            if worker.provider_observation_sha256 != persisted.sha256:
                raise AnytimeArtifactError("worker result belongs to another provider observation")
            candidate = evaluation.candidate
            wall_seconds = evaluation.observed_wall_seconds
        result = append_anytime_turn(
            header=self.header.ledger_header,
            loop=loop,
            base_prompt=base_prompt,
            records=records,
            checkpoints=checkpoints,
            provider=persisted.observation,
            candidate=candidate,
            observed_wall_seconds=wall_seconds,
        )
        self._write_json(directory / "turn-record.json", result.record)
        self._inject("after_turn_record")
        if result.checkpoint is not None:
            self._write_json(
                Path("checkpoints") / f"{scientific_call_index:04d}.json",
                result.checkpoint,
                before_publish="checkpoint_before_publish",
            )
            self._inject("after_checkpoint")
        return result.record

    def persist_terminal(
        self,
        *,
        loop: AnytimeLoopPolicy,
        base_prompt: tuple[ChatMessage, ...],
        terminal_kind: AttemptTerminalKind,
        reason: str,
    ) -> AnytimeAttemptTerminal:
        """Replay the prefix and derive, then publish, one terminal tombstone."""

        _validate_component("terminal reason", reason)
        records, checkpoints = _load_prefix(self.staging)
        verified = verify_anytime_ledger(
            header=self.header.ledger_header,
            loop=loop,
            base_prompt=base_prompt,
            records=records,
            checkpoints=checkpoints,
        )
        if terminal_kind == "success" and verified.terminal_reason != "call_budget":
            raise AnytimeArtifactError("success tombstone requires a completed call budget")
        if terminal_kind == "scientific_failure" and verified.terminal_reason not in {
            "resource_cap",
            "provider_submitted_error",
            "ambiguous_submission",
        }:
            raise AnytimeArtifactError("scientific failure requires a scientific terminal ledger")
        if terminal_kind in {"infrastructure_failure", "controller_failure"}:
            if verified.terminal_reason is not None:
                raise AnytimeArtifactError("local failure cannot replace a scientific terminal")
            request_submitted = False
            possibly_charged = False
        else:
            if not records:
                raise AnytimeArtifactError("scientific terminal requires a consumed request")
            request_submitted = True
            last_provider = records[-1].provider
            possibly_charged = isinstance(
                last_provider,
                AnytimeProviderAmbiguousSubmission,
            ) or (
                isinstance(last_provider, AnytimeProviderSubmittedError)
                and last_provider.possibly_charged
            )
        retry_eligible = (
            terminal_kind == "infrastructure_failure"
            and self.header.identity.infrastructure_attempt_index
            < loop.infrastructure.max_attempts_per_trajectory
        )
        terminal = AnytimeAttemptTerminal(
            attempt_header_sha256=self.header.sha256,
            terminal_kind=terminal_kind,
            reason=reason,
            request_submitted=request_submitted,
            possibly_charged=possibly_charged,
            retry_eligible=retry_eligible,
            ledger_records=verified.records_verified,
            ledger_head_sha256=verified.ledger_head_sha256,
            resource_snapshot=verified.resource_snapshot,
        )
        self._write_json("terminal.json", terminal)
        return terminal

    def seal_and_promote(
        self,
        *,
        loop: AnytimeLoopPolicy,
        base_prompt: tuple[ChatMessage, ...],
    ) -> AnytimeAttemptAudit:
        """Seal exact checksums, audit fresh bytes, and atomically rename to final."""

        terminal = _load_model(self.staging / "terminal.json", AnytimeAttemptTerminal)
        files = _inventory(self.staging, exclude={"manifest.json"})
        manifest = AnytimeAttemptManifest(
            attempt_header_sha256=self.header.sha256,
            terminal_sha256=terminal.sha256,
            files=files,
        )
        self._write_json(
            "manifest.json",
            manifest,
            before_publish="seal_before_manifest_publish",
        )
        self._inject("after_manifest_publish")
        return self._promote_sealed(loop=loop, base_prompt=base_prompt)

    def _promote_sealed(
        self,
        *,
        loop: AnytimeLoopPolicy,
        base_prompt: tuple[ChatMessage, ...],
    ) -> AnytimeAttemptAudit:
        _audit_attempt_directory(self.staging, loop=loop, base_prompt=base_prompt)
        _make_read_only(self.staging)
        if self.final.exists() or self.final.is_symlink():
            raise AnytimeArtifactError("final attempt path already exists")
        _rename_directory_noreplace(self.staging, self.final)
        _fsync_directory(self.final.parent)
        self._inject("after_atomic_promotion")
        return _audit_attempt_directory(self.final, loop=loop, base_prompt=base_prompt)


def _inventory(root: Path, *, exclude: set[str]) -> tuple[AnytimeArtifactChecksum, ...]:
    files: list[AnytimeArtifactChecksum] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AnytimeArtifactError("symbolic links are forbidden in attempt artifacts")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if _TEMP_PATTERN.fullmatch(path.name):
            raise AnytimeArtifactError("attempt contains an unpublished temporary file")
        if relative in exclude:
            continue
        files.append(
            AnytimeArtifactChecksum(
                path=relative,
                sha256=_file_sha256(path),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(files)


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise AnytimeArtifactError("symbolic links are forbidden in attempt artifacts")
        path.chmod(0o400 if path.is_file() else 0o500)
    root.chmod(0o500)


def _load_prefix(
    directory: Path,
) -> tuple[tuple[AnytimeTurnRecord, ...], tuple[AnytimeCheckpointRecord, ...]]:
    records = tuple(
        _load_model(path, AnytimeTurnRecord)
        for path in sorted(directory.glob("turns/[0-9][0-9][0-9][0-9]/turn-record.json"))
    )
    checkpoints = tuple(
        _load_model(path, AnytimeCheckpointRecord)
        for path in sorted(directory.glob("checkpoints/[0-9][0-9][0-9][0-9].json"))
    )
    return records, checkpoints


def _validate_turn_artifacts(
    directory: Path,
    records: tuple[AnytimeTurnRecord, ...],
    terminal: AnytimeAttemptTerminal,
) -> None:
    record_calls = {record.scientific_call_index for record in records}
    intent_paths = sorted(directory.glob("turns/[0-9][0-9][0-9][0-9]/dispatch-intent.json"))
    intent_calls = {int(path.parent.name) for path in intent_paths}
    extra_calls = intent_calls - record_calls
    if extra_calls:
        expected_unsubmitted = {len(records) + 1}
        if (
            terminal.terminal_kind != "infrastructure_failure"
            or extra_calls != expected_unsubmitted
        ):
            raise AnytimeArtifactError("attempt has a non-ledger dispatch that is not unsubmitted")
        pending = directory / _turn_directory(next(iter(extra_calls)))
        native_error = _load_model(
            pending / "provider-native-error.json",
            NativeNormalizedError,
        )
        if native_error.request_submitted or (pending / "provider-observation.json").exists():
            raise AnytimeArtifactError("infrastructure tombstone hides a submitted request")
        pending_intent = _load_model(
            pending / "dispatch-intent.json",
            AnytimeDispatchIntent,
        )
        _validate_native_request_binding(
            pending_intent,
            native_error.request_id,
            native_error.logical_request_sha256,
        )
    if record_calls - intent_calls:
        raise AnytimeArtifactError("ledger turn has no dispatch intent artifact")
    for record in records:
        turn = directory / _turn_directory(record.scientific_call_index)
        intent = _load_model(turn / "dispatch-intent.json", AnytimeDispatchIntent)
        provider = _load_model(
            turn / "provider-observation.json",
            AnytimePersistedProviderObservation,
        )
        if intent.prepared.logical_request != record.logical_request:
            raise AnytimeArtifactError("dispatch intent differs from ledger logical request")
        if provider.dispatch_intent_sha256 != intent.sha256:
            raise AnytimeArtifactError("provider observation differs from dispatch intent")
        if provider.observation != record.provider:
            raise AnytimeArtifactError("provider observation differs from ledger record")
        if isinstance(provider.observation, AnytimeProviderSuccess):
            response_path = turn / "provider-native-response.json"
            response = _load_model(response_path, NativeNormalizedResponse)
            _validate_native_request_binding(
                intent,
                response.request_id,
                response.logical_request_sha256,
            )
            expected_observation = _success_observation_from_native(
                response,
                _file_sha256(response_path),
            )
            if provider.observation != expected_observation:
                raise AnytimeArtifactError(
                    "provider success fields are not derived from native response bytes"
                )
            evaluation = _load_model(turn / "evaluation.json", AnytimePersistedEvaluation)
            worker = turn / "worker-result.json"
            if _file_sha256(worker) != evaluation.worker_artifact_sha256:
                raise AnytimeArtifactError("worker artifact bytes differ from evaluation")
            worker_fact = _load_model(worker, AnytimeWorkerEvaluationArtifact)
            expected_evaluation = _persisted_evaluation_from_worker(
                worker_fact,
                evaluation.worker_artifact_sha256,
            )
            if evaluation != expected_evaluation:
                raise AnytimeArtifactError(
                    "evaluation fields are not derived from worker-result bytes"
                )
            if worker_fact.provider_observation_sha256 != provider.sha256:
                raise AnytimeArtifactError("worker result belongs to another provider observation")
            if evaluation.candidate != record.candidate:
                raise AnytimeArtifactError("evaluation candidate differs from ledger record")
            if evaluation.observed_wall_seconds != record.observed_wall_seconds:
                raise AnytimeArtifactError("evaluation wall time differs from ledger record")
        elif isinstance(provider.observation, AnytimeProviderSubmittedError):
            error_path = turn / "provider-native-error.json"
            error = _load_model(error_path, NativeNormalizedError)
            _validate_native_request_binding(
                intent,
                error.request_id,
                error.logical_request_sha256,
            )
            expected_observation = _submitted_error_observation_from_native(
                error,
                _file_sha256(error_path),
            )
            if provider.observation != expected_observation:
                raise AnytimeArtifactError(
                    "provider error fields are not derived from native error bytes"
                )
            if provider.observed_wall_seconds != record.observed_wall_seconds:
                raise AnytimeArtifactError("provider-error wall time differs from ledger record")


def _validate_terminal_policy(
    terminal: AnytimeAttemptTerminal,
    *,
    header: AnytimeAttemptHeader,
    verified: Any,
    loop: AnytimeLoopPolicy,
) -> None:
    if terminal.terminal_kind == "success":
        if verified.terminal_reason != "call_budget" or not terminal.request_submitted:
            raise AnytimeArtifactError("success terminal differs from completed ledger policy")
    elif terminal.terminal_kind == "scientific_failure":
        if (
            verified.terminal_reason
            not in {
                "resource_cap",
                "provider_submitted_error",
                "ambiguous_submission",
            }
            or not terminal.request_submitted
        ):
            raise AnytimeArtifactError("scientific terminal differs from ledger policy")
    elif terminal.terminal_kind == "infrastructure_failure":
        if verified.terminal_reason is not None:
            raise AnytimeArtifactError("infrastructure terminal cannot replace scientific terminal")
        if terminal.request_submitted or terminal.possibly_charged:
            raise AnytimeArtifactError("infrastructure failure must be definitely unsubmitted")
    else:
        if verified.terminal_reason is not None:
            raise AnytimeArtifactError("controller terminal cannot replace scientific terminal")
        if terminal.request_submitted or terminal.possibly_charged:
            raise AnytimeArtifactError("controller terminal must be definitely unsubmitted")
    expected_charge = (
        verified.resource_snapshot.possibly_charged_requests > 0
        if terminal.terminal_kind in {"success", "scientific_failure"}
        else False
    )
    if terminal.possibly_charged != expected_charge:
        raise AnytimeArtifactError("terminal charge state differs from verified ledger resources")
    expected_retry = (
        terminal.terminal_kind == "infrastructure_failure"
        and header.identity.infrastructure_attempt_index
        < loop.infrastructure.max_attempts_per_trajectory
    )
    if terminal.retry_eligible != expected_retry:
        raise AnytimeArtifactError("terminal retry state differs from bounded attempt policy")


def _audit_attempt_directory(
    directory: Path,
    *,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
) -> AnytimeAttemptAudit:
    if directory.is_symlink() or not directory.is_dir():
        raise AnytimeArtifactError("attempt directory is missing or unsafe")
    manifest_path = directory / "manifest.json"
    manifest = _load_model(manifest_path, AnytimeAttemptManifest)
    actual = _inventory(directory, exclude={"manifest.json"})
    if actual != manifest.files:
        raise AnytimeArtifactError("attempt manifest does not match fresh file bytes")
    header = _load_model(directory / "header.json", AnytimeAttemptHeader)
    expected_names = {
        header.identity.attempt_name,
        f"{header.identity.attempt_name}.incomplete",
    }
    if directory.name not in expected_names:
        raise AnytimeArtifactError("attempt directory name differs from its header")
    if (
        directory.parent.name != header.identity.trajectory_id
        or directory.parent.parent.name != "attempts"
        or directory.parent.parent.parent.name != header.identity.phase_id
        or directory.parent.parent.parent.parent.name != header.identity.study_id
        or directory.parent.parent.parent.parent.parent.name != "source"
    ):
        raise AnytimeArtifactError("attempt directory layout differs from its header")
    terminal = _load_model(directory / "terminal.json", AnytimeAttemptTerminal)
    if manifest.attempt_header_sha256 != header.sha256:
        raise AnytimeArtifactError("manifest differs from attempt header")
    if manifest.terminal_sha256 != terminal.sha256:
        raise AnytimeArtifactError("manifest differs from terminal tombstone")
    if terminal.attempt_header_sha256 != header.sha256:
        raise AnytimeArtifactError("terminal differs from attempt header")
    records, checkpoints = _load_prefix(directory)
    verified = verify_anytime_ledger(
        header=header.ledger_header,
        loop=loop,
        base_prompt=base_prompt,
        records=records,
        checkpoints=checkpoints,
    )
    if (
        terminal.ledger_records != verified.records_verified
        or terminal.ledger_head_sha256 != verified.ledger_head_sha256
        or terminal.resource_snapshot != verified.resource_snapshot
    ):
        raise AnytimeArtifactError("terminal tombstone differs from verified ledger")
    _validate_terminal_policy(terminal, header=header, verified=verified, loop=loop)
    _validate_turn_artifacts(directory, records, terminal)
    return AnytimeAttemptAudit(
        identity=header.identity,
        directory=str(directory),
        manifest_file_sha256=_file_sha256(manifest_path),
        terminal=terminal,
        ledger_records=len(records),
    )


def audit_anytime_attempt(
    directory: str | Path,
    *,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
) -> AnytimeAttemptAudit:
    """Freshly checksum and replay-verify one promoted attempt."""

    return _audit_attempt_directory(
        Path(directory),
        loop=AnytimeLoopPolicy.model_validate_json(loop.model_dump_json()),
        base_prompt=tuple(
            ChatMessage.model_validate_json(message.model_dump_json()) for message in base_prompt
        ),
    )


def _cleanup_private_temps(staging: Path) -> None:
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise AnytimeArtifactError("symbolic links are forbidden in attempt staging")
        if path.is_file() and _TEMP_PATTERN.fullmatch(path.name):
            path.unlink()
            _fsync_directory(path.parent)


def _open_writer(
    root: str | Path,
    header: AnytimeAttemptHeader,
    *,
    fault: FaultInjector | None,
) -> AnytimeAttemptWriter:
    return AnytimeAttemptWriter(root, header, fault=fault, create=False)


def _repair_checkpoints(
    writer: AnytimeAttemptWriter,
    *,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
) -> None:
    records, observed = _load_prefix(writer.staging)
    expected = rebuild_anytime_checkpoints(
        header=writer.header.ledger_header,
        loop=loop,
        base_prompt=base_prompt,
        records=records,
    )
    observed_by_call = {item.identity.scientific_call_index: item for item in observed}
    if len(observed_by_call) != len(observed):
        raise AnytimeArtifactError("duplicate checkpoint artifacts")
    for checkpoint in expected:
        call = checkpoint.identity.scientific_call_index
        existing = observed_by_call.pop(call, None)
        if existing is not None:
            if existing != checkpoint:
                raise AnytimeArtifactError("persisted checkpoint differs from rebuilt checkpoint")
            continue
        writer._write_json(
            Path("checkpoints") / f"{call:04d}.json",
            checkpoint,
            before_publish="checkpoint_before_publish",
        )
        writer._inject("after_checkpoint")
    if observed_by_call:
        raise AnytimeArtifactError("attempt contains unexpected checkpoint artifacts")


def recover_anytime_attempt(
    root: str | Path,
    identity: AnytimeAttemptIdentity,
    *,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
    fault: FaultInjector | None = None,
) -> AnytimeRecoveryDecision:
    """Recover one attempt offline without ever replaying a provider request."""

    final = anytime_attempt_directory(root, identity, incomplete=False)
    staging = anytime_attempt_directory(root, identity, incomplete=True)
    if final.exists() or final.is_symlink():
        audit = audit_anytime_attempt(final, loop=loop, base_prompt=base_prompt)
        return AnytimeRecoveryDecision(
            action="complete",
            directory=audit.directory,
            request_submitted=audit.terminal.request_submitted,
            possibly_charged=audit.terminal.possibly_charged,
        )
    if not staging.exists():
        return AnytimeRecoveryDecision(
            action="missing",
            request_submitted=False,
            possibly_charged=False,
        )
    if staging.is_symlink() or not staging.is_dir():
        raise AnytimeArtifactError("attempt staging is not a safe directory")
    _cleanup_private_temps(staging)
    header = _load_model(staging / "header.json", AnytimeAttemptHeader)
    if header.identity != identity:
        raise AnytimeArtifactError("attempt staging identity differs from trust anchor")
    writer = _open_writer(root, header, fault=fault)

    manifest_path = staging / "manifest.json"
    if manifest_path.exists():
        audit = writer._promote_sealed(loop=loop, base_prompt=base_prompt)
        return AnytimeRecoveryDecision(
            action="finalized",
            directory=audit.directory,
            request_submitted=audit.terminal.request_submitted,
            possibly_charged=audit.terminal.possibly_charged,
        )

    records, _ = _load_prefix(staging)
    next_call = len(records) + 1
    turn = staging / _turn_directory(next_call) if next_call <= 12 else None
    if turn is not None and (turn / "dispatch-intent.json").is_file():
        intent = _load_model(turn / "dispatch-intent.json", AnytimeDispatchIntent)
        provider_path = turn / "provider-observation.json"
        if not provider_path.is_file():
            response_path = turn / "provider-native-response.json"
            error_path = turn / "provider-native-error.json"
            if response_path.is_file() and error_path.is_file():
                raise AnytimeArtifactError("dispatch has both response and error facts")
            if response_path.is_file():
                native_response = _load_model(response_path, NativeNormalizedResponse)
                _validate_native_request_binding(
                    intent,
                    native_response.request_id,
                    native_response.logical_request_sha256,
                )
                observation = _success_observation_from_native(
                    native_response,
                    _file_sha256(response_path),
                )
                writer._persist_provider_observation(
                    scientific_call_index=next_call,
                    observation=observation,
                    observed_wall_seconds=observation.provider_seconds,
                )
            elif error_path.is_file():
                native_error = _load_model(error_path, NativeNormalizedError)
                if native_error.failed_at_utc < native_error.started_at_utc:
                    raise AnytimeArtifactError("native error timestamps are reversed")
                _validate_native_request_binding(
                    intent,
                    native_error.request_id,
                    native_error.logical_request_sha256,
                )
                if native_error.request_submitted:
                    observation = _submitted_error_observation_from_native(
                        native_error,
                        _file_sha256(error_path),
                    )
                    writer._persist_provider_observation(
                        scientific_call_index=next_call,
                        observation=observation,
                        observed_wall_seconds=observation.provider_seconds,
                    )
                else:
                    # A durable M2 error proves transport did not submit.  It
                    # authorizes a separate whole-attempt retry and consumes no
                    # scientific call.
                    pass
            else:
                ambiguous = AnytimeProviderAmbiguousSubmission(
                    logical_request_sha256=intent.logical_request_sha256,
                    dispatch_intent_sha256=intent.sha256,
                    provider_seconds=0.0,
                )
                writer._persist_provider_observation(
                    scientific_call_index=next_call,
                    observation=ambiguous,
                    observed_wall_seconds=0.0,
                )
        if not (turn / "provider-observation.json").is_file():
            # The only non-observation terminal fact admitted here is a fully
            # validated unsubmitted NativeNormalizedError.
            native_error = _load_model(
                turn / "provider-native-error.json",
                NativeNormalizedError,
            )
            if native_error.request_submitted:
                raise AnytimeArtifactError("submitted native error lacks derived observation")
        else:
            persisted = _load_model(
                turn / "provider-observation.json",
                AnytimePersistedProviderObservation,
            )
            if (
                isinstance(persisted.observation, AnytimeProviderSuccess)
                and not (turn / "evaluation.json").is_file()
            ):
                return AnytimeRecoveryDecision(
                    action="resume_worker_evaluation",
                    directory=str(staging),
                    scientific_call_index=next_call,
                    request_submitted=True,
                    possibly_charged=False,
                )
            writer.persist_derived_turn(
                loop=loop,
                base_prompt=base_prompt,
                scientific_call_index=next_call,
            )

    _repair_checkpoints(writer, loop=loop, base_prompt=base_prompt)
    records, checkpoints = _load_prefix(staging)
    verified = verify_anytime_ledger(
        header=header.ledger_header,
        loop=loop,
        base_prompt=base_prompt,
        records=records,
        checkpoints=checkpoints,
    )
    terminal_path = staging / "terminal.json"
    if not terminal_path.is_file():
        if verified.terminal_reason == "call_budget":
            writer.persist_terminal(
                loop=loop,
                base_prompt=base_prompt,
                terminal_kind="success",
                reason="call_budget_complete",
            )
        elif verified.terminal_reason is not None:
            writer.persist_terminal(
                loop=loop,
                base_prompt=base_prompt,
                terminal_kind="scientific_failure",
                reason=verified.terminal_reason,
            )
        else:
            # No dispatch intent exists for the next call.  Therefore the
            # interrupted boundary is definitely unsubmitted and may consume
            # the one separately stored whole-attempt retry.
            writer.persist_terminal(
                loop=loop,
                base_prompt=base_prompt,
                terminal_kind="infrastructure_failure",
                reason="interrupted_before_dispatch",
            )
    terminal = _load_model(terminal_path, AnytimeAttemptTerminal)
    audit = writer.seal_and_promote(loop=loop, base_prompt=base_prompt)
    return AnytimeRecoveryDecision(
        action="finalized",
        directory=audit.directory,
        request_submitted=terminal.request_submitted,
        possibly_charged=terminal.possibly_charged,
    )


def create_anytime_retry(
    root: str | Path,
    previous_identity: AnytimeAttemptIdentity,
    *,
    ledger_header: AnytimeLedgerHeader,
    loop: AnytimeLoopPolicy,
    base_prompt: tuple[ChatMessage, ...],
    secrets: tuple[str, ...] = (),
    fault: FaultInjector | None = None,
) -> AnytimeAttemptWriter:
    """Create the sole bounded retry as a distinct immutable attempt."""

    if previous_identity.infrastructure_attempt_index != 1:
        raise AnytimeArtifactError("only the primary attempt can authorize a retry")
    previous = anytime_attempt_directory(root, previous_identity, incomplete=False)
    audit = audit_anytime_attempt(previous, loop=loop, base_prompt=base_prompt)
    if not audit.terminal.retry_eligible:
        raise AnytimeArtifactError("previous attempt does not authorize an infrastructure retry")
    if ledger_header.infrastructure_attempt_index != 2:
        raise AnytimeArtifactError("retry ledger header must use attempt index two")
    if ledger_header.trajectory_id != previous_identity.trajectory_id:
        raise AnytimeArtifactError("retry belongs to another trajectory")
    if ledger_header.trajectory_execution_sha256 != previous_identity.trajectory_execution_sha256:
        raise AnytimeArtifactError("retry execution hash differs from primary attempt")
    identity = AnytimeAttemptIdentity(
        study_id=previous_identity.study_id,
        phase_id=previous_identity.phase_id,
        trajectory_id=previous_identity.trajectory_id,
        infrastructure_attempt_index=2,
        trajectory_execution_sha256=previous_identity.trajectory_execution_sha256,
        prior_attempt_manifest_sha256=audit.manifest_file_sha256,
    )
    return AnytimeAttemptWriter.create(
        root,
        identity=identity,
        ledger_header=ledger_header,
        secrets=secrets,
        fault=fault,
    )
