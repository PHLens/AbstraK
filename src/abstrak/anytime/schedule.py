"""Pure deterministic scheduling for version-one anytime DSL studies."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from abstrak.anytime.contracts import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    AnytimeModel,
    AnytimeStudySpec,
)
from abstrak.providers.contracts import sha256_json


class AnytimeScheduleError(ValueError):
    """Raised when study axes cannot produce an unambiguous schedule."""


class AnytimeScheduleCell(AnytimeModel):
    """One trajectory in the immutable total execution order."""

    schema_version: Literal["abstrak-anytime-schedule-cell.v1"] = (
        "abstrak-anytime-schedule-cell.v1"
    )
    cohort_id: str = Field(pattern=IDENTIFIER_PATTERN)
    ordinal: int = Field(ge=0)
    cohort_ordinal: int = Field(ge=0)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    replicate: int = Field(ge=1)
    target_order_index: int = Field(ge=0)

    @property
    def trajectory_id(self) -> str:
        return (
            f"{self.cohort_id}-{self.task_id}-{self.agent_id}-"
            f"{self.target_id}-r{self.replicate}"
        )

    @property
    def block_key(self) -> tuple[str, str, str, int]:
        return (self.cohort_id, self.task_id, self.agent_id, self.replicate)


def _gate_authorization_binding_sha256(
    *,
    study_id: str,
    spec_sha256: str,
    cohort_id: str,
    evidence_sha256: str,
) -> str:
    return sha256_json(
        {
            "schema_version": "abstrak-anytime-gate-authorization-binding.v1",
            "study_id": study_id,
            "spec_sha256": spec_sha256,
            "cohort_id": cohort_id,
            "evidence_sha256": evidence_sha256,
        }
    )


class AnytimeGateAuthorizationReceipt(AnytimeModel):
    """Integrity-bound authorization for one ``core_gate`` cohort.

    The evidence itself remains in its owning artifact.  This receipt binds its
    digest to one exact study, spec, and cohort, and cannot authorize any other
    planned cells.
    """

    schema_version: Literal["abstrak-anytime-gate-authorization.v1"] = (
        "abstrak-anytime-gate-authorization.v1"
    )
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    cohort_id: str = Field(pattern=IDENTIFIER_PATTERN)
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    authorized: Literal[True] = True
    authorization_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def authorization_binds_all_inputs(self) -> AnytimeGateAuthorizationReceipt:
        expected = _gate_authorization_binding_sha256(
            study_id=self.study_id,
            spec_sha256=self.spec_sha256,
            cohort_id=self.cohort_id,
            evidence_sha256=self.evidence_sha256,
        )
        if self.authorization_sha256 != expected:
            raise ValueError("gate authorization binding hash mismatch")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class AnytimeSchedule(AnytimeModel):
    """Materialized *planned* schedule bound to one exact anytime study spec.

    ``cells`` is the immutable full plan used for cardinality and hashing; it is
    not an executable queue.  Runners must call :meth:`executable_cells`, which
    excludes gated cohorts unless supplied a valid authorization receipt.
    """

    schema_version: Literal["abstrak-anytime-schedule.v1"] = (
        "abstrak-anytime-schedule.v1"
    )
    spec: AnytimeStudySpec
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    cells: tuple[AnytimeScheduleCell, ...] = Field(
        min_length=1,
        description="Full planned cell set; use executable_cells() before execution",
    )

    @model_validator(mode="after")
    def cells_exactly_match_spec(self) -> AnytimeSchedule:
        if self.spec_sha256 != self.spec.sha256:
            raise ValueError("anytime schedule spec hash mismatch")
        expected = _build_cells(self.spec)
        if self.cells != expected:
            raise ValueError("anytime schedule cells do not match the deterministic study spec")
        identifiers = tuple(cell.trajectory_id for cell in self.cells)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("anytime schedule contains duplicate trajectory IDs")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)

    @property
    def expected_trajectories(self) -> int:
        return len(self.cells)

    @property
    def scientific_request_ceiling(self) -> int:
        return self.spec.scientific_request_ceiling

    @property
    def operational_request_ceiling(self) -> int:
        """Request-only retry ceiling, not an all-resource operational envelope.

        Operational aggregation for tokens, GPU time, compile/evaluation work,
        provider time, and wall time belongs to M4 attempt/resume accounting.
        """

        return self.spec.operational_request_ceiling

    def cells_for_cohort(self, cohort_id: str) -> tuple[AnytimeScheduleCell, ...]:
        """Return planned cells for inspection, regardless of activation state."""

        self.spec.cohort(cohort_id)
        return tuple(cell for cell in self.cells if cell.cohort_id == cohort_id)

    def executable_cells(
        self,
        gate_authorizations: tuple[AnytimeGateAuthorizationReceipt, ...] = (),
    ) -> tuple[AnytimeScheduleCell, ...]:
        """Return the safe execution subset after validating every gate receipt."""

        authorized_cohorts: set[str] = set()
        for untrusted_receipt in gate_authorizations:
            try:
                receipt = AnytimeGateAuthorizationReceipt.model_validate(
                    untrusted_receipt.model_dump(mode="python")
                )
            except (AttributeError, ValidationError) as error:
                raise AnytimeScheduleError("invalid gate authorization receipt") from error
            if receipt.study_id != self.spec.study_id:
                raise AnytimeScheduleError(
                    "gate authorization belongs to a different study"
                )
            if receipt.spec_sha256 != self.spec_sha256:
                raise AnytimeScheduleError(
                    "gate authorization belongs to a different study spec"
                )
            try:
                cohort = self.spec.cohort(receipt.cohort_id)
            except ValueError as error:
                raise AnytimeScheduleError(
                    "gate authorization references an unknown cohort"
                ) from error
            if cohort.activation != "core_gate":
                raise AnytimeScheduleError(
                    "gate authorization may only target a core_gate cohort"
                )
            if receipt.cohort_id in authorized_cohorts:
                raise AnytimeScheduleError(
                    "duplicate gate authorization for cohort " + receipt.cohort_id
                )
            authorized_cohorts.add(receipt.cohort_id)

        return tuple(
            cell
            for cell in self.cells
            if self.spec.cohort(cell.cohort_id).activation == "always"
            or cell.cohort_id in authorized_cohorts
        )


def _cohort_seed(seed: int, cohort_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{cohort_id}".encode()).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _build_cells(spec: AnytimeStudySpec) -> tuple[AnytimeScheduleCell, ...]:
    cells: list[AnytimeScheduleCell] = []
    for cohort in spec.cohorts:
        generator = random.Random(_cohort_seed(spec.seed, cohort.id))
        balanced = list(cohort.target_ids)
        generator.shuffle(balanced)
        balanced_base = tuple(balanced)
        cohort_ordinal = 0
        block_index = 0
        for task_id in cohort.task_ids:
            for replicate in cohort.replicates:
                offset = block_index % len(balanced_base)
                ordered_targets = (*balanced_base[offset:], *balanced_base[:offset])
                for target_order_index, target_id in enumerate(ordered_targets):
                    cells.append(
                        AnytimeScheduleCell(
                            cohort_id=cohort.id,
                            ordinal=len(cells),
                            cohort_ordinal=cohort_ordinal,
                            task_id=task_id,
                            agent_id=cohort.agent_id,
                            target_id=target_id,
                            replicate=replicate,
                            target_order_index=target_order_index,
                        )
                    )
                    cohort_ordinal += 1
                block_index += 1
    return tuple(cells)


def build_anytime_schedule(spec: AnytimeStudySpec) -> AnytimeSchedule:
    """Build the exact balanced schedule declared by ``spec``."""

    cells = _build_cells(spec)
    identifiers = tuple(cell.trajectory_id for cell in cells)
    if len(identifiers) != len(set(identifiers)):
        duplicates = tuple(
            identifier for identifier, count in Counter(identifiers).items() if count > 1
        )
        raise AnytimeScheduleError(
            "anytime axes produce ambiguous trajectory IDs: " + ", ".join(duplicates)
        )
    return AnytimeSchedule(spec=spec, spec_sha256=spec.sha256, cells=cells)


def build_anytime_gate_authorization(
    schedule: AnytimeSchedule,
    *,
    cohort_id: str,
    evidence_sha256: str,
) -> AnytimeGateAuthorizationReceipt:
    """Bind trusted gate evidence to one gated cohort in one exact schedule."""

    try:
        cohort = schedule.spec.cohort(cohort_id)
    except ValueError as error:
        raise AnytimeScheduleError("cannot authorize an unknown cohort") from error
    if cohort.activation != "core_gate":
        raise AnytimeScheduleError("only a core_gate cohort requires authorization")
    return AnytimeGateAuthorizationReceipt(
        study_id=schedule.spec.study_id,
        spec_sha256=schedule.spec_sha256,
        cohort_id=cohort_id,
        evidence_sha256=evidence_sha256,
        authorization_sha256=_gate_authorization_binding_sha256(
            study_id=schedule.spec.study_id,
            spec_sha256=schedule.spec_sha256,
            cohort_id=cohort_id,
            evidence_sha256=evidence_sha256,
        ),
    )
