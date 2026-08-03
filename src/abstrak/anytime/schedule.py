"""Pure deterministic scheduling for version-one anytime DSL studies."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Literal

from pydantic import Field, model_validator

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


class AnytimeSchedule(AnytimeModel):
    """Materialized replayable schedule bound to one exact anytime study spec."""

    schema_version: Literal["abstrak-anytime-schedule.v1"] = (
        "abstrak-anytime-schedule.v1"
    )
    spec: AnytimeStudySpec
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    cells: tuple[AnytimeScheduleCell, ...] = Field(min_length=1)

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
        return self.spec.operational_request_ceiling

    def cells_for_cohort(self, cohort_id: str) -> tuple[AnytimeScheduleCell, ...]:
        self.spec.cohort(cohort_id)
        return tuple(cell for cell in self.cells if cell.cohort_id == cohort_id)


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
