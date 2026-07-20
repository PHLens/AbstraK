"""Deterministic schedules for the frozen A100 R1 rapid study."""

from __future__ import annotations

import random
import re
from typing import Literal

from pydantic import Field, model_validator

from abstrak.canary.contracts import IDENTIFIER_PATTERN, CanaryModel
from abstrak.providers.contracts import sha256_json

SCHEDULE_SEED = 20260717
R1_AGENTS = ("deepseek-v4-flash", "deepseek-v4-pro")
R1_TARGETS = ("triton-a100", "tilelang-a100", "cute-a100")
R1_CANARIES = ("row-reduction-scale", "matmul-bias")
R1_TASKS = (
    "rmsnorm-static",
    "layernorm-static",
    "gemm-static",
    "gemm-bias-relu-static",
)
R1_REPLICATES = (1, 2)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ScheduleError(ValueError):
    """Raised when a requested schedule violates the frozen R1 contract."""


class ScheduleCell(CanaryModel):
    """One trajectory in its immutable execution order."""

    phase: Literal["shakeout", "formal"]
    ordinal: int = Field(ge=0)
    task_id: str = Field(pattern=IDENTIFIER_PATTERN)
    agent_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    replicate: int = Field(ge=1)
    target_order_index: int = Field(ge=0)

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (
            self.phase,
            self.task_id,
            self.agent_id,
            self.target_id,
            self.replicate,
        )

    @property
    def block_key(self) -> tuple[str, str, int]:
        return (self.task_id, self.agent_id, self.replicate)

    @property
    def trajectory_id(self) -> str:
        return f"{self.phase}-{self.task_id}-{self.agent_id}-{self.target_id}-r{self.replicate}"


class R1StudySchedule(CanaryModel):
    """The complete 12-trajectory shakeout and 48-trajectory formal schedule."""

    schema_version: Literal["abstrak-canary-schedule.v1"] = "abstrak-canary-schedule.v1"
    seed: int = SCHEDULE_SEED
    agents: tuple[str, ...]
    targets: tuple[str, ...]
    canaries: tuple[str, ...]
    tasks: tuple[str, ...]
    replicates: tuple[int, ...]
    shakeout: tuple[ScheduleCell, ...]
    formal: tuple[ScheduleCell, ...]

    @model_validator(mode="after")
    def matrix_matches_contract(self) -> R1StudySchedule:
        if not (
            len(self.agents) == 2
            and len(self.targets) == 3
            and len(self.canaries) == 2
            and len(self.tasks) == 4
            and self.replicates == R1_REPLICATES
        ):
            raise ValueError("schedule axes do not match the frozen 2x3x2/2x4x3x2 contract")
        for name, values in (
            ("agents", self.agents),
            ("targets", self.targets),
            ("canaries", self.canaries),
            ("tasks", self.tasks),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
            if any(_IDENTIFIER.fullmatch(value) is None for value in values):
                raise ValueError(f"{name} contain invalid identifiers")
        expected_shakeout = len(self.agents) * len(self.targets) * len(self.canaries)
        expected_formal = (
            len(self.agents) * len(self.targets) * len(self.tasks) * len(self.replicates)
        )
        if len(self.shakeout) != expected_shakeout:
            raise ValueError("shakeout does not cover the requested Cartesian product")
        if len(self.formal) != expected_formal:
            raise ValueError("formal schedule does not cover the requested Cartesian product")
        for phase, cells in (("shakeout", self.shakeout), ("formal", self.formal)):
            if tuple(cell.ordinal for cell in cells) != tuple(range(len(cells))):
                raise ValueError(f"{phase} ordinals must be contiguous from zero")
            keys = [cell.key for cell in cells]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{phase} schedule contains a duplicate cell")
            if any(cell.phase != phase for cell in cells):
                raise ValueError(f"{phase} schedule contains a cell from another phase")

        expected_shakeout_keys = {
            ("shakeout", canary, agent, target, 1)
            for canary in self.canaries
            for agent in self.agents
            for target in self.targets
        }
        expected_formal_keys = {
            ("formal", task, agent, target, replicate)
            for task in self.tasks
            for agent in self.agents
            for replicate in self.replicates
            for target in self.targets
        }
        if {cell.key for cell in self.shakeout} != expected_shakeout_keys:
            raise ValueError("shakeout cell identities do not match the frozen axes")
        if {cell.key for cell in self.formal} != expected_formal_keys:
            raise ValueError("formal cell identities do not match the frozen axes")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _axis(name: str, values: tuple[str, ...], expected_size: int) -> tuple[str, ...]:
    if len(values) != expected_size:
        raise ScheduleError(f"R1 requires exactly {expected_size} {name}")
    if len(values) != len(set(values)):
        raise ScheduleError(f"{name} must be unique")
    invalid = [value for value in values if _IDENTIFIER.fullmatch(value) is None]
    if invalid:
        raise ScheduleError(f"{name} contain invalid identifiers: {', '.join(invalid)}")
    return values


def build_r1_schedule(
    *,
    agents: tuple[str, ...] = R1_AGENTS,
    targets: tuple[str, ...] = R1_TARGETS,
    canaries: tuple[str, ...] = R1_CANARIES,
    tasks: tuple[str, ...] = R1_TASKS,
    seed: int = SCHEDULE_SEED,
) -> R1StudySchedule:
    """Build the exact R1 matrix with target order shuffled inside each block."""

    agents = _axis("agents", agents, 2)
    targets = _axis("targets", targets, 3)
    canaries = _axis("canaries", canaries, 2)
    tasks = _axis("tasks", tasks, 4)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ScheduleError("seed must be an integer")

    shakeout: list[ScheduleCell] = []
    for canary in canaries:
        for agent in agents:
            for target_index, target in enumerate(targets):
                shakeout.append(
                    ScheduleCell(
                        phase="shakeout",
                        ordinal=len(shakeout),
                        task_id=canary,
                        agent_id=agent,
                        target_id=target,
                        replicate=1,
                        target_order_index=target_index,
                    )
                )

    generator = random.Random(seed)
    formal: list[ScheduleCell] = []
    for task in tasks:
        for agent in agents:
            for replicate in R1_REPLICATES:
                block_targets = list(targets)
                generator.shuffle(block_targets)
                for target_index, target in enumerate(block_targets):
                    formal.append(
                        ScheduleCell(
                            phase="formal",
                            ordinal=len(formal),
                            task_id=task,
                            agent_id=agent,
                            target_id=target,
                            replicate=replicate,
                            target_order_index=target_index,
                        )
                    )

    return R1StudySchedule(
        seed=seed,
        agents=agents,
        targets=targets,
        canaries=canaries,
        tasks=tasks,
        replicates=R1_REPLICATES,
        shakeout=tuple(shakeout),
        formal=tuple(formal),
    )
