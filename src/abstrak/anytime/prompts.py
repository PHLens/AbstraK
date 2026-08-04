"""Frozen base-prompt policy for the anytime DSL A100 study.

The renderer admits only the workload's public whitelist view and the selected
target's balanced card.  It deliberately has no access to sealed generators,
trusted experts, reference source, other targets' observations, or profiler
output.
"""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from abstrak.anytime.contracts import AnytimeModel
from abstrak.anytime.workloads import (
    DEFAULT_ASSET_ROOT,
    PinnedAnytimeWorkloadInputs,
    validate_anytime_workload_inputs,
)
from abstrak.providers.contracts import ChatMessage, MessageRole, sha256_json


class AnytimePromptError(ValueError):
    """Raised when public prompt inputs cannot be reconstructed exactly."""


class AnytimeBasePromptPolicy(AnytimeModel):
    """Exact public information and response contract for the first turn."""

    schema_version: Literal["abstrak-anytime-base-prompt-policy.v1"] = (
        "abstrak-anytime-base-prompt-policy.v1"
    )
    renderer_version: Literal["anytime-base-prompt-renderer.v1"] = (
        "anytime-base-prompt-renderer.v1"
    )
    system_message: str = Field(min_length=1)
    user_section_order: tuple[str, ...] = (
        "study_notice",
        "public_workload_json",
        "selected_target_card",
        "response_contract",
    )
    workload_disclosure: Literal["public-whitelist-view-only"] = (
        "public-whitelist-view-only"
    )
    target_card_disclosure: Literal["selected-target-card-only"] = (
        "selected-target-card-only"
    )
    sealed_cases_disclosed: Literal[False] = False
    trusted_expert_source_disclosed: Literal[False] = False
    reference_source_disclosed: Literal[False] = False
    other_target_results_disclosed: Literal[False] = False
    profiler_output_disclosed: Literal[False] = False
    response_contract: Literal["one-fenced-python-source-no-prose"] = (
        "one-fenced-python-source-no-prose"
    )

    @field_validator("user_section_order")
    @classmethod
    def section_order_is_exact(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        expected = (
            "study_notice",
            "public_workload_json",
            "selected_target_card",
            "response_contract",
        )
        if value != expected:
            raise ValueError("base-prompt sections must use the frozen canonical order")
        return value

    @model_validator(mode="after")
    def system_message_states_the_boundary(self) -> AnytimeBasePromptPolicy:
        required = ("fixed-call", "development feedback", "selected DSL target")
        if any(fragment not in self.system_message for fragment in required):
            raise ValueError("base-prompt system message omits a frozen study boundary")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def build_anytime_base_prompt_policy() -> AnytimeBasePromptPolicy:
    """Return the reviewed version-one prompt policy."""

    return AnytimeBasePromptPolicy(
        system_message=(
            "You optimize one fixed-shape GPU kernel for the selected DSL target. "
            "This is a fixed-call study: return one complete candidate per call and use "
            "only the bounded development feedback supplied by the controller. Do not use "
            "tools, retrieval, persistent provider state, or any undeclared runtime."
        )
    )


def _read_target_card(root: Path, relative_path: str, expected_sha256: str) -> str:
    try:
        resolved_root = root.resolve(strict=True)
        candidate = resolved_root / relative_path
        metadata = candidate.stat(follow_symlinks=False)
        path = candidate.resolve(strict=True)
    except OSError as error:
        raise AnytimePromptError(f"cannot resolve selected target card: {error}") from error
    if (
        resolved_root not in path.parents
        or candidate.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise AnytimePromptError("selected target card must be a regular in-repository file")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise AnytimePromptError(
            f"selected target card SHA-256 mismatch: expected {expected_sha256}, found {actual}"
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AnytimePromptError("selected target card is not UTF-8") from error


def render_anytime_base_prompt(
    inputs: PinnedAnytimeWorkloadInputs,
    *,
    task_id: str,
    target_id: str,
    repository_root: str | Path = DEFAULT_ASSET_ROOT,
    policy: AnytimeBasePromptPolicy | None = None,
) -> tuple[ChatMessage, ...]:
    """Render one deterministic public prompt without opening private assets."""

    root = Path(repository_root).expanduser()
    try:
        trusted = validate_anytime_workload_inputs(inputs, repository_root=root)
        workload = trusted.manifest.workload(task_id)
        card = trusted.manifest.target_card(target_id)
    except ValueError as error:
        raise AnytimePromptError(f"invalid public prompt inputs: {error}") from error
    checked_policy = AnytimeBasePromptPolicy.model_validate_json(
        (policy or build_anytime_base_prompt_policy()).model_dump_json()
    )
    public_workload = json.dumps(
        workload.public_view().model_dump(mode="json"),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    target_card = _read_target_card(root, card.source_path, card.source_sha256)
    user_message = "\n\n".join(
        (
            "STUDY NOTICE\nOptimize only the public development contract below.",
            f"PUBLIC WORKLOAD JSON\n{public_workload}",
            f"SELECTED TARGET CARD: {target_id}\n{target_card}",
            (
                "RESPONSE CONTRACT\nReturn exactly one fenced Python source block and no "
                "prose. The source must define ModelNew and use only the selected target stack."
            ),
        )
    )
    return (
        ChatMessage(role=MessageRole.SYSTEM, content=checked_policy.system_message),
        ChatMessage(role=MessageRole.USER, content=user_message),
    )
