from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from abstrak.anytime.contracts import AnytimeResourceBudget
from abstrak.canary.contracts import R1_AGENT_LOOP_POLICY, AgentBudget
from abstrak.canary.manifests import load_study_spec
from abstrak.canary.matrix import build_matrix_schedule
from abstrak.canary.targets import list_target_ids
from abstrak.canary.tasks import list_task_ids
from abstrak.providers.contracts import sha256_json
from abstrak.providers.manifests import GenerationConfig, ModelManifest, ProviderManifest


def test_frozen_canary_budget_and_loop_policy_golden_hashes() -> None:
    assert sha256_json(AgentBudget()) == (
        "5f44e8ffb71c5a06c1b8df888a3e9a3cceb5d5c464b14061921614e2b99f63eb"
    )
    assert R1_AGENT_LOOP_POLICY.sha256 == (
        "5c7ba9c21a8f6a03cca677ea6d8c7845e201378dd833e8fbd3d1714738228543"
    )
    assert R1_AGENT_LOOP_POLICY.model_dump(mode="json") == {
        "schema_version": "canary-agent-loop-policy.v1",
        "response_parser": "agent_marker",
        "stop_policy": "agent",
        "final_selection": "last",
        "latency_ceiling_ms": None,
    }


def test_old_budget_still_rejects_five_calls_while_anytime_accepts_twelve() -> None:
    with pytest.raises(ValidationError):
        AgentBudget(max_calls=5)

    budget = AnytimeResourceBudget(
        max_scientific_calls=12,
        max_total_output_tokens=12 * 16384,
        max_compile_attempts=12,
        max_evaluation_attempts=12,
        max_gpu_seconds=12 * 600.0,
    )
    assert budget.max_scientific_calls == 12


def test_frozen_capability_study_bytes_and_spec_hash_are_unchanged() -> None:
    repository = Path(__file__).resolve().parents[1]
    path = repository / "benchmarks" / "capability-gate-a100" / "study.json"
    pinned = load_study_spec(path)

    assert pinned.sha256 == "876b18e75d86e77c6e2e4cd47038f60719ba6108943ddc754086ea82685ecd00"
    assert pinned.spec.sha256 == (
        "539f39178586ee7c9d028c817268bfe6ababffef6da00298b57bfc5e63402669"
    )
    assert build_matrix_schedule(pinned.spec).sha256 == (
        "40c372285875337ebd62529d72b2dd5bc2f6d123cbb2940a93c7482d2537983e"
    )
    assert pinned.spec.schema_version == "abstrak-matrix-study-spec.v1"


def test_frozen_r1_task_and_target_registry_ids_are_unchanged() -> None:
    assert list_task_ids() == (
        "gemm-bias-relu-static",
        "gemm-static",
        "layernorm-static",
        "matmul-bias",
        "rmsnorm-static",
        "row-reduction-scale",
    )
    assert list_target_ids() == ("cute-a100", "tilelang-a100", "triton-a100")


def test_provider_v1_remains_chat_completions_only() -> None:
    provider = ProviderManifest(id="provider", api_key_env="PROVIDER_API_KEY")
    model = ModelManifest(
        id="model",
        provider=provider.id,
        api_model="model",
        model_id_policy="exact",
        expected_returned_model="model",
        generation=GenerationConfig(max_completion_tokens=1024),
    )

    assert provider.protocol == "chat_completions"
    assert model.interface == "chat_completions"
    with pytest.raises(ValidationError):
        ProviderManifest(
            id="provider",
            api_key_env="PROVIDER_API_KEY",
            protocol="responses",
        )
    payload = model.model_dump(mode="json")
    payload["interface"] = "responses"
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(payload)
