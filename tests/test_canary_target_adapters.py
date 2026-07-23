from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from abstrak.canary.capabilities import CORE_PACK
from abstrak.canary.fallback import StaticValidationIssue, validate_candidate_source
from abstrak.canary.target_adapters import (
    DEFAULT_TARGET_ADAPTER_REGISTRY,
    TargetAdapterRegistry,
    TargetValidationResult,
    validate_target_source,
)
from abstrak.canary.targets import get_target_stack
from abstrak.canary.tasks import load_oracle_source


def test_kernelbench_adapter_strictly_preserves_legacy_validation() -> None:
    target = get_target_stack("triton-a100")
    source = load_oracle_source("row-reduction-scale", "triton").replace(
        "return output", "return torch.sum(output)"
    )

    legacy = validate_candidate_source(source, target.backend)
    adapted = validate_target_source(source, target)

    assert DEFAULT_TARGET_ADAPTER_REGISTRY.adapter_ids == (
        "kernelbench",
        "tilelang-capability-core",
        "tilelang-capability-sched",
        "tilelang-capability-map",
        "tilelang-capability-full",
    )
    assert adapted.valid == legacy.valid
    assert adapted.errors == legacy.errors
    assert adapted.warnings == ()
    assert dict(adapted.metadata) == {}


def test_unknown_adapter_fails_closed() -> None:
    target = get_target_stack("triton-a100").model_copy(update={"adapter": "unregistered-adapter"})

    result = validate_target_source("class ModelNew: pass\n", target)

    assert not result.valid
    assert result.error_codes == ("unknown_target_adapter",)
    assert "unregistered-adapter" in result.errors[0].message


def test_capability_adapter_rejects_a_mismatched_target_contract() -> None:
    target = get_target_stack("tilelang-a100").model_copy(
        update={
            "id": CORE_PACK.target_id,
            "adapter": CORE_PACK.adapter_id,
            "version": "wrong-version",
        }
    )

    result = validate_target_source("class ModelNew: pass\n", target)

    assert not result.valid
    assert result.error_codes == ("capability_target_mismatch",)


def test_registry_extensions_and_results_are_immutable() -> None:
    warning = StaticValidationIssue(code="audit_note", message="custom adapter ran")

    def validator(_source: str, _target: object) -> TargetValidationResult:
        return TargetValidationResult(
            valid=True,
            warnings=(warning,),
            metadata={"used_capabilities": ("custom",)},
        )

    original = TargetAdapterRegistry()
    extended = original.with_validator("custom-adapter", validator)  # type: ignore[arg-type]
    target = get_target_stack("triton-a100").model_copy(update={"adapter": "custom-adapter"})
    result = validate_target_source("class ModelNew: pass\n", target, registry=extended)

    assert original.adapter_ids == ()
    assert extended.adapter_ids == ("custom-adapter",)
    assert result.warning_codes == ("audit_note",)
    assert result.metadata["used_capabilities"] == ("custom",)
    with pytest.raises(ValueError, match="already registered"):
        extended.with_validator("custom-adapter", validator)  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        extended.validators = ()
    with pytest.raises(TypeError):
        result.metadata["changed"] = True  # type: ignore[index]


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf")])
def test_adapter_metadata_must_be_finite_json(value: object) -> None:
    with pytest.raises(TypeError, match="finite JSON"):
        TargetValidationResult(valid=True, metadata={"invalid": value})
