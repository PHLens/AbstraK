"""Offline source-validation adapters selected by frozen target contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from abstrak.canary.contracts import IDENTIFIER_PATTERN, TargetStackSpec
from abstrak.canary.fallback import StaticValidationIssue, validate_candidate_source

TargetSourceValidator = Callable[[str, TargetStackSpec], "TargetValidationResult"]
_ADAPTER_ID = re.compile(IDENTIFIER_PATTERN)


@dataclass(frozen=True)
class TargetValidationResult:
    """One adapter's immutable, runtime-free source validation result."""

    valid: bool
    errors: tuple[StaticValidationIssue, ...] = ()
    warnings: tuple[StaticValidationIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors = tuple(self.errors)
        warnings = tuple(self.warnings)
        metadata = dict(self.metadata)
        if self.valid and errors:
            raise ValueError("valid target validation results cannot contain errors")
        if not self.valid and not errors:
            raise ValueError("invalid target validation results require at least one error")
        if any(not isinstance(issue, StaticValidationIssue) for issue in (*errors, *warnings)):
            raise TypeError("target validation issues must be StaticValidationIssue values")
        if any(not isinstance(key, str) for key in metadata):
            raise TypeError("target validation metadata keys must be strings")
        try:
            json.dumps(metadata, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise TypeError("target validation metadata must be finite JSON data") from error
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.errors)

    @property
    def warning_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.warnings)


@dataclass(frozen=True)
class TargetAdapterValidator:
    """One immutable adapter ID to validator binding."""

    adapter_id: str
    validator: TargetSourceValidator

    def __post_init__(self) -> None:
        if _ADAPTER_ID.fullmatch(self.adapter_id) is None:
            raise ValueError(f"invalid target adapter ID: {self.adapter_id!r}")
        if not callable(self.validator):
            raise TypeError("target adapter validator must be callable")


@dataclass(frozen=True)
class TargetAdapterRegistry:
    """An immutable registry; extensions create a new value with ``with_validator``."""

    validators: tuple[TargetAdapterValidator, ...] = ()

    def __post_init__(self) -> None:
        validators = tuple(self.validators)
        identifiers = tuple(binding.adapter_id for binding in validators)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("target adapter IDs must be unique")
        object.__setattr__(self, "validators", validators)

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        return tuple(binding.adapter_id for binding in self.validators)

    def with_validator(
        self,
        adapter_id: str,
        validator: TargetSourceValidator,
    ) -> TargetAdapterRegistry:
        """Return a new registry containing one additional validator binding."""

        if adapter_id in self.adapter_ids:
            raise ValueError(f"target adapter is already registered: {adapter_id}")
        binding = TargetAdapterValidator(adapter_id=adapter_id, validator=validator)
        return TargetAdapterRegistry(validators=(*self.validators, binding))

    def validator_for(self, adapter_id: str) -> TargetSourceValidator | None:
        for binding in self.validators:
            if binding.adapter_id == adapter_id:
                return binding.validator
        return None


def _validate_kernelbench(source: str, target: TargetStackSpec) -> TargetValidationResult:
    legacy = validate_candidate_source(source, target.backend)
    return TargetValidationResult(valid=legacy.valid, errors=legacy.errors)


DEFAULT_TARGET_ADAPTER_REGISTRY = TargetAdapterRegistry().with_validator(
    "kernelbench",
    _validate_kernelbench,
)


def validate_target_source(
    source: str,
    target: TargetStackSpec,
    *,
    registry: TargetAdapterRegistry = DEFAULT_TARGET_ADAPTER_REGISTRY,
) -> TargetValidationResult:
    """Validate source with the adapter named by the hash-bound target contract."""

    validator = registry.validator_for(target.adapter)
    if validator is None:
        issue = StaticValidationIssue(
            code="unknown_target_adapter",
            message=(f"target {target.id!r} selects unregistered adapter {target.adapter!r}"),
        )
        return TargetValidationResult(valid=False, errors=(issue,))
    result = validator(source, target)
    if not isinstance(result, TargetValidationResult):
        raise TypeError(
            f"target adapter {target.adapter!r} returned {type(result).__name__}, "
            "expected TargetValidationResult"
        )
    return result
