"""Fail-closed offline target-use qualification contracts and validators.

This module performs AST inspection and validates synthetic supervisor/launch
attestations.  It never imports a DSL runtime and never executes candidate code.
Even a fully passing offline fixture is reported as ``pending-m9``: trusted OS
containment and a real target launch can only be established by the separately
authorized M9 GPU preflight.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, ValidationError, field_validator, model_validator

from abstrak.anytime.contracts import IDENTIFIER_PATTERN, SHA256_PATTERN, AnytimeModel
from abstrak.anytime.isolation import (
    AnytimeCandidateInvocation,
    AnytimeProcessIsolationContract,
    AnytimeTargetBackend,
    verify_anytime_candidate_invocation,
    verify_anytime_isolation_contract,
)
from abstrak.providers.contracts import sha256_json


class AnytimeQualificationError(ValueError):
    """Raised when a qualification trust anchor is internally inconsistent."""


class AnytimeStaticValidationIssue(AnytimeModel):
    """Stable source diagnostic emitted without importing a target runtime."""

    schema_version: Literal["abstrak-anytime-static-validation-issue.v1"] = (
        "abstrak-anytime-static-validation-issue.v1"
    )
    code: str = Field(pattern=IDENTIFIER_PATTERN)
    message: str = Field(min_length=1, max_length=1000)
    line: int | None = Field(default=None, ge=1)
    column: int | None = Field(default=None, ge=0)


class AnytimeTargetStaticPolicy(AnytimeModel):
    """One target's independent default-deny Python/DSL surface."""

    schema_version: Literal["abstrak-anytime-target-static-policy.v1"] = (
        "abstrak-anytime-target-static-policy.v1"
    )
    backend: AnytimeTargetBackend
    policy_id: str = Field(pattern=IDENTIFIER_PATTERN)
    allowed_imports: tuple[str, ...] = Field(min_length=1)
    allowed_exact_calls: tuple[str, ...] = Field(min_length=1)
    allowed_call_prefixes: tuple[str, ...] = Field(min_length=1)
    allowed_decorators: tuple[str, ...] = Field(min_length=1)
    required_any_decorators: tuple[str, ...] = Field(min_length=1)
    required_exact_calls: tuple[str, ...] = ()
    required_any_operations: tuple[str, ...] = Field(min_length=1)

    @field_validator(
        "allowed_imports",
        "allowed_exact_calls",
        "allowed_call_prefixes",
        "allowed_decorators",
        "required_any_decorators",
        "required_exact_calls",
        "required_any_operations",
    )
    @classmethod
    def entries_are_unique_and_ordered(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("target policy entries must be unique")
        if values != tuple(sorted(values)):
            raise ValueError("target policy entries must use canonical sorted order")
        return values

    @property
    def sha256(self) -> str:
        return sha256_json(self)


_PYTHON_SCAFFOLD_CALLS = (
    "abs",
    "bool",
    "enumerate",
    "float",
    "int",
    "len",
    "max",
    "min",
    "range",
    "super",
    "super.__init__",
    "tuple",
)
_SAFE_TORCH_ALLOCATION_CALLS = (
    "torch.empty",
    "torch.empty_like",
    "torch.empty_strided",
    "torch.zeros",
)


def _sorted(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(values))


_TRITON_POLICY = AnytimeTargetStaticPolicy(
    backend="triton",
    policy_id="anytime-triton-default-deny.v1",
    allowed_imports=_sorted(("torch", "torch.nn", "triton", "triton.language")),
    allowed_exact_calls=_sorted(
        (
            *_PYTHON_SCAFFOLD_CALLS,
            *_SAFE_TORCH_ALLOCATION_CALLS,
            "triton.Config",
            "triton.cdiv",
            "triton.next_power_of_2",
        )
    ),
    allowed_call_prefixes=("triton.language.",),
    allowed_decorators=_sorted(("triton.autotune", "triton.heuristics", "triton.jit")),
    required_any_decorators=_sorted(("triton.autotune", "triton.jit")),
    required_any_operations=_sorted(("triton.language.load", "triton.language.store")),
)

_TILELANG_POLICY = AnytimeTargetStaticPolicy(
    backend="tilelang",
    policy_id="anytime-tilelang-default-deny.v1",
    allowed_imports=_sorted(("tilelang", "tilelang.language", "torch", "torch.nn")),
    allowed_exact_calls=_sorted(
        (*_PYTHON_SCAFFOLD_CALLS, *_SAFE_TORCH_ALLOCATION_CALLS, "tilelang.compile")
    ),
    allowed_call_prefixes=("tilelang.language.",),
    allowed_decorators=("tilelang.language.prim_func",),
    required_any_decorators=("tilelang.language.prim_func",),
    required_exact_calls=("tilelang.compile",),
    required_any_operations=_sorted(
        (
            "tilelang.language.Kernel",
            "tilelang.language.Parallel",
            "tilelang.language.copy",
            "tilelang.language.gemm",
            "tilelang.language.reduce_sum",
        )
    ),
)

_CUTE_POLICY = AnytimeTargetStaticPolicy(
    backend="cute",
    policy_id="anytime-cute-default-deny.v1",
    allowed_imports=_sorted(("cutlass", "cutlass.cute", "torch", "torch.nn")),
    allowed_exact_calls=_sorted(
        (*_PYTHON_SCAFFOLD_CALLS, *_SAFE_TORCH_ALLOCATION_CALLS, "cutlass.cute.compile")
    ),
    allowed_call_prefixes=("cutlass.cute.",),
    allowed_decorators=_sorted(("cutlass.cute.jit", "cutlass.cute.kernel")),
    required_any_decorators=_sorted(("cutlass.cute.jit", "cutlass.cute.kernel")),
    required_exact_calls=("cutlass.cute.compile",),
    required_any_operations=_sorted(
        (
            "cutlass.cute.copy",
            "cutlass.cute.gemm",
            "cutlass.cute.make_tensor",
            "cutlass.cute.mma",
        )
    ),
)

_POLICIES: dict[AnytimeTargetBackend, AnytimeTargetStaticPolicy] = {
    "triton": _TRITON_POLICY,
    "tilelang": _TILELANG_POLICY,
    "cute": _CUTE_POLICY,
}


def get_anytime_target_static_policy(
    backend: AnytimeTargetBackend,
) -> AnytimeTargetStaticPolicy:
    """Return a defensive copy of one target-specific policy."""

    try:
        return _POLICIES[backend].model_copy(deep=True)
    except KeyError:
        raise AnytimeQualificationError(f"unsupported anytime target backend: {backend}") from None


class AnytimeStaticValidationResult(AnytimeModel):
    """Hash-bound result of one target-specific AST validation."""

    schema_version: Literal["abstrak-anytime-static-validation-result.v1"] = (
        "abstrak-anytime-static-validation-result.v1"
    )
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    backend: AnytimeTargetBackend
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    policy_sha256: str = Field(pattern=SHA256_PATTERN)
    valid: bool
    backend_signature_present: bool
    target_operation_count: int = Field(ge=0)
    issues: tuple[AnytimeStaticValidationIssue, ...] = ()

    @model_validator(mode="after")
    def validity_is_derived_from_issues(self) -> AnytimeStaticValidationResult:
        if self.valid != (not self.issues):
            raise ValueError("static-validation validity must be derived from issues")
        if self.backend_signature_present != (self.target_operation_count > 0):
            raise ValueError("backend signature must include a target operation")
        return self

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _attribute_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, ast.Attribute | ast.Subscript):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


class _ImportBindings(ast.NodeVisitor):
    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            local = item.asname or item.name.partition(".")[0]
            resolved = item.name if item.asname else item.name.partition(".")[0]
            self.aliases[local] = resolved

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module is None:
            return
        for item in node.names:
            if item.name != "*":
                self.aliases[item.asname or item.name] = f"{node.module}.{item.name}"

    def resolve(self, path: str) -> str:
        root, separator, suffix = path.partition(".")
        resolved = self.aliases.get(root, root)
        return f"{resolved}.{suffix}" if separator else resolved


_PRIVATE_SOURCE_MARKERS = re.compile(
    r"(?:^|[_-])(expert|oracle|private|reference_source|sealed)(?:$|[_-])",
    re.IGNORECASE,
)
_FORBIDDEN_PATH_FRAGMENT = re.compile(
    r"(?:^|[\"'])(?:/|[A-Za-z]:\\)|(?:^|/)(?:etc|home|proc|repo|sys|workspace)/|"
    r"(?:^|/)\.git(?:/|$)|(?:^|/)\.\.(?:/|$)|file://",
    re.IGNORECASE,
)
_FRAME_ATTRIBUTES = frozenset(
    {
        "cr_frame",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "gi_frame",
        "tb_frame",
    }
)
_SAFE_INPUT_METHODS = frozenset({"dim", "numel", "size", "stride"})


def _issue(code: str, message: str, node: ast.AST | None = None) -> AnytimeStaticValidationIssue:
    return AnytimeStaticValidationIssue(
        code=code,
        message=message,
        line=getattr(node, "lineno", None),
        column=getattr(node, "col_offset", None),
    )


class _DefaultDenyInspector(ast.NodeVisitor):
    def __init__(
        self,
        *,
        policy: AnytimeTargetStaticPolicy,
        imports: _ImportBindings,
        forward_inputs: frozenset[str],
        decorated_kernels: frozenset[str],
        compiled_callables: frozenset[str],
    ) -> None:
        self.policy = policy
        self.imports = imports
        self.forward_inputs = forward_inputs
        self.decorated_kernels = decorated_kernels
        self.compiled_callables = compiled_callables
        self.calls: list[str] = []
        self.decorators: list[str] = []
        self.issues: list[AnytimeStaticValidationIssue] = []

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            if item.name not in self.policy.allowed_imports:
                self.issues.append(
                    _issue(
                        "import_not_allowed",
                        f"{self.policy.backend} policy does not allow import {item.name!r}",
                        node,
                    )
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module is None:
            self.issues.append(_issue("import_not_allowed", "relative imports are forbidden", node))
            return
        if node.module not in self.policy.allowed_imports:
            self.issues.append(
                _issue(
                    "import_not_allowed",
                    f"{self.policy.backend} policy does not allow import {node.module!r}",
                    node,
                )
            )
        if any(item.name == "*" for item in node.names):
            self.issues.append(
                _issue("dynamic_lookup_forbidden", "wildcard imports are forbidden", node)
            )
        if any(item.name.startswith("_") for item in node.names):
            self.issues.append(
                _issue(
                    "dynamic_lookup_forbidden",
                    "private or dunder imports are forbidden",
                    node,
                )
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_decorators(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.issues.append(
            _issue("async_control_forbidden", "async candidate functions are forbidden", node)
        )
        self._record_decorators(node)
        self.generic_visit(node)

    def _record_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            raw = _attribute_path(target)
            if raw is None:
                self.issues.append(
                    _issue("decorator_not_allowed", "dynamic decorators are forbidden", decorator)
                )
                continue
            resolved = self.imports.resolve(raw)
            self.decorators.append(resolved)
            if resolved not in self.policy.allowed_decorators:
                self.issues.append(
                    _issue(
                        "decorator_not_allowed",
                        f"{self.policy.backend} policy does not allow decorator {resolved!r}",
                        decorator,
                    )
                )

    def visit_Call(self, node: ast.Call) -> None:
        path = self._resolved_call_path(node)
        if path is None:
            self.issues.append(
                _issue(
                    "dynamic_call_forbidden",
                    "indirect or dynamically looked-up calls are forbidden",
                    node,
                )
            )
        else:
            self.calls.append(path)
            if not self._call_is_allowed(node, path):
                code = self._call_issue_code(node, path)
                self.issues.append(
                    _issue(
                        code,
                        f"{self.policy.backend} default-deny policy rejects call {path!r}",
                        node,
                    )
                )
            self._check_input_mutation_call(node, path)
        self.generic_visit(node)

    def _resolved_call_path(self, node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Subscript):
            raw_base = _attribute_path(node.func.value)
            if raw_base in self.decorated_kernels:
                return f"{raw_base}.__target_launch__"
            return None
        raw = _attribute_path(node.func)
        if raw is not None:
            return self.imports.resolve(raw)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "__init__"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
        ):
            return "super.__init__"
        return None

    def _call_is_allowed(self, node: ast.Call, path: str) -> bool:
        if path in self.policy.allowed_exact_calls:
            return True
        if path in self.policy.allowed_decorators:
            return True
        if any(path.startswith(prefix) for prefix in self.policy.allowed_call_prefixes):
            return True
        if path.endswith(".__target_launch__"):
            return path.removesuffix(".__target_launch__") in self.decorated_kernels
        if path in self.compiled_callables:
            return True
        raw = _attribute_path(node.func)
        if raw is not None:
            root, _, method = raw.partition(".")
            if root in self.forward_inputs and method in _SAFE_INPUT_METHODS:
                return True
        return False

    def _call_issue_code(self, node: ast.Call, path: str) -> str:
        raw = _attribute_path(node.func)
        root = raw.partition(".")[0] if raw is not None else ""
        if path == "open" or path.startswith(("os.", "pathlib.", "io.")):
            return "filesystem_access_forbidden"
        if path in {
            "__import__",
            "compile",
            "eval",
            "exec",
            "getattr",
            "globals",
            "locals",
            "vars",
        }:
            return "dynamic_lookup_forbidden"
        if path.startswith(("inspect.", "sys._getframe")):
            return "frame_inspection_forbidden"
        if path.startswith("torch.") or root in self.forward_inputs:
            return "framework_fallback_forbidden"
        return "call_not_allowed"

    def _check_input_mutation_call(self, node: ast.Call, path: str) -> None:
        raw = _attribute_path(node.func)
        if raw is None:
            return
        root, _, method = raw.partition(".")
        if root in self.forward_inputs and (method.endswith("_") or method in {"set", "resize"}):
            self.issues.append(
                _issue(
                    "input_mutation_forbidden",
                    f"input-mutating call {path!r} is forbidden",
                    node,
                )
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _FRAME_ATTRIBUTES:
            self.issues.append(
                _issue(
                    "frame_inspection_forbidden",
                    f"frame attribute {node.attr!r} is forbidden",
                    node,
                )
            )
        if node.attr.startswith("__") and node.attr.endswith("__") and node.attr != "__init__":
            self.issues.append(
                _issue(
                    "dynamic_lookup_forbidden",
                    f"dunder attribute {node.attr!r} is forbidden",
                    node,
                )
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "__builtins__":
            self.issues.append(
                _issue("dynamic_lookup_forbidden", "__builtins__ access is forbidden", node)
            )
        if _PRIVATE_SOURCE_MARKERS.search(node.id):
            self.issues.append(
                _issue(
                    "private_asset_reference",
                    f"private-looking symbol {node.id!r} is forbidden",
                    node,
                )
            )
        if isinstance(node.ctx, ast.Store) and node.id in self.forward_inputs:
            self.issues.append(
                _issue(
                    "input_mutation_forbidden",
                    f"rebinding input {node.id!r} is forbidden",
                    node,
                )
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_mutation_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_mutation_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_mutation_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._check_mutation_target(target)
        self.generic_visit(node)

    def _check_mutation_target(self, target: ast.AST) -> None:
        if _root_name(target) in self.forward_inputs:
            self.issues.append(
                _issue(
                    "input_mutation_forbidden",
                    "candidate source may not mutate an input",
                    target,
                )
            )

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            self.issues.append(
                _issue(
                    "framework_fallback_forbidden",
                    "Python matrix multiplication is forbidden",
                    node,
                )
            )
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.issues.append(
            _issue("unbounded_control_flow", "while loops are rejected by the offline policy", node)
        )
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.issues.append(
            _issue("dynamic_call_forbidden", "lambda call targets are forbidden", node)
        )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.issues.append(
            _issue("global_state_forbidden", "global declarations are forbidden", node)
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.issues.append(
            _issue("global_state_forbidden", "nonlocal declarations are forbidden", node)
        )

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            if _FORBIDDEN_PATH_FRAGMENT.search(node.value):
                self.issues.append(
                    _issue(
                        "filesystem_path_forbidden",
                        "filesystem-looking string is forbidden",
                        node,
                    )
                )
            if _PRIVATE_SOURCE_MARKERS.search(node.value):
                self.issues.append(
                    _issue("private_asset_reference", "private benchmark marker is forbidden", node)
                )


def _forward_inputs(tree: ast.Module) -> frozenset[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "forward":
            arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            result.update(argument.arg for argument in arguments if argument.arg != "self")
    return frozenset(result)


def _decorated_kernel_names(
    tree: ast.Module,
    imports: _ImportBindings,
    policy: AnytimeTargetStaticPolicy,
) -> frozenset[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            raw = _attribute_path(target)
            if raw is not None and imports.resolve(raw) in policy.allowed_decorators:
                result.add(node.name)
    rebound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in result:
            rebound.add(node.id)
    return frozenset(result - rebound)


def _compiled_callable_names(
    tree: ast.Module,
    imports: _ImportBindings,
    policy: AnytimeTargetStaticPolicy,
) -> frozenset[str]:
    assignments: dict[str, list[bool]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            path = _attribute_path(target)
            if path is not None:
                is_compile = False
                if isinstance(value, ast.Call):
                    raw = _attribute_path(value.func)
                    is_compile = (
                        raw is not None and imports.resolve(raw) in policy.required_exact_calls
                    )
                assignments.setdefault(path, []).append(is_compile)
    return frozenset(path for path, sources in assignments.items() if sources == [True])


def _deduplicate_issues(
    issues: list[AnytimeStaticValidationIssue],
) -> tuple[AnytimeStaticValidationIssue, ...]:
    result: list[AnytimeStaticValidationIssue] = []
    seen: set[tuple[str, int | None, int | None, str]] = set()
    for issue in issues:
        key = (issue.code, issue.line, issue.column, issue.message)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return tuple(result)


def validate_anytime_candidate_source(
    source: str,
    *,
    target_id: str,
    backend: AnytimeTargetBackend,
) -> AnytimeStaticValidationResult:
    """Validate source with one backend policy; no target package is imported."""

    policy = get_anytime_target_static_policy(backend)
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as error:
        issue = AnytimeStaticValidationIssue(
            code="syntax_error",
            message=str(getattr(error, "msg", error)),
            line=getattr(error, "lineno", None),
            column=getattr(error, "offset", None),
        )
        return AnytimeStaticValidationResult(
            target_id=target_id,
            backend=backend,
            source_sha256=source_sha256,
            policy_sha256=policy.sha256,
            valid=False,
            backend_signature_present=False,
            target_operation_count=0,
            issues=(issue,),
        )

    imports = _ImportBindings()
    imports.visit(tree)
    decorated = _decorated_kernel_names(tree, imports, policy)
    compiled = _compiled_callable_names(tree, imports, policy)
    inspector = _DefaultDenyInspector(
        policy=policy,
        imports=imports,
        forward_inputs=_forward_inputs(tree),
        decorated_kernels=decorated,
        compiled_callables=compiled,
    )
    inspector.visit(tree)

    present_decorators = set(inspector.decorators)
    present_calls = set(inspector.calls)
    has_decorator = bool(present_decorators.intersection(policy.required_any_decorators))
    if not has_decorator:
        inspector.issues.append(
            _issue("missing_target_decorator", f"source lacks a {backend} kernel decorator")
        )
    for required_call in policy.required_exact_calls:
        if required_call not in present_calls:
            inspector.issues.append(
                _issue("missing_target_compile", f"source must call {required_call!r}")
            )
    operation_count = sum(1 for call in inspector.calls if call in policy.required_any_operations)
    if operation_count == 0:
        inspector.issues.append(
            _issue(
                "dummy_target_signature",
                f"{backend} signature does not contain a recognized target operation",
            )
        )
    if backend == "triton":
        for required in ("triton.language.load", "triton.language.store"):
            if required not in present_calls:
                inspector.issues.append(
                    _issue("missing_target_operation", f"Triton source must call {required!r}")
                )

    issues = _deduplicate_issues(inspector.issues)
    signature = (
        has_decorator
        and operation_count > 0
        and all(required in present_calls for required in policy.required_exact_calls)
    )
    if backend == "triton":
        signature = signature and {
            "triton.language.load",
            "triton.language.store",
        }.issubset(present_calls)
    return AnytimeStaticValidationResult(
        target_id=target_id,
        backend=backend,
        source_sha256=source_sha256,
        policy_sha256=policy.sha256,
        valid=not issues,
        backend_signature_present=signature,
        target_operation_count=operation_count if signature else 0,
        issues=issues,
    )


class AnytimeCandidateQualificationBinding(AnytimeModel):
    """Frozen source, target, invocation and execution identity."""

    schema_version: Literal["abstrak-anytime-candidate-qualification-binding.v1"] = (
        "abstrak-anytime-candidate-qualification-binding.v1"
    )
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    backend: AnytimeTargetBackend
    target_stack_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_source_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_invocation_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_binding_sha256: str = Field(pattern=SHA256_PATTERN)
    static_policy_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def policy_matches_backend(self) -> AnytimeCandidateQualificationBinding:
        if self.static_policy_sha256 != get_anytime_target_static_policy(self.backend).sha256:
            raise ValueError("qualification binding uses the wrong target static policy")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def build_anytime_qualification_binding(
    *,
    invocation: AnytimeCandidateInvocation,
    target_stack_sha256: str,
    execution_binding_sha256: str,
) -> AnytimeCandidateQualificationBinding:
    """Bind a validated candidate-visible request to non-visible controller digests."""

    trusted = verify_anytime_candidate_invocation(invocation)
    policy = get_anytime_target_static_policy(trusted.public_runtime.backend)
    return AnytimeCandidateQualificationBinding(
        target_id=trusted.public_runtime.target_id,
        backend=trusted.public_runtime.backend,
        target_stack_sha256=target_stack_sha256,
        candidate_source_sha256=trusted.source.source_sha256,
        candidate_invocation_sha256=trusted.sha256,
        execution_binding_sha256=execution_binding_sha256,
        static_policy_sha256=policy.sha256,
    )


class AnytimeSyntheticRuntimeObservation(AnytimeModel):
    """Scripted supervisor fact used only to test fail-closed derivation."""

    schema_version: Literal["abstrak-anytime-synthetic-runtime-observation.v1"] = (
        "abstrak-anytime-synthetic-runtime-observation.v1"
    )
    origin: Literal["offline-synthetic-fixture"] = "offline-synthetic-fixture"
    observer_role: Literal["reference-qualifier-supervisor"] = "reference-qualifier-supervisor"
    candidate_invocation_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_source_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_binding_sha256: str = Field(pattern=SHA256_PATTERN)
    terminal_status: Literal["completed", "timeout", "oom", "crash"]
    expected_output_count: int = Field(ge=1, le=16)
    observed_output_count: int = Field(ge=0, le=16)
    outputs_finite: bool
    inputs_unchanged: bool
    ipc_envelope_valid: bool
    elapsed_seconds: float = Field(ge=0, le=3600)
    timing_source: Literal["qualifier-monotonic-clock"] = "qualifier-monotonic-clock"
    candidate_reported_timing_accepted: Literal[False] = False
    real_os_containment_observed: Literal[False] = False

    @field_validator("elapsed_seconds")
    @classmethod
    def elapsed_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("synthetic runtime elapsed time must be finite")
        return 0.0 if value == 0.0 else value

    @property
    def sha256(self) -> str:
        return sha256_json(self)


class _SyntheticLaunchPayload(AnytimeModel):
    """Fields shared by target-specific scripted launch evidence."""

    origin: Literal["offline-synthetic-fixture"] = "offline-synthetic-fixture"
    observer_role: Literal["reference-qualifier"] = "reference-qualifier"
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target_stack_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_source_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_invocation_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_binding_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_mode: Literal["runtime", "lowered", "runtime-and-lowered"] = "runtime-and-lowered"
    runtime_launch_count: int = Field(ge=0, le=1000000)
    launched_kernel_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    lowered_code_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    core_operation_attributed: bool
    fallback_detected: bool
    dummy_signature_only: bool


class AnytimeTritonSyntheticLaunchPayload(_SyntheticLaunchPayload):
    schema_version: Literal["abstrak-anytime-triton-synthetic-launch.v1"] = (
        "abstrak-anytime-triton-synthetic-launch.v1"
    )
    backend: Literal["triton"] = "triton"
    runtime_observer: Literal["triton-launch-hook.v1"] = "triton-launch-hook.v1"
    lowered_capture: Literal["triton-ttir-or-ptx.v1"] = "triton-ttir-or-ptx.v1"


class AnytimeTileLangSyntheticLaunchPayload(_SyntheticLaunchPayload):
    schema_version: Literal["abstrak-anytime-tilelang-synthetic-launch.v1"] = (
        "abstrak-anytime-tilelang-synthetic-launch.v1"
    )
    backend: Literal["tilelang"] = "tilelang"
    runtime_observer: Literal["tilelang-launch-hook.v1"] = "tilelang-launch-hook.v1"
    lowered_capture: Literal["tilelang-lowered-cuda.v1"] = "tilelang-lowered-cuda.v1"


class AnytimeCuteSyntheticLaunchPayload(_SyntheticLaunchPayload):
    schema_version: Literal["abstrak-anytime-cute-synthetic-launch.v1"] = (
        "abstrak-anytime-cute-synthetic-launch.v1"
    )
    backend: Literal["cute"] = "cute"
    runtime_observer: Literal["cute-dsl-launch-hook.v1"] = "cute-dsl-launch-hook.v1"
    lowered_capture: Literal["cute-dsl-lowered-cuda.v1"] = "cute-dsl-lowered-cuda.v1"


AnytimeSyntheticLaunchPayload: TypeAlias = Annotated[
    AnytimeTritonSyntheticLaunchPayload
    | AnytimeTileLangSyntheticLaunchPayload
    | AnytimeCuteSyntheticLaunchPayload,
    Field(discriminator="backend"),
]


class AnytimeSyntheticLaunchAttestation(AnytimeModel):
    """Hash-bound scripted evidence; its origin can never satisfy the live gate."""

    schema_version: Literal["abstrak-anytime-synthetic-launch-attestation.v1"] = (
        "abstrak-anytime-synthetic-launch-attestation.v1"
    )
    payload: AnytimeSyntheticLaunchPayload
    payload_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def digest_matches_payload(self) -> AnytimeSyntheticLaunchAttestation:
        if self.payload_sha256 != sha256_json(self.payload):
            raise ValueError("launch attestation payload digest mismatch")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def attest_anytime_synthetic_launch(
    payload: AnytimeSyntheticLaunchPayload,
) -> AnytimeSyntheticLaunchAttestation:
    """Seal a scripted payload for offline validator tests only."""

    return AnytimeSyntheticLaunchAttestation(
        payload=payload,
        payload_sha256=sha256_json(payload),
    )


class AnytimeOfflineQualificationDecision(AnytimeModel):
    """Offline result with no representable formal-qualified state."""

    schema_version: Literal["abstrak-anytime-offline-qualification-decision.v1"] = (
        "abstrak-anytime-offline-qualification-decision.v1"
    )
    binding_sha256: str = Field(pattern=SHA256_PATTERN)
    isolation_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    static_validation_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_observation_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    launch_attestation_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    status: Literal["rejected", "pending-m9"]
    rejection_codes: tuple[str, ...]
    formal_target_use_qualified: Literal[False] = False
    real_os_containment_observed: Literal[False] = False
    trusted_gpu_launch_observed: Literal[False] = False
    next_gate: Literal["m9-trusted-gpu-preflight"] = "m9-trusted-gpu-preflight"

    @field_validator("rejection_codes")
    @classmethod
    def rejection_codes_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("qualification rejection codes must be unique")
        return values

    @model_validator(mode="after")
    def status_matches_rejections(self) -> AnytimeOfflineQualificationDecision:
        if self.status == "rejected" and not self.rejection_codes:
            raise ValueError("rejected qualification requires at least one reason")
        if self.status == "pending-m9" and self.rejection_codes:
            raise ValueError("pending M9 qualification cannot contain rejection reasons")
        if self.status == "pending-m9" and (
            self.runtime_observation_sha256 is None or self.launch_attestation_sha256 is None
        ):
            raise ValueError("pending M9 requires complete synthetic fixture bindings")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _strict_revalidate(model_type: type[AnytimeModel], value: object) -> AnytimeModel:
    if isinstance(value, AnytimeModel):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True)
    return model_type.model_validate_json(encoded)


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def qualify_anytime_candidate_offline(
    *,
    binding: AnytimeCandidateQualificationBinding,
    invocation: AnytimeCandidateInvocation,
    isolation_contract: AnytimeProcessIsolationContract,
    runtime_observation: object | None,
    launch_attestation: object | None,
) -> AnytimeOfflineQualificationDecision:
    """Derive a fail-closed offline decision without executing candidate code."""

    try:
        trusted_binding = _strict_revalidate(AnytimeCandidateQualificationBinding, binding)
        assert isinstance(trusted_binding, AnytimeCandidateQualificationBinding)
        trusted_invocation = verify_anytime_candidate_invocation(invocation)
        trusted_isolation = verify_anytime_isolation_contract(isolation_contract)
    except (AnytimeQualificationError, ValueError, TypeError) as error:
        raise AnytimeQualificationError(f"invalid qualification trust anchor: {error}") from error

    rejections: list[str] = []
    if (
        trusted_binding.target_id != trusted_invocation.public_runtime.target_id
        or trusted_binding.backend != trusted_invocation.public_runtime.backend
        or trusted_binding.candidate_source_sha256 != trusted_invocation.source.source_sha256
        or trusted_binding.candidate_invocation_sha256 != trusted_invocation.sha256
    ):
        _append_unique(rejections, "candidate_binding_mismatch")

    static = validate_anytime_candidate_source(
        trusted_invocation.source.text,
        target_id=trusted_binding.target_id,
        backend=trusted_binding.backend,
    )
    if static.policy_sha256 != trusted_binding.static_policy_sha256:
        _append_unique(rejections, "static_policy_mismatch")
    for code in static.error_codes:
        _append_unique(rejections, f"static-{code}")

    runtime: AnytimeSyntheticRuntimeObservation | None = None
    if runtime_observation is None:
        _append_unique(rejections, "runtime_observation_missing")
    else:
        try:
            candidate_runtime = _strict_revalidate(
                AnytimeSyntheticRuntimeObservation,
                runtime_observation,
            )
            assert isinstance(candidate_runtime, AnytimeSyntheticRuntimeObservation)
            runtime = candidate_runtime
        except (ValidationError, TypeError, ValueError):
            _append_unique(rejections, "runtime_observation_malformed")
    if runtime is not None:
        if (
            runtime.candidate_invocation_sha256 != trusted_binding.candidate_invocation_sha256
            or runtime.candidate_source_sha256 != trusted_binding.candidate_source_sha256
            or runtime.execution_binding_sha256 != trusted_binding.execution_binding_sha256
        ):
            _append_unique(rejections, "runtime_binding_mismatch")
        if runtime.terminal_status != "completed":
            _append_unique(rejections, f"candidate_{runtime.terminal_status}")
        if runtime.observed_output_count != runtime.expected_output_count:
            _append_unique(rejections, "output_count_mismatch")
        if runtime.expected_output_count != trusted_invocation.public_abi.output_count:
            _append_unique(rejections, "runtime_abi_mismatch")
        if not runtime.outputs_finite:
            _append_unique(rejections, "nonfinite_output")
        if not runtime.inputs_unchanged:
            _append_unique(rejections, "input_mutation_detected")
        if not runtime.ipc_envelope_valid:
            _append_unique(rejections, "runtime_ipc_invalid")

    attestation: AnytimeSyntheticLaunchAttestation | None = None
    if launch_attestation is None:
        _append_unique(rejections, "launch_evidence_missing")
    else:
        try:
            candidate_attestation = _strict_revalidate(
                AnytimeSyntheticLaunchAttestation,
                launch_attestation,
            )
            assert isinstance(candidate_attestation, AnytimeSyntheticLaunchAttestation)
            attestation = candidate_attestation
        except (ValidationError, TypeError, ValueError):
            _append_unique(rejections, "launch_evidence_malformed")
    if attestation is not None:
        payload = attestation.payload
        if (
            payload.target_id != trusted_binding.target_id
            or payload.backend != trusted_binding.backend
            or payload.target_stack_sha256 != trusted_binding.target_stack_sha256
            or payload.candidate_source_sha256 != trusted_binding.candidate_source_sha256
            or payload.candidate_invocation_sha256 != trusted_binding.candidate_invocation_sha256
            or payload.execution_binding_sha256 != trusted_binding.execution_binding_sha256
        ):
            _append_unique(rejections, "launch_evidence_binding_mismatch")
        has_runtime = (
            payload.runtime_launch_count >= 1 and payload.launched_kernel_sha256 is not None
        )
        has_lowered = payload.lowered_code_sha256 is not None
        if payload.evidence_mode in {"runtime", "runtime-and-lowered"} and not has_runtime:
            _append_unique(rejections, "target_launch_not_observed")
        if payload.evidence_mode in {"lowered", "runtime-and-lowered"} and not has_lowered:
            _append_unique(rejections, "lowered_code_missing")
        if payload.evidence_mode == "runtime" and has_lowered:
            _append_unique(rejections, "launch_evidence_mode_mismatch")
        if payload.evidence_mode == "lowered" and has_runtime:
            _append_unique(rejections, "launch_evidence_mode_mismatch")
        if not payload.core_operation_attributed:
            _append_unique(rejections, "core_operation_not_attributed")
        if payload.fallback_detected:
            _append_unique(rejections, "framework_fallback_detected")
        if payload.dummy_signature_only:
            _append_unique(rejections, "dummy_target_signature")

    status: Literal["rejected", "pending-m9"] = "rejected" if rejections else "pending-m9"
    return AnytimeOfflineQualificationDecision(
        binding_sha256=trusted_binding.sha256,
        isolation_contract_sha256=trusted_isolation.sha256,
        static_validation_sha256=static.sha256,
        runtime_observation_sha256=runtime.sha256 if runtime is not None else None,
        launch_attestation_sha256=attestation.sha256 if attestation is not None else None,
        status=status,
        rejection_codes=tuple(rejections),
    )
