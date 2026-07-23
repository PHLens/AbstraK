"""Frozen TileLang capability packs, target cards, and offline validation."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from pydantic import Field, model_validator

from abstrak.canary.contracts import IDENTIFIER_PATTERN, CanaryModel
from abstrak.canary.fallback import StaticValidationIssue, validate_candidate_source
from abstrak.providers.contracts import sha256_json

BASE_CAPABILITY_BIT = 0b001
SCHEDULE_CAPABILITY_BIT = 0b010
MAPPING_CAPABILITY_BIT = 0b100
ALL_CAPABILITY_BITS = BASE_CAPABILITY_BIT | SCHEDULE_CAPABILITY_BIT | MAPPING_CAPABILITY_BIT

PackId = Literal["tileops-core", "tileops-sched", "tileops-map", "tileops-full"]

_PACK_SHAPES: Mapping[str, tuple[str, str, int]] = MappingProxyType(
    {
        "tileops-core": ("tilelang-a100-core", "tilelang-capability-core", 0b001),
        "tileops-sched": ("tilelang-a100-sched", "tilelang-capability-sched", 0b011),
        "tileops-map": ("tilelang-a100-map", "tilelang-capability-map", 0b101),
        "tileops-full": ("tilelang-a100-full", "tilelang-capability-full", 0b111),
    }
)

_MAPPING_APIS = (
    "T.alloc_local",
    "T.annotate_layout",
    "T.get_thread_binding",
    "T.serial",
    "T.sync_threads",
    "T.unroll",
    "T.vectorized",
    "T.warp_reduce_max",
    "T.warp_reduce_min",
    "T.warp_reduce_sum",
    "tilelang.layout.make_swizzled_layout",
)


class CapabilityPackSpec(CanaryModel):
    """One hashable capability contract exposed through a TileLang target."""

    schema_version: Literal["tilelang-capability-pack.v1"] = "tilelang-capability-pack.v1"
    id: PackId
    target_id: str = Field(pattern=IDENTIFIER_PATTERN)
    adapter_id: str = Field(pattern=IDENTIFIER_PATTERN)
    bitmask: int = Field(ge=BASE_CAPABILITY_BIT, le=ALL_CAPABILITY_BITS)
    kernel_threads: tuple[int, ...]
    pipeline_stages: tuple[int, ...]
    gemm_policies: tuple[Literal["Square", "FullRow", "FullCol"], ...] = ()
    mapping_apis: tuple[str, ...] = ()

    @model_validator(mode="after")
    def fields_match_frozen_pack(self) -> CapabilityPackSpec:
        expected_target, expected_adapter, expected_mask = _PACK_SHAPES[self.id]
        if (self.target_id, self.adapter_id, self.bitmask) != (
            expected_target,
            expected_adapter,
            expected_mask,
        ):
            raise ValueError("target, adapter, and bitmask must match the frozen pack ID")
        has_schedule = bool(self.bitmask & SCHEDULE_CAPABILITY_BIT)
        has_mapping = bool(self.bitmask & MAPPING_CAPABILITY_BIT)
        expected_threads = (64, 128, 256) if has_schedule else (128,)
        expected_stages = (0, 1, 2, 3) if has_schedule else (0,)
        expected_policies = ("Square", "FullRow", "FullCol") if has_schedule else ()
        expected_mapping = _MAPPING_APIS if has_mapping else ()
        if self.kernel_threads != expected_threads:
            raise ValueError("kernel thread domain does not match the pack bitmask")
        if self.pipeline_stages != expected_stages:
            raise ValueError("pipeline stage domain does not match the pack bitmask")
        if self.gemm_policies != expected_policies:
            raise ValueError("GEMM policy domain does not match the pack bitmask")
        if self.mapping_apis != expected_mapping:
            raise ValueError("mapping API domain does not match the pack bitmask")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self)


def _pack(pack_id: PackId) -> CapabilityPackSpec:
    target_id, adapter_id, bitmask = _PACK_SHAPES[pack_id]
    has_schedule = bool(bitmask & SCHEDULE_CAPABILITY_BIT)
    has_mapping = bool(bitmask & MAPPING_CAPABILITY_BIT)
    return CapabilityPackSpec(
        id=pack_id,
        target_id=target_id,
        adapter_id=adapter_id,
        bitmask=bitmask,
        kernel_threads=(64, 128, 256) if has_schedule else (128,),
        pipeline_stages=(0, 1, 2, 3) if has_schedule else (0,),
        gemm_policies=("Square", "FullRow", "FullCol") if has_schedule else (),
        mapping_apis=_MAPPING_APIS if has_mapping else (),
    )


CORE_PACK = _pack("tileops-core")
SCHED_PACK = _pack("tileops-sched")
MAP_PACK = _pack("tileops-map")
FULL_PACK = _pack("tileops-full")

_CAPABILITY_PACKS: Mapping[str, CapabilityPackSpec] = MappingProxyType(
    {pack.id: pack for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)}
)


def list_capability_pack_ids() -> tuple[str, ...]:
    """Return capability pack IDs in increasing bitmask order."""

    return tuple(pack.id for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK))


def get_capability_pack(pack_id: str) -> CapabilityPackSpec:
    """Return an isolated copy of a frozen capability pack."""

    try:
        return _CAPABILITY_PACKS[pack_id].model_copy(deep=True)
    except KeyError:
        raise ValueError(f"unknown capability pack: {pack_id}") from None


def minimum_pack_for_bitmask(bitmask: int) -> CapabilityPackSpec:
    """Return the smallest pack containing all requested capability bits."""

    if isinstance(bitmask, bool) or not isinstance(bitmask, int):
        raise TypeError("capability bitmask must be an integer")
    if bitmask < 0 or bitmask & ~ALL_CAPABILITY_BITS:
        raise ValueError(f"invalid capability bitmask: {bitmask}")
    required = bitmask | BASE_CAPABILITY_BIT
    by_mask = {
        CORE_PACK.bitmask: CORE_PACK,
        SCHED_PACK.bitmask: SCHED_PACK,
        MAP_PACK.bitmask: MAP_PACK,
        FULL_PACK.bitmask: FULL_PACK,
    }
    return by_mask[required].model_copy(deep=True)


def _resolve_pack(pack: str | CapabilityPackSpec) -> CapabilityPackSpec:
    if isinstance(pack, str):
        return get_capability_pack(pack)
    if not isinstance(pack, CapabilityPackSpec):
        raise TypeError("pack must be a capability pack ID or CapabilityPackSpec")
    registered = get_capability_pack(pack.id)
    if registered != pack:
        raise ValueError(f"capability pack differs from frozen registry: {pack.id}")
    return pack


_BASE_CARD_APIS = (
    "T.Kernel",
    "T.Parallel",
    "T.Pipelined",
    "T.Tensor",
    "T.alloc_fragment",
    "T.alloc_shared",
    "T.cast",
    "T.ceildiv",
    "T.clear",
    "T.copy",
    "T.erf",
    "T.exp",
    "T.fill",
    "T.float16",
    "T.float32",
    "T.gemm",
    "T.if_then_else",
    "T.infinity",
    "T.int32",
    "T.int64",
    "T.max",
    "T.min",
    "T.pow",
    "T.prim_func",
    "T.reduce_max",
    "T.reduce_min",
    "T.reduce_sum",
    "T.rsqrt",
    "T.sqrt",
    "T.tanh",
    "tilelang.compile",
)


def render_tilelang_target_card(pack: str | CapabilityPackSpec) -> str:
    """Render deterministic Agent-visible documentation from one pack spec."""

    resolved = _resolve_pack(pack)
    threads = ", ".join(str(value) for value in resolved.kernel_threads)
    stages = ", ".join(str(value) for value in resolved.pipeline_stages)
    lines = [
        f"# TileLang 0.1.12 / NVIDIA A100 / {resolved.id}",
        "",
        f"Target ID: `{resolved.target_id}`",
        f"Capability contract SHA-256: `{resolved.sha256}`",
        "",
        "Return one Python source defining `ModelNew`. Use exactly these imports:",
        "",
        "```python",
        "import torch",
        "import tilelang",
        "import tilelang.language as T",
        "from torch import nn",
        "```",
        "",
        'Define kernels with `@T.prim_func` and compile them for `target="cuda"`.',
        "Only direct calls from the surface below are accepted.",
        "",
        "## Base surface",
        "",
        *[f"- `{name}`" for name in _BASE_CARD_APIS],
        "",
        "## Control domains",
        "",
        f"- `T.Kernel(..., threads=...)`: {{{threads}}}",
        f"- `T.Pipelined(..., num_stages=...)`: {{{stages}}}",
        "- `T.Parallel(...)`: positional extents only",
        "- shared/fragment allocation: positional shape; dtype positional or keyword",
        "- `T.gemm` transpose flags: literal booleans only",
        "- high-level reductions: literal `dim` and `clear` controls only",
    ]
    if resolved.gemm_policies:
        policies = ", ".join(f"T.GemmWarpPolicy.{name}" for name in resolved.gemm_policies)
        lines.append(f"- `T.gemm(..., policy=...)`: {{{policies}}}")
    else:
        lines.append("- `T.gemm(...)`: default policy only")
    if resolved.mapping_apis:
        lines.extend(
            (
                "",
                "## Mapping surface",
                "",
                *[f"- `{name}`" for name in resolved.mapping_apis],
                "- `T.get_thread_binding`: dimension 0 only",
                "- `T.alloc_local`: positional shape; dtype positional or keyword",
                "- `T.serial`: positive literal ranges of at most 4096 iterations",
                "- `T.vectorized`: widths 2, 4, or 8",
                "- `T.unroll`: extent and factor at most 16",
                "- `T.sync_threads`: no arguments",
                "- layout dictionaries: automatic swizzle of the same buffer only",
            )
        )
    lines.extend(
        (
            "",
            "Use FP32 intermediates wherever the task contract requires them.",
            "The evaluator imports the source and calls `ModelNew.forward` with CUDA tensors.",
            "",
        )
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class CapabilityValidationResult:
    """Static capability validation and minimum-pack inference for one source."""

    pack_id: str
    valid: bool
    errors: tuple[StaticValidationIssue, ...]
    warnings: tuple[StaticValidationIssue, ...] = ()
    used_capabilities: tuple[str, ...] = ()
    minimum_pack_id: str | None = None
    minimum_pack_bitmask: int | None = None
    pack_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.valid and self.errors:
            raise ValueError("valid capability results cannot contain errors")
        if not self.valid and not self.errors:
            raise ValueError("invalid capability results require errors")
        if tuple(sorted(set(self.used_capabilities))) != self.used_capabilities:
            raise ValueError("used capabilities must be sorted and unique")
        if (self.minimum_pack_id is None) != (self.minimum_pack_bitmask is None):
            raise ValueError("minimum pack ID and bitmask must be supplied together")

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.errors)

    @property
    def warning_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.warnings)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "capability_pack_id": self.pack_id,
                "capability_pack_sha256": self.pack_sha256,
                "used_capabilities": self.used_capabilities,
                "minimum_pack_id": self.minimum_pack_id,
                "minimum_pack_bitmask": self.minimum_pack_bitmask,
            }
        )


_NOT_CONSTANT = object()
_CONSTANT_BINOPS: Mapping[type[ast.operator], Any] = MappingProxyType(
    {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
    }
)
_CONSTANT_UNARYOPS: Mapping[type[ast.unaryop], Any] = MappingProxyType(
    {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Invert: operator.invert}
)


def _fold_constant(node: ast.AST, constants: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, int | float | str | bool | None):
            if isinstance(value, float) and not math.isfinite(value):
                return _NOT_CONSTANT
            if type(value) is int and abs(value) > 2**63:
                return _NOT_CONSTANT
            return value
        return _NOT_CONSTANT
    if isinstance(node, ast.Name):
        return constants.get(node.id, _NOT_CONSTANT)
    if isinstance(node, ast.Tuple | ast.List):
        values = tuple(_fold_constant(item, constants) for item in node.elts)
        return _NOT_CONSTANT if _NOT_CONSTANT in values else values
    if isinstance(node, ast.UnaryOp):
        operand = _fold_constant(node.operand, constants)
        function = _CONSTANT_UNARYOPS.get(type(node.op))
        if operand is _NOT_CONSTANT or function is None or isinstance(operand, bool | str | tuple):
            return _NOT_CONSTANT
        try:
            value = function(operand)
        except (ArithmeticError, MemoryError, TypeError, ValueError):
            return _NOT_CONSTANT
        return value if isinstance(value, int | float) and abs(value) <= 2**63 else _NOT_CONSTANT
    if isinstance(node, ast.BinOp):
        left = _fold_constant(node.left, constants)
        right = _fold_constant(node.right, constants)
        function = _CONSTANT_BINOPS.get(type(node.op))
        if (
            left is _NOT_CONSTANT
            or right is _NOT_CONSTANT
            or function is None
            or isinstance(left, bool | str | tuple)
            or isinstance(right, bool | str | tuple)
        ):
            return _NOT_CONSTANT
        if isinstance(node.op, ast.LShift | ast.RShift) and (
            type(right) is not int or not 0 <= right <= 63
        ):
            return _NOT_CONSTANT
        try:
            value = function(left, right)
        except (ArithmeticError, MemoryError, TypeError, ValueError):
            return _NOT_CONSTANT
        return value if isinstance(value, int | float) and abs(value) <= 2**63 else _NOT_CONSTANT
    return _NOT_CONSTANT


class _BindingCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.counts: defaultdict[str, int] = defaultdict(int)

    def _bind(self, name: str) -> None:
        self.counts[name] += 1

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store | ast.Del):
            self._bind(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self._bind(node.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        self._bind(node.asname or node.name.partition(".")[0])

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self._bind(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self._bind(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self._bind(node.name)
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None:
            self._bind(node.rest)
        self.generic_visit(node)


def _binding_counts(tree: ast.Module) -> Mapping[str, int]:
    collector = _BindingCollector()
    collector.visit(tree)
    return MappingProxyType(dict(collector.counts))


class _ScopeBindingCollector(_BindingCollector):
    """Collect bindings in one lexical scope without descending into child scopes."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._bind(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _scope_binding_counts(
    statements: list[ast.stmt],
    arguments: ast.arguments | None = None,
) -> Mapping[str, int]:
    collector = _ScopeBindingCollector()
    if arguments is not None:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            collector._bind(argument.arg)
        if arguments.vararg is not None:
            collector._bind(arguments.vararg.arg)
        if arguments.kwarg is not None:
            collector._bind(arguments.kwarg.arg)
    for statement in statements:
        collector.visit(statement)
    return MappingProxyType(dict(collector.counts))


def _module_constants(tree: ast.Module) -> Mapping[str, Any]:
    constants: dict[str, Any] = {}
    bindings = _binding_counts(tree)
    for statement in tree.body:
        name: str | None = None
        value_node: ast.AST | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            name = statement.targets[0].id
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            name = statement.target.id
            value_node = statement.value
        if name is None or value_node is None:
            continue
        if bindings.get(name) != 1:
            continue
        value = _fold_constant(value_node, constants)
        if value is not _NOT_CONSTANT:
            constants[name] = value
    return MappingProxyType(constants)


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


class _FunctionReturnCollector(ast.NodeVisitor):
    """Collect returns in one lexical function scope, excluding nested definitions."""

    def __init__(self) -> None:
        self.returns: list[ast.Return] = []

    def visit_Return(self, node: ast.Return) -> None:
        self.returns.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _tilelang_builder_names(tree: ast.Module) -> frozenset[str]:
    """Find immutable module-level builders that directly return a compiled kernel."""

    bindings = _scope_binding_counts(tree.body)
    builders: set[str] = set()
    for statement in tree.body:
        if (
            not isinstance(statement, ast.FunctionDef)
            or statement.decorator_list
            or bindings.get(statement.name) != 1
            or not statement.body
        ):
            continue
        collector = _FunctionReturnCollector()
        for child in statement.body:
            collector.visit(child)
        final = statement.body[-1]
        if (
            len(collector.returns) == 1
            and collector.returns[0] is final
            and isinstance(final, ast.Return)
            and isinstance(final.value, ast.Call)
            and _attribute_path(final.value.func) == "tilelang.compile"
        ):
            builders.add(statement.name)
    return frozenset(builders)


def _canonical_prim_funcs(
    statements: list[ast.stmt],
    bindings: Mapping[str, int],
) -> frozenset[str]:
    return frozenset(
        statement.name
        for statement in statements
        if isinstance(statement, ast.FunctionDef)
        and len(statement.decorator_list) == 1
        and _attribute_path(statement.decorator_list[0]) == "T.prim_func"
        and bindings.get(statement.name) == 1
    )


def _prim_func_scope_contracts(
    tree: ast.Module,
) -> tuple[
    frozenset[str],
    Mapping[int, frozenset[str]],
    Mapping[int, frozenset[str]],
]:
    """Return canonical primfuncs and bindings for each executable lexical scope."""

    module_bindings = _scope_binding_counts(tree.body)
    module_prim_funcs = _canonical_prim_funcs(tree.body, module_bindings)
    local_prim_funcs: dict[int, frozenset[str]] = {}
    local_bindings: dict[int, frozenset[str]] = {}

    class ScopeCollector(ast.NodeVisitor):
        def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            bindings = _scope_binding_counts(node.body, node.args)
            local_bindings[id(node)] = frozenset(bindings)
            local_prim_funcs[id(node)] = _canonical_prim_funcs(node.body, bindings)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node)

    ScopeCollector().visit(tree)
    return (
        module_prim_funcs,
        MappingProxyType(local_prim_funcs),
        MappingProxyType(local_bindings),
    )


def _issue(code: str, message: str, node: ast.AST | None = None) -> StaticValidationIssue:
    return StaticValidationIssue(
        code=code,
        message=message,
        line=getattr(node, "lineno", None),
        column=getattr(node, "col_offset", None),
    )


def _validate_imports(tree: ast.Module) -> tuple[StaticValidationIssue, ...]:
    issues: list[StaticValidationIssue] = []
    top_level = {id(statement) for statement in tree.body}
    required_imports = {
        ("import", "torch", None),
        ("import", "tilelang", None),
        ("import", "tilelang.language", "T"),
        ("from", "torch", "nn"),
    }
    observed: list[tuple[str, str, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        if id(node) not in top_level:
            issues.append(
                _issue(
                    "capability_import_forbidden",
                    "imports must appear at module scope",
                    node,
                )
            )
        if isinstance(node, ast.Import):
            for imported in node.names:
                key = ("import", imported.name, imported.asname)
                observed.append(key)
                if len(node.names) == 1 and key in required_imports:
                    continue
                if imported.name.startswith("tvm"):
                    code = "capability_tvm_escape"
                    message = f"TVM import {imported.name!r} is not allowed"
                elif imported.name.startswith("tilelang"):
                    code = "capability_import_alias"
                    message = f"TileLang import must use the canonical spelling: {imported.name!r}"
                else:
                    code = "capability_import_forbidden"
                    message = f"import {imported.name!r} is not allowed"
                issues.append(_issue(code, message, node))
        else:
            if (
                node.level == 0
                and node.module == "torch"
                and len(node.names) == 1
                and node.names[0].name == "nn"
                and node.names[0].asname is None
            ):
                observed.append(("from", "torch", "nn"))
                continue
            module = node.module or ""
            if module.startswith("tvm"):
                code = "capability_tvm_escape"
                message = f"TVM import {module!r} is not allowed"
            elif module.startswith("tilelang"):
                code = "capability_import_alias"
                message = "TileLang from-imports and star imports are not allowed"
            else:
                code = "capability_import_forbidden"
                message = f"from-import {module!r} is not allowed"
            issues.append(_issue(code, message, node))
    counts = {key: observed.count(key) for key in required_imports}
    for kind, module, name in sorted(required_imports):
        if counts[(kind, module, name)] == 1:
            continue
        spelling = (
            f"from {module} import {name}"
            if kind == "from"
            else f"import {module}" + (f" as {name}" if name is not None else "")
        )
        issues.append(
            _issue(
                "capability_missing_import",
                f"source must contain exactly one `{spelling}`",
            )
        )
    return tuple(issues)


_RAW_CODE_PATTERNS = (
    re.compile(r"\b__global__\b"),
    re.compile(r"extern\s+[\"']C[\"']"),
    re.compile(r"\b(?:__asm__|asm)\s*\("),
    re.compile(r"\b(?:ptx|cubin)\b", re.IGNORECASE),
    re.compile(r"\bCUDASourceCodeKernel\b"),
    re.compile(r"\bload_inline\b"),
    re.compile(r"\bRawKernel\b"),
)
_RAW_CODE_CALLEES = {"CUDASourceCodeKernel", "RawKernel", "load_inline"}


def _raw_code_issues(tree: ast.Module) -> tuple[StaticValidationIssue, ...]:
    issues: list[StaticValidationIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(pattern.search(node.value) is not None for pattern in _RAW_CODE_PATTERNS):
                issues.append(
                    _issue(
                        "capability_raw_cuda",
                        "raw CUDA, PTX, and custom-extension strings are not allowed",
                        node,
                    )
                )
        elif isinstance(node, ast.Call):
            path = _attribute_path(node.func)
            callee = path.rpartition(".")[2] if path is not None else None
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            if callee in _RAW_CODE_CALLEES:
                issues.append(
                    _issue(
                        "capability_raw_cuda",
                        f"raw CUDA or custom-extension call {callee!r} is not allowed",
                        node,
                    )
                )
    return tuple(issues)


_DYNAMIC_CALLS = {
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "locals",
    "open",
    "setattr",
    "vars",
}

_FRAMEWORK_COMPUTE_NAMES = {
    "abs",
    "add",
    "addmm",
    "amax",
    "amin",
    "bmm",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_min",
    "clone",
    "conv1d",
    "conv2d",
    "conv3d",
    "div",
    "einsum",
    "erf",
    "exp",
    "gelu",
    "layer_norm",
    "linear",
    "matmul",
    "max",
    "mean",
    "min",
    "mm",
    "mul",
    "norm",
    "pow",
    "prod",
    "relu",
    "rms_norm",
    "rsqrt",
    "sigmoid",
    "silu",
    "softmax",
    "sqrt",
    "square",
    "std",
    "sub",
    "sum",
    "tanh",
    "var",
    "__add__",
    "__matmul__",
    "__mul__",
    "__sub__",
}

_TORCH_SCAFFOLD_CALLS = {
    "torch.device",
    "torch.empty",
    "torch.empty_like",
    "torch.empty_strided",
    "torch.zeros",
    "torch.zeros_like",
}

_BASE_SIMPLE_CALLS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "T.Tensor": (2, 2),
        "T.alloc_fragment": (2, 2),
        "T.alloc_shared": (2, 2),
        "T.cast": (2, 2),
        "T.ceildiv": (2, 2),
        "T.clear": (1, 1),
        "T.copy": (2, 2),
        "T.erf": (1, 1),
        "T.exp": (1, 1),
        "T.fill": (2, 2),
        "T.infinity": (1, 1),
        "T.max": (2, 2),
        "T.min": (2, 2),
        "T.pow": (2, 2),
        "T.rsqrt": (1, 1),
        "T.sqrt": (1, 1),
        "T.tanh": (1, 1),
        "T.if_then_else": (3, 3),
        "T.float16": (1, 1),
        "T.float32": (1, 1),
        "T.int32": (1, 1),
        "T.int64": (1, 1),
    }
)

_WARP_REDUCTIONS = {"T.warp_reduce_sum", "T.warp_reduce_max", "T.warp_reduce_min"}
_PROTECTED_NAMES = {"T", "tilelang", "torch", "nn"}
_ALLOWED_TILELANG_ATTRIBUTES = frozenset(
    {
        *_BASE_SIMPLE_CALLS,
        *_WARP_REDUCTIONS,
        "T.GemmWarpPolicy",
        "T.GemmWarpPolicy.FullCol",
        "T.GemmWarpPolicy.FullRow",
        "T.GemmWarpPolicy.Square",
        "T.Kernel",
        "T.Parallel",
        "T.Pipelined",
        "T.alloc_local",
        "T.annotate_layout",
        "T.gemm",
        "T.get_thread_binding",
        "T.prim_func",
        "T.reduce_max",
        "T.reduce_min",
        "T.reduce_sum",
        "T.serial",
        "T.sync_threads",
        "T.unroll",
        "T.vectorized",
        "tilelang.compile",
        "tilelang.layout",
        "tilelang.layout.make_swizzled_layout",
    }
)


def _is_tilelang_symbol_value(node: ast.AST) -> bool:
    path = _attribute_path(node)
    if path is not None and (
        path == "T" or path.startswith("T.") or path == "tilelang" or path.startswith("tilelang.")
    ):
        return True
    if isinstance(node, ast.Call):
        return False
    return any(_is_tilelang_symbol_value(child) for child in ast.iter_child_nodes(node))


def _contains_tilelang_symbol(node: ast.AST) -> bool:
    path = _attribute_path(node)
    if path is not None and (
        path == "T" or path.startswith("T.") or path == "tilelang" or path.startswith("tilelang.")
    ):
        return True
    return any(_contains_tilelang_symbol(child) for child in ast.iter_child_nodes(node))


class _CapabilityInspector(ast.NodeVisitor):
    def __init__(
        self,
        constants: Mapping[str, Any],
        tilelang_builders: frozenset[str],
        module_prim_funcs: frozenset[str],
        local_prim_funcs: Mapping[int, frozenset[str]],
        local_bindings: Mapping[int, frozenset[str]],
        module_statements: frozenset[int],
    ) -> None:
        self.constants = constants
        self.tilelang_builders = tilelang_builders
        self.module_prim_funcs = module_prim_funcs
        self.local_prim_funcs = local_prim_funcs
        self.local_bindings = local_bindings
        self.module_statements = module_statements
        self.errors: list[StaticValidationIssue] = []
        self.warnings: list[StaticValidationIssue] = []
        self.used: set[str] = set()
        self.required_bitmask = BASE_CAPABILITY_BIT
        self._classes: list[str] = []
        self._functions: list[str] = []
        self._function_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self._prim_functions: list[bool] = []
        self._has_prim_func = False
        self._has_compile = False
        self._tvm_nodes: set[int] = set()
        self._model_class_count = 0
        self._model_init_count = 0
        self._model_forward_count = 0
        self._kernel_binding_count = 0
        self._approved_kernel_targets: set[int] = set()

    def finish(self) -> None:
        if not self._has_prim_func:
            self.errors.append(
                _issue(
                    "capability_missing_entrypoint", "source must define an `@T.prim_func` kernel"
                )
            )
        if not self._has_compile:
            self.errors.append(
                _issue(
                    "capability_missing_entrypoint", "source must call `tilelang.compile` directly"
                )
            )
        if self._model_class_count != 1:
            self.errors.append(
                _issue(
                    "capability_model_contract",
                    "source must define exactly one `ModelNew(nn.Module)` class",
                )
            )
        if self._model_init_count != 1 or self._model_forward_count != 1:
            self.errors.append(
                _issue(
                    "capability_model_contract",
                    "ModelNew must define exactly one __init__ and one forward method",
                )
            )
        has_kernel_binding_error = any(
            issue.code == "capability_kernel_binding" for issue in self.errors
        )
        if self._kernel_binding_count != 1 and not has_kernel_binding_error:
            self.errors.append(
                _issue(
                    "capability_kernel_binding",
                    "ModelNew.__init__ must bind self.kernel exactly once from a TileLang builder",
                )
            )

    def _error(self, code: str, message: str, node: ast.AST | None = None) -> None:
        self.errors.append(_issue(code, message, node))

    def _mark(self, capability: str, bit: int = BASE_CAPABILITY_BIT) -> None:
        self.used.add(capability)
        self.required_bitmask |= bit

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name in _PROTECTED_NAMES:
            self._error(
                "capability_symbol_rebinding",
                f"protected symbol {node.name!r} cannot be rebound",
                node,
            )
        if node.name == "ModelNew":
            is_module_top_level = id(node) in self.module_statements
            if is_module_top_level:
                self._model_class_count += 1
            else:
                self._error(
                    "capability_model_contract",
                    "ModelNew must be defined at module scope",
                    node,
                )
            valid_base = (
                len(node.bases) == 1
                and _attribute_path(node.bases[0]) == "nn.Module"
                and not node.keywords
                and not node.decorator_list
            )
            if not valid_base:
                self._error(
                    "capability_model_contract",
                    "ModelNew must directly inherit only from nn.Module",
                    node,
                )
            for statement in node.body:
                is_docstring = (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                )
                if isinstance(statement, ast.FunctionDef) or is_docstring:
                    continue
                self._error(
                    "capability_model_contract",
                    "ModelNew may contain only __init__, forward, and an optional docstring",
                    statement,
                )
        self._classes.append(node.name)
        self.generic_visit(node)
        self._classes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name == "ModelNew":
            self._error(
                "capability_model_contract",
                "ModelNew must remain the exported class entrypoint",
                node,
            )
        if node.name in _PROTECTED_NAMES:
            self._error(
                "capability_symbol_rebinding",
                f"protected symbol {node.name!r} cannot be rebound",
                node,
            )
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        if node.args.vararg is not None:
            arguments = (*arguments, node.args.vararg)
        if node.args.kwarg is not None:
            arguments = (*arguments, node.args.kwarg)
        for argument in arguments:
            if argument.arg in _PROTECTED_NAMES:
                self._error(
                    "capability_symbol_rebinding",
                    f"protected symbol {argument.arg!r} cannot be used as an argument",
                    argument,
                )
        defaults = (
            *node.args.defaults,
            *(item for item in node.args.kw_defaults if item is not None),
        )
        for default in defaults:
            if _contains_tilelang_symbol(default):
                self._error(
                    "capability_symbol_alias",
                    "TileLang symbols cannot be captured by function defaults",
                    default,
                )
        is_model_method = self._classes == ["ModelNew"] and not self._functions
        if is_model_method:
            if isinstance(node, ast.AsyncFunctionDef) or node.name not in {"__init__", "forward"}:
                self._error(
                    "capability_model_contract",
                    "ModelNew supports only synchronous __init__ and forward methods",
                    node,
                )
            elif node.name == "__init__":
                self._model_init_count += 1
            else:
                self._model_forward_count += 1
                self._validate_forward_contract(node)
        is_prim_func = False
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            path = _attribute_path(target)
            if path == "T.prim_func" and not isinstance(decorator, ast.Call):
                is_prim_func = True
                self._has_prim_func = True
                self._mark("T.prim_func")
            elif path is not None and (path.startswith("T.") or path.startswith("tilelang.")):
                self._error(
                    "capability_unknown_symbol",
                    f"decorator {path!r} is outside the capability surface",
                    decorator,
                )
            else:
                self._error(
                    "capability_decorator_forbidden",
                    "only the exact @T.prim_func decorator is allowed",
                    decorator,
                )
        if is_prim_func and len(node.decorator_list) != 1:
            self._error(
                "capability_decorator_forbidden",
                "@T.prim_func cannot be combined with other decorators",
                node,
            )
        self._functions.append(node.name)
        self._function_nodes.append(node)
        self._prim_functions.append(is_prim_func)
        self.generic_visit(node)
        self._prim_functions.pop()
        self._function_nodes.pop()
        self._functions.pop()

    def _validate_forward_contract(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
        valid = len(body) == 1 and isinstance(body[0], ast.Return)
        value = body[0].value if valid else None
        valid = (
            valid
            and isinstance(value, ast.Call)
            and _attribute_path(value.func) == "self.kernel"
            and not value.keywords
            and not any(isinstance(argument, ast.Starred) for argument in value.args)
        )
        if not valid:
            self._error(
                "capability_forward_contract",
                "ModelNew.forward must directly return one `self.kernel(...)` call",
                node,
            )

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._error(
            "capability_dynamic_access",
            "lambda expressions are not allowed in capability candidates",
            node,
        )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store | ast.Del) and node.id in _PROTECTED_NAMES:
            self._error(
                "capability_symbol_rebinding",
                f"protected symbol {node.id!r} cannot be rebound",
                node,
            )
        if isinstance(node.ctx, ast.Store | ast.Del) and node.id == "ModelNew":
            self._error(
                "capability_model_contract",
                "ModelNew cannot be rebound or deleted",
                node,
            )
        if isinstance(node.ctx, ast.Load) and node.id == "tvm" and id(node) not in self._tvm_nodes:
            self._tvm_nodes.add(id(node))
            self._error("capability_tvm_escape", "TVM symbols are not allowed", node)
        if isinstance(node.ctx, ast.Load) and node.id in {"__builtins__", "builtins"}:
            self._error(
                "capability_dynamic_access",
                f"dynamic builtins access through {node.id!r} is not allowed",
                node,
            )
        self.generic_visit(node)

    def _validate_pattern_binding(self, name: str | None, node: ast.AST) -> None:
        if name in {*_PROTECTED_NAMES, "ModelNew"}:
            self._error(
                "capability_symbol_rebinding",
                f"protected symbol {name!r} cannot be captured by a match pattern",
                node,
            )

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self._validate_pattern_binding(node.name, node)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self._validate_pattern_binding(node.name, node)
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self._validate_pattern_binding(node.rest, node)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._validate_pattern_binding(node.name, node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        path = _attribute_path(node)
        if (
            isinstance(node.ctx, ast.Store | ast.Del)
            and path == "self.kernel"
            and id(node) not in self._approved_kernel_targets
        ):
            self._error(
                "capability_kernel_binding",
                "self.kernel can only be assigned once in ModelNew.__init__",
                node,
            )
        if (
            isinstance(node.ctx, ast.Store | ast.Del)
            and path is not None
            and path.startswith(("T.", "tilelang.", "torch.", "nn.", "ModelNew."))
        ):
            self._error(
                "capability_symbol_rebinding",
                f"protected symbol {path!r} cannot be rebound",
                node,
            )
        if node.attr.startswith("__") and not (
            node.attr == "__init__"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "super"
            and not node.value.args
            and not node.value.keywords
        ):
            self._error(
                "capability_dynamic_access",
                f"dunder attribute access {node.attr!r} is not allowed",
                node,
            )
        if path is not None and (path.startswith("tvm.") or path.startswith("tilelang.tvm")):
            if id(node) not in self._tvm_nodes:
                self._tvm_nodes.add(id(node))
                self._error("capability_tvm_escape", f"TVM symbol {path!r} is not allowed", node)
        if (
            isinstance(node.ctx, ast.Load)
            and path is not None
            and (path.startswith("T.") or path.startswith("tilelang."))
            and path not in _ALLOWED_TILELANG_ATTRIBUTES
        ):
            self._error(
                "capability_unknown_symbol",
                f"TileLang symbol {path!r} is outside the capability surface",
                node,
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        kernel_targets = [
            target
            for target in node.targets
            if isinstance(target, ast.Attribute) and _attribute_path(target) == "self.kernel"
        ]
        if kernel_targets:
            self._approved_kernel_targets.update(id(target) for target in kernel_targets)
            valid = (
                self._in_model_method("__init__")
                and len(node.targets) == 1
                and len(kernel_targets) == 1
                and self._is_tilelang_kernel_factory(node.value)
            )
            if valid:
                self._kernel_binding_count += 1
            else:
                self._error(
                    "capability_kernel_binding",
                    "self.kernel must be assigned directly from tilelang.compile "
                    "or a verified builder",
                    node,
                )
        if _is_tilelang_symbol_value(node.value):
            self._error(
                "capability_symbol_alias",
                "TileLang symbols cannot be assigned to aliases",
                node,
            )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Attribute) and _attribute_path(node.target) == "self.kernel":
            self._approved_kernel_targets.add(id(node.target))
            self._error(
                "capability_kernel_binding",
                "annotated self.kernel assignments are not allowed",
                node,
            )
        if node.value is not None and _is_tilelang_symbol_value(node.value):
            self._error(
                "capability_symbol_alias",
                "TileLang symbols cannot be assigned to aliases",
                node,
            )
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if _is_tilelang_symbol_value(node.value):
            self._error(
                "capability_symbol_alias",
                "TileLang symbols cannot be assigned to aliases",
                node,
            )
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and _is_tilelang_symbol_value(node.value):
            self._error(
                "capability_symbol_alias",
                "TileLang symbols cannot escape as Python values",
                node,
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if self._functions and not self._prim_functions[-1]:
            self._error(
                "framework_compute_fallback",
                "Python arithmetic outside an @T.prim_func kernel is not allowed",
                node,
            )
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if self._functions and not self._prim_functions[-1]:
            self._error(
                "framework_compute_fallback",
                "Python arithmetic outside an @T.prim_func kernel is not allowed",
                node,
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if self._functions and not self._prim_functions[-1]:
            self._error(
                "framework_compute_fallback",
                "Python comparisons outside an @T.prim_func kernel are not allowed",
                node,
            )
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if self._functions and not self._prim_functions[-1]:
            self._error(
                "framework_compute_fallback",
                "Python boolean operations outside an @T.prim_func kernel are not allowed",
                node,
            )
        self.generic_visit(node)

    def _in_model_method(self, name: str) -> bool:
        return self._classes == ["ModelNew"] and self._functions == [name]

    def _is_super_init_call(self, node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "__init__":
            return False
        owner = node.func.value
        return (
            isinstance(owner, ast.Call)
            and isinstance(owner.func, ast.Name)
            and owner.func.id == "super"
            and not owner.args
            and not owner.keywords
            and not node.args
            and not node.keywords
        )

    def _is_allowed_python_scaffold_call(self, node: ast.Call, path: str | None) -> bool:
        if self._in_model_method("__init__"):
            if self._is_super_init_call(node):
                return True
            if isinstance(node.func, ast.Name) and node.func.id == "super":
                return not node.args and not node.keywords
            if isinstance(node.func, ast.Name) and node.func.id in self.tilelang_builders:
                return not node.keywords and not any(
                    isinstance(argument, ast.Starred) for argument in node.args
                )
        return self._in_model_method("forward") and path == "self.kernel"

    def _is_tilelang_kernel_factory(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        path = _attribute_path(node.func)
        if path == "tilelang.compile":
            return True
        return (
            isinstance(node.func, ast.Name)
            and node.func.id in self.tilelang_builders
            and not node.keywords
            and not any(isinstance(argument, ast.Starred) for argument in node.args)
        )

    def visit_Call(self, node: ast.Call) -> None:
        path = _attribute_path(node.func)
        if isinstance(node.func, ast.Name) and node.func.id in _DYNAMIC_CALLS:
            self._error(
                "capability_dynamic_access",
                f"dynamic call {node.func.id!r} is not allowed",
                node,
            )
        elif path is not None and path.rpartition(".")[2] in {"__getattr__", "__getattribute__"}:
            self._error(
                "capability_dynamic_access",
                f"dynamic attribute call {path!r} is not allowed",
                node,
            )
        elif isinstance(node.func, ast.Subscript | ast.Call):
            self._error(
                "capability_indirect_call",
                "subscripted and call-produced callees are not allowed",
                node,
            )
        elif path is None and _is_tilelang_symbol_value(node.func):
            self._error(
                "capability_dynamic_access",
                "TileLang symbols must be called directly",
                node,
            )
        elif path is not None and (path == "T" or path.startswith("T.")):
            self._validate_t_call(path, node)
        elif path is not None and (path == "tilelang" or path.startswith("tilelang.")):
            self._validate_tilelang_call(path, node)
        elif path is not None and path.startswith("torch."):
            if path not in _TORCH_SCAFFOLD_CALLS or self._in_model_method("forward"):
                self._error(
                    "framework_compute_fallback",
                    f"PyTorch call {path!r} is not allowed in a capability candidate",
                    node,
                )
        elif path is not None and path.startswith("nn."):
            self._error(
                "framework_compute_fallback",
                f"PyTorch module call {path!r} is not allowed",
                node,
            )
        elif path is not None and path.rpartition(".")[2] in _FRAMEWORK_COMPUTE_NAMES:
            self._error(
                "framework_compute_fallback",
                f"framework-style compute call {path!r} is not allowed",
                node,
            )
        elif self._is_allowed_python_scaffold_call(node, path):
            pass
        else:
            display = path or type(node.func).__name__
            self._error(
                "capability_call_forbidden",
                f"non-DSL call {display!r} is outside the Python scaffold allowlist",
                node,
            )
        self.generic_visit(node)

    def _validate_t_call(self, path: str, node: ast.Call) -> None:
        if path == "T.Kernel":
            self._validate_kernel(node)
        elif path == "T.Parallel":
            self._mark(path)
            self._validate_arity(path, node, 1, 3)
            self._reject_keywords(path, node, set())
        elif path == "T.Pipelined":
            self._validate_pipelined(node)
        elif path == "T.gemm":
            self._validate_gemm(node)
        elif path in {"T.reduce_sum", "T.reduce_max", "T.reduce_min"}:
            self._validate_reduction(path, node)
        elif path in {"T.alloc_fragment", "T.alloc_shared"}:
            self._mark(path)
            self._validate_allocation(path, node)
        elif path in _BASE_SIMPLE_CALLS:
            self._mark(path)
            minimum, maximum = _BASE_SIMPLE_CALLS[path]
            self._validate_arity(path, node, minimum, maximum)
            self._reject_keywords(path, node, set())
        elif path == "T.get_thread_binding":
            self._mark(path, MAPPING_CAPABILITY_BIT)
            self._validate_thread_binding(node)
        elif path in {"T.serial", "T.unroll", "T.vectorized"}:
            self._mark(path, MAPPING_CAPABILITY_BIT)
            self._validate_loop(path, node)
        elif path == "T.alloc_local":
            self._mark(path, MAPPING_CAPABILITY_BIT)
            self._validate_allocation(path, node)
        elif path in _WARP_REDUCTIONS:
            self._mark(path, MAPPING_CAPABILITY_BIT)
            self._validate_arity(path, node, 1, 1)
            self._reject_keywords(path, node, set())
        elif path == "T.sync_threads":
            self._mark(path, MAPPING_CAPABILITY_BIT)
            self._validate_arity(path, node, 0, 0)
            self._reject_keywords(path, node, set())
        elif path == "T.annotate_layout":
            self._mark(path, MAPPING_CAPABILITY_BIT)
            self._validate_layout_annotation(node)
        else:
            self._error(
                "capability_unknown_symbol",
                f"TileLang call {path!r} is outside the capability surface",
                node,
            )

    def _validate_tilelang_call(self, path: str, node: ast.Call) -> None:
        if path == "tilelang.compile":
            self._mark(path)
            self._has_compile = True
            self._validate_compile(node)
        elif path == "tilelang.layout.make_swizzled_layout":
            self._mark(path, MAPPING_CAPABILITY_BIT)
            self._validate_arity(path, node, 1, 1)
            self._reject_keywords(path, node, set())
        else:
            self._error(
                "capability_unknown_symbol",
                f"TileLang call {path!r} is outside the capability surface",
                node,
            )

    def _validate_arity(
        self,
        path: str,
        node: ast.Call,
        minimum: int,
        maximum: int,
    ) -> None:
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            self._error(
                "capability_argument_dynamic",
                f"{path} does not allow starred arguments",
                node,
            )
            return
        if not minimum <= len(node.args) <= maximum:
            expected = str(minimum) if minimum == maximum else f"{minimum}..{maximum}"
            self._error(
                "capability_argument_domain",
                f"{path} requires {expected} positional arguments",
                node,
            )

    def _reject_keywords(self, path: str, node: ast.Call, allowed: set[str]) -> None:
        for keyword in node.keywords:
            if keyword.arg is None:
                self._error(
                    "capability_argument_dynamic",
                    f"{path} does not allow expanded keyword arguments",
                    keyword,
                )
            elif keyword.arg not in allowed:
                self._error(
                    "capability_argument_forbidden",
                    f"{path} argument {keyword.arg!r} is not exposed",
                    keyword,
                )

    def _keyword(self, node: ast.Call, name: str) -> ast.AST | None:
        return next((keyword.value for keyword in node.keywords if keyword.arg == name), None)

    def _constant(self, path: str, name: str, node: ast.AST) -> Any:
        value = _fold_constant(node, self.constants)
        if value is _NOT_CONSTANT:
            self._error(
                "capability_argument_dynamic",
                f"{path} argument {name!r} must be statically constant-foldable",
                node,
            )
        return value

    def _validate_kernel(self, node: ast.Call) -> None:
        path = "T.Kernel"
        self._mark(path)
        self._validate_arity(path, node, 1, 3)
        self._reject_keywords(path, node, {"threads"})
        value_node = self._keyword(node, "threads")
        value = 128 if value_node is None else self._constant(path, "threads", value_node)
        if value is _NOT_CONSTANT:
            return
        if type(value) is not int or value not in {64, 128, 256}:
            self._error(
                "capability_argument_domain",
                "T.Kernel threads must be one of 64, 128, or 256",
                value_node or node,
            )
        elif value != 128:
            self._mark("schedule.kernel_threads", SCHEDULE_CAPABILITY_BIT)

    def _validate_pipelined(self, node: ast.Call) -> None:
        path = "T.Pipelined"
        self._mark(path)
        self._validate_arity(path, node, 1, 2)
        self._reject_keywords(path, node, {"num_stages"})
        value_node = self._keyword(node, "num_stages")
        value = 0 if value_node is None else self._constant(path, "num_stages", value_node)
        if value is _NOT_CONSTANT:
            return
        if type(value) is not int or value not in {0, 1, 2, 3}:
            self._error(
                "capability_argument_domain",
                "T.Pipelined num_stages must be one of 0, 1, 2, or 3",
                value_node or node,
            )
        elif value != 0:
            self._mark("schedule.pipeline_stages", SCHEDULE_CAPABILITY_BIT)

    def _validate_gemm(self, node: ast.Call) -> None:
        path = "T.gemm"
        self._mark(path)
        self._validate_arity(path, node, 3, 3)
        self._reject_keywords(path, node, {"transpose_A", "transpose_B", "policy"})
        for name in ("transpose_A", "transpose_B"):
            value_node = self._keyword(node, name)
            if value_node is None:
                continue
            value = self._constant(path, name, value_node)
            if value is not _NOT_CONSTANT and type(value) is not bool:
                self._error(
                    "capability_argument_domain",
                    f"T.gemm {name} must be a boolean literal",
                    value_node,
                )
        policy_node = self._keyword(node, "policy")
        if policy_node is None:
            return
        policy = _attribute_path(policy_node)
        policies = {
            "T.GemmWarpPolicy.Square",
            "T.GemmWarpPolicy.FullRow",
            "T.GemmWarpPolicy.FullCol",
        }
        if policy not in policies:
            self._error(
                "capability_argument_domain",
                "T.gemm policy must be a registered T.GemmWarpPolicy literal",
                policy_node,
            )
        else:
            self._mark("schedule.gemm_policy", SCHEDULE_CAPABILITY_BIT)

    def _validate_reduction(self, path: str, node: ast.Call) -> None:
        self._mark(path)
        self._validate_arity(path, node, 2, 2)
        self._reject_keywords(path, node, {"dim", "clear"})
        dim_node = self._keyword(node, "dim")
        if dim_node is not None:
            dim = self._constant(path, "dim", dim_node)
            if dim is not _NOT_CONSTANT and (type(dim) is not int or not -4 <= dim <= 3):
                self._error(
                    "capability_argument_domain",
                    f"{path} dim must be an integer in [-4, 3]",
                    dim_node,
                )
        clear_node = self._keyword(node, "clear")
        if clear_node is not None:
            clear = self._constant(path, "clear", clear_node)
            if clear is not _NOT_CONSTANT and type(clear) is not bool:
                self._error(
                    "capability_argument_domain",
                    f"{path} clear must be a boolean literal",
                    clear_node,
                )

    def _validate_allocation(self, path: str, node: ast.Call) -> None:
        self._reject_keywords(path, node, {"dtype"})
        dtype_node = self._keyword(node, "dtype")
        positional = len(node.args)
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            self._error(
                "capability_argument_dynamic",
                f"{path} does not allow starred arguments",
                node,
            )
            return
        valid = (positional == 2 and dtype_node is None) or (
            positional == 1 and dtype_node is not None
        )
        if not valid:
            self._error(
                "capability_argument_domain",
                f"{path} requires positional shape and positional or keyword dtype",
                node,
            )

    def _validate_thread_binding(self, node: ast.Call) -> None:
        path = "T.get_thread_binding"
        self._validate_arity(path, node, 0, 1)
        self._reject_keywords(path, node, {"dim"})
        dim_node = self._keyword(node, "dim")
        if node.args and dim_node is not None:
            self._error(
                "capability_argument_domain",
                "T.get_thread_binding dimension cannot be supplied twice",
                node,
            )
            return
        value_node = node.args[0] if node.args else dim_node
        if value_node is not None:
            value = self._constant(path, "dimension", value_node)
            if value is not _NOT_CONSTANT and (type(value) is not int or value != 0):
                self._error(
                    "capability_argument_domain",
                    "T.get_thread_binding only exposes threadIdx.x dimension 0",
                    value_node,
                )

    def _loop_range(self, path: str, node: ast.Call, maximum_args: int) -> int | None:
        self._validate_arity(path, node, 1, maximum_args)
        if not 1 <= len(node.args) <= maximum_args:
            return None
        values = [
            self._constant(path, f"range[{index}]", value) for index, value in enumerate(node.args)
        ]
        if any(value is _NOT_CONSTANT for value in values):
            return None
        if any(type(value) is not int for value in values):
            self._error(
                "capability_argument_domain",
                f"{path} range arguments must be integer literals",
                node,
            )
            return None
        if len(values) == 1:
            start, stop, step = 0, values[0], 1
        elif len(values) == 2:
            start, stop, step = values[0], values[1], 1
        else:
            start, stop, step = values
        if step <= 0 or step > 16:
            self._error(
                "capability_argument_domain",
                f"{path} step must be in [1, 16]",
                node,
            )
            return None
        return max(0, (stop - start + step - 1) // step)

    def _validate_loop(self, path: str, node: ast.Call) -> None:
        if path == "T.serial":
            self._reject_keywords(path, node, set())
            extent = self._loop_range(path, node, 3)
            maximum = 4096
        elif path == "T.vectorized":
            self._reject_keywords(path, node, set())
            extent = self._loop_range(path, node, 2)
            maximum = 8
        else:
            self._reject_keywords(path, node, {"explicit", "unroll_factor"})
            extent = self._loop_range(path, node, 3)
            maximum = 16
            explicit_node = self._keyword(node, "explicit")
            if explicit_node is not None:
                explicit = self._constant(path, "explicit", explicit_node)
                if explicit is not _NOT_CONSTANT and type(explicit) is not bool:
                    self._error(
                        "capability_argument_domain",
                        "T.unroll explicit must be a boolean literal",
                        explicit_node,
                    )
            factor_node = self._keyword(node, "unroll_factor")
            if factor_node is not None:
                factor = self._constant(path, "unroll_factor", factor_node)
                if factor is not _NOT_CONSTANT and (
                    type(factor) is not int or not 1 <= factor <= 16
                ):
                    self._error(
                        "capability_argument_domain",
                        "T.unroll unroll_factor must be in [1, 16]",
                        factor_node,
                    )
        if extent is None:
            return
        if path == "T.vectorized" and extent not in {2, 4, 8}:
            self._error(
                "capability_argument_domain",
                "T.vectorized width must be 2, 4, or 8",
                node,
            )
        elif extent > maximum:
            self._error(
                "capability_argument_domain",
                f"{path} extent must not exceed {maximum}",
                node,
            )

    def _validate_layout_annotation(self, node: ast.Call) -> None:
        path = "T.annotate_layout"
        self._validate_arity(path, node, 1, 1)
        self._reject_keywords(path, node, set())
        if len(node.args) != 1 or not isinstance(node.args[0], ast.Dict):
            self._error(
                "capability_argument_domain",
                "T.annotate_layout requires one literal buffer-to-layout dictionary",
                node,
            )
            return
        mapping = node.args[0]
        if not mapping.keys:
            self._error(
                "capability_argument_domain",
                "T.annotate_layout dictionary cannot be empty",
                mapping,
            )
        for key, value in zip(mapping.keys, mapping.values, strict=True):
            if key is None or not isinstance(value, ast.Call):
                self._error(
                    "capability_argument_domain",
                    "layout values must directly call make_swizzled_layout",
                    value,
                )
                continue
            if _attribute_path(value.func) != "tilelang.layout.make_swizzled_layout":
                self._error(
                    "capability_argument_domain",
                    "only automatic make_swizzled_layout values are allowed",
                    value,
                )
                continue
            if (
                len(value.args) != 1
                or value.keywords
                or ast.dump(key, include_attributes=False)
                != ast.dump(value.args[0], include_attributes=False)
            ):
                self._error(
                    "capability_argument_domain",
                    "make_swizzled_layout must receive the dictionary's buffer key",
                    value,
                )

    def _validate_compile(self, node: ast.Call) -> None:
        path = "tilelang.compile"
        self._validate_arity(path, node, 1, 1)
        self._reject_keywords(path, node, {"out_idx", "target"})
        allowed_prim_funcs = self.module_prim_funcs
        if self._function_nodes:
            function_id = id(self._function_nodes[-1])
            local = self.local_prim_funcs.get(function_id, frozenset())
            shadowed = self.local_bindings.get(function_id, frozenset())
            allowed_prim_funcs = local | (self.module_prim_funcs - shadowed)
        if (
            len(node.args) != 1
            or not isinstance(node.args[0], ast.Name)
            or node.args[0].id not in allowed_prim_funcs
        ):
            self._error(
                "capability_compile_kernel",
                "tilelang.compile must receive a uniquely bound @T.prim_func function",
                node,
            )
        target_node = self._keyword(node, "target")
        if target_node is None:
            self._error(
                "capability_argument_domain",
                'tilelang.compile must explicitly set target="cuda"',
                node,
            )
        else:
            target = self._constant(path, "target", target_node)
            if target is not _NOT_CONSTANT and target != "cuda":
                self._error(
                    "capability_argument_domain",
                    'tilelang.compile target must equal "cuda"',
                    target_node,
                )
        out_node = self._keyword(node, "out_idx")
        if out_node is None:
            self._error(
                "capability_argument_domain",
                "tilelang.compile must declare out_idx",
                node,
            )
        else:
            out_idx = self._constant(path, "out_idx", out_node)
            valid = type(out_idx) is int and out_idx >= 0
            if isinstance(out_idx, tuple):
                valid = bool(out_idx) and all(
                    type(value) is int and value >= 0 for value in out_idx
                )
            if out_idx is not _NOT_CONSTANT and not valid:
                self._error(
                    "capability_argument_domain",
                    "tilelang.compile out_idx must be a non-negative integer or tuple",
                    out_node,
                )


def validate_tilelang_capability_source(
    source: str,
    pack: str | CapabilityPackSpec,
) -> CapabilityValidationResult:
    """Validate one candidate against a strict, monotonic TileLang capability pack."""

    resolved = _resolve_pack(pack)
    legacy = validate_candidate_source(source, "tilelang")
    if not legacy.valid:
        return CapabilityValidationResult(
            pack_id=resolved.id,
            pack_sha256=resolved.sha256,
            valid=False,
            errors=legacy.errors,
        )

    tree = ast.parse(source)
    errors = list(_validate_imports(tree))
    errors.extend(_raw_code_issues(tree))

    module_prim_funcs, local_prim_funcs, local_bindings = _prim_func_scope_contracts(tree)
    inspector = _CapabilityInspector(
        _module_constants(tree),
        _tilelang_builder_names(tree),
        module_prim_funcs,
        local_prim_funcs,
        local_bindings,
        frozenset(id(statement) for statement in tree.body),
    )
    inspector.visit(tree)
    inspector.finish()
    errors.extend(inspector.errors)

    minimum = minimum_pack_for_bitmask(inspector.required_bitmask)
    missing_bits = inspector.required_bitmask & ~resolved.bitmask
    if missing_bits:
        errors.append(
            _issue(
                "capability_pack_violation",
                f"source requires {minimum.id} (bitmask {minimum.bitmask}), "
                f"but target pack is {resolved.id} (bitmask {resolved.bitmask})",
            )
        )
    used = tuple(sorted(inspector.used))
    return CapabilityValidationResult(
        pack_id=resolved.id,
        pack_sha256=resolved.sha256,
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(inspector.warnings),
        used_capabilities=used,
        minimum_pack_id=minimum.id,
        minimum_pack_bitmask=minimum.bitmask,
    )
