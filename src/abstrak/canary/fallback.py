"""Offline source validation for canary-study kernel candidates."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Literal

TargetName = Literal["triton", "tilelang", "cute", "cuda"]


@dataclass(frozen=True)
class StaticValidationIssue:
    code: str
    message: str
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class StaticValidationResult:
    target: str
    valid: bool
    errors: tuple[StaticValidationIssue, ...]

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.errors)


_COMPUTE_CALLS = frozenset(
    {
        "addmm",
        "batch_norm",
        "bmm",
        "einsum",
        "group_norm",
        "instance_norm",
        "layer_norm",
        "linear",
        "matmul",
        "matrix_norm",
        "mean",
        "mm",
        "norm",
        "normalize",
        "relu",
        "relu_",
        "rms_norm",
        "sum",
        "vector_norm",
    }
)
_COMPUTE_MODULES = frozenset(
    {
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "GroupNorm",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
        "LayerNorm",
        "Linear",
        "RMSNorm",
        "ReLU",
    }
)
_DSL_PREFIXES = ("triton.", "tilelang.", "cutlass.cute.")
_CUDA_KERNEL_PATTERN = re.compile(r"\b__global__\b")


def _attribute_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


class _ImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.modules: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            self.modules.add(imported.name)
            local_name = imported.asname or imported.name.partition(".")[0]
            resolved = imported.name if imported.asname else imported.name.partition(".")[0]
            self.aliases[local_name] = resolved

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module is None:
            return
        self.modules.add(node.module)
        for imported in node.names:
            if imported.name == "*":
                continue
            local_name = imported.asname or imported.name
            self.aliases[local_name] = f"{node.module}.{imported.name}"

    def resolve(self, name: str) -> str:
        root, separator, suffix = name.partition(".")
        resolved_root = self.aliases.get(root, root)
        return f"{resolved_root}.{suffix}" if separator else resolved_root


class _SourceInspector(ast.NodeVisitor):
    def __init__(self, imports: _ImportCollector) -> None:
        self.imports = imports
        self.calls: set[str] = set()
        self.decorators: set[str] = set()
        self.issues: list[StaticValidationIssue] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_decorators(node.decorator_list)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        raw_name = _attribute_name(node.func)
        if raw_name is not None:
            resolved = self.imports.resolve(raw_name)
            self.calls.add(resolved)
            self._check_compute_call(node, raw_name, resolved)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            self.issues.append(
                StaticValidationIssue(
                    code="framework_compute_fallback",
                    message="Python @ matrix multiplication is not allowed in a target kernel",
                    line=node.lineno,
                    column=node.col_offset,
                )
            )
        self.generic_visit(node)

    def _record_decorators(self, decorators: list[ast.expr]) -> None:
        for decorator in decorators:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            raw_name = _attribute_name(target)
            if raw_name is not None:
                self.decorators.add(self.imports.resolve(raw_name))

    def _check_compute_call(self, node: ast.Call, raw_name: str, resolved: str) -> None:
        final_name = resolved.rpartition(".")[2]
        if resolved.startswith(_DSL_PREFIXES):
            return
        is_torch_call = resolved == "torch" or resolved.startswith("torch.")
        is_tensor_method = (
            "." in raw_name and raw_name.partition(".")[0] not in self.imports.aliases
        )
        if final_name in _COMPUTE_CALLS and (is_torch_call or is_tensor_method):
            self.issues.append(
                StaticValidationIssue(
                    code="framework_compute_fallback",
                    message=f"framework compute call {raw_name!r} is not allowed",
                    line=node.lineno,
                    column=node.col_offset,
                )
            )
        if final_name in _COMPUTE_MODULES and resolved.startswith("torch.nn."):
            self.issues.append(
                StaticValidationIssue(
                    code="framework_compute_fallback",
                    message=f"framework compute module {raw_name!r} is not allowed",
                    line=node.lineno,
                    column=node.col_offset,
                )
            )


def _has_module(imports: _ImportCollector, prefix: str) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for module in imports.modules)


def _has_prefix(values: set[str], prefixes: tuple[str, ...]) -> bool:
    return any(value.startswith(prefixes) for value in values)


def _has_backend_signature(
    source: str,
    target: TargetName,
    imports: _ImportCollector,
    inspector: _SourceInspector,
) -> bool:
    if target == "triton":
        return _has_module(imports, "triton") and bool(
            {"triton.jit", "triton.autotune"}.intersection(inspector.decorators)
        )
    if target == "tilelang":
        has_primitive = _has_prefix(inspector.decorators, ("tilelang.language.prim_func",))
        has_compile = _has_prefix(inspector.calls, ("tilelang.compile",))
        return _has_module(imports, "tilelang") and (has_primitive or has_compile)
    if target == "cute":
        has_cute_import = any(
            value == "cutlass.cute" or value.startswith("cutlass.cute.")
            for value in (*imports.modules, *imports.aliases.values())
        )
        has_cute_entrypoint = _has_prefix(
            inspector.decorators,
            ("cutlass.cute.kernel", "cutlass.cute.jit"),
        ) or _has_prefix(inspector.calls, ("cutlass.cute.compile",))
        return has_cute_import and has_cute_entrypoint
    has_load_inline = any(call.endswith(".load_inline") for call in inspector.calls)
    return has_load_inline and _CUDA_KERNEL_PATTERN.search(source) is not None


def validate_candidate_source(source: str, target: str) -> StaticValidationResult:
    """Reject obvious framework fallbacks and candidates for the wrong target stack."""

    if target not in {"triton", "tilelang", "cute", "cuda"}:
        issue = StaticValidationIssue(
            code="unknown_target",
            message=f"unsupported target {target!r}",
        )
        return StaticValidationResult(target=target, valid=False, errors=(issue,))

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        issue = StaticValidationIssue(
            code="syntax_error",
            message=error.msg,
            line=error.lineno,
            column=error.offset,
        )
        return StaticValidationResult(target=target, valid=False, errors=(issue,))

    imports = _ImportCollector()
    imports.visit(tree)
    inspector = _SourceInspector(imports)
    inspector.visit(tree)
    typed_target: TargetName = target  # type: ignore[assignment]
    if not _has_backend_signature(source, typed_target, imports, inspector):
        inspector.issues.append(
            StaticValidationIssue(
                code="missing_backend_signature",
                message=f"candidate does not contain the frozen {target} backend signature",
            )
        )

    errors = tuple(inspector.issues)
    return StaticValidationResult(target=target, valid=not errors, errors=errors)
