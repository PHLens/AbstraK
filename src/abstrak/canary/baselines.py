"""Hash-bound PyTorch baselines for the four R1 formal tasks.

The sources in this module are inert strings. Importing the registry does not
import PyTorch or initialize a GPU runtime.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

FORMAL_TASK_IDS = (
    "gemm-bias-relu-static",
    "gemm-static",
    "layernorm-static",
    "rmsnorm-static",
)
BASELINE_VARIANTS = ("compile", "eager", "vendor")


class BaselineRegistryError(ValueError):
    """Raised when a formal baseline ID or source does not match the registry."""


@dataclass(frozen=True)
class BaselineSource:
    """One immutable, content-addressed baseline implementation."""

    task_id: str
    variant: str
    source: str
    source_sha256: str


def _model_source(
    task_id: str,
    arguments: str,
    body: tuple[str, ...],
    *,
    compiled: bool = False,
    functional: bool = False,
) -> str:
    imports = "import torch\nfrom torch import nn\n"
    if functional:
        imports += "from torch.nn import functional as F\n"
    decorator = (
        '    @torch.compile(mode="max-autotune-no-cudagraphs")\n' if compiled else ""
    )
    rendered_body = "".join(f"        {line}\n" for line in body)
    return (
        f'"""Registered {task_id} PyTorch baseline."""\n\n'
        f"{imports}\n\n"
        "class ModelNew(nn.Module):\n"
        f"{decorator}"
        f"    def forward(self, {arguments}) -> torch.Tensor:\n"
        f"{rendered_body}"
    )


_REFERENCE_BODIES: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType(
    {
        "rmsnorm-static": (
            "x: torch.Tensor, gamma: torch.Tensor",
            (
                "x_fp32 = x.to(torch.float32)",
                "mean_square = torch.mean(x_fp32 * x_fp32, dim=1, keepdim=True)",
                "normalized = x_fp32 * torch.rsqrt(mean_square + 1e-5)",
                "return (normalized * gamma.to(torch.float32)).to(torch.float16)",
            ),
        ),
        "layernorm-static": (
            "x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor",
            (
                "x_fp32 = x.to(torch.float32)",
                "mean = torch.mean(x_fp32, dim=1, keepdim=True)",
                "centered = x_fp32 - mean",
                "variance = torch.mean(centered * centered, dim=1, keepdim=True)",
                "normalized = centered * torch.rsqrt(variance + 1e-5)",
                "affine = normalized * gamma.to(torch.float32) + beta.to(torch.float32)",
                "return affine.to(torch.float16)",
            ),
        ),
        "gemm-static": (
            "a: torch.Tensor, b: torch.Tensor",
            (
                "product = torch.matmul(a.to(torch.float32), b.to(torch.float32))",
                "return product.to(torch.float16)",
            ),
        ),
        "gemm-bias-relu-static": (
            "a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor",
            (
                "product = torch.matmul(a.to(torch.float32), b.to(torch.float32))",
                "affine = product + bias.to(torch.float32)",
                "return torch.relu(affine).to(torch.float16)",
            ),
        ),
    }
)

_VENDOR_BODIES: Mapping[str, tuple[str, tuple[str, ...], bool]] = MappingProxyType(
    {
        "rmsnorm-static": (
            "x: torch.Tensor, gamma: torch.Tensor",
            ("return F.rms_norm(x, (4096,), gamma, eps=1e-5)",),
            True,
        ),
        "layernorm-static": (
            "x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor",
            ("return F.layer_norm(x, (4096,), gamma, beta, eps=1e-5)",),
            True,
        ),
        "gemm-static": (
            "a: torch.Tensor, b: torch.Tensor",
            ("return torch.matmul(a, b)",),
            False,
        ),
        "gemm-bias-relu-static": (
            "a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor",
            (
                "product = torch.matmul(a, b)",
                "affine = product.to(torch.float32) + bias.to(torch.float32)",
                "return torch.relu(affine).to(torch.float16)",
            ),
            False,
        ),
    }
)


def _make_record(task_id: str, variant: str, source: str) -> BaselineSource:
    return BaselineSource(
        task_id=task_id,
        variant=variant,
        source=source,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def _build_registry() -> Mapping[tuple[str, str], BaselineSource]:
    records: dict[tuple[str, str], BaselineSource] = {}
    for task_id, (arguments, body) in _REFERENCE_BODIES.items():
        eager = _model_source(task_id, arguments, body)
        compiled = _model_source(task_id, arguments, body, compiled=True)
        vendor_arguments, vendor_body, functional = _VENDOR_BODIES[task_id]
        vendor = _model_source(
            task_id,
            vendor_arguments,
            vendor_body,
            functional=functional,
        )
        for variant, source in (
            ("compile", compiled),
            ("eager", eager),
            ("vendor", vendor),
        ):
            records[(task_id, variant)] = _make_record(task_id, variant, source)
    return MappingProxyType(records)


_BASELINES = _build_registry()


def list_baseline_task_ids() -> tuple[str, ...]:
    """Return task IDs with registered common-baseline candidates."""

    return tuple(sorted({task_id for task_id, _ in _BASELINES}))


def list_baseline_variants(task_id: str) -> tuple[str, ...]:
    """Return the registered variants for one formal task in stable order."""

    variants = tuple(sorted(variant for known_task, variant in _BASELINES if known_task == task_id))
    if not variants:
        raise BaselineRegistryError(f"no baselines registered for task: {task_id}")
    return variants


def get_baseline_source(task_id: str, variant: str) -> BaselineSource:
    """Return one immutable baseline record."""

    try:
        return _BASELINES[(task_id, variant)]
    except KeyError:
        raise BaselineRegistryError(f"unknown baseline: {task_id}/{variant}") from None


def load_baseline_source(task_id: str, variant: str) -> str:
    """Load one registered source string without importing it."""

    return get_baseline_source(task_id, variant).source


def validate_baseline_source(
    task_id: str,
    source: str,
    *,
    source_sha256: str | None = None,
) -> BaselineSource:
    """Return the matching record only when source bytes and SHA are registered."""

    actual_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_sha256 is not None and actual_sha256 != source_sha256:
        raise BaselineRegistryError("baseline source does not match its declared SHA-256")
    for variant in list_baseline_variants(task_id):
        record = _BASELINES[(task_id, variant)]
        if record.source_sha256 == actual_sha256 and record.source == source:
            return record
    raise BaselineRegistryError(f"unregistered baseline source for task: {task_id}")


def validate_baseline_registry() -> None:
    """Check formal coverage, source hashes, syntax, and the ModelNew entry point."""

    if list_baseline_task_ids() != tuple(sorted(FORMAL_TASK_IDS)):
        raise BaselineRegistryError("baseline registry does not cover exactly the formal tasks")
    for task_id in FORMAL_TASK_IDS:
        if list_baseline_variants(task_id) != BASELINE_VARIANTS:
            raise BaselineRegistryError(f"baseline variants are incomplete for task: {task_id}")
        for variant in BASELINE_VARIANTS:
            record = _BASELINES[(task_id, variant)]
            if record.task_id != task_id or record.variant != variant:
                raise BaselineRegistryError(f"baseline registry key mismatch: {task_id}/{variant}")
            actual_sha256 = hashlib.sha256(record.source.encode("utf-8")).hexdigest()
            if actual_sha256 != record.source_sha256:
                raise BaselineRegistryError(f"baseline SHA-256 mismatch: {task_id}/{variant}")
            try:
                tree = ast.parse(record.source)
            except SyntaxError as error:
                raise BaselineRegistryError(
                    f"invalid baseline syntax for {task_id}/{variant}: {error.msg}"
                ) from error
            entry_points = {
                node.name for node in tree.body if isinstance(node, ast.ClassDef)
            }
            if "ModelNew" not in entry_points:
                raise BaselineRegistryError(
                    f"baseline is missing ModelNew: {task_id}/{variant}"
                )
