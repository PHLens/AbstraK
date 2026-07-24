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
CAPABILITY_TASK_IDS = (
    "gelu-static",
    "gated-silu-static",
    "gemm-large-k-static",
    "gemm-bias-relu-mirror-static",
    "gemm-small-k-irregular-static",
    "row-sum-static",
    "row-softmax-static",
    "rmsnorm-wide-static",
)
BASELINE_VARIANTS = ("compile", "eager", "vendor")
R1_SCOPE = "r1"
CAPABILITY_GATE_SCOPE = "capability-gate"


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


_R1_REFERENCE_BODIES: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType(
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

_R1_VENDOR_BODIES: Mapping[str, tuple[str, tuple[str, ...], bool]] = MappingProxyType(
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

_CAPABILITY_REFERENCE_BODIES: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType(
    {
        "gelu-static": (
            "x: torch.Tensor",
            (
                "x_fp32 = x.to(torch.float32)",
                "scale = 0.7071067811865476",
                "return (0.5 * x_fp32 * (1.0 + torch.erf(x_fp32 * scale))).to(torch.float16)",
            ),
        ),
        "gated-silu-static": (
            "x: torch.Tensor, gate: torch.Tensor",
            (
                "x_fp32 = x.to(torch.float32)",
                "gate_fp32 = gate.to(torch.float32)",
                "silu = x_fp32 / (1.0 + torch.exp(-x_fp32))",
                "return (silu * gate_fp32).to(torch.float16)",
            ),
        ),
        "gemm-large-k-static": (
            "a: torch.Tensor, b: torch.Tensor",
            (
                "product = torch.matmul(a.to(torch.float32), b.to(torch.float32))",
                "return product.to(torch.float16)",
            ),
        ),
        "gemm-bias-relu-mirror-static": (
            "a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor",
            (
                "product = torch.matmul(a.to(torch.float32), b.to(torch.float32))",
                "affine = product + bias.to(torch.float32)",
                "return torch.relu(affine).to(torch.float16)",
            ),
        ),
        "gemm-small-k-irregular-static": (
            "a: torch.Tensor, b: torch.Tensor",
            (
                "product = torch.matmul(a.to(torch.float32), b.to(torch.float32))",
                "return product.to(torch.float16)",
            ),
        ),
        "row-sum-static": (
            "x: torch.Tensor",
            ("return torch.sum(x, dim=1, dtype=torch.float32)",),
        ),
        "row-softmax-static": (
            "x: torch.Tensor",
            (
                "x_fp32 = x.to(torch.float32)",
                "shifted = x_fp32 - torch.amax(x_fp32, dim=1, keepdim=True)",
                "numerator = torch.exp(shifted)",
                "return (numerator / torch.sum(numerator, dim=1, keepdim=True)).to(torch.float16)",
            ),
        ),
        "rmsnorm-wide-static": (
            "x: torch.Tensor, gamma: torch.Tensor",
            (
                "x_fp32 = x.to(torch.float32)",
                "mean_square = torch.mean(x_fp32 * x_fp32, dim=1, keepdim=True)",
                "normalized = x_fp32 * torch.rsqrt(mean_square + 1e-5)",
                "return (normalized * gamma.to(torch.float32)).to(torch.float16)",
            ),
        ),
    }
)

_CAPABILITY_VENDOR_BODIES: Mapping[str, tuple[str, tuple[str, ...], bool]] = MappingProxyType(
    {
        "gelu-static": (
            "x: torch.Tensor",
            ("return F.gelu(x.to(torch.float32), approximate=\"none\").to(torch.float16)",),
            True,
        ),
        "gated-silu-static": (
            "x: torch.Tensor, gate: torch.Tensor",
            (
                "return (F.silu(x.to(torch.float32)) * gate.to(torch.float32)).to(torch.float16)",
            ),
            True,
        ),
        "gemm-large-k-static": (
            "a: torch.Tensor, b: torch.Tensor",
            ("return torch.matmul(a, b)",),
            False,
        ),
        "gemm-bias-relu-mirror-static": (
            "a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor",
            (
                "product = torch.matmul(a, b)",
                "return torch.relu(product.to(torch.float32) + "
                "bias.to(torch.float32)).to(torch.float16)",
            ),
            False,
        ),
        "gemm-small-k-irregular-static": (
            "a: torch.Tensor, b: torch.Tensor",
            ("return torch.matmul(a, b)",),
            False,
        ),
        "row-sum-static": (
            "x: torch.Tensor",
            ("return torch.sum(input=x, dim=-1, dtype=torch.float32)",),
            False,
        ),
        "row-softmax-static": (
            "x: torch.Tensor",
            ("return F.softmax(x, dim=1, dtype=torch.float32).to(torch.float16)",),
            True,
        ),
        "rmsnorm-wide-static": (
            "x: torch.Tensor, gamma: torch.Tensor",
            (
                "return F.rms_norm(x.to(torch.float32), (4096,), "
                "gamma.to(torch.float32), eps=1e-5).to(torch.float16)",
            ),
            True,
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


def _build_registry(
    reference_bodies: Mapping[str, tuple[str, tuple[str, ...]]],
    vendor_bodies: Mapping[str, tuple[str, tuple[str, ...], bool]],
) -> Mapping[tuple[str, str], BaselineSource]:
    records: dict[tuple[str, str], BaselineSource] = {}
    if set(reference_bodies) != set(vendor_bodies):
        raise BaselineRegistryError("reference and vendor baseline task IDs differ")
    for task_id, (arguments, body) in reference_bodies.items():
        eager = _model_source(task_id, arguments, body)
        compiled = _model_source(task_id, arguments, body, compiled=True)
        vendor_arguments, vendor_body, functional = vendor_bodies[task_id]
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


_R1_BASELINES = _build_registry(_R1_REFERENCE_BODIES, _R1_VENDOR_BODIES)
_CAPABILITY_BASELINES = _build_registry(
    _CAPABILITY_REFERENCE_BODIES,
    _CAPABILITY_VENDOR_BODIES,
)
if set(_R1_BASELINES) & set(_CAPABILITY_BASELINES):
    raise BaselineRegistryError("baseline IDs must be unique across registry scopes")
_BASELINE_REGISTRIES: Mapping[str, Mapping[tuple[str, str], BaselineSource]] = MappingProxyType(
    {
        R1_SCOPE: _R1_BASELINES,
        CAPABILITY_GATE_SCOPE: _CAPABILITY_BASELINES,
    }
)
_BASELINES = MappingProxyType({**_R1_BASELINES, **_CAPABILITY_BASELINES})


def _registry_for_scope(scope: str) -> Mapping[tuple[str, str], BaselineSource]:
    try:
        return _BASELINE_REGISTRIES[scope]
    except KeyError:
        raise BaselineRegistryError(f"unknown baseline registry scope: {scope}") from None


def list_baseline_task_ids(scope: str = R1_SCOPE) -> tuple[str, ...]:
    """Return task IDs with registered common-baseline candidates."""

    return tuple(sorted({task_id for task_id, _ in _registry_for_scope(scope)}))


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


def validate_baseline_registry(*, scope: str = R1_SCOPE) -> None:
    """Check formal coverage, source hashes, syntax, and the ModelNew entry point."""

    expected_task_ids = FORMAL_TASK_IDS if scope == R1_SCOPE else CAPABILITY_TASK_IDS
    registry = _registry_for_scope(scope)
    if list_baseline_task_ids(scope=scope) != tuple(sorted(expected_task_ids)):
        raise BaselineRegistryError("baseline registry does not cover exactly the formal tasks")
    for task_id in expected_task_ids:
        if list_baseline_variants(task_id) != BASELINE_VARIANTS:
            raise BaselineRegistryError(f"baseline variants are incomplete for task: {task_id}")
        for variant in BASELINE_VARIANTS:
            record = registry[(task_id, variant)]
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
