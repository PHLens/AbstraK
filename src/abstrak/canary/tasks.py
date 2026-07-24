"""Pinned task-pack registry for the A100 R1 canary study.

This module deliberately does not import PyTorch. Task sources are loaded as
hash-verified text and are imported only inside an evaluator process.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from abstrak.canary.contracts import InputCaseSpec, TaskPackSpec

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[3] / "benchmarks" / "r1-a100"
CAPABILITY_GATE_ASSET_ROOT = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "capability-gate-a100"
)


class TaskRegistryError(ValueError):
    """Raised when a task ID or pinned task asset is invalid."""


R1_SCOPE = "r1"
CAPABILITY_GATE_SCOPE = "capability-gate"


@dataclass(frozen=True)
class PinnedAsset:
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class TaskAssets:
    source: PinnedAsset
    oracles: Mapping[str, PinnedAsset]


@dataclass(frozen=True)
class _TaskRegistryScope:
    packs: Mapping[str, TaskPackSpec]
    assets: Mapping[str, TaskAssets]
    asset_root: Path = DEFAULT_ASSET_ROOT


_ROW_REDUCTION_SOURCE = PinnedAsset(
    relative_path="tasks/row_reduction_scale.py",
    sha256="00057e63fd2b5be59044173fad8a0cc3b4d573021b8486880f089bf534c2f2cf",
)
_ROW_REDUCTION_TRITON_ORACLE = PinnedAsset(
    relative_path="oracles/triton/row_reduction_scale.py",
    sha256="037ec6d894daa9c59f01a43739fb18296a3c3aaf10d0946c5d07c04116f48fb8",
)
_ROW_REDUCTION_TILELANG_ORACLE = PinnedAsset(
    relative_path="oracles/tilelang/row_reduction_scale.py",
    sha256="f725e8dce6ccb192a0cc0f8c7b85a988807c9a7445ab2dcbfc92f92cd4978d2a",
)
_ROW_REDUCTION_CUTE_ORACLE = PinnedAsset(
    relative_path="oracles/cute/row_reduction_scale.py",
    sha256="c7fa2a82fde0ac5191f16ddd867bbc7c35656b413cb983e47811bdabab2466d8",
)
_MATMUL_BIAS_SOURCE = PinnedAsset(
    relative_path="tasks/matmul_bias.py",
    sha256="56c37bc614bbaca3c6e09d278960283c5c22726e8876026e7bedb6f063d561da",
)
_MATMUL_BIAS_TRITON_ORACLE = PinnedAsset(
    relative_path="oracles/triton/matmul_bias.py",
    sha256="ee2f018ab72ca51425d01762ba1a93008465225b6e4f81ea853cdc3281cf3aac",
)
_MATMUL_BIAS_TILELANG_ORACLE = PinnedAsset(
    relative_path="oracles/tilelang/matmul_bias.py",
    sha256="1f19050b78a01ea010b8dd488001b8d9924701d7f00d954a5b8d2352b620a16d",
)
_MATMUL_BIAS_CUTE_ORACLE = PinnedAsset(
    relative_path="oracles/cute/matmul_bias.py",
    sha256="ff67983fd0a66b37105821491db87a8fe9288921a9b4f14e6de9bb3b9043ae23",
)


def _scientific_asset(kind: str, backend: str | None, sha256: str) -> PinnedAsset:
    directory = "tasks" if backend is None else f"oracles/{backend}"
    return PinnedAsset(relative_path=f"{directory}/{kind}.py", sha256=sha256)


_RMSNORM_SOURCE = _scientific_asset(
    "rmsnorm_static", None, "7bf18a0db0bcdc98d645823fbbdc53b2b6bddb47939417e57e73379f875597cb"
)
_RMSNORM_ORACLES = MappingProxyType(
    {
        "triton": _scientific_asset(
            "rmsnorm_static",
            "triton",
            "a78610048c3973cb8284aca2f0f078ca4601c939d8c749e10985546d84054f64",
        ),
        "tilelang": _scientific_asset(
            "rmsnorm_static",
            "tilelang",
            "31ab488dbed16cc358b0de0028125b6a7255c537f821ebcb69292c2332520a28",
        ),
        "cute": _scientific_asset(
            "rmsnorm_static",
            "cute",
            "b3ce19069eb0db26445b6913430419825a8fb7c4e0f54758acc0f6ac781a9ca4",
        ),
    }
)
_LAYERNORM_SOURCE = _scientific_asset(
    "layernorm_static", None, "26464a3b5e134c9ddd062c62627a73c76351edc2baa79ffd3bc5afd1aab47137"
)
_LAYERNORM_ORACLES = MappingProxyType(
    {
        "triton": _scientific_asset(
            "layernorm_static",
            "triton",
            "45b1fa8efa8b35fc00fe26dcb65d2d4014da921bd697cab6ac46b207bb713ca9",
        ),
        "tilelang": _scientific_asset(
            "layernorm_static",
            "tilelang",
            "b2f41c1fb1a8904253f3b560c313e8fc8cc08d4a7f6a2516a9fedd9064ba687c",
        ),
        "cute": _scientific_asset(
            "layernorm_static",
            "cute",
            "a94b52ceb65d8ff615e72c5f74df9d525bacb0219ba096892516d49dd4ccf7fd",
        ),
    }
)
_GEMM_SOURCE = _scientific_asset(
    "gemm_static", None, "d777fe3b008e3137de474c85e6ba5e4fc38188f032d5836afa4ff5b7e756f7cb"
)
_GEMM_ORACLES = MappingProxyType(
    {
        "triton": _scientific_asset(
            "gemm_static",
            "triton",
            "d6d1fefbdc4d47607525998e29f68159dd161630c37d8422e726adfca40faca9",
        ),
        "tilelang": _scientific_asset(
            "gemm_static",
            "tilelang",
            "ddb4b2da2fb0381c453031ef62d97aa3a1cb1d3dad658e6e066fc6bc838405b1",
        ),
        "cute": _scientific_asset(
            "gemm_static",
            "cute",
            "3c465f69eb4b3f65d4b1e1afebf7e262c017d2a7069e2bef63fd3d9ce98be0e0",
        ),
    }
)
_GEMM_BIAS_RELU_SOURCE = _scientific_asset(
    "gemm_bias_relu_static",
    None,
    "53cb92a3cd9a0ccc1b21612b4f60631d4f9694f1da9d22e906756cfb07f5b332",
)
_GEMM_BIAS_RELU_ORACLES = MappingProxyType(
    {
        "triton": _scientific_asset(
            "gemm_bias_relu_static",
            "triton",
            "d30eae2dd7688b41a37d5c5c61f2c119125cbd7f13f3b95fc6ec2acc34b4cfa3",
        ),
        "tilelang": _scientific_asset(
            "gemm_bias_relu_static",
            "tilelang",
            "b0aafea981975ad22ce582a374fa308968c8669d6a447afc125160875593b24d",
        ),
        "cute": _scientific_asset(
            "gemm_bias_relu_static",
            "cute",
            "0bba9d6b8827cb179279f657b67199429e6e7286b50455aad0b6f3cc8f86ef86",
        ),
    }
)

_SCIENTIFIC_DEV_CASES = (
    InputCaseSpec(id="dev-random-1", kind="random", seed=2026071701),
    InputCaseSpec(id="dev-random-2", kind="random", seed=2026071702),
)
_SCIENTIFIC_SEALED_CASES = (
    InputCaseSpec(id="sealed-random-1", kind="random", seed=2026071801),
    InputCaseSpec(id="sealed-random-2", kind="random", seed=2026071802),
    InputCaseSpec(id="sealed-random-3", kind="random", seed=2026071803),
    InputCaseSpec(id="sealed-random-4", kind="random", seed=2026071804),
    InputCaseSpec(id="sealed-constant", kind="constant", seed=2026071805, value=0.25),
)

_R1_TASK_PACKS: Mapping[str, TaskPackSpec] = MappingProxyType(
    {
        "row-reduction-scale": TaskPackSpec(
            id="row-reduction-scale",
            specification=(
                "Given a contiguous FP16 tensor x with shape (1024, 1024), sum each row "
                "using FP32 accumulation, multiply every row sum by 0.5, and return a "
                "contiguous FP16 tensor with shape (1024,)."
            ),
            source_path=_ROW_REDUCTION_SOURCE.relative_path,
            source_sha256=_ROW_REDUCTION_SOURCE.sha256,
            dtype="fp16",
            reference_precision="fp32",
            input_shapes=((1024, 1024),),
            parameters=(
                ("rows", 1024),
                ("columns", 1024),
                ("scale", 0.5),
                ("output_dtype", "fp16"),
            ),
            atol=1e-2,
            rtol=1e-2,
            fallback_policy="forbid_framework_ops",
            dev_cases=(
                InputCaseSpec(id="dev-random-1", kind="random", seed=2026071701),
                InputCaseSpec(id="dev-random-2", kind="random", seed=2026071702),
            ),
            sealed_cases=(
                InputCaseSpec(id="sealed-random-1", kind="random", seed=2026071801),
                InputCaseSpec(id="sealed-random-2", kind="random", seed=2026071802),
                InputCaseSpec(id="sealed-random-3", kind="random", seed=2026071803),
                InputCaseSpec(id="sealed-random-4", kind="random", seed=2026071804),
                InputCaseSpec(
                    id="sealed-constant",
                    kind="constant",
                    seed=2026071805,
                    value=0.25,
                ),
            ),
        ),
        "matmul-bias": TaskPackSpec(
            id="matmul-bias",
            specification=(
                "Given contiguous FP16 tensors a, b, and bias with shapes (256, 256), "
                "(256, 256), and (256,), compute a @ b using FP32 accumulation, add "
                "the bias in FP32, and return a contiguous FP16 tensor with shape "
                "(256, 256)."
            ),
            source_path=_MATMUL_BIAS_SOURCE.relative_path,
            source_sha256=_MATMUL_BIAS_SOURCE.sha256,
            dtype="fp16",
            reference_precision="fp32",
            input_shapes=((256, 256), (256, 256), (256,)),
            parameters=(
                ("m", 256),
                ("n", 256),
                ("k", 256),
                ("epilogue", "bias"),
                ("output_dtype", "fp16"),
            ),
            atol=1e-2,
            rtol=1e-2,
            fallback_policy="forbid_framework_ops",
            dev_cases=(
                InputCaseSpec(id="dev-random-1", kind="random", seed=2026071901),
                InputCaseSpec(id="dev-random-2", kind="random", seed=2026071902),
            ),
            sealed_cases=(
                InputCaseSpec(id="sealed-random-1", kind="random", seed=2026072001),
                InputCaseSpec(id="sealed-random-2", kind="random", seed=2026072002),
                InputCaseSpec(id="sealed-random-3", kind="random", seed=2026072003),
                InputCaseSpec(id="sealed-random-4", kind="random", seed=2026072004),
                InputCaseSpec(
                    id="sealed-constant",
                    kind="constant",
                    seed=2026072005,
                    value=0.125,
                ),
            ),
        ),
        "rmsnorm-static": TaskPackSpec(
            id="rmsnorm-static",
            specification=(
                "Given contiguous FP16 tensors x and gamma with shapes (4096, 4096) and "
                "(4096,), apply row-wise RMSNorm using FP32 accumulation and epsilon "
                "1e-5, multiply by gamma in FP32, and return contiguous FP16 output with "
                "shape (4096, 4096)."
            ),
            source_path=_RMSNORM_SOURCE.relative_path,
            source_sha256=_RMSNORM_SOURCE.sha256,
            dtype="fp16",
            input_shapes=((4096, 4096), (4096,)),
            parameters=(
                ("rows", 4096),
                ("columns", 4096),
                ("epsilon", 1e-5),
                ("affine", "gamma"),
                ("output_dtype", "fp16"),
            ),
            atol=1e-2,
            rtol=1e-2,
            fallback_policy="forbid_framework_ops",
            dev_cases=_SCIENTIFIC_DEV_CASES,
            sealed_cases=_SCIENTIFIC_SEALED_CASES,
        ),
        "layernorm-static": TaskPackSpec(
            id="layernorm-static",
            specification=(
                "Given contiguous FP16 tensors x, gamma, and beta with shapes "
                "(4096, 4096), (4096,), and (4096,), apply row-wise LayerNorm using "
                "FP32 mean and variance, epsilon 1e-5, and FP32 affine arithmetic, then "
                "return contiguous FP16 output with shape (4096, 4096)."
            ),
            source_path=_LAYERNORM_SOURCE.relative_path,
            source_sha256=_LAYERNORM_SOURCE.sha256,
            dtype="fp16",
            input_shapes=((4096, 4096), (4096,), (4096,)),
            parameters=(
                ("rows", 4096),
                ("columns", 4096),
                ("epsilon", 1e-5),
                ("affine", "gamma-beta"),
                ("output_dtype", "fp16"),
            ),
            atol=1e-2,
            rtol=1e-2,
            fallback_policy="forbid_framework_ops",
            dev_cases=_SCIENTIFIC_DEV_CASES,
            sealed_cases=_SCIENTIFIC_SEALED_CASES,
        ),
        "gemm-static": TaskPackSpec(
            id="gemm-static",
            specification=(
                "Given contiguous FP16 tensors a and b with shapes (1024, 4096) and "
                "(4096, 4096), compute a @ b using FP32 accumulation and return a "
                "contiguous FP16 tensor with shape (1024, 4096)."
            ),
            source_path=_GEMM_SOURCE.relative_path,
            source_sha256=_GEMM_SOURCE.sha256,
            dtype="fp16",
            input_shapes=((1024, 4096), (4096, 4096)),
            parameters=(
                ("m", 1024),
                ("n", 4096),
                ("k", 4096),
                ("output_dtype", "fp16"),
            ),
            atol=1e-2,
            rtol=1e-2,
            fallback_policy="forbid_framework_ops",
            dev_cases=_SCIENTIFIC_DEV_CASES,
            sealed_cases=_SCIENTIFIC_SEALED_CASES,
        ),
        "gemm-bias-relu-static": TaskPackSpec(
            id="gemm-bias-relu-static",
            specification=(
                "Given contiguous FP16 tensors a, b, and bias with shapes (1024, 4096), "
                "(4096, 4096), and (4096,), compute a @ b using FP32 accumulation, add "
                "bias and apply ReLU in FP32, then return a contiguous FP16 tensor with "
                "shape (1024, 4096)."
            ),
            source_path=_GEMM_BIAS_RELU_SOURCE.relative_path,
            source_sha256=_GEMM_BIAS_RELU_SOURCE.sha256,
            dtype="fp16",
            input_shapes=((1024, 4096), (4096, 4096), (4096,)),
            parameters=(
                ("m", 1024),
                ("n", 4096),
                ("k", 4096),
                ("epilogue", "bias-relu"),
                ("output_dtype", "fp16"),
            ),
            atol=1e-2,
            rtol=1e-2,
            fallback_policy="forbid_framework_ops",
            dev_cases=_SCIENTIFIC_DEV_CASES,
            sealed_cases=_SCIENTIFIC_SEALED_CASES,
        ),
    }
)

_R1_TASK_ASSETS: Mapping[str, TaskAssets] = MappingProxyType(
    {
        "row-reduction-scale": TaskAssets(
            source=_ROW_REDUCTION_SOURCE,
            oracles=MappingProxyType(
                {
                    "triton": _ROW_REDUCTION_TRITON_ORACLE,
                    "tilelang": _ROW_REDUCTION_TILELANG_ORACLE,
                    "cute": _ROW_REDUCTION_CUTE_ORACLE,
                }
            ),
        ),
        "matmul-bias": TaskAssets(
            source=_MATMUL_BIAS_SOURCE,
            oracles=MappingProxyType(
                {
                    "triton": _MATMUL_BIAS_TRITON_ORACLE,
                    "tilelang": _MATMUL_BIAS_TILELANG_ORACLE,
                    "cute": _MATMUL_BIAS_CUTE_ORACLE,
                }
            ),
        ),
        "rmsnorm-static": TaskAssets(source=_RMSNORM_SOURCE, oracles=_RMSNORM_ORACLES),
        "layernorm-static": TaskAssets(source=_LAYERNORM_SOURCE, oracles=_LAYERNORM_ORACLES),
        "gemm-static": TaskAssets(source=_GEMM_SOURCE, oracles=_GEMM_ORACLES),
        "gemm-bias-relu-static": TaskAssets(
            source=_GEMM_BIAS_RELU_SOURCE,
            oracles=_GEMM_BIAS_RELU_ORACLES,
        ),
    }
)


_CAPABILITY_DEV_CASES = (
    InputCaseSpec(id="dev-random-1", kind="random", seed=2026072401),
    InputCaseSpec(id="dev-random-2", kind="random", seed=2026072402),
)
_CAPABILITY_SEALED_CASES = (
    InputCaseSpec(id="sealed-random-1", kind="random", seed=2026072501),
    InputCaseSpec(id="sealed-random-2", kind="random", seed=2026072502),
    InputCaseSpec(id="sealed-random-3", kind="random", seed=2026072503),
    InputCaseSpec(id="sealed-random-4", kind="random", seed=2026072504),
    InputCaseSpec(id="sealed-zero", kind="zero", seed=2026072505),
)


def _capability_task(
    task_id: str,
    specification: str,
    source_sha256: str,
    input_shapes: tuple[tuple[int, ...], ...],
    parameters: tuple[tuple[str, int | float | str | bool], ...],
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> TaskPackSpec:
    return TaskPackSpec(
        id=task_id,
        specification=specification,
        source_path=f"tasks/{task_id.replace('-', '_')}.py",
        source_sha256=source_sha256,
        dtype="fp16",
        reference_precision="fp32",
        input_shapes=input_shapes,
        parameters=parameters,
        atol=atol,
        rtol=rtol,
        fallback_policy="forbid_framework_ops",
        dev_cases=_CAPABILITY_DEV_CASES,
        sealed_cases=_CAPABILITY_SEALED_CASES,
    )


_CAPABILITY_GATE_TASK_PACKS: Mapping[str, TaskPackSpec] = MappingProxyType(
    {
        "gelu-static": _capability_task(
            "gelu-static",
            (
                "Given a contiguous FP16 tensor x with shape (8192, 4096), apply exact "
                "erf-based GELU using FP32 intermediate math and return FP16 with the same shape."
            ),
            "929106a7135ec938bc6b323b301d8dca9e5e6126923d0556435f53e19750b60f",
            ((8192, 4096),),
            (
                ("rows", 8192),
                ("columns", 4096),
                ("approximation", "none"),
                ("output_dtype", "fp16"),
            ),
        ),
        "gated-silu-static": _capability_task(
            "gated-silu-static",
            (
                "Given contiguous FP16 tensors x and gate with shape (8192, 4096), compute "
                "silu(x) * gate using FP32 intermediate math and return FP16 with the same shape."
            ),
            "7f4b063d8e6c9ea72076c0219343ba59582a2ca8a0d37f52202a89fdecdfb022",
            ((8192, 4096), (8192, 4096)),
            (
                ("rows", 8192),
                ("columns", 4096),
                ("epilogue", "silu-times-gate"),
                ("output_dtype", "fp16"),
            ),
        ),
        "gemm-large-k-static": _capability_task(
            "gemm-large-k-static",
            (
                "Given contiguous FP16 tensors A (1024, 4096) and B (4096, 4096), compute "
                "A @ B with FP32 accumulation and return FP16 shape (1024, 4096)."
            ),
            "643e7d8b591b39b35932b457a4c038110a6811e3e75de4b23272a6ca61728afa",
            ((1024, 4096), (4096, 4096)),
            (("m", 1024), ("n", 4096), ("k", 4096), ("output_dtype", "fp16")),
        ),
        "gemm-bias-relu-mirror-static": _capability_task(
            "gemm-bias-relu-mirror-static",
            (
                "Given contiguous FP16 tensors A (4096, 4096), B (4096, 1024), and bias "
                "(1024,), compute A @ B with FP32 accumulation, add bias and apply ReLU "
                "in FP32, then return FP16 shape (4096, 1024)."
            ),
            "dbca54684fb9175226394cbfb61678f1f03551dd18f285cbceb2a4a4ea685c57",
            ((4096, 4096), (4096, 1024), (1024,)),
            (
                ("m", 4096),
                ("n", 1024),
                ("k", 4096),
                ("epilogue", "bias-relu"),
                ("output_dtype", "fp16"),
            ),
        ),
        "gemm-small-k-irregular-static": _capability_task(
            "gemm-small-k-irregular-static",
            (
                "Given contiguous FP16 tensors A (8191, 80) and B (80, 8179), compute "
                "A @ B with FP32 accumulation and return FP16 shape (8191, 8179)."
            ),
            "130c1301537f21828ab4764e47773de4e52a23e77b3d88b96c2213204f22fe3f",
            ((8191, 80), (80, 8179)),
            (("m", 8191), ("n", 8179), ("k", 80), ("output_dtype", "fp16")),
        ),
        "row-sum-static": _capability_task(
            "row-sum-static",
            (
                "Given a contiguous FP16 tensor x with shape (16384, 4096), sum the final "
                "dimension using FP32 accumulation and return FP32 shape (16384,)."
            ),
            "89d80ff45a8894e8a6a73d236ad77c4258393c8598f176794d627074950518b0",
            ((16384, 4096),),
            (("rows", 16384), ("columns", 4096), ("output_dtype", "fp32")),
            rtol=1e-3,
        ),
        "row-softmax-static": _capability_task(
            "row-softmax-static",
            (
                "Given a contiguous FP16 tensor x with shape (8192, 4096), compute stable "
                "row-wise softmax as FP32 max, exp, sum, and divide operations, then return "
                "FP16 with the same shape."
            ),
            "12d59716bb27aa7322cb287108827fd69c7b259385ab4832df7ff5bda7da2c05",
            ((8192, 4096),),
            (("rows", 8192), ("columns", 4096), ("output_dtype", "fp16")),
            atol=1e-3,
        ),
        "rmsnorm-wide-static": _capability_task(
            "rmsnorm-wide-static",
            (
                "Given contiguous FP16 tensors x (8192, 4096) and gamma (4096,), apply "
                "row-wise RMSNorm using FP32 mean-square math and epsilon 1e-5, then return "
                "FP16 shape (8192, 4096)."
            ),
            "284b0523b56af85dcf23977049e7c789871629ac74e2466a22221d436fe10b85",
            ((8192, 4096), (4096,)),
            (
                ("rows", 8192),
                ("columns", 4096),
                ("epsilon", 1e-5),
                ("output_dtype", "fp16"),
            ),
        ),
    }
)
CAPABILITY_GATE_TASK_IDS = tuple(_CAPABILITY_GATE_TASK_PACKS)


def _capability_assets(
    task_id: str,
    source_sha256: str,
    expert_sha256: str,
) -> TaskAssets:
    filename = f"{task_id.replace('-', '_')}.py"
    return TaskAssets(
        source=PinnedAsset(relative_path=f"tasks/{filename}", sha256=source_sha256),
        oracles=MappingProxyType(
            {
                "tilelang": PinnedAsset(
                    relative_path=f"experts/{filename}",
                    sha256=expert_sha256,
                )
            }
        ),
    )


_CAPABILITY_GATE_TASK_ASSETS: Mapping[str, TaskAssets] = MappingProxyType(
    {
        "gelu-static": _capability_assets(
            "gelu-static",
            "929106a7135ec938bc6b323b301d8dca9e5e6126923d0556435f53e19750b60f",
            "64d62fc67569c90ebf094bb0bff25acf371fadb4ae1a5927407f3770e6e042e6",
        ),
        "gated-silu-static": _capability_assets(
            "gated-silu-static",
            "7f4b063d8e6c9ea72076c0219343ba59582a2ca8a0d37f52202a89fdecdfb022",
            "5cc0f723b5de15ecbc07ab2d6ec073a0a9cc9f51e9ded077972c57dc88e7ab9f",
        ),
        "gemm-large-k-static": _capability_assets(
            "gemm-large-k-static",
            "643e7d8b591b39b35932b457a4c038110a6811e3e75de4b23272a6ca61728afa",
            "ae8939062532b042c8fd14f8e624d6af6f4d6b7d61d0c7df6d849fbf0df743d1",
        ),
        "gemm-bias-relu-mirror-static": _capability_assets(
            "gemm-bias-relu-mirror-static",
            "dbca54684fb9175226394cbfb61678f1f03551dd18f285cbceb2a4a4ea685c57",
            "1d44b40ec759eca28e706e477460e1ace00b00710e3835c83789a2b7aeb15529",
        ),
        "gemm-small-k-irregular-static": _capability_assets(
            "gemm-small-k-irregular-static",
            "130c1301537f21828ab4764e47773de4e52a23e77b3d88b96c2213204f22fe3f",
            "742cb8b6b9a4529ace2853176f5fad4af4752c965188b072eb034c41d5ee96f2",
        ),
        "row-sum-static": _capability_assets(
            "row-sum-static",
            "89d80ff45a8894e8a6a73d236ad77c4258393c8598f176794d627074950518b0",
            "c4335afd0423e751e51017128488c4d360d3ea7d30ef3ab1da3ac9efa0f763c3",
        ),
        "row-softmax-static": _capability_assets(
            "row-softmax-static",
            "12d59716bb27aa7322cb287108827fd69c7b259385ab4832df7ff5bda7da2c05",
            "7f7ad5d75fd84c5b29ae81d49cef3c19657370ecd301ed5c13a0b9c9da93ac9a",
        ),
        "rmsnorm-wide-static": _capability_assets(
            "rmsnorm-wide-static",
            "284b0523b56af85dcf23977049e7c789871629ac74e2466a22221d436fe10b85",
            "44f22d22a9898434f1aa43b3c28e43056a92098a753deffb6f090d991b0e662c",
        ),
    }
)

_TASK_REGISTRIES: Mapping[str, _TaskRegistryScope] = MappingProxyType(
    {
        R1_SCOPE: _TaskRegistryScope(
            packs=_R1_TASK_PACKS,
            assets=_R1_TASK_ASSETS,
        ),
        CAPABILITY_GATE_SCOPE: _TaskRegistryScope(
            packs=_CAPABILITY_GATE_TASK_PACKS,
            assets=_CAPABILITY_GATE_TASK_ASSETS,
            asset_root=CAPABILITY_GATE_ASSET_ROOT,
        ),
    }
)


def _build_global_indexes(
    registries: Mapping[str, _TaskRegistryScope],
) -> tuple[Mapping[str, TaskPackSpec], Mapping[str, TaskAssets]]:
    packs: dict[str, TaskPackSpec] = {}
    assets: dict[str, TaskAssets] = {}
    owners: dict[str, str] = {}
    for scope, registry in registries.items():
        task_ids = set(registry.packs) | set(registry.assets)
        for task_id in sorted(task_ids):
            previous_scope = owners.get(task_id)
            if previous_scope is not None:
                raise TaskRegistryError(
                    f"task ID is registered in multiple scopes: {task_id} "
                    f"({previous_scope}, {scope})"
                )
            owners[task_id] = scope
        packs.update(registry.packs)
        assets.update(registry.assets)
    return MappingProxyType(packs), MappingProxyType(assets)


_TASK_PACKS, _TASK_ASSETS = _build_global_indexes(_TASK_REGISTRIES)


def _registry_for_scope(scope: str) -> _TaskRegistryScope:
    try:
        return _TASK_REGISTRIES[scope]
    except KeyError:
        raise TaskRegistryError(f"unknown task registry scope: {scope}") from None


def _registry_for_task(task_id: str) -> _TaskRegistryScope:
    for registry in _TASK_REGISTRIES.values():
        if task_id in registry.packs:
            return registry
    raise TaskRegistryError(f"unknown task pack: {task_id}")


def list_task_ids(scope: str = R1_SCOPE) -> tuple[str, ...]:
    """Return registered task IDs in stable order."""

    return tuple(sorted(_registry_for_scope(scope).packs))


def get_task_pack(task_id: str) -> TaskPackSpec:
    """Return an isolated copy of a registered task-pack contract."""

    try:
        return _TASK_PACKS[task_id].model_copy(deep=True)
    except KeyError:
        raise TaskRegistryError(f"unknown task pack: {task_id}") from None


def get_task_assets(task_id: str) -> TaskAssets:
    """Return pinned source and oracle references for a task pack."""

    try:
        return _TASK_ASSETS[task_id]
    except KeyError:
        raise TaskRegistryError(f"unknown task pack: {task_id}") from None


def _resolve_asset_root(asset_root: str | Path | None) -> Path:
    configured_root = DEFAULT_ASSET_ROOT if asset_root is None else Path(asset_root).expanduser()
    try:
        resolved_root = configured_root.resolve(strict=True)
    except OSError as error:
        message = f"cannot resolve task asset root {configured_root}: {error}"
        raise TaskRegistryError(message) from error
    if not resolved_root.is_dir():
        raise TaskRegistryError(f"task asset root is not a directory: {resolved_root}")
    return resolved_root


def _resolve_asset_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        not relative.parts
        or relative.is_absolute()
        or "\\" in relative_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise TaskRegistryError(f"unsafe task asset path: {relative_path!r}")
    try:
        resolved_path = root.joinpath(*relative.parts).resolve(strict=True)
        resolved_path.relative_to(root)
    except (OSError, ValueError) as error:
        raise TaskRegistryError(f"task asset escaped or is missing: {relative_path}") from error
    if not resolved_path.is_file():
        raise TaskRegistryError(f"task asset is not a regular file: {relative_path}")
    return resolved_path


def load_pinned_asset(
    asset: PinnedAsset,
    *,
    asset_root: str | Path | None = None,
) -> str:
    """Load UTF-8 source only after path containment and SHA-256 checks."""

    if _SHA256_PATTERN.fullmatch(asset.sha256) is None:
        raise TaskRegistryError(f"invalid SHA-256 for task asset: {asset.relative_path}")
    root = _resolve_asset_root(asset_root)
    path = _resolve_asset_path(root, asset.relative_path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise TaskRegistryError(f"cannot read task asset {asset.relative_path}: {error}") from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != asset.sha256:
        raise TaskRegistryError(
            f"task asset SHA-256 mismatch for {asset.relative_path}: "
            f"expected {asset.sha256}, found {actual_sha256}"
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TaskRegistryError(f"task asset is not UTF-8: {asset.relative_path}") from error


def load_task_source(task_id: str, *, asset_root: str | Path | None = None) -> str:
    """Load one registered public task fixture."""

    resolved_root = _registry_for_task(task_id).asset_root if asset_root is None else asset_root
    return load_pinned_asset(get_task_assets(task_id).source, asset_root=resolved_root)


def load_oracle_source(
    task_id: str,
    target_id: str,
    *,
    asset_root: str | Path | None = None,
) -> str:
    """Load one registered trusted oracle fixture."""

    assets = get_task_assets(task_id)
    try:
        oracle = assets.oracles[target_id]
    except KeyError:
        raise TaskRegistryError(f"no {target_id} oracle registered for task {task_id}") from None
    resolved_root = _registry_for_task(task_id).asset_root if asset_root is None else asset_root
    return load_pinned_asset(oracle, asset_root=resolved_root)


def validate_task_registry(
    *,
    scope: str = R1_SCOPE,
    asset_root: str | Path | None = None,
) -> None:
    """Validate task contracts, cross-references, paths, and content hashes."""

    registry = _registry_for_scope(scope)
    resolved_root = registry.asset_root if asset_root is None else asset_root
    if set(registry.packs) != set(registry.assets):
        raise TaskRegistryError("task contracts and asset registry have different task IDs")
    for task_id in sorted(registry.packs):
        task_pack = registry.packs[task_id]
        assets = registry.assets[task_id]
        if task_pack.id != task_id:
            raise TaskRegistryError(f"task registry key does not match contract ID: {task_id}")
        if (
            task_pack.source_path != assets.source.relative_path
            or task_pack.source_sha256 != assets.source.sha256
        ):
            raise TaskRegistryError(f"task source reference mismatch: {task_id}")
        load_pinned_asset(assets.source, asset_root=resolved_root)
        if not assets.oracles:
            raise TaskRegistryError(f"task has no registered oracle: {task_id}")
        for oracle in assets.oracles.values():
            load_pinned_asset(oracle, asset_root=resolved_root)
