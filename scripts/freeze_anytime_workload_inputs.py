#!/usr/bin/env python3
"""Freeze the offline-only workload input manifest without importing GPU code."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from abstrak.anytime.isolation import build_anytime_process_isolation_contract
from abstrak.anytime.workloads import (
    BASELINE_VARIANTS,
    DEFAULT_CANDIDATE_MAX_MEMORY_BYTES,
    DEFAULT_CANDIDATE_MAX_WALL_SECONDS,
    DEFAULT_INPUT_MANIFEST,
    ENVIRONMENT_ASSET_PATHS,
    EXPECTED_WHEELHOUSE_ARCHIVE_SHA256,
    KERNELBENCH_REVISION,
    PINNED_INPUT_MANIFEST_SHA256,
    TARGET_IDS,
    WORKLOAD_SOURCE_INPUTS,
    AnytimeBaselineSourceInput,
    AnytimeEnvironmentContract,
    AnytimeExpertSourceInput,
    AnytimeGeneratorCase,
    AnytimeGeneratorContract,
    AnytimeKernelBenchLineage,
    AnytimeNumericalAdversary,
    AnytimeStateTransferContract,
    AnytimeTargetCardInput,
    AnytimeTensorInput,
    AnytimeTimingInputContract,
    AnytimeToleranceContract,
    AnytimeWorkloadInputManifest,
    AnytimeWorkloadPack,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "l1-2-standard-matmul",
        "family": "gemm",
        "specification": (
            "Compute C = A @ B for fixed row-major FP16 A[2048,8192] and "
            "B[8192,4096], returning FP16 C[2048,4096] with FP32 accumulation."
        ),
        "arguments": (
            ("a", (2048, 8192), "fp16", "uniform finite matrix values"),
            ("b", (8192, 4096), "fp16", "uniform finite matrix values"),
        ),
        "state": (),
        "output": (2048, 4096),
    },
    {
        "id": "l1-8-irregular-matmul",
        "family": "gemm",
        "specification": (
            "Compute C = A @ B for fixed irregular FP16 A[8205,2949] and "
            "B[2949,5921], returning FP16 C[8205,5921] with FP32 accumulation."
        ),
        "arguments": (
            ("a", (8205, 2949), "fp16", "uniform finite matrix values"),
            ("b", (2949, 5921), "fp16", "uniform finite matrix values"),
        ),
        "state": (),
        "output": (8205, 5921),
    },
    {
        "id": "l1-40-layernorm",
        "family": "reduction",
        "specification": (
            "Apply affine LayerNorm over the final [64,256,256] dimensions of fixed FP16 "
            "x[16,64,256,256], using epsilon 1e-5 and an identical cloned weight/bias state."
        ),
        "arguments": (("x", (16, 64, 256, 256), "fp16", "uniform finite activation values"),),
        "state": (
            ("ln_weight", (64, 256, 256)),
            ("ln_bias", (64, 256, 256)),
        ),
        "output": (16, 64, 256, 256),
    },
    {
        "id": "l1-24-logsoftmax",
        "family": "reduction",
        "specification": (
            "Compute LogSoftmax along dimension 1 for fixed FP16 x[4096,393216], returning "
            "FP16 output of the same shape with numerically stable FP32 reductions."
        ),
        "arguments": (("x", (4096, 393216), "fp16", "uniform finite logits"),),
        "state": (),
        "output": (4096, 393216),
    },
    {
        "id": "l1-93-masked-cumsum",
        "family": "scan",
        "specification": (
            "Compute cumsum(x * mask, dim=1) for fixed FP16 x[32768,32768] and boolean mask "
            "of the same shape, returning FP16 output with FP32 scan accumulation."
        ),
        "arguments": (
            ("x", (32768, 32768), "fp16", "uniform finite scan values"),
            ("mask", (32768, 32768), "bool", "deterministic boolean mask"),
        ),
        "state": (),
        "output": (32768, 32768),
    },
    {
        "id": "l1-97-scaled-dot-product-attention",
        "family": "attention",
        "specification": (
            "Compute unmasked scaled dot-product attention for fixed FP16 Q, K, and V tensors "
            "of shape [32,32,512,1024], returning FP16 output of the same shape."
        ),
        "arguments": (
            ("q", (32, 32, 512, 1024), "fp16", "uniform finite query values"),
            ("k", (32, 32, 512, 1024), "fp16", "uniform finite key values"),
            ("v", (32, 32, 512, 1024), "fp16", "uniform finite value values"),
        ),
        "state": (),
        "output": (32, 32, 512, 1024),
    },
    {
        "id": "l1-85-asymmetric-depthwise-conv2d",
        "family": "convolution",
        "specification": (
            "Apply bias-free depthwise Conv2d to fixed FP16 x[32,128,128,256] with 128 groups, "
            "kernel [3,7], unit stride, no padding, and cloned weight state."
        ),
        "arguments": (("x", (32, 128, 128, 256), "fp16", "uniform finite image values"),),
        "state": (("conv_weight", (128, 1, 3, 7)),),
        "output": (32, 128, 126, 250),
    },
    {
        "id": "l2-14-gemm-divide-sum-scaling",
        "family": "gemm-fusion",
        "specification": (
            "For fixed FP16 x[1024,8192] and cloned weight[8192,8192], compute x @ weight.T, "
            "divide by 2, sum dimension 1 with keepdim, then multiply by 1.5."
        ),
        "arguments": (("x", (1024, 8192), "fp16", "uniform finite activation values"),),
        "state": (("weight", (8192, 8192)),),
        "output": (1024, 1),
    },
    {
        "id": "l2-99-matmul-gelu-softmax",
        "family": "gemm-fusion",
        "specification": (
            "For fixed FP16 x[1024,8192] and cloned Linear(8192,8192) weight/bias, compute "
            "linear, exact GELU, then Softmax along dimension 1."
        ),
        "arguments": (("x", (1024, 8192), "fp16", "uniform finite activation values"),),
        "state": (
            ("linear_weight", (8192, 8192)),
            ("linear_bias", (8192,)),
        ),
        "output": (1024, 8192),
    },
    {
        "id": "l2-1-conv2d-relu-biasadd",
        "family": "conv-fusion",
        "specification": (
            "For fixed FP16 x[128,64,128,128], apply cloned Conv2d(64,128,3), ReLU, and a "
            "cloned broadcast bias[128,1,1], returning FP16 [128,128,126,126]."
        ),
        "arguments": (("x", (128, 64, 128, 128), "fp16", "uniform finite image values"),),
        "state": (
            ("conv_weight", (128, 64, 3, 3)),
            ("conv_bias", (128,)),
            ("broadcast_bias", (128, 1, 1)),
        ),
        "output": (128, 128, 126, 126),
    },
    {
        "id": "l2-2-convtranspose2d-bias-clamp-scale",
        "family": "conv-fusion",
        "specification": (
            "For fixed FP16 x[128,64,128,128], apply cloned ConvTranspose2d(64,64,3, "
            "stride=2,padding=1,output_padding=1), broadcast bias, clamp, scale by 2, clamp, "
            "and divide by 2."
        ),
        "arguments": (("x", (128, 64, 128, 128), "fp16", "uniform finite image values"),),
        "state": (
            ("conv_transpose_weight", (64, 64, 3, 3)),
            ("conv_transpose_bias", (64,)),
            ("broadcast_bias", (64, 1, 1)),
        ),
        "output": (128, 64, 256, 256),
    },
    {
        "id": "l2-85-conv2d-groupnorm-scale-pool-clamp",
        "family": "conv-fusion",
        "specification": (
            "For fixed FP16 x[128,8,128,128], apply cloned Conv2d(8,64,3), GroupNorm(16), "
            "channel scale, MaxPool2d(4), and clamp to [0,1]."
        ),
        "arguments": (("x", (128, 8, 128, 128), "fp16", "uniform finite image values"),),
        "state": (
            ("conv_weight", (64, 8, 3, 3)),
            ("conv_bias", (64,)),
            ("groupnorm_weight", (64,)),
            ("groupnorm_bias", (64,)),
            ("scale", (64, 1, 1)),
        ),
        "output": (128, 64, 31, 31),
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_kernelbench_root(root: Path) -> Path:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != KERNELBENCH_REVISION:
        raise SystemExit(f"KernelBench revision mismatch: {revision}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise SystemExit("KernelBench checkout must be clean before freezing source hashes")
    return root.resolve()


def _generator(task_index: int) -> AnytimeGeneratorContract:
    seed = 100_000 + task_index * 100
    return AnytimeGeneratorContract(
        dev_cases=(
            AnytimeGeneratorCase(
                id="dev-uniform",
                seed=seed + 1,
                recipe="PCG64 CPU tensors; finite uniform values in [-1,1]; boolean p=0.5",
            ),
            AnytimeGeneratorCase(
                id="dev-structured",
                seed=seed + 2,
                recipe="PCG64 CPU tensors; alternating signs and repeated rows; boolean stripes",
            ),
        ),
        sealed_cases=(
            AnytimeGeneratorCase(
                id="sealed-boundary",
                seed=seed + 11,
                recipe="PCG64 CPU tensors; shape tails and FP16 boundary magnitudes",
            ),
            AnytimeGeneratorCase(
                id="sealed-cancellation",
                seed=seed + 12,
                recipe="PCG64 CPU tensors; paired signs and cancellation-sensitive values",
            ),
            AnytimeGeneratorCase(
                id="sealed-sparse",
                seed=seed + 13,
                recipe="PCG64 CPU tensors; deterministic zeros, sparse masks, and repeated values",
            ),
            AnytimeGeneratorCase(
                id="sealed-dynamic-range",
                seed=seed + 14,
                recipe="PCG64 CPU tensors; finite mixed small/large values within FP16 range",
            ),
        ),
        timing_case=AnytimeGeneratorCase(
            id="timing-selection",
            seed=seed + 21,
            recipe="PCG64 CPU tensors; finite uniform values in [-0.5,0.5]; boolean p=0.5",
        ),
    )


def _workload(
    definition: dict[str, Any],
    source: tuple[str, int, int, str, str],
    task_index: int,
) -> AnytimeWorkloadPack:
    task_id, level, problem_id, source_path, source_sha256 = source
    if task_id != definition["id"]:
        raise SystemExit("task definitions and frozen source inputs are out of order")
    state = definition["state"]
    inputs = tuple(
        AnytimeTensorInput(
            name=name,
            shape=shape,
            dtype=dtype,
            role="argument",
            distribution=distribution,
        )
        for name, shape, dtype, distribution in definition["arguments"]
    ) + tuple(
        AnytimeTensorInput(
            name=name,
            shape=shape,
            dtype="fp16",
            role="state",
            distribution="controller-cloned frozen state snapshot",
        )
        for name, shape in state
    )
    state_slot = "frozen-state-v1" if state else None
    return AnytimeWorkloadPack(
        id=task_id,
        family=definition["family"],
        specification=definition["specification"],
        lineage=AnytimeKernelBenchLineage(
            level=level,
            problem_id=problem_id,
            source_path=source_path,
            source_sha256=source_sha256,
        ),
        inputs=inputs,
        output_shape=definition["output"],
        generator=_generator(task_index),
        state_transfer=AnytimeStateTransferContract(
            parameterized=bool(state),
            initialization_seed=(200_000 + task_index if state else None),
            mode=("snapshot-and-clone-state-dict" if state else "stateless"),
            state_dtype=("fp16" if state else None),
            reference_state_slot=state_slot,
            candidate_state_slot=state_slot,
        ),
        tolerance=AnytimeToleranceContract(atol=0.01, rtol=0.01),
        numerical_adversaries=(
            AnytimeNumericalAdversary(
                id="boundary-shape-and-range",
                sealed_case_id="sealed-boundary",
                purpose="Exercise non-power-of-two tails and safe FP16 range boundaries.",
            ),
            AnytimeNumericalAdversary(
                id="cancellation-and-order",
                sealed_case_id="sealed-cancellation",
                purpose="Detect incorrect accumulation precision or reassociation sensitivity.",
            ),
            AnytimeNumericalAdversary(
                id="sparse-and-repeated",
                sealed_case_id="sealed-sparse",
                purpose="Detect mask, zero, broadcast, and repeated-value mistakes.",
            ),
            AnytimeNumericalAdversary(
                id="mixed-dynamic-range",
                sealed_case_id="sealed-dynamic-range",
                purpose="Detect overflow-prone or unstable reductions while inputs remain finite.",
            ),
        ),
        timing_input=AnytimeTimingInputContract(
            generator_case_id="timing-selection",
            state_slot=state_slot,
        ),
    )


def build_manifest(kernelbench_root: Path) -> AnytimeWorkloadInputManifest:
    root = _checked_kernelbench_root(kernelbench_root)
    for _, _, _, relative_path, expected in WORKLOAD_SOURCE_INPUTS:
        actual = _sha256(root / relative_path)
        if actual != expected:
            raise SystemExit(f"KernelBench source hash mismatch for {relative_path}: {actual}")

    cards = tuple(
        AnytimeTargetCardInput(
            target_id=target_id,
            backend=backend,
            source_path=f"benchmarks/r1-a100/targets/{backend}.md",
            source_sha256=_sha256(REPOSITORY_ROOT / f"benchmarks/r1-a100/targets/{backend}.md"),
        )
        for target_id, backend in zip(
            TARGET_IDS,
            ("triton", "tilelang", "cute"),
            strict=True,
        )
    )
    isolation = build_anytime_process_isolation_contract(
        max_wall_seconds=DEFAULT_CANDIDATE_MAX_WALL_SECONDS,
        max_memory_bytes=DEFAULT_CANDIDATE_MAX_MEMORY_BYTES,
    )
    environment_hashes = {
        field: _sha256(REPOSITORY_ROOT / relative_path)
        for field, relative_path in ENVIRONMENT_ASSET_PATHS.items()
    }
    return AnytimeWorkloadInputManifest(
        workloads=tuple(
            _workload(definition, source, index)
            for index, (definition, source) in enumerate(
                zip(TASKS, WORKLOAD_SOURCE_INPUTS, strict=True),
                start=1,
            )
        ),
        target_cards=cards,
        experts=tuple(
            AnytimeExpertSourceInput(
                task_id=task_id,
                target_id=target_id,
                source_path=(f"benchmarks/anytime-dsl-a100/experts/{task_id}/{target_id}.py"),
            )
            for task_id, *_ in WORKLOAD_SOURCE_INPUTS
            for target_id in TARGET_IDS
        ),
        baselines=tuple(
            AnytimeBaselineSourceInput(
                task_id=task_id,
                variant=variant,
                source_path=(f"benchmarks/anytime-dsl-a100/baselines/{task_id}/{variant}.py"),
            )
            for task_id, *_ in WORKLOAD_SOURCE_INPUTS
            for variant in BASELINE_VARIANTS
        ),
        environment=AnytimeEnvironmentContract(
            worker_bootstrap_sha256=environment_hashes["worker_bootstrap_sha256"],
            worker_update_sha256=environment_hashes["worker_update_sha256"],
            isolation_contract_sha256=isolation.sha256,
            lock_sha256=environment_hashes["lock_sha256"],
            wheelhouse_archive_sha256=EXPECTED_WHEELHOUSE_ARCHIVE_SHA256,
        ),
    )


def _render(manifest: AnytimeWorkloadInputManifest) -> bytes:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return f"{payload}\n".encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernelbench-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    manifest = build_manifest(arguments.kernelbench_root)
    rendered = _render(manifest)
    raw_sha256 = hashlib.sha256(rendered).hexdigest()
    if arguments.check:
        if not arguments.output.is_file() or arguments.output.read_bytes() != rendered:
            raise SystemExit("frozen anytime workload manifest is stale")
        if PINNED_INPUT_MANIFEST_SHA256 != raw_sha256:
            raise SystemExit("PINNED_INPUT_MANIFEST_SHA256 is stale")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(rendered)
    print(f"raw_sha256={raw_sha256}")
    print(f"manifest_sha256={manifest.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
