"""Compile and evaluate one trusted KernelBench backend example."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path

EXPECTED_KERNELBENCH_COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
CANDIDATES = {
    "triton": "model_new_ex_add_triton.py",
    "tilelang": "model_new_ex_add_tilelang.py",
    "cute": "model_new_ex_add_cute.py",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernelbench-root", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-correct-trials", type=int, default=2)
    parser.add_argument("--num-perf-trials", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.kernelbench_root.expanduser().resolve()
    prompts = root / "src" / "kernelbench" / "prompts"
    reference_path = prompts / "model_ex_add.py"
    candidate_path = prompts / CANDIDATES[arguments.target]
    if not reference_path.is_file() or not candidate_path.is_file():
        raise SystemExit(f"invalid KernelBench checkout: {root}")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if commit != EXPECTED_KERNELBENCH_COMMIT:
        raise SystemExit(
            f"KernelBench commit mismatch: expected {EXPECTED_KERNELBENCH_COMMIT}, found {commit}"
        )

    sys.path.insert(0, str(root / "src"))
    import torch
    from kernelbench import eval as kernel_eval
    from kernelbench.kernel_static_checker import validate_kernel_static

    reference = reference_path.read_text(encoding="utf-8")
    candidate = candidate_path.read_text(encoding="utf-8")
    valid, errors, warnings = validate_kernel_static(
        candidate,
        backend=arguments.target,
        precision="fp16",
    )
    if not valid:
        print(
            json.dumps(
                {
                    "target": arguments.target,
                    "static_errors": [str(error) for error in errors],
                    "static_warnings": [str(warning) for warning in warnings],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    with contextlib.redirect_stdout(sys.stderr):
        result = kernel_eval.eval_kernel_against_ref(
            original_model_src=reference,
            custom_model_src=candidate,
            num_correct_trials=arguments.num_correct_trials,
            num_perf_trials=arguments.num_perf_trials,
            measure_performance=True,
            timing_method="cuda_event",
            verbose=False,
            device=torch.device(arguments.device),
            backend=arguments.target,
            precision=torch.float16,
            check_for_excessive_speedup=True,
            excessive_speedup_threshold=1000,
        )

    report = {
        "schema_version": "abstrak-kernelbench-backend-smoke.v1",
        "kernelbench_commit": commit,
        "target": arguments.target,
        "device": arguments.device,
        "compiled": bool(result and result.compiled),
        "correctness": bool(result and result.correctness),
        "runtime_ms": result.runtime if result else None,
        "reference_runtime_ms": result.ref_runtime if result else None,
        "static_warnings": [str(warning) for warning in warnings],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(
        result is None
        or not result.compiled
        or not result.correctness
        or result.runtime <= 0
        or result.ref_runtime <= 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
