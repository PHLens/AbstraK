# AbstraK

AbstraK is the controlled experimental harness for studying how an agent's
effective GPU-kernel authoring abstraction depends on model capability,
workload structure, hardware, and resource budget.

The repository starts deliberately small. The first milestone is environment
and provider conformance, not an optimization policy.

## Quick start

```bash
uv sync
uv run abstrak-doctor
uv run pytest
uv run ruff check .
```

Run the stricter probe on a GPU worker:

```bash
uv run abstrak-doctor --require-gpu
```

The local controller does not require a GPU. Compilation, qualification, and
performance measurement must run on a declared GPU worker with a frozen
software manifest.

## Repository layout

- `configs/`: versioned provider, model, hardware, target, task, and study manifests.
- `src/abstrak/`: shared controller and experiment infrastructure.
- `tests/`: unit and conformance tests that do not require hidden benchmark data.
- `artifacts/`: local run output; only its storage contract is tracked by Git.

## Experimental invariants

- The controlled track uses one orchestration loop and tool schema for every model.
- Provider retries, fallback, context compaction, and tool execution must be explicit.
- Every request, response, patch, compiler result, and budget event is recorded.
- Development feedback and sealed qualification run in separate processes.
- Model credentials, generated kernels, traces, and raw benchmark results are not committed.
- Published summaries must be reproducible from an immutable artifact bundle and manifest hash.

## Current scope

The initial dependency set contains only the API transport and structured-config
building blocks. Triton, TileLang, CuTe DSL, CUDA toolchains, and profiler
dependencies will be installed in target-specific GPU images after oracle
readiness checks, rather than being coupled to the local controller environment.
