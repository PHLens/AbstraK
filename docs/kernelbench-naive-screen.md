# KernelBench Naive Screen

This screen is the earliest, deliberately weak baseline for AbstraK. It asks
whether two model profiles show different out-of-the-box capability across
three KernelBench DSL backends before memory, repair loops, workflow logic,
profiling feedback, or target switching are introduced.

## Frozen Behavior

- Models: `deepseek-v4-flash` and `deepseek-v4-pro` from the user config.
- Targets: KernelBench `triton`, `tilelang`, and `cute` (CuTe DSL).
- Prompt: the pinned checkout's `zero_shot` components, without examples,
  hardware information, a system message, or extra AbstraK guidance.
- Generation: one request, one candidate, temperature zero, no retry or repair.
- Precision: FP16 so all three KernelBench backends share one precision.
- Raw outputs, extraction failures, provider usage, and returned model IDs are
  retained in private ignored artifacts.
- Generation and evaluation are separate sealed bundles. Evaluation verifies
  the candidate checksums before execution and records its device, trial
  settings, Python, Torch/CUDA, and available DSL package versions.

The screen is descriptive with one replicate. A negative result does not prove
model/target equivalence and cannot by itself reject the full research question.
It can expose floor effects, ceiling effects, backend unavailability, or a weak
choice of two same-family model aliases.

## Controller Setup

Clone and pin KernelBench outside this repository:

```bash
git clone https://github.com/ScalingIntelligence/KernelBench.git /path/to/KernelBench
git -C /path/to/KernelBench checkout 423217d9fda91e0c2d67e4a43bf62f96f6d104f1
export KERNELBENCH_ROOT=/path/to/KernelBench
```

Validate the smoke matrix without credentials or network calls:

```bash
uv run abstrak-kernelbench validate \
  --study configs/studies/kernelbench-naive-smoke.yaml
```

Generate six candidates, one for each model/target cell:

```bash
uv run abstrak-kernelbench generate \
  --study configs/studies/kernelbench-naive-smoke.yaml \
  --expected-requests 6 \
  --live
```

The 24-cell screen uses `kernelbench-naive-screen.yaml`. Generation is
billable, so the CLI requires the explicit `--live` acknowledgement.

## GPU Handoff

The controller and GPU worker both use Python 3.10, matching the pinned
KernelBench package constraint. The GPU extra pins stable PyTorch 2.13.0 from
the CUDA 12.6 wheel index, TileLang, and CuTe DSL; PyTorch supplies its matching
Triton version. The bootstrap stages the persistent wheelhouse archive into
container-local storage and rebuilds `/tmp/abstrak-gpu-venv` offline. Source,
the wheel archive, and run artifacts remain on the persistent volume. On the
A100 worker, update the persistent checkout, build, and validate:

```bash
scripts/update-worker.sh
scripts/bootstrap-a100.sh
source scripts/activate-a100.sh

for target in triton tilelang cute; do
  python scripts/smoke-kernelbench-backend.py \
    --kernelbench-root "$KERNELBENCH_ROOT" \
    --target "$target"
done
```

After a container refresh, rerun the update, bootstrap, and activation commands.
The update uses the public HTTPS remote through the persistent Git helper bundle;
the offline environment rebuild uses the persistent wheel archive and does not
contact package indexes.

Then run cells serially:

```bash
abstrak-kernelbench evaluate \
  --run artifacts/kernelbench-naive/kernelbench-naive-smoke/<run-id> \
  --execute-generated-code \
  --device cuda:0

abstrak-kernelbench summarize \
  --run artifacts/kernelbench-naive/kernelbench-naive-smoke/<run-id>
```

Each candidate runs in a separate subprocess with a timeout. Credentials are
removed from the worker environment. This is process isolation, not yet the
final hostile-code sandbox; formal experiments still require the worker
isolation contract from the pilot plan.

## Metrics

The first summary reports, per model and target:

- compile rate and correctness rate;
- `fast_1` and `fast_2` relative to KernelBench's PyTorch reference;
- performance coverage, so speed is not reported without correctness;
- geometric-mean and median performance ratio among correct candidates;
- failure status counts and raw per-cell artifacts.

These are screening metrics. Production qualification against `B*`, repeated
runs, multi-shape robustness, equivalence margins, and Agent-aware routing are
outside this naive baseline.
