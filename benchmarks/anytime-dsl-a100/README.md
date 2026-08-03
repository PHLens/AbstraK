# Anytime DSL A100 Inputs

`manifests/inputs.json` is the content-pinned, offline M6 input manifest for the twelve-workload
anytime study. It freezes workload contracts, deterministic dev/sealed/timing recipes, KernelBench
source hashes, target cards, environment expectations, and the complete expert/baseline source-slot
matrix. It does not contain measured performance or claim that a kernel has run.

Regenerate and check it from the clean pinned KernelBench checkout without importing Torch or a DSL:

```bash
uv run python scripts/freeze_anytime_workload_inputs.py \
  --kernelbench-root /path/to/KernelBench
uv run python scripts/freeze_anytime_workload_inputs.py \
  --kernelbench-root /path/to/KernelBench \
  --check
```

The 36 paths under `experts/` and 36 paths under `baselines/` are reserved source slots. They are
intentionally absent and marked `pending_live_materialization` in the offline manifest. M9 must
materialize and hash the reviewed sources, validate correctness, target use, applicability, timing,
and environment evidence, then write an explicitly pinned successor manifest. Until then the floor
validator returns `invalid_floor`; it never treats these reservations as implementations.

The three target-card inputs reuse the independently content-addressed R1 cards. Each has exactly one
unrelated VectorAdd example, so the study does not introduce task-specific example leakage or unequal
few-shot counts across targets.
