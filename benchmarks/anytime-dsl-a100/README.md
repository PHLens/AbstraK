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

## M8 Offline Freeze And Rehearsal

The reviewed logical freeze is split into `manifests/formal-study.json`,
`manifests/shakeout-study.json`, and `manifests/offline-freeze.json`. It pins the study axes,
balanced schedule hashes, public base-prompt policy, timing/winner/continuation thresholds, native
provider dependency conformance, M6 input digests, and the source-code inventory. The freeze is
explicitly non-live: provider readiness is not endpoint authorization, and environment/floor evidence
remain M9 blockers.

Generate the files during review, then use the pinned check before any live authorization:

```bash
uv run python scripts/freeze_anytime_offline.py
uv run python scripts/freeze_anytime_offline.py --check --require-clean
```

The optional `--rehearse DIRECTORY` flag is available only with `--check`. It writes a complete
48-trajectory fake-provider/fake-worker shakeout to a new directory, including one request-before-
dispatch crash and bounded retry, 192 scripted responses, phase journals, pending-M9 qualification
fixtures (one execution-bound artifact per successful trajectory turn), invalid-floor output, analysis
tables, and figures. Candidate source is statically inspected only; no provider credentials, network
request, SSH connection, GPU API, or candidate execution is performed. Pending qualification is kept
out of compiled/correct rates. The rehearsal is synthetic evidence and cannot authorize M9 or formal
scoring.
