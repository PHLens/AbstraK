# KernelBench Agent Pilot

This is the small exploratory harness for the DSL-target pilot. It is separate from
`src/abstrak/anytime`; the existing naive KernelBench evaluator supplies the reference loader,
static checker, correctness check, and timing primitive.

The default study is [`configs/studies/kernelbench-agent-pilot.yaml`](../configs/studies/kernelbench-agent-pilot.yaml).
It contains 2 models x 4 KernelBench tasks x 3 targets and defaults to 4 turns per trajectory:

- models: `deepseek-v4-flash` through Chat Completions and `gpt-5.6-luna` through Responses;
  both use `xhigh` reasoning;
- targets: `triton`, `tilelang`, and `cute`;
- tasks: Level 1 problems 1, 3, and 40, plus Level 2 problem 76;
- precision: `fp16`.

## Separate stages

Evaluate one already-written candidate:

```bash
uv run abstrak-kernelbench agent-eval \
  --candidate /path/to/candidate.py \
  --task level1:1 --target triton \
  --ssh-host a100-r1 --worker-root /srv/AbstraK \
  --worker-kernelbench-root /srv/KernelBench
```

Collect model turns. Each parseable response is evaluated immediately by the SSH worker, and
the result is appended to the next model turn:

```bash
uv run abstrak-kernelbench agent-collect \
  --study configs/studies/kernelbench-agent-pilot.yaml \
  --kernelbench-root /srv/local/KernelBench \
  --ssh-host a100-r1 --worker-root /srv/AbstraK \
  --worker-kernelbench-root /srv/KernelBench \
  --device cuda:0 --iterations 4 --live
```

The local checkout is used to render the pinned task prompt. `--worker-root` and
`--worker-kernelbench-root` are paths as seen by the remote host. Credentials come from the
existing `--auth`/`ABSTRAK_AUTH` mechanism or the model-specific environment variables in the
study YAML.

Derive metrics and render figures independently:

```bash
uv run abstrak-kernelbench agent-analyze --run artifacts/kernelbench-agent/<run-id>
uv run abstrak-kernelbench agent-plot --run artifacts/kernelbench-agent/<run-id>
```

Or run all three stages in sequence:

```bash
uv run abstrak-kernelbench agent-pipeline \
  --study configs/studies/kernelbench-agent-pilot.yaml \
  --kernelbench-root /srv/local/KernelBench \
  --ssh-host a100-r1 --worker-root /srv/AbstraK \
  --worker-kernelbench-root /srv/KernelBench --live
```

## Artifacts and figures

Each collection run writes only raw material under:

```text
artifacts/kernelbench-agent/<run-id>/raw/
  run.json
  attempts.jsonl
  candidates/<trajectory>/iteration-XXX.py
  responses/<trajectory>/iteration-XXX.json
  worker-logs/<trajectory>/iteration-XXX.log
```

`agent-analyze` writes `analysis/metrics.json` and `analysis/metrics.csv`. `agent-plot` writes:

- `figures/01_anytime_performance_profiles.{png,pdf}`: model x workload panels, one line per
  target, iteration on the x-axis, and best correct speedup on the y-axis;
- `figures/02_winner_map_and_oracle_gain.{png,pdf}`: workload x iteration winner map plus the
  final-iteration descriptive gain of choosing a workload-specific target over the best fixed
  target.

Missing or not-yet-correct candidates are absent from the winner map. Aggregate utility uses
`1.0x` for those cells, so the oracle comparison is descriptive and single-replicate only; it is
not a statistical claim.

The implementation has been verified with fake provider/worker responses and synthetic plotting
fixtures. No provider request, SSH connection, GPU evaluation, or generated-kernel experiment is
run by the repository tests.
