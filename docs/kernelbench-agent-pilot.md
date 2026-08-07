# KernelBench Agent Pilot

This is the small exploratory harness for the DSL-target pilot. It is separate from
`src/abstrak/anytime`; the existing naive KernelBench evaluator supplies the reference loader,
static checker, correctness check, and timing primitive.

The default study is [`configs/studies/kernelbench-agent-pilot.yaml`](../configs/studies/kernelbench-agent-pilot.yaml).
It contains 2 models x 4 KernelBench tasks x 3 targets and defaults to 4 turns per trajectory:

- models: `deepseek-v4-flash` through Chat Completions and `gpt-5.6-luna` through Responses;
  both use `xhigh` reasoning; DeepSeek maps that to native `reasoning_effort=max`, enforces the
  configured output cap with `max_tokens`, streams reasoning/output, and has a 1200-second timeout;
- targets: `triton`, `tilelang`, and `cute`;
- tasks: Level 1 problem 1 Square Matmul and problem 24 LogSoftmax, plus Level 2 problem 1
  Conv2D+ReLU+BiasAdd and problem 76 GEMM+Add+ReLU;
- precision: `fp16`.

## Separate stages

Start with the DeepSeek-only 1 workload x 1 target x 2 turn smoke study if the OpenAI
credentials are not configured yet:

```bash
uv run abstrak-kernelbench agent-collect \
  --study configs/studies/kernelbench-agent-smoke.yaml \
  --kernelbench-root /home/cambricon/KernelBench \
  --ssh-host a100-r1 \
  --worker-root /workspace/volume/lipenghui/AbstraK \
  --worker-kernelbench-root /workspace/volume/lipenghui/KernelBench \
  --run-id agent-smoke-001 --live
```

After the smoke passes, `configs/studies/kernelbench-agent-deepseek-pilot.yaml` runs the same
4 workloads x 3 targets x 4 turns as the default pilot using only `deepseek-v4-flash`.

Then analyze and plot the returned `run_directory` with the commands below.

Evaluate one already-written candidate:

```bash
uv run abstrak-kernelbench agent-eval \
  --candidate /path/to/candidate.py \
  --task level1:1 --target triton \
  --ssh-host a100-r1 --worker-root /srv/AbstraK \
  --worker-kernelbench-root /srv/KernelBench
```

Collect model turns. Each parseable response is evaluated immediately by the SSH worker. The next
request contains the initial task and concise target contract, the latest assistant response and
its evaluator feedback, plus the best correct implementation when the latest attempt regresses.
Older turns are discarded. For thinking-mode Chat Completions models, the latest assistant's
returned `reasoning_content` is preserved.

DeepSeek Chat Completions are consumed as a stream. `--live` reports cumulative reasoning and
answer character counts without printing the reasoning text; the completed response is aggregated
into the same response artifact used by later turns.

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

For the default two-model study, the auth file needs these four names (replace the example
values locally):

```json
{
  "schema_version": "auth.v1",
  "environment": {
    "ABSTRAK_DEEPSEEK_API_KEY": "...",
    "ABSTRAK_DEEPSEEK_BASE_URL": "...",
    "ABSTRAK_OPENAI_API_KEY": "...",
    "ABSTRAK_OPENAI_BASE_URL": "..."
  }
}
```

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
  worker-logs/<trajectory>/iteration-XXX.result.json
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
