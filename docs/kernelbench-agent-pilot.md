# KernelBench Agent DSL-Affinity Pilot

This is the minimal exploratory harness for asking whether a fixed model benefits from selecting
Triton, TileLang, or CuTe per KernelBench workload. It reuses the existing KernelBench reference
loader, correctness check, timing primitive, and SSH worker. It does not add a router, retrieval
system, reference collector, or new artifact schema.

The primary comparison is best-of-targets versus best-fixed-target at the same observed search
budget. Results are descriptive single-replicate evidence for deciding whether a larger study is
worthwhile; they are not a statistical or target-purity claim.

## Study Configs

All affinity studies use `fp16`, three targets, `xhigh` reasoning, a 65536-token output cap, a
1200-second provider timeout, and a 900-second evaluator timeout.

| Config | Models | Workloads | Turns | Maximum requests | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `kernelbench-agent-affinity-qualification.yaml` | 1 | 4 | 2 | 24 | Cheap end-to-end and target-toolchain check |
| `kernelbench-agent-affinity-deepseek.yaml` | 1 | 12 | 4 | 144 | Primary DeepSeek pilot |
| `kernelbench-agent-affinity-full.yaml` | 2 | 12 | 4 | 288 | DeepSeek and GPT comparison |
| `kernelbench-agent-affinity-stress.yaml` | 1 | 2 | 4 | 24 | Optional convolution stress diagnostic |

The 12-workload matrix intentionally samples different programming styles:

| Stratum | KernelBench tasks | Sampling hypothesis |
| --- | --- | --- |
| `triton-affinity` | L1/P36 RMSNorm; L1/P47 middle-axis Sum; L1/P41 dilated MaxPool1D | Blocked indexing and reductions are accessible in Triton |
| `tilelang-affinity` | L1/P3 Batched Matmul; L2/P76 GEMM+Add+ReLU; L2/P99 Matmul+GELU+Softmax | Tiled tensor-core pipelines and fused epilogues suit TileLang |
| `cute-affinity` | L1/P8 irregular Matmul; L1/P16 transposed-A Matmul; L1/P17 transposed-B Matmul | Explicit layout, copy, and MMA control can benefit CuTe |
| `control` | L1/P1 square Matmul; L1/P5 matrix-scalar; L1/P7 small-K Matmul | No target winner is preregistered |

These strata are workload-selection hypotheses, not expected-result labels. The qualification subset
uses L1/P36, L1/P3, L1/P17, and L1/P1. The optional stress subset uses L1/P82 Depthwise Conv2D and
L2/P1 Conv2D+ReLU+BiasAdd.

The older `kernelbench-agent-smoke.yaml` and `kernelbench-agent-pilot.yaml` remain loadable, but the
affinity configs above define the current validation direction.

## Harness Behavior

Each target receives the complete existing target card, including its runnable scaffold, in the
initial prompt. The next request contains only the initial task/target prompt, latest assistant
response, concise evaluator feedback, and the best correct implementation when the latest attempt
regresses. Thinking-mode Chat Completions retain the returned `reasoning_content` in that bounded
history.

The Agent worker opts into stricter target checks than the shared naive evaluator. Obvious PyTorch
computation/wrapper fallback and the explicit raw-CUDA escape markers `load_inline`,
`cpp_extension`, and `__global__` are rejected when static checks are enabled. Compilation,
execution, and KernelBench correctness remain the final runnable checks. The shared naive path keeps
its existing defaults.

DeepSeek Chat Completions are streamed. Live progress reports cumulative reasoning/output character
counts without printing reasoning text. A response ending at `finish_reason=length` is recorded as
`output_truncated`, and the next configured turn becomes a code-only retry. Available usage is kept
for successful, truncated, interrupted, and provider-error requests.

The normal POSIX CLI collector installs a main-thread wall-clock deadline around each native
provider request. The transport timeout remains the fallback in environments where a signal
deadline cannot be installed.

## Run In Stages

Start with qualification. Replace the SSH host and remote paths with the current worker values:

```bash
uv run abstrak-kernelbench agent-collect \
  --study configs/studies/kernelbench-agent-affinity-qualification.yaml \
  --kernelbench-root /home/cambricon/KernelBench \
  --ssh-host a100-r1 \
  --worker-root /srv/AbstraK \
  --worker-kernelbench-root /srv/KernelBench \
  --run-id affinity-qualification-001 \
  --live
```

Then analyze and plot the returned `run_directory` independently:

```bash
uv run abstrak-kernelbench agent-analyze \
  --run artifacts/kernelbench-agent/affinity-qualification-001

uv run abstrak-kernelbench agent-plot \
  --run artifacts/kernelbench-agent/affinity-qualification-001
```

For the primary DeepSeek matrix, change only the study and run ID:

```bash
uv run abstrak-kernelbench agent-collect \
  --study configs/studies/kernelbench-agent-affinity-deepseek.yaml \
  --kernelbench-root /home/cambricon/KernelBench \
  --ssh-host a100-r1 \
  --worker-root /srv/AbstraK \
  --worker-kernelbench-root /srv/KernelBench \
  --run-id affinity-deepseek-001 \
  --live
```

The same functions are available as one command when staged diagnosis is not needed:

```bash
uv run abstrak-kernelbench agent-pipeline \
  --study configs/studies/kernelbench-agent-affinity-deepseek.yaml \
  --kernelbench-root /home/cambricon/KernelBench \
  --ssh-host a100-r1 \
  --worker-root /srv/AbstraK \
  --worker-kernelbench-root /srv/KernelBench \
  --run-id affinity-deepseek-001 \
  --live
```

Credentials come from `--auth`/`ABSTRAK_AUTH` or the model-specific environment names in the study
YAML. For the two-model full study, the auth environment mapping uses:

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

One already-written candidate can still be diagnosed directly with `agent-eval`:

```bash
uv run abstrak-kernelbench agent-eval \
  --candidate /path/to/candidate.py \
  --task level1:36 \
  --target triton \
  --ssh-host a100-r1 \
  --worker-root /srv/AbstraK \
  --worker-kernelbench-root /srv/KernelBench
```

## Token Accounting

The token curve uses actual provider usage rather than varying `max_output_tokens`:

```text
attempt_tokens = input_tokens + output_tokens
cumulative_tokens(k) = sum(attempt_tokens[1..k])
```

Reasoning tokens are already part of output tokens; cached input tokens are already part of input
tokens. Neither is added again. A point is exact only when both input and output usage are known.
The exact token curve stops at the first unknown request, while later attempts remain visible on the
iteration curve. Models have separate token grids because their token counts are not treated as
equivalent compute units.

At total budget `B`, analysis reports:

- `best fixed`: the best one target in hindsight, receiving `B` per workload;
- `free best-of`: each target receives `B`, an opportunity upper bound rather than a deployable
  policy;
- `equal split`: each target receives `floor(B / 3)`, with the remainder left unspent.

Generation speedup and deployable fallback utility remain separate. A correct but slower kernel is
shown below `1.0x` in the profile, while deployment utility falls back to the KernelBench reference
at `1.0x`. Correctness coverage is plotted beside utility.

## Optional Measured References

Existing manually measured DSL implementations can be overlaid during analysis:

```csv
task_ref,target,speedup,label
level1-problem36,triton,2.14,expert-triton-v1
level1-problem36,tilelang,2.31,expert-tilelang-v1
level1-problem36,cute,2.08,expert-cute-v1
```

```bash
uv run abstrak-kernelbench agent-analyze \
  --run artifacts/kernelbench-agent/affinity-deepseek-001 \
  --reference-file references/a100-dsl-references.csv
```

Rows may be partial. References never block collection or core aggregation, and the harness does not
collect or qualify them. The input file SHA is recorded in derived metrics. A reference winner is
reported only when all three target rows exist for that workload. `agent-pipeline` accepts the same
optional `--reference-file` argument.

## Artifacts And Figures

Collection writes raw artifacts only:

```text
artifacts/kernelbench-agent/<run-id>/raw/
  run.json
  attempts.jsonl
  candidates/<trajectory>/iteration-XXX.py
  responses/<trajectory>/iteration-XXX.json
  worker-logs/<trajectory>/iteration-XXX.log
  worker-logs/<trajectory>/iteration-XXX.result.json
```

`agent-analyze` writes `analysis/metrics.json` and `analysis/metrics.csv`, including iteration rows,
exact token rows, tie-aware winners, strategy aggregates, per-target health counts, and optional
reference rows. `agent-plot` writes:

- `01_anytime_performance_profiles.{png,pdf}`: best correct speedup by Agent turn;
- `01b_token_performance_profiles.{png,pdf}`: best correct speedup by cumulative tokens, with exact
  prefix endpoints marked when later budgets are censored;
- `02_winner_map_and_oracle_gain.{png,pdf}`: tie-aware workload winners and final-turn free-oracle
  gain over best fixed;
- `03_token_budget_portfolio.{png,pdf}`: best-fixed, free-best-of, and equal-split utility with a
  separate correctness-coverage panel.

Runs without exact token usage continue to produce the two iteration-based figure bases. Optional
measured references appear as hollow diamonds in both profile figures.

Repository tests use scripted provider/worker responses and synthetic metrics only. They make no
provider request, SSH connection, GPU evaluation, or generated-kernel experiment.
