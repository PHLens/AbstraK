# KernelBench DSL Target Affinity Pilot: Minimal Implementation Plan

## Status And Scope

- Status: implementation draft for review.
- Scope: make the smallest additive changes to the existing exploratory
  `kernelbench-agent-study.v1` harness.
- This draft does not authorize provider, SSH, or GPU execution.
- The existing collect/analyze/plot split and one-command pipeline remain the user interface.
- Existing local streaming, timeout, truncation, and reasoning-history changes are finalized and
  committed separately before this work. `run.sh` remains untracked.

## Goal

The pilot asks one narrow question:

> For a fixed model and search budget, does selecting Triton, TileLang, or CuTe per workload produce
> a useful correctness/performance gain over using one fixed target for every workload?

The first study is descriptive. It needs enough evidence to decide whether target selection deserves
further investment; it does not need a new benchmark platform or paper-grade provenance system.

The required outputs are:

1. per-workload best-correct performance versus Agent iteration;
2. per-workload best-correct performance versus cumulative model tokens;
3. workload-level target winners, including ties and no-correct states;
4. best-fixed-target versus best-of-targets, with correctness coverage shown beside performance;
5. an optional manually measured DSL-reference overlay.

## Reuse The Existing Harness

| Existing capability | Decision |
| --- | --- |
| `KernelBenchAgentStudy` model/task/target/turn configuration | Reuse unchanged |
| Bounded conversation history, evaluator feedback, and best incumbent | Reuse unchanged |
| DeepSeek streaming and reasoning-content history | Finish the current local patch, then reuse |
| KernelBench compile, correctness, and CUDA-event timing | Reuse |
| `AgentAttemptRecord` token, latency, status, and artifact fields | Reuse; do not add a v2 attempt |
| `raw/run.json`, `attempts.jsonl`, responses, candidates, and worker logs | Reuse unchanged |
| `agent-collect`, `agent-analyze`, `agent-plot`, and `agent-pipeline` | Reuse |
| Existing iteration analyzer and figures | Extend in place |

No new run store, database, router, retrieval layer, reference collector, or schema family is added.

## What Is Removed From The Previous Draft

The following are explicitly not part of the first implementation:

- `kernelbench-agent-affinity-*.v1` study/run/attempt schemas;
- prompt/reference asset registries and per-attempt SHA chains;
- environment and timing-protocol hashes on every attempt;
- a twenty-source target-BKR registry;
- three-repeat reference qualification and a reference bundle;
- reference coverage as a precondition for Agent collection;
- a new `agent-reference-collect` command;
- a separate `agent_purity.py` policy engine;
- static proof of the complete DSL definition-to-launch data path;
- stage-to-stage automatic gates;
- online token stopping, adaptive allocation, routing, or run merging.

Qualification, references, and stress workloads are useful diagnostics. None blocks collection of the
main pilot.

## The Only Hard Boundaries

Four boundaries are necessary for the result to mean what it says:

1. A performance point must compile and pass KernelBench correctness.
2. The candidate must pass the existing target-specific implementation check, with obvious PyTorch
   compute fallback treated as an error rather than a warning.
3. A token-axis point is exact only when both input and output token counts are known. Unknown token
   usage does not invalidate iteration-axis results.
4. A best-fixed/best-of aggregate is emitted only for the coordinates actually required by that
   aggregate. An incomplete run still produces per-workload diagnostics.

There is no reference preflight, environment preflight, support-pack coverage gate, or full-matrix
collection gate.

## Workloads

The workload list remains a configuration choice, not a harness feature. The main config uses twelve
headline workloads grouped through the existing free-form `KernelBenchTask.stratum` field.

| Stratum | KernelBench workloads | Hypothesis |
| --- | --- | --- |
| `triton-affinity` | L1/P36 RMSNorm; L1/P47 middle-axis Sum; L1/P41 dilated MaxPool1D | blocked indexing and reductions are accessible in Triton |
| `tilelang-affinity` | L1/P3 Batched Matmul; L2/P76 GEMM+Add+ReLU; L2/P99 Matmul+GELU+Softmax | tiled tensor-core pipelines and fused epilogues suit TileLang |
| `cute-affinity` | L1/P8 irregular Matmul; L1/P16 transposed-A Matmul; L1/P17 transposed-B Matmul | explicit layout/copy/MMA control can benefit CuTe |
| `control` | L1/P1 square Matmul; L1/P5 matrix-scalar; L1/P7 small-K Matmul | no target winner is preregistered |

These strata are labels for grouping and interpretation, not target ground truth. The measured winner
comes only from correct generated candidates.

Convolution support-boundary tasks stay in a separate optional config so the analyzer needs no stress
exclusion rules:

- L1/P82 Depthwise Conv2D;
- L2/P1 Conv2D+ReLU+BiasAdd.

## Minimal Prompt Change

The three existing A100 target cards already contain balanced VectorAdd scaffolds:

- Triton: decorator, masked kernel, grid, and launch;
- TileLang: `@T.prim_func`, `tilelang.compile`, `out_idx`, and compiled callable;
- CuTe: DLPack bridge, `@cute.kernel`, `@cute.jit`, `.launch`, `cute.compile`, and cache.

The current runner removes those scaffolds at `## Model scaffold and launch example`. The minimum
change is to stop truncating the cards and include each complete, already hash-pinned card in the
initial prompt.

Prompt order remains:

1. KernelBench task prompt;
2. complete target card;
3. existing runnable-output contract.

No new support-pack files, prompt renderer, prompt manifest, or prompt hash fields are required. The
examples remain workload-independent and use the same VectorAdd semantic coverage for all targets.

If the qualification run still shows repeated TileLang/CuTe API-shape failures after this change,
adding matched GEMM examples is a later evidence-driven change and requires a new study config. It is
not implemented speculatively.

## Minimal Target Validity

Keep KernelBench's current static checker and backend-specific check. For Agent evaluation only,
promote these existing warnings to forbidden checks:

```text
pytorch_wrap
torch_computation_ops
```

Retain the existing strict categories for code bypass, timing manipulation, thread injection, and
lazy evaluation. Reject the three obvious raw-CUDA escape markers `load_inline`, `cpp_extension`, and
`__global__` in a Triton, TileLang, or CuTe candidate.

Do not add a general purity schema or attempt to statically prove every launch binding. Compilation,
runtime execution, and correctness already reject non-runnable implementations. Final winning source
files remain available for manual inspection before any research claim.

Static validation continues to use the current `EvaluationResult.static_errors` and
`static_warnings`; no result or attempt schema changes are needed.

## Token Accounting

The experiment does not vary `max_output_tokens` to manufacture a budget curve. Each completed Agent
request contributes one observation at its cumulative actual cost:

```text
attempt_tokens = input_tokens + output_tokens
cumulative_tokens(k) = sum(attempt_tokens[1..k])
```

`reasoning_tokens` is already included in output tokens and is not added again. Cached input tokens
are already included in input tokens and are reported only as a diagnostic subset.

The existing attempt fields are sufficient. The provider exposes one shared usage parser so success,
output truncation, interrupted stream, and provider-error paths populate the same existing token
fields when usage is available. For older artifacts or error paths with only raw response usage, the
analyzer may recover it from `response_path`.

For each trajectory:

- exact usage continues the cumulative-token curve;
- missing input or output usage marks that request `unknown`;
- the exact token curve stops at the first unknown request;
- later attempts remain visible on the iteration curve;
- missing usage is never counted as zero or estimated from text length.

Token grids are data-derived per model from the sorted union of exact cumulative request boundaries,
plus zero. This avoids a new study field or CLI option. Models are analyzed separately because their
token counts are not treated as equivalent compute units.

At budget `B`, analysis selects the latest exact completed request with cumulative cost at most `B`.
Before the first request boundary the state is exact with no candidate. A trajectory with all
configured turns and exact usage carries its final incumbent forward; a trajectory after an unknown
request is censored instead. Equal-split lookup uses this same step function directly even when
`floor(B/3)` is not one of the displayed grid points.

## Metrics

For workload `w`, target `t`, and an iteration or exact token budget `B`, let `S(w,t,B)` be the best
correct speedup observed by that point. It is `NA` before the first correct candidate.

Keep generation quality and deployable fallback separate:

```text
correct(w,t,B) = whether any correct candidate exists by B
generation_speedup(w,t,B) = S(w,t,B) or NA
deployment_utility(w,t,B) = max(1, S(w,t,B)) if correct else 1
```

Correct-but-slower kernels remain below `1.0x` in generation plots. Aggregate deployment utility may
fall back to the KernelBench reference, but correctness coverage is always plotted beside it.

At a common budget `B`:

```text
fixed_score(t,B) = geomean_w deployment_utility(w,t,B)
best_fixed(B) = max_t fixed_score(t,B)

best_of_oracle(B) = geomean_w max_t deployment_utility(w,t,B)

equal_split_best_of(B_total)
  = geomean_w max_t deployment_utility(w,t,floor(B_total / 3))
```

Interpretation:

- `best_fixed` is the best single target selected in hindsight for all workloads;
- `best_of_oracle` gives the winning target the full budget and is an opportunity upper bound, not a
  deployable router;
- `equal_split_best_of` is the optional equal-total-token exhaustive comparator;
- the remainder of `B_total / 3` is left unspent so target ordering cannot affect the result.

Winner and best-fixed ties are stored as target lists and displayed as ties. A first target may remain
in legacy scalar fields for compatibility, but figures and interpretation must not present it as a
unique winner.

The current `kernelbench-agent-metrics.v1` output is extended additively with:

```text
trajectory_usage_rows
token_curve_rows
token_winners
token_aggregates
target_health_rows
reference_rows                 # empty when no optional reference file is supplied
```

Existing iteration rows remain. `target_health_rows` reports attempts, static-check passes, compiled
count, and correct count per `(model,target)` so a validator/toolchain floor is distinguishable from
poor generated performance.

## Optional DSL Reference Overlay

Reference kernels are diagnostic and never block collection or aggregation. When measured reference
points already exist, `agent-analyze` optionally accepts one small CSV:

```text
task_ref,target,speedup,label
level1-problem36,triton,2.14,expert-triton-v1
level1-problem36,tilelang,2.31,expert-tilelang-v1
level1-problem36,cute,2.08,expert-cute-v1
```

Rules are deliberately small:

- `task_ref` and target must exist in the run;
- speedup must be finite and positive;
- `(task_ref,target)` must be unique;
- partial rows are allowed and plotted as available reference points;
- a reference winner is shown only when all three target rows exist;
- the file SHA is recorded in derived metrics so the plotted input is identifiable.

The user is responsible for measuring reference speedups with the same A100, KernelBench commit,
precision, and evaluator settings. The harness does not verify this or collect the references. These
points are called measured references, not optimal DSL or hardware limits.

This overlay answers a useful diagnostic question without changing the main claim: a low-level target
may have the fastest expert reference while a higher-level target is more reachable by the Agent at a
finite budget.

## Configs

Add configuration files only; they use `kernelbench-agent-study.v1` unchanged.

```text
configs/studies/kernelbench-agent-affinity-qualification.yaml
configs/studies/kernelbench-agent-affinity-deepseek.yaml
configs/studies/kernelbench-agent-affinity-full.yaml
configs/studies/kernelbench-agent-affinity-stress.yaml
```

Common settings:

- targets: `triton`, `tilelang`, `cute`;
- precision: FP16;
- reasoning effort: `xhigh`;
- maximum output tokens: 65536;
- provider timeout: 1200 seconds for both models;
- evaluator timeout: 900 seconds;
- correctness trials: 5;
- performance trials: 100 CUDA-event trials.

| Config | Models | Tasks | Turns | Trajectories | Maximum requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| qualification | `deepseek-v4-flash` | 4 representative | 2 | 12 | 24 |
| deepseek | `deepseek-v4-flash` | 12 headline | 4 | 36 | 144 |
| full | `deepseek-v4-flash` + `gpt-5.6-luna` | 12 headline | 4 | 72 | 288 |
| stress | chosen model(s) | 2 convolution | configured separately | separate | separate |

The qualification config uses L1/P36, L1/P3, L1/P17, and L1/P1. It is recommended for debugging but
is not checked by or required for the main config. The stress config is a separate run and never
silently enters headline aggregates.

## CLI And Artifacts

The normal workflow stays unchanged:

```text
agent-collect  --study ...
agent-analyze  --run ... [--reference-file optional.csv]
agent-plot     --run ...
agent-pipeline --study ... [--reference-file optional.csv]
```

Collection, analysis, and plotting remain independently runnable for debugging. The pipeline calls
the same three functions in sequence; it does not introduce a second execution path.

The existing artifact layout remains unchanged. Only derived metric fields and figure files grow:

```text
artifacts/kernelbench-agent/<run-id>/
  raw/run.json
  raw/attempts.jsonl
  raw/responses/...
  raw/candidates/...
  raw/worker-logs/...
  analysis/metrics.json
  analysis/metrics.csv
  figures/...
```

The optional reference CSV remains an explicit CLI input and is not copied into prompts or provider
requests.

## Figures

Keep the existing iteration figure and add only the views needed for the research question:

1. `01_anytime_performance_profiles`: existing iteration-axis per-workload curves.
2. `01b_token_performance_profiles`: the same best-correct curves on cumulative tokens; unknown
   prefixes end with an explicit censored marker.
3. `02_winner_map_and_oracle_gain`: tie-aware workload winner map and final-iteration oracle gain.
4. `03_token_budget_portfolio`: best-fixed, free best-of oracle, and equal-split curves, with
   correctness coverage in a separate panel.

Optional measured-reference points are overlaid on the two profile figures. No standalone reference
dashboard is added.

## Implementation Milestones

Each milestone ends in one focused commit. No milestone runs a real Agent experiment.

### M0: Finish The Existing Provider Patch

Files already modified in the current worktree:

- `configs/studies/kernelbench-agent-smoke.yaml`;
- `docs/kernelbench-agent-pilot.md`;
- `src/abstrak/evaluation/agent_contracts.py`;
- `src/abstrak/evaluation/agent_provider.py`;
- `src/abstrak/evaluation/agent_runner.py`;
- `tests/test_kernelbench_agent_loop.py`.

Finish streaming timeout, reasoning-history, truncation recovery, and error-artifact usage handling
without affinity-specific behavior. Leave `run.sh` untracked.

Focused verification:

```bash
uv run pytest tests/test_kernelbench_agent_loop.py tests/test_kernelbench_agent_cli.py -q
```

Suggested commit: `fix: finalize streamed agent response handling`

### M1: Make All Three Target Cards Usable

Files:

- modify `src/abstrak/evaluation/agent_runner.py`;
- modify `src/abstrak/evaluation/agent_worker.py`;
- modify `src/abstrak/evaluation/worker.py`;
- extend `tests/test_kernelbench_agent_loop.py`;
- extend `tests/test_kernelbench_agent_worker.py`.

Changes:

- include the complete existing target card instead of truncating its scaffold;
- let the Agent worker opt into PyTorch wrapper/computation errors and the three explicit raw-CUDA
  escape checks through optional shared-worker arguments;
- keep naive/shared worker callers on their current default policy;
- keep all result and artifact schemas unchanged.

Tests assert that the initial prompts contain the existing target scaffolds and that valid minimal
Triton, TileLang, and CuTe submissions reach evaluation while obvious fallback is rejected.

Suggested commit: `fix: expose runnable DSL target scaffolds`

### M2: Add The Affinity Study Configs

Files:

- add the four YAML files listed under Configs;
- extend the existing agent contract/config tests.

Tests assert only model IDs, target set, task IDs/strata, turn counts, timeouts, and matrix arithmetic.
There are no fake reference hashes or preflight assets.

Suggested commit: `config: define DSL target affinity pilot`

### M3: Add Token-Aware Analysis And Optional References

Files:

- modify `src/abstrak/evaluation/agent_provider.py` only if a shared usage parser is not completed in
  M0;
- modify `src/abstrak/evaluation/agent_runner.py` to preserve available error usage in existing
  attempt fields;
- modify `src/abstrak/evaluation/agent_analysis.py`;
- modify `src/abstrak/evaluation/cli.py` for optional `--reference-file`;
- extend `tests/test_kernelbench_agent_analysis.py` and CLI tests.

Implement exact-prefix token accounting, tie-aware winners, fallback utility plus correctness
coverage, best-fixed, free best-of, equal split, target-health summaries, and the optional four-column
reference loader.

Focused synthetic cases:

- input 100 + output 50 costs 150 even when cached=20 and reasoning=30;
- truncation usage is counted when available;
- unknown usage ends only the exact token prefix;
- correct-but-slower remains below one in generation view and equals one in deployment utility;
- all incorrect, target tie, and best-fixed tie;
- total budget 10 splits as 3/3/3 with one unspent token;
- missing and partial reference files do not block analysis.

Suggested commit: `feat: analyze target opportunity by token budget`

### M4: Add The Two Token-Aware Figure Views

Files:

- modify `src/abstrak/evaluation/agent_figures.py`;
- extend figure and CLI tests;
- update `docs/kernelbench-agent-pilot.md` after its M0 edit is committed.

Render `01b_token_performance_profiles` and `03_token_budget_portfolio`; make the existing winner map
tie-aware and add correctness coverage. Keep the existing iteration figure for debugging.

The offline acceptance fixture is deliberately small:

```text
2 workloads x 3 targets x 2 turns with scripted provider/worker
  -> existing raw artifacts
  -> iteration and token metrics
  -> all four non-empty PNG/PDF figure bases
```

Suggested commit: `feat: plot token-aware DSL target opportunity`

## Completion Criteria

The refactor is complete when:

- old v1 study files and raw runs still load;
- complete target-card examples reach the Agent prompt;
- existing attempt artifacts retain available token usage on success and failure paths;
- one synthetic run produces iteration curves, token curves, target winners, best-fixed, best-of,
  equal-split, and correctness coverage;
- absence of a reference CSV does not change collection or core analysis;
- collect/analyze/plot and pipeline use the same implementation functions;
- offline tests make no provider request, SSH connection, or GPU call.

No reference qualification, environment fingerprint, full target BKR matrix, repeated Agent samples,
or statistical claim is required to finish this implementation.

## How To Read The First Pilot

| Observation | Interpretation |
| --- | --- |
| TileLang/CuTe improve after complete target cards are exposed | the old prompt, not necessarily model capability, was the main floor |
| A target has only static failures across workloads | inspect checker/toolchain before interpreting it as a target loss |
| Different workloads have different correct winners | target diversity exists in the sampled budget |
| Free best-of beats best-fixed but equal-split does not | opportunity exists, but exhaustive multi-target search is not cost-effective |
| Equal-split also beats best-fixed | strongest initial signal for investing in a selector |
| Expert reference winner differs from Agent winner | target ceiling and finite-budget Agent reachability differ |
| One target dominates correctness and performance | no evidence yet that target selection is worth further investment |

Only after this pilot shows a stable opportunity should a later study add repeated trajectories,
formal target-purity attribution, reference qualification, hardware manuals/RAG, or a learned router.
