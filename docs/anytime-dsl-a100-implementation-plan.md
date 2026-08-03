# Anytime DSL A100 Infrastructure And Experiment Plan

## Status And Execution Boundary

- Plan status: accepted for infrastructure implementation.
- Repository baseline: `b876719c76d992415874215930a10521d9655d0c`.
- Current execution boundary: infrastructure-only. Do not issue provider requests, connect to a GPU
  worker, execute generated code, or start an Agent trajectory while Milestones M0-M8 are being
  implemented.
- Existing `kernelbench-naive-study.v1`, `abstrak-canary-study.v1`,
  `abstrak-matrix-study-spec.v1`, R1 artifacts, and TileLang capability-gate artifacts are frozen.
  New behavior must use new schema identifiers and an independent artifact root.

## Decision

The experiment will extend AbstraK rather than create a second `TileAgent` controller. AbstraK's
provider boundary, A100 worker, dev/sealed separation, immutable trajectory ledger, deterministic
schedule, and timing machinery remain the implementation substrate. The new study owns only its
versioned contracts, provider-native request policies, fixed-call anytime semantics, checkpoint
artifacts, task/target assets, analysis, and figures.

The primary scientific question is whether the reachable end-to-end performance of an
`Agent + target card + DSL/compiler stack` changes materially with workload and optimization budget.
The experiment does not claim to measure a DSL's theoretical ceiling or isolate language syntax from
documentation, compiler maturity, primitives, and model familiarity.

## Frozen Study Shape

The intended full study contains two independently analyzed cohorts:

| Cohort | Agent | Workloads | Targets | Replicates | Calls | Trajectories | Call ceiling |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary | `deepseek-v4-flash` | 12 | 3 | 4 | 12 | 144 | 1,728 |
| Robustness | `gpt-5.6-luna` | 6 | 3 | 3 | 12 | 54 | 648 |
| Total | separate analyses | - | - | - | - | 198 | 2,376 |

`2,376` is the scientific model-call ceiling, not the trajectory count. Infrastructure attempts are
stored separately and have their own operational ceiling. Each provider uses its native protocol and
native `xhigh` reasoning setting; the two providers' reasoning effort and token counts are not treated
as equivalent compute.

The full study is conditional. A non-scoring four-workload shakeout must pass before either formal
cohort is authorized. After the shared six-workload formal core, the remaining six primary workloads
run only if the preregistered core rule permits expansion.

## Architecture

```text
src/abstrak/anytime/                    new v1 anytime contracts and pure infrastructure
src/abstrak/providers/                  versioned native Chat/Responses transport support
benchmarks/anytime-dsl-a100/            frozen tasks, cards, experts and study manifests
artifacts/anytime-dsl-a100/             ignored immutable run artifacts
docs/anytime-dsl-a100-*.md              protocol, implementation and eventual result record
```

The package boundary is deliberate:

- `abstrak.canary` remains compatible with every existing v1 artifact and command.
- `abstrak.anytime` may call stable canary primitives but does not widen a frozen v1 model in place.
- Provider v2 code normalizes different wire protocols behind one no-tools, one-request logical
  contract; it does not force DeepSeek and GPT onto the same HTTP endpoint shape.
- Immutable artifacts are the fact source. SQLite, DuckDB, CSV, or Parquet may be generated only as
  disposable analysis indexes derived from verified artifacts.

## Scientific And Engineering Invariants

1. One iteration means one submitted provider request. Parse, static-check, compile, correctness,
   timeout, and fallback failures consume the iteration; infrastructure failures do not silently do
   so.
2. Every scientific trajectory executes exactly 12 calls unless a declared resource cap or
   infrastructure terminal state prevents continuation. There is no performance-triggered early
   stop.
3. Turn context is reconstructed deterministically from the frozen base prompt, current incumbent,
   previous candidate, and previous bounded feedback. Full provider conversation state,
   `previous_response_id`, automatic compaction, tools, retrieval, and cross-trajectory memory are
   forbidden.
4. Checkpoints `1, 4, 8, 12` identify the incumbent after that many consumed scientific calls. They
   are immutable snapshots, not separately prompted low-budget policies.
5. Development feedback never includes sealed cases, sealed results, trusted expert source, other
   targets' results, or profiler output.
6. Candidate and reference/qualifier code execute in different processes. Candidate code cannot read
   task generators, sealed seeds, reference source, expert source, controller artifacts, provider
   credentials, or the host repository.
7. Target use is fail-closed. Qualification requires target-specific static validation and runtime or
   lowered-code evidence that the core operation launched through the declared DSL stack.
8. Every retained `task x target` cell has a stable trusted expert path before the first scoring Agent
   request. Every task has a common strong baseline envelope `B*`.
9. PyTorch eager is retained as a plot reference and deployable fallback, but formal qualification and
   useful-performance claims are measured against `B*`.
10. Timing selection and formal measurement use different processes and samples. The exact search and
    checkpoint timing protocols are part of the study hash.
11. Model, prompt, task, target, environment, budget, context policy, timing, thresholds, schedule, and
    analysis versions are hash-bound before a scoring request.
12. No failed, timed-out, incorrect, unstable, unsupported, or infrastructure-censored cell is deleted
    from the denominator after unblinding.

## Milestones

### M0: Implementation Plan And Compatibility Fence

Deliverables:

- This implementation plan, including the infra-only execution boundary and formal stage gates.
- A documented new namespace, schema-version policy, artifact root, and commit sequence.

Verification:

- Existing worktree is clean before the plan commit.
- Markdown and `git diff --check` pass.
- No existing source contract or artifact is modified.

Commit: `docs: add anytime DSL A100 implementation plan`

### M1: Versioned Study, Loop, Budget And Checkpoint Contracts

Implement strict frozen models under `abstrak.anytime` for:

- agent identity and native protocol selection;
- per-agent generation settings, including required `xhigh` and optional sampling parameters;
- fixed-call loop/context policy;
- multi-axis resource budget and infrastructure-attempt policy;
- checkpoint identities and resource snapshots;
- per-cohort task/target/replicate axes;
- a deterministic, balanced schedule and scientific/operational request ceilings.

The models must express 144 primary and 54 robustness trajectories without pretending calls are
trajectories. Unknown fields, duplicate axes, invalid checkpoints, cohort collisions, unbounded
retries, and inconsistent request ceilings fail closed.

Verification:

- Pure unit tests cover the exact `198 / 2,376` cardinalities and deterministic hashes.
- Existing v1 contract and schedule tests remain byte-for-byte compatible.
- No provider, worker, evaluator, or Agent loop is invoked.

Commit: `feat: add anytime study and budget contracts`

### M2: Provider-Native Chat And Responses Infrastructure

Add a versioned provider client boundary that supports:

- DeepSeek through native Chat Completions;
- GPT through native Responses;
- exactly one non-streaming request with no tools, cache, fallback, router, or implicit retry;
- protocol-specific output-token and `xhigh` parameter rendering;
- normalized text, request/response ID, returned model, finish/status, input/cached/output/reasoning
  tokens, elapsed time, sanitized request, and raw SDK response;
- offline conformance checks that fail when `xhigh` is omitted or changed while rendering the
  sanitized wire request. Real endpoint acceptance remains pending until M9.

Temperature and top-p are not inserted by a common runtime normalizer. Omitted values remain omitted.
Responses requests do not use `previous_response_id` and request `store=false` where supported.

Verification:

- Scripted transports exercise both protocols and malformed response shapes.
- Tests prove one physical transport call, no unsupported-parameter dropping, no secret leakage, and
  correct token normalization.
- Tests are offline; local auth files are not read.

Commit: `feat: add native provider protocols for anytime studies`

### M3: Deterministic Context, Checkpoint And Resource Ledger

Implement pure turn-context construction and append-only checkpoint records without wiring them to a
live loop. The ledger records:

- consumed scientific call index and infrastructure-attempt index;
- base prompt, incumbent, previous candidate and bounded previous feedback hashes;
- candidate and incumbent source hashes;
- provider usage and latency;
- compile/correctness/timing status and elapsed resources;
- cumulative calls, tokens, compile/eval counts, GPU seconds and wall time;
- checkpoint snapshot identity at calls `1, 4, 8, 12`.

The context builder has one canonical ordering and truncation rule. A verifier reconstructs every turn
request from earlier ledger records instead of checking only the initial-message prefix.

Verification:

- Synthetic histories cover success, parse failure, duplicate source, compile failure, wrong result,
  timeout and incumbent replacement.
- Tampered history, feedback, resource totals, or checkpoint hashes fail verification.
- No completion client or candidate evaluator is called.

Commit: `feat: add deterministic anytime checkpoint ledger`

### M4: Crash-Safe Attempt Artifacts And Resume Index

Create new anytime attempt artifacts with:

- `<attempt>.incomplete` staging followed by atomic promotion;
- immutable terminal success, scientific failure, infrastructure failure, and controller-failure
  tombstones;
- explicit `request_submitted` and `possibly_charged` state;
- one bounded infrastructure retry stored as a separate attempt, never an overwrite;
- incremental phase index for efficient resume, followed by a full checksum audit at phase close;
- derived-analysis index generation that cannot mutate source artifacts.

Verification:

- Fault-injection tests cover interruption before request, during response persistence, during worker
  evaluation, during checkpoint persistence, during seal, and during atomic promotion.
- Resume never duplicates an ambiguous provider request and never requires manually deleting a final
  directory.
- Existing `TrajectoryStore` and v1 resume tests remain unchanged.

Commit: `feat: add crash-safe anytime artifact staging`

### M5: Candidate Isolation And Target-Use Qualification

Split reference/qualifier and candidate execution into separate processes with constrained IPC. The
candidate sandbox receives only candidate source, public task ABI, runtime libraries, input tensors,
and an output channel. It cannot read the repository or any private benchmark asset.

Replace the legacy cross-DSL denylist with target-specific default-deny validation and contracts for
runtime or lowered-code launch evidence for Triton, TileLang, and CuTe DSL. Add adversarial controls
for frame inspection, filesystem reads, dynamic imports/lookups, framework fallback, dummy DSL
signatures, input mutation, nonfinite output, hangs, OOM, and forged timing.

Verification:

- Offline sandbox/IPC and validator fixtures reject every hostile control.
- Scripted attestations prove that missing, malformed, or wrong-target launch evidence fails closed.
- Real trusted target launch evidence remains pending until the M9 trusted GPU preflight.
- No model request, SSH worker, GPU code, or model-generated code is used.

Commit: `feat: isolate anytime candidates and verify target use`

### M6: Twelve Workload Packs, Per-Target Experts And B-Star

Build the twelve frozen FP16 KernelBench-derived task packs. Each pack owns deterministic dev and
sealed generators, initialization/state transfer, tolerances, numerical adversaries, timing input,
and source-lineage metadata. Parameterized Level-2 modules must bind identical reference and candidate
state rather than relying on sequential random construction.

Add the 36 trusted `task x target` expert source inputs, target cards with balanced unrelated examples,
common eager / Inductor / vendor baseline source inputs, and environment contracts that bind Python,
Torch, CUDA, driver, Triton, TileLang, CuTe/CUTLASS DSL, CUDA bindings, KernelBench revision, worker
revision, isolation mode, and lock/archive digests.

Verification:

- Every source, task, target, card, baseline, expert and environment input is hash-bound and passes
  offline schema, cross-reference, static and leakage checks.
- Fake floor evidence proves that incomplete, unstable or mismatched `task x target` results block the
  study with `invalid_floor`.
- Real expert correctness/timing, launch verification, environment observation and `B*` construction
  remain pending until the M9 trusted GPU preflight.
- No model request, SSH worker, GPU code, or model-generated code is used.

Commit sequence may be split by workload family and ends with:
`feat: complete anytime workload and floor inputs`

### M7: Generic Anytime Analysis And Figures

Implement artifact-only reconstruction for:

- qualification/eligible/compile/correctness rates by turn and checkpoint;
- per-workload target winner sets and ties;
- eager-reference speedup and `B*`-relative qualification;
- fixed-target versus per-workload hindsight oracle gain;
- iteration-matched and common-wall-clock matched comparisons;
- missingness, infrastructure censoring, replicate disagreement and target-floor outcomes;
- clustered uncertainty by semantic workload family without treating timing samples as independent
  Agent replicates.

Formal checkpoint plots use independently sealed and clean-process measurements. A complete 1-12 dev
curve may be shown as exploratory unless every plotted turn is independently retimed. The analysis
produces deterministic tables plus PNG and vector SVG/PDF figures with a figure manifest.

Verification:

- Synthetic fixtures cover crossover, one-target dominance, tie, floor, unstable timing,
  infrastructure missingness, early budget exhaustion and model-dependent rankings.
- Report regeneration depends only on immutable artifacts and frozen analysis code.

Commit: `feat: add anytime DSL analysis and figures`

### M8: Offline End-To-End Dry Run And Freeze

Run an entirely scripted fake-provider/fake-worker study through schedule, context reconstruction,
candidate records, checkpoints, crash/retry paths, sealed qualification, analysis and figures. Freeze
the logical study manifests, asset inputs, thresholds, task groups, randomization, timing policies and
code hashes. Live environment and floor evidence are separate M9 outputs and are not fabricated by the
offline freeze.

Exit criteria:

- Full offline test suite, Ruff, lock check and artifact tamper tests pass.
- The repository and worker revisions intended for live preflight are clean and pinned.
- Scientific and operational ceilings are printed before authorization.
- No provider credential, live request, SSH worker, GPU code, or generated code has been used.

Commit: `test: freeze anytime DSL offline study infrastructure`

### M9: Live Conformance And Non-Scoring Shakeout — Deferred

This milestone requires a new explicit authorization because it connects to the GPU worker, performs
billable provider requests, and executes generated GPU code. First observe and hash the worker
environment, then run all trusted expert, baseline and target-launch gates to construct the real
per-target floor and `B*`. Only after that floor is valid may provider-native `xhigh` conformance and
the non-scoring 192-call shakeout run.

The shakeout passes only if each retained target produces stable correct Agent candidates in at least
two workloads from two different workload families, infrastructure censoring remains below the frozen
limit, target-use evidence is complete, and checkpoint artifacts reproduce exactly. If one uniform
target-card revision has already been consumed and the floor persists, stop with a target-stack
usability result rather than expanding the matrix.

### M10: Formal Core And Conditional Reserve — Deferred

The six workloads common to both Agents form the first formal core. The six remaining primary-only
workloads are an all-or-none reserve authorized only by the frozen core decision. Formal data never
shares caches, candidates, feedback, or conversation state across trajectories.

The full-study continuation gate requires interpretable target coverage, at least two stable target
winners across at least two workload families, positive practical fixed-versus-oracle gain under both
iteration and common-wall-clock budgets, and no result driven only by unsupported targets or timing
noise. Passing this gate justifies a later held-out routing study; it does not itself demonstrate a
deployable router.

## Timing And Winner Policy To Freeze Before M8

The current repository uses several historical timing policies, including 5/100 dev timing and a
25/200/3 clean-process formal protocol. The anytime study must choose and hash one search protocol and
one checkpoint protocol rather than silently inherit a CLI default. The final margin must exceed the
measured baseline noise. A 3% winner margin is not coherent with an accepted 5% timing CV unless the
paired uncertainty rule independently establishes the difference.

For every workload/checkpoint, report the raw replicate points and preserve all targets within the
preregistered practical-equivalence band as ties. `P(best)` is descriptive with three or four Agent
replicates and must not resample individual CUDA timing trials as if they were independent Agent runs.

## Definition Of Infrastructure Complete

Offline infrastructure is complete only after M8. In particular, a schedule that can count 2,376
requests is not sufficient. Completion requires protocol-native `xhigh` rendering, fixed-call context
semantics, checkpoint provenance, multi-axis resource accounting, crash-safe resume, candidate
isolation, target-use evidence contracts, complete per-target floor inputs, artifact-only generic
analysis, synthetic decision fixtures, and an offline end-to-end reconstruction. Real endpoint
acceptance, target launches, expert correctness/timing and `B*` remain M9 gates.

Until then, commands that could perform a provider request or execute generated code remain outside
the authorized scope of this implementation sequence.
