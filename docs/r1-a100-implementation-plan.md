# AbstraK A100 R1 Implementation Plan

## Objective

以纵向切片尽快跑通第一条 `Flash × Triton × row-reduction` trajectory，再扩展到 12 条 shakeout 和 48 条正式 trajectories。保留现有 `kernelbench-naive` screen；新实现使用独立的 `abstrak.canary` namespace 和 `abstrak-canary-study.v1` schema。

## Module Boundary

```text
src/abstrak/canary/
  contracts.py    frozen study, task, target, job, result, event contracts
  tasks.py        AbstraK-owned static task and input-case registry
  evaluator.py    dev/sealed/oracle/baseline evaluation core
  worker.py       one JSON job to one terminal JSON result
  remote.py       local and SSH worker transports
  artifacts.py    append-only trajectory artifacts and first/final snapshots
  loop.py         fixed four-turn Agent state machine
  schedule.py     deterministic shakeout/formal cell expansion
  analysis.py     post-run fixed/oracle aggregation
  cli.py          validate, run-cell, run-study, worker, summarize

benchmarks/r1-a100/
  tasks/          four scientific tasks and two canaries
  targets/        Triton, TileLang, and CuTe cards
  oracles/        trusted target implementations, never injected into prompts
```

Pinned KernelBench remains an unmodified dependency. AbstraK reuses its backend loading, static checks, and CUDA timing helpers, while owning explicit input cases, tolerance, mutation/fallback checks, and dev/sealed separation.

## Agent Protocol

Each turn returns one complete `ModelNew` code block and one exact `CONTINUE` or `FINISH` marker. The controller evaluates the complete candidate and appends structured feedback containing only status, truncated compiler/runtime error, maximum numerical error, and dev latency.

```text
START -> MODEL_CALL -> PARSE -> DEV_EVAL
      -> FEEDBACK -> MODEL_CALL (at most four calls)
      -> FINISH/BUDGET/FAILURE -> SEALED_EVAL(first, final)
```

There is no patch tool, arbitrary shell, cross-trajectory memory, implicit provider retry, or sealed-result feedback.

## Milestones

### M0: Implementation Contract

- Freeze this module boundary, commit sequence, and exit criteria.
- Keep the research protocol in `r1-a100-rapid-validation-plan.md` authoritative for experiment semantics.

Exit: clean documentation-only commit.

### M1: Trusted Local Vertical Slice

- Add strict canary contracts and task registry.
- Implement row-reduction task cases and a trusted Triton candidate fixture.
- Implement one-process evaluator support for explicit dev/sealed cases, correctness, mutation detection, and timing-result normalization.
- Cover valid, wrong-result, malformed, and timeout results without a live provider.

Exit: trusted `row-reduction × Triton` job round-trips locally and all offline tests pass.

### M2: Fixed Four-Turn Agent Loop

- Add the response parser, state machine, budget accounting, structured feedback, and trajectory store.
- Reuse `ProviderClient` through a narrow completion protocol.
- Verify first/final snapshots, early finish, four-call exhaustion, provider failure, and crash-safe event persistence with fake provider and worker transports.

Exit: a fake model repairs a failing candidate into a correct terminal candidate and the trajectory can be replayed from artifacts.

### M3: Remote Worker And Canary CLI

- Add canonical JSON `WorkerJob/WorkerResult` transport over local subprocess and SSH stdin/stdout.
- Run every job in an ephemeral workspace with credential scrubbing, process-group timeout, and a post-job GPU health result.
- Add `validate`, `worker`, and `run-cell` CLI commands.

Exit: trusted local controller -> A100 worker -> local artifact round-trip, followed by one supervised live Flash/Triton canary trajectory.

### M4: Shakeout Matrix

- Add matmul+bias canary, TileLang/CuTe target cards, adapters, and six trusted canary paths.
- Expand and freeze the 12-cell shakeout schedule.
- Run shakeout, allow at most one uniform card/harness revision, rerun, and freeze hashes.

Exit: no retained target has a per-target Agent floor.

### M5: Formal R1 Matrix

- Add four scientific task packs, twelve trusted target oracles, and four stable `B*` records.
- Freeze the 48-cell schedule and pre/post provider sentinels.
- Run all cells serially and retain every terminal result.

Exit: complete immutable raw matrix. Hindsight fixed/oracle aggregation may be implemented after collection, but every required raw field must be frozen before execution.

## Commit Policy

Each milestone is one or more self-contained commits. A commit must include its focused tests and pass `uv run pytest`, `uv run ruff check .`, `uv lock --check`, and `git diff --check` as applicable. Generated candidates, credentials, remote workspaces, and run artifacts remain outside Git.

## Deferred

General Agent plugins, parallel scheduling, UI, profiler counters, target switching, selector implementation, multi-GPU support, and capability-versus-abstraction interpretation are outside the vertical slice.
