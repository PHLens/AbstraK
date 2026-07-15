# Progress Log: AbstraK Gate R Pilot

## Session: 2026-07-14

### Planning Bootstrap

- **Status:** `complete`
- **Started:** 2026-07-14 21:33 CST
- **Scope:** 只建立 persistent pilot planning files，不执行 toolchain 安装、GPU workload 或 Agent trajectories。

#### Actions Taken

- 定位并完整读取 Planning with Files skill、三个模板和 session recovery 流程。
- 运行 session catch-up；项目中没有可恢复的旧 planning files 或未同步 planning context。
- 确认开始计划前 `main` 与 `origin/main` 对齐且工作区干净。
- 对照 proposal 的 RQ、evaluation contract、Go/No-Go、roadmap 和 6–8 周 pilot protocol。
- 将用户最新决定纳入计划：fixed model ID 不阻塞 pilot，DeepSeek 优先，OpenAI deferred，不统计 API 货币费用。
- 将 pilot 拆成七个有 exit criteria 的 phases，从 manifest/protocol freeze 到 Gate R decision。
- 写入候选 targets、workloads、A100/4090、Agent strata、nested budget、oracle 和 baseline 依赖。
- 建立 Planning with Files 的错误记录和恢复约定。

#### Files Created

- `task_plan.md`：完整 pilot phase plan、scope、invariants、Go/No-Go 和 schedule。
- `findings.md`：repo/proposal findings、resource facts、technical decisions、risks 和 open questions。
- `progress.md`：本文件，记录实际实施、测试与错误。

### Phase 1: Pilot Contract Freeze and Manifest Foundation

- **Status:** `in_progress`
- **Started:** plan accepted for execution, implementation not yet started
- **Next implementation unit:** 定义 `hardware.v1`、`target.v1`、`task.v1`、`agent.v1`、`budget.v1` 和 `study.v1` 的最小 schema 与 cross-reference contract。

#### Completed Within Phase 1

- [x] 记录 pilot scope 与 deferred scope。
- [x] 记录 candidate matrix 和 pre-Agent oracle requirement。
- [x] 记录 mutable model alias policy。
- [x] 记录 resource-accounting policy。
- [x] 记录 R.1/R.2/R.3 三层决策链、baseline ladder 和分支结果；具体 `delta_Q/delta_C` 与增量 margin 留在 Phase 1 freeze。

#### Remaining

- [ ] 实现并测试 manifests。
- [ ] 获取 A100/4090 worker inventory。
- [ ] 审计三个 target 的 `sm80/sm89` 支持。
- [ ] 冻结 workload contracts、split、thresholds 和 study version。
- [ ] 冻结 R.1/R.2/R.3 的 exact comparator semantics、practical margins 和 analysis fields。

### Independent Plan Review

- **Status:** `complete`
- Engineering and scientific-rigor reviews completed and recorded in `findings.md`.
- Required revisions: replicate-aware matrix, anytime checkpoint semantics, exploratory alias eligibility, deterministic utility rule, earlier worker/artifact contracts, sandbox recovery, frozen baseline replay semantics, randomized scheduling and realistic timeline.
- Additional rigor revisions: cross-family P1, target-stack claim boundary, semantic-member statistical unit, operational leakage scan, headline threshold/ambiguity band and zero-human primary track.

### Plan Revision After Review

- **Status:** `complete`
- Initial review recomputed P1/full-P2 from 12/36 to 48/144 trajectories using two semantic packs per family and two Agent replicates per cell; the later layered revision identifies 96 as the mandatory two-Agent, dual-hardware minimum.
- Reframed `B_L` as an anytime checkpoint under one `B_H`-declared policy.
- Added qualification-first serial-wall-clock decision rule, secondary caps, ambiguity band and Pareto reporting.
- Added exploratory mutable-alias eligibility without weakening strict `pilot_ready`.
- Moved worker/job, minimal adapter, failure taxonomy, generalized artifact bundle and generated-code isolation into Phase 1–2.
- Froze cross-family P1, randomized/time-blocked execution, baseline replay/fresh-run semantics and infrastructure reschedule policy before unblinding.
- Changed the primary empirical claim from pure representation effect to Agent–target-stack fit.
- Extended the schedule from 8 weeks to a 10–16 week exit-criteria-driven plan.
- Final re-review closed replicate-pairing, semantic-independence, Gate aggregation and tie-margin issues, and tightened CuTe phase ordering, `B_L` qualification, replicate identity, hostile-code exit criteria and ambiguous billing semantics.

### Layered Gate R Revision

- **Status:** `complete`
- Reframed Gate R as R.1 target-selection necessity, R.2 Agent incremental value and R.3 low-cost predict-and-exploit feasibility.
- Added global/per-Agent calibration fixed baselines, hindsight fixed bounds and fixed-vs-cell-oracle qualification/cost comparisons.
- Split workload and hardware matched contrasts; an A100 negative result no longer skips the second-hardware test.
- Changed the minimum complete two-Agent matrix from 48 directly-to-144 into P1 48, mandatory dual-hardware P2a 96, and optional third-Agent P2b 144.
- Replaced one mixed Go/No-Go list with explicit fixed, per-Agent default, context rule, direct exploration and Gate P branches.

## Verification Results

- Planning skill recovery check：无旧 planning context，符合预期。
- `task_plan.md` heading audit：Phase 1–7 顺序已修复并确认。
- Matrix arithmetic audit：审查前为 12/36；当前为 P1 48、mandatory P2a 96、optional P2b 144，分别带对应 anytime checkpoints。
- Markdown audit：无 trailing whitespace；文件以 newline 结束；无 Obsidian-style internal links、Markdown tables 和 Mermaid。
- Scope audit：没有开始 GPU/Agent 实验，没有修改 LoreForge proposal。
- Secret boundary：planning files 只记录配置路径和环境变量概念，不包含 API key。
- Rendering policy：planning files 使用普通 Markdown links/lists，不使用 Obsidian wikilinks或 Mermaid。
- Final structure：七个 phases，状态计数为一个 `in_progress`、六个 `pending`。
- Final matrix：P1 48、mandatory dual-hardware P2a 96、optional third-Agent P2b 144；每条 trajectory 对应一个 anytime checkpoint。
- Planning-bootstrap scope：当时 `git status` 只显示三份新 planning files；这些文件随后已提交。本轮修订仍只涉及这三份 tracked planning files。
- Layered-revision verification：`git diff --check` 通过；Phase 1–7 数量为七、状态为一个 `in_progress` 加六个 `pending`；Obsidian wikilink、Markdown table 与 Mermaid render-risk scan 均无命中。

## Error Log

- **2026-07-14 21:xx CST, attempt 1:** 第二段 `task_plan.md` patch 以重复的 `Status: pending` 为锚点，Phase 3 被插入文件末尾。
- **Resolution:** 先检查所有 heading，再以唯一的 `Phase 4` heading 为插入锚点移动 Phase 3；后续大段 patch 必须使用唯一 heading。

## Next Actions

1. 冻结 R.1/R.2/R.3 comparator semantics、`delta_Q/delta_C`、Agent incremental margin 和 R.3 break-even rule。
2. 从 Phase 1 manifest schema 与 analysis result fields 开始实施，不直接跳到 Agent loop。
3. 每完成一个 phase，更新 `task_plan.md` status、`findings.md` discoveries 和本文件的验证结果。

## 5-Question Reboot Check

- **Where am I?** Phase 1，pilot contract freeze 与 manifest foundation；当前只完成计划定义。
- **Where am I going?** Single-cell qualifier、full oracle matrix、Agent runtime、calibration、P1/P2 和 Gate R decision。
- **What is the goal?** 依次验证 target selection 是否必要、Agent information 是否提供 context-only 之外的增量价值，以及差异能否被低成本预测和利用。
- **What have I learned?** 见 `findings.md`。
- **What have I done?** 完成 planning bootstrap，见本 session log。

---

本文件记录实际发生的工作，不预先把计划中的未来任务写成完成状态。

## Session: 2026-07-15

### KernelBench Naive Screen Skeleton

- **Status:** `controller_complete_gpu_pending`
- **Scope:** 先实现 DeepSeek Flash/Pro × Triton/TileLang/CuTeDSL 的单轮 naive baseline；本轮不发起真实 API 请求，不安装或运行 GPU toolchain。

#### Implemented

- 新增严格 `kernelbench-naive-study.v1`、cell、generation、evaluation 和 summary contracts。
- 新增 pinned KernelBench checkout adapter，复现官方 zero-shot prompt 与 first-code-block extraction policy。
- 新增单请求 matrix generator；实验级 manifest 覆盖 P0.1 token/output contract，但不修改用户全局配置。
- 新增私有原子 artifact store；生成/evaluation bundles 分离、只读 sealing、secret scan、checksum verification 和 generation reference hash。
- 新增 serial subprocess GPU evaluator/worker；CLI 必须显式确认 billable request 数和 generated-code execution。
- 新增 per-profile/target compile、correctness、performance coverage、ratio、`fast_1` 和 `fast_2` 汇总。
- 新增 6-cell smoke 与 24-cell screen manifests、运行文档和 CLI entrypoint。

#### Verification

- `.python-version`、package metadata、Ruff target 与 `uv.lock` 已固定到 CPython 3.10；`uv run python --version` 实测为 `3.10.20`。
- `uv run ruff check .` 通过。
- `uv run pytest` 在 Python 3.10.20 通过：68 tests passed，且没有 warning。
- `uv lock --check` 与 `git diff --check` 通过。
- Smoke 和 screen 都在 KernelBench pinned commit 上离线 validate，矩阵大小分别为 6 和 24。
- Tampered/extra artifact、commit mismatch、非法 TileLang FP32、无 code block、请求数不匹配和缺少执行授权均有 negative tests。
- 本轮没有真实 provider call、GPU evaluation 或 generated-code execution。

#### Next Actions

1. 用户提供 GPU worker 环境后，先完成 Torch/KernelBench/Triton/TileLang/CuTeDSL inventory 与 known-correct backend smoke。
2. 先运行一个 `Flash × Triton × square GEMM` live generation/evaluation cell，检查实际 artifact 与 worker provenance。
3. 单 cell 通过后运行 6-cell smoke；只有结果不是明显 floor/ceiling 时才扩展到 24-cell screen。

### A100 GPU Environment

- **Status:** `complete_for_trusted_toolchain_smoke`
- **Runtime:** CPython `3.10.20` + PyTorch `2.13.0+cu126` + Triton `3.7.1` + TileLang `0.1.12` + CuTe DSL `4.6.1`。
- **Persistence:** source、locks、Python runtime 和带 SHA-256 的离线 wheel archive 保存在 `/workspace/volume/lipenghui`；GPU venv/cache/wheel staging 放在容器本地 `/tmp`，可由 `scripts/bootstrap-a100.sh` 重建。

#### Verified

- Bootstrap 对 Python、Torch/CUDA、driver、A100 SM80、GPU package imports 和 Torch FP16 CUDA 运算均已通过。
- `abstrak-doctor --require-gpu` 已通过。
- KernelBench trusted Triton/TileLang/CuTe add examples 均 compile 且 correctness 通过，并记录 candidate/reference runtime。
- 远端 `pytest` 通过 68 tests，Ruff 通过。
- 未发起 provider API 请求，未执行任何模型生成 kernel。

#### Environment Boundaries

- Live smoke 期间已以 mode `0600` 向当前容器临时部署 `~/.abstrak/config.yaml`/`auth.json`；secrets 不在仓库或持久 artifacts 中。
- 正式执行模型生成代码前，仍需完成 worker isolation 与 hang/OOM quarantine gate；当前镜像无 Docker。

### First Live 6-Cell Naive Smoke

- **Status:** `complete_floor_detected_no_24_cell_expansion`
- **Study/run:** study hash `d09216ec7035707c803e8cde90540f68dee2ac506123fb72010b3c281e5b4341`；run `20260715T110357.954897Z-e3afdf778b`。
- **Execution exception:** 用户明确授权在当前 remote Docker 中执行 generated code；这只适用于 exploratory smoke，不改变正式 pilot 的 worker-isolation exit gate。

#### Execution

- Remote controller 到 DeepSeek official endpoint 的所有请求被 `[Errno 104] Connection reset by peer`；failed run `20260715T110146.001822Z-c23e15442b` 保留为 infrastructure artifact，不计入结果。六次均已标记 `request_submitted=true/possibly_charged`，没有在原 run 中 retry。
- Local controller 在 pinned KernelBench commit 上完成 6/6 single-turn generations。Generation archive SHA-256 为 `917460b076210ae04f7e87b4feeb0c76d768c90aa46ea86ebf62a590f3913f3d`，远端校验后才解包。
- A100 evaluator 重新验证每个 sealed generation bundle，完成 6/6 terminal evaluations 和 metrics summary，无 missing evaluations。

#### Results

- Flash×Triton：compile/correct，`0.620 ms` vs PyTorch `0.518 ms`，performance ratio `0.835x`。
- Pro×Triton：compile/correct，`0.781 ms` vs PyTorch `0.517 ms`，performance ratio `0.662x`。
- Flash/Pro×TileLang：均缺少当前 `@T.prim_func` API；关闭 static check 后仍因不存在的 `tilelang.kernel` 失败。
- Flash/Pro×CuTe：均未通过 static check；关闭 static check 后仍因 placeholder path、C++ standard 和 binding declaration 错误失败。

#### Decision

- 两个 DeepSeek profiles 的 target ranking 相同，没有 Agent-dependent reversal；Flash 在唯一成功的 Triton cell 上快于 Pro，但这只是 within-target 差异。
- TileLang/CuTe 的 4/4 failures 构成明显 floor effect。按预注册规则不扩展到 24-cell screen，先重新设计 target assets 或 target set。
- 本结果证明 target selection 在当前设置下退化为 fixed Triton，不证明 R.2 没有研究价值；同 family、单 task、单 replicate 且两 targets floor，不具备 equivalence-test 条件。
