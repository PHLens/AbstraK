# Task Plan: AbstraK Gate R Pilot

## Goal

在不预设 Agent-aware target selection 必然有效的前提下，完成一个可复现的 Gate R pilot：建立三种 target stacks、两类 workload、A100/RTX 4090、多个 Agent 与 anytime resource checkpoints 上的 oracle matrix，验证是否存在稳定、可预测且具有实际成本意义的 Agent–target-stack crossover，并据此做出继续建设 progressive policy 或停止的 Go/No-Go 决策。

## Current Phase

**Phase 1: Pilot Contract Freeze and Manifest Foundation** (`in_progress`)

当前只编写和审查 pilot 计划。下一次实施从 Phase 1 的 manifest schema 与 protocol freeze 开始。

## Completed Preconditions

- [x] 初始化 AbstraK 仓库并推送 `main`。
- [x] 完成 P0.1 provider conformance、usage normalization、artifact sealing 与 secret redaction。
- [x] 将默认运行配置迁移到 `~/.abstrak/config.yaml` 和 `~/.abstrak/auth.json`。
- [x] 真实验证 DeepSeek V4 Flash/Pro endpoint；短请求的 provider token 与可见输入基本一致。
- [x] 决定初期使用 DeepSeek API，OpenAI 后续按需要加入。
- [x] 决定 pilot 不以固定 checkpoint/model ID 作为启动门槛；所有服务别名必须记录请求 ID、返回 ID、时间、provider、配置 hash 和重复 conformance 结果。

## Pilot Boundary

### Included

- Whole-workload representation target 选择，不做单个 Kernel 内的 region-level target mixing。
- 三个候选 transformation layers：Triton、TileLang、CuTeDSL。
- 两个 workload strata：memory/data-movement dominated family 与 low-precision GEMM family。
- 两个 NVIDIA hardware strata：A100 (`sm80`) 与 RTX 4090 (`sm89`)。
- DeepSeek V4 Flash/Pro 用于 infrastructure bring-up；正式 P1 使用一个 DeepSeek tier 加一个 GLM/Qwen external family，P2 再加入第三 capability tier。
- 正确性、robustness、integration、性能、tokens、turns、compile/profile runs、GPU time、wall time 与 human intervention 的分项记录。
- Representation oracle、budget-matched per-Agent target oracle 和 production oracle `B*` 的严格分离。
- Gate R 所需 fixed/rule/Compiler-library-selector/feature-only/model-ID/profile/top-2/successive-halving/parallel/oracle baselines。

### Deferred

- OpenAI 官方 API、缓存费用和货币化 API 成本作为可选扩展，不阻塞 pilot。
- 固定模型 snapshot 作为 paper-scale reproducibility 加固项，不阻塞 insight-seeking pilot。
- Progressive probe/continue/switch policy 仅在 Gate R 通过后实现。
- MoE、dynamic shape/ragged workload、multi-GPU、跨 vendor hardware 和新 backend onboarding。
- 新 Semantic IR、通用 dialect 或 Compiler extension；只有重复 failure evidence 触发 Gate C 后才考虑。
- 人类参与者实验与长期维护成本研究。

## Frozen Candidate Matrix

在 Phase 1–3 oracle audit 完成前，以下是候选矩阵，不得开始正式 Agent 数据采集：

```text
Targets:   Triton | TileLang | CuTeDSL
Workloads: 2 semantically distinct, lineage-isolated RMSNorm-family members
           2 semantically distinct, lineage-isolated W4A8 family members
Hardware:  A100 (sm80) | RTX 4090 (sm89)
Agents P1: one DeepSeek tier | one GLM/Qwen external family
Agent P2:  one additional capability tier, initially DeepSeek V4 Pro
Replicates: 2 independent Agent runs per Agent–task–target–hardware cell
Budgets:   B_L anytime checkpoint nested inside one B_H-declared trajectory
```

正式矩阵中的 `target` 是完整 target stack：representation/DSL、Compiler/backend、standard primitives、autotuner、documentation pack 和允许的 examples。Pilot 的主张是 Agent–target-stack fit；除非增加 factorial asset ablation，否则不把 observed difference 因果归于 representation 本身。

如果 CuTeDSL 无法在两个 workload 和两个 GPU 上建立可复现的 representation oracle，必须在观察任何正式 Agent 结果之前替换为 CUDA C++ 或收缩 target claim。Dense/fused GEMM 只能作为 low-precision oracle 未就绪时的 harness control，不能支撑量化适配结论。

## Cross-Phase Invariants

- Agent、target、workload、hardware、budget 和全部软件版本由显式 manifest 标识并参与 artifact hash。
- 同一 semantic member 的 shapes、target realizations、timing seeds 和派生实现始终位于同一 split。
- Development feedback 与 sealed qualification 隔离；sealed result 不回流当前或其他 held-out trajectories。
- 每个 retained task–target–hardware cell 必须先有 known qualified representation path。
- 每个 Agent–task–target–hardware–replicate trajectory 从 clean workspace 启动，最多执行一次预注册 orchestration policy。
- Agent 只获知 `B_H`，并被要求尽快完成；`B_L` 是同一 anytime policy 的只读 prefix checkpoint，不解释为 budget-conditioned Agent behavior。
- 每个 workload family 在 pilot 中至少包含两个语义不同且 lineage-isolated 的 members；不同 repository、shape pack 或轻微派生不构成新的 semantic member。每个 member 的 task pack 内含多个 shapes/corner cases。
- 每个 Agent–task–target–hardware cell 至少运行两个 clean-workspace replicates，用于估计 Agent sampling/service variability；replicate 不替代 semantic sample。
- 失败、超时、编译错误和未 qualification 实例全部保存，按 budget cap 作为 censored failure。
- 不把 shape、seed 或 target realization 当成独立统计样本。
- 不使用单一加权“总成本”；tokens、turns、compile/profile runs、GPU time、wall time 和 human time 分轴报告。
- Pilot 初期不计算 API 货币费用，但保留 provider input/output token 与本地可见 transcript token。

## Pre-Registered Decision Rule

- Qualification 是硬约束：所有 required shapes/corner cases 正确，integration constraints 通过，aggregate relative-to-`B*` 的置信下界达到唯一 headline `tau`，且 worst-shape score 不低于冻结 floor。
- `B*` 是每个 shape 上 applicable strong baselines 的 envelope；shape distribution、aggregate function、threshold 和 worst-shape floor 在 Phase 1 冻结。
- `tau=0.8/0.9/0.95` 等只作为 sensitivity；headline `tau`、timing confidence method、最大复测次数和 indifference/ambiguity band 在 P1 前只能有一套冻结值。区间跨越门槛且复测耗尽时记为 ambiguous，不强制归类为 pass/fail。
- Autonomous primary track 不允许人工 patch/takeover。发生人工修复的 artifact 进入单独 engineering ledger，不计为 autonomous qualified result。
- 在满足所有 secondary resource caps 的 qualified runs 中，primary cost 是 isolated charged serial wall-clock-to-qualified：包括 provider latency、compile、test 和 profile，不包括外部排队时间。
- 两个 targets 的 primary cost 差异小于预注册 equivalence band 时，先比较 worst-shape score，再比较 provider total tokens；每一级 tie-break 都有独立 practical-equivalence/uncertainty band，任何一级未超过 margin 就继续，全部未超过则记为 tie。
- Tokens、GPU time、compile/profile runs 和其他资源继续单独报告 Pareto frontier，不用任意加权和覆盖 primary decision rule。
- Actionable crossover 只在 matched semantic members 经 replicate aggregation 后出现 stable qualification change，或 stable-qualified winner 的 primary cost gap 超过 utility margin 时成立。

### Replicate Aggregation and Target Oracle

- Replicate IDs 不跨 targets 配对，也不构造“同 seed”假设；mutable service API 的两次 sampling 没有共同随机变量。
- Primary oracle unit 是 `Agent × semantic member × hardware × checkpoint` cell。每个 target 的两个 replicates先聚合为 `stable-qualified`（2/2）、`unstable`（1/2）或 `failed`（0/2）。
- Target oracle 只在 stable-qualified targets 中选择；先比较 median serial wall-clock cost，再按带 margin 的 worst-shape score 和 token tie-break。只有 unstable target 时，该 cell 记为 ambiguous/no-stable-oracle。
- Per-run winner 只作 descriptive debugging，不进入 crossover、regret 或 Gate R 计数。
- Replicate disagreement 不删除、不按半个样本计数，并在 Gate aggregation 中保守地视为不稳定证据。

## Budget Event Semantics

- `B_L/B_H` 都是包含 turns、tokens、compile/test/profile、GPU 和 wall time guardrails 的资源向量；Agent 只看见 `B_H`。
- 每个 action 在开始前按 manifest 声明 reservation。剩余资源不足以覆盖 reservation 时不得启动该 action。
- LLM completion、compile、test 和 profile 等 in-flight action 只能在 reservation 与 timeout 内完成；超过 reservation 记为 harness/budget violation，而不是静默 overshoot。
- `B_L` snapshot 是按同一 ledger 回放时，在任一 `B_L` guardrail 阻止下一 action 前的最后完整 event boundary。
- `B_L` 只用于 anytime qualification/cost curve。真正“被告知低预算”的 Agent policy 若需要研究，必须作为独立 matched study 运行。

## Analysis Units and Replication

- 独立 scientific unit 是语义不同且 lineage-isolated 的 semantic member；同一 member 的 hardware、target、shape、timing 和 Agent replicates 是 clustered/repeated observations。
- Workload-family×hardware 的四个 blocks 只用于 pragmatic Gate R screening，不解释为四个独立科学样本。
- 每个 cell 的两个 Agent replicates 用于暴露 sampling/service instability，不足以单独估计稳定的 node-level probability distribution。
- Pilot 的 probability/proper-score 分析按 semantic task 聚类并报告 raw outcome table 与 uncertainty；ECE 仅在预注册 prediction-instance minimum 达到时作为 exploratory metric。
- Paper-scale study 的 semantic member 数和 replicate 数由 pilot variance/power sensitivity 重新冻结，不从 shapes/seeds制造样本量。

## Phases

### Phase 1: Pilot Contract Freeze and Manifest Foundation

**Objective:** 在运行任何 Kernel Agent 之前冻结实验单位、资源向量、版本边界、split 和 Go/No-Go 逻辑。

#### Tasks

- [ ] 定义并测试 `hardware.v1`：只包含稳定 GPU capability/SKU；定义 `worker.v1`：host、driver/toolchain/container、GPU assignment、isolation 和 profiler policy；每条 run 另存 observed environment fingerprint。
- [ ] 定义并测试 `target.v1` manifest：完整 target stack，包括 representation、Compiler/backend version、supported architectures、adapter、documentation/examples hashes、standard primitives、autotuner 和 cache policy。
- [ ] 定义并测试 `task.v1`：semantic family/member、source lineage、dtype/layout、shape distribution、corner cases、tolerance、integration、shape dispatch/autotuning allowance、all-shape correctness、per-shape `B*` envelope、aggregate score、worst-shape floor 和 split。
- [ ] 定义并测试 `agent.v1`：provider profile、requested/returned model identity policy、prompt/tool/context/retrieval、decoding、timeout 和 stop rules；replicate ID 不属于 Agent identity。
- [ ] 定义并测试 `budget.v1`：resource caps、per-action reservations、timeouts、event-boundary checkpoint、violation classification，以及 `B_L` anytime prefix 与 `B_H` continuation。
- [ ] 定义并测试 `study.v1` 和 run/trajectory manifest：交叉引用、semantic task/replicate IDs、sampling seed（若 provider 支持）、blocked-randomized schedule、thresholds、split hashes、early-stop、baseline 和 artifact schema version。
- [ ] 定义 `exploratory-study-eligible` provider gate：严格 `pilot_ready` 保持不变；mutable alias 需要 transport/action ready、显式 service-alias provenance、batch 前后 conformance 和 reproducibility warning。
- [ ] 冻结每个 family 至少两个真正不同的 semantic members 及 derivation graph；建立 source-lineage registry，禁止同源实现跨 calibration/holdout。
- [ ] 定义 semantic-family hierarchy，并对 calibration/holdout 与 oracle/docs/examples 做 exact hash、token/AST near-duplicate 和人工语义 overlap audit；pretraining contamination 只记录风险，不伪称可完全排除。
- [ ] 冻结 primary decision rule、secondary caps、tie rule、censoring、missing-cell 和 infrastructure-reschedule semantics。
- [ ] 定义 minimal `worker-job.v1/result.v1`、target-adapter interface、common failure taxonomy 和 qualification result contract，供 Phase 2 直接实现。
- [ ] 设计新的 content-addressed `artifact-bundle.v1`：source tree/patch、logs、binaries、nested worker artifacts、hash chain、atomic incomplete/terminal state 和 checksum；不扩张 provider-specific store。
- [ ] 定义 generated-code worker isolation：network/filesystem/process boundaries、GPU assignment、resource controls、timeout/cancel、health check、reset/quarantine、cleanup 和 idempotent job ID。
- [ ] 对 A100 和 RTX 4090 运行只读 worker inventory，确认独占方式、时钟稳定性、profiler 权限和远程执行边界。
- [ ] 审计 Triton、TileLang、CuTeDSL 对 `sm80/sm89`、目标 dtype/layout 和 profiler 的实际支持。
- [ ] 预注册唯一 headline `tau`、sensitivity thresholds、baseline/timing confidence、ambiguity band、actionable utility margin、replicate aggregation 和 P1/P2 early-stop 条件。
- [ ] 明确服务型模型 alias 的 pilot 记录协议：不阻塞运行，但禁止声称 checkpoint-level reproducibility。

#### Deliverables

- `configs/{hardware,workers,targets,tasks,agents,budgets,studies}/` 下的 schema-compatible examples 与 frozen pilot manifests。
- Manifest cross-reference validator、stable/content hash，以及 duplicate ID、cycle、missing reference 和 hash mismatch negative tests。
- `pilot-protocol.v1`：matrix、replicates、split、resource vector、decision rule、thresholds、baseline semantics、model alias policy 与 allowed feedback。
- A100/4090 worker inventory artifacts 和 target support audit。
- `worker-job/result.v1`、minimal adapter/failure contract、`artifact-bundle.v1` 和 sandbox threat model。

#### Exit Criteria

- [ ] 所有 manifest 可离线验证，未知字段、交叉引用错误和不安全配置会 fail closed。
- [ ] 四个 workload–hardware blocks、每 family 两个 semantically distinct members、两个 replicates、三个 target 和 Agent/budget strata 均有稳定 ID。
- [ ] Strict `pilot_ready` 与 exploratory study eligibility 在代码、CLI、docs 和 tests 中有不同名称与行为。
- [ ] Worker/job/artifact contracts 可在无 GPU 的 fake worker 上完成 round-trip、idempotency、cancel、crash-recovery 和 checksum tests。
- [ ] Phase 1 只冻结 CuTeDSL retain/replace 的证据要求、决策人和 deadline；最终决定由 Phase 3 oracle audit 产生。
- [ ] Monetary API cost 和 fixed model ID 明确标记为 deferred，不影响其他资源计量。

**Status:** `in_progress`

### Phase 2: Single-Cell Qualification and Oracle Vertical Slice

**Objective:** 在完全不调用 LLM 的情况下，稳定判定一个候选 Kernel 是否 production-qualified。

**Initial cell:** `RMSNorm-family × Triton × A100`。

#### Tasks

- [ ] 实现 workload contract、input generator、reference semantics 和 dtype-aware tolerance。
- [ ] 建立 development shapes 与 sealed qualification shapes/seeds/corner cases 的不可逆分离。
- [ ] 实现 GPU worker request/result protocol，不让 controller 假定本机存在 GPU 或 CUDA toolchain。
- [ ] 实现最小 Triton target adapter 和 common failure taxonomy；Phase 3 只扩展该接口，不重新定义。
- [ ] 实现 generalized content-addressed artifact bundle，不把 Kernel artifacts 塞进 provider conformance store。
- [ ] 在隔离 worker 上执行 generated code：无外网、ephemeral writable workspace、只读 inputs/docs、host/process/GPU resource limits 和强制 cleanup。
- [ ] 实现 compile、load、correctness、robustness、warmup、timing、independent rerun 和 integration checks。
- [ ] 审计 applicable production baselines，选出当前 cell 的 `B*`，禁止默认使用 PyTorch eager。
- [ ] 引入一个 known qualified Triton implementation 作为 representation oracle，但不进入 Agent-visible materials。
- [ ] 建立 positive control、numerically wrong control、shape-fragile control 和 correct-but-slow control。
- [ ] 记录 raw timings、environment fingerprint、source hash、manifest hashes 和 terminal qualification reason。
- [ ] 重复运行以估计 timing variance，冻结 warmup、measurement count、outlier 和 clock policy。
- [ ] 注入 infinite host loop、long-running kernel、GPU OOM、process crash 和 worker restart，验证 timeout/cancel、health check、quarantine 与幂等恢复。
- [ ] 注入 NaN/Inf、OOB sentinel、nondeterministic output、timing outlier、cache contamination 和 sealed-data access，验证 qualifier fail closed。

#### Deliverables

- Qualification runner 与 worker adapter。
- Minimal Triton adapter、common failure taxonomy、sandbox recovery tests 和 generalized artifact bundle。
- RMSNorm task pack、Triton representation oracle 和 production baseline audit。
- Sealed qualification artifact bundle 与 checksum verifier。
- 正负控制测试和 timing stability report。

#### Exit Criteria

- [ ] Known qualified path 稳定通过 correctness、robustness、integration 和 performance threshold。
- [ ] 四类负/正控制得到预期判定，失败层级可解释。
- [ ] Selection/tuning run 与 final measurement run 使用独立 process 和 timing samples。
- [ ] 从 clean workspace 可重放并得到相同 terminal classification；中断后 artifact 明确处于 incomplete 或 recoverable terminal state。
- [ ] 所有 hostile controls 都在预注册 caps 内终止，无 network/filesystem/process escape；任务后 worker 必须 health-check pass 或进入 quarantine，未恢复前不得运行下一 measurement。

**Status:** `pending`

### Phase 3: Target Adapters and Full Oracle Readiness

**Objective:** 将 Phase 2 的共同 qualification contract 扩展到完整 candidate matrix，并在 Agent 运行前证明每个 retained representation space 可达。

#### Tasks

- [ ] 将 Phase 2 的 adapter interface 扩展为 scaffold、compile、diagnose、run、profile、package 和 version probe。
- [ ] 完成 TileLang/CuTeDSL adapters，并验证 Triton adapter；target-specific output 统一映射到 frozen failure taxonomy。
- [ ] 为 RMSNorm-family 在 A100/4090 上建立全部 retained representation oracles。
- [ ] 为 W4A8 fused-dequant GEMM 在 A100/4090 上建立全部 retained representation oracles。
- [ ] 对每个 cell 审计强 production baselines，包括 applicable vendor、expert、Compiler 和 domain implementation。
- [ ] 记录 target standard primitives、examples、autotuning、compile cache 和 cold-start cost，避免不等价 asset comparison。
- [ ] 验证同一 task pack 在不同 targets 使用相同 semantics、shape distribution、tolerance、integration 和 measurement protocol。
- [ ] 对无 oracle cell 在解盲前执行 fix、replace 或 claim shrink，并记录原因。
- [ ] 运行 adapter conformance、oracle regression 和 cross-hardware repeatability tests。
- [ ] 测试 adapter version mismatch、compile/autotune cache isolation、artifact content hash 和 per-hardware environment compatibility。

#### Deliverables

- 三个 target adapters、documentation-pack manifests 和 conformance reports。
- `oracle-matrix.v1`：representation availability、qualified path、`B*`、performance band 与 N/A reason。
- 两类 workload 的 A100/4090 sealed task packs。
- Target asset-equivalence audit 与 retained-cell decision。

#### Exit Criteria

- [ ] 每个 retained task–target–hardware cell 都有 independently remeasured known qualified path。
- [ ] 主要 compile/runtime failures 可稳定映射到 representation/compiler/environment categories。
- [ ] 不需要新增完整 backend；若需要则该 cell 转入 capability-extension，不进入 Gate R routing claim。
- [ ] 在 oracle matrix 完成前不保存任何正式 Agent trajectory。
- [ ] 正式数据采集前完成 CuTeDSL retain/replace/claim-shrink 决策，不允许看到 Agent 结果后修改 target label。

**Status:** `pending`

### Phase 4: Controlled Agent Runtime and Trajectory Ledger

**Objective:** 在不引入 target router 的情况下，为所有 model–target cells 提供相同 orchestration、tools、反馈和资源计量。

#### Tasks

- [ ] 定义最小 Agent state machine：inspect、edit、compile、test、profile、finish/abstain；禁止隐式 retry、fallback 和跨任务 memory。
- [ ] 冻结 common system contract、tool schema、context policy、error parser、stop policy 和 target-specific documentation injection 规则。
- [ ] 实现 clean workspace lifecycle、candidate source capture、patch history、command allowlist、timeouts 和 deterministic tool result envelope。
- [ ] 实现 append-only trajectory ledger：attempt、action、diff、compile/test/profile result、failure category、tokens、turns、GPU/wall time 和 human intervention。
- [ ] 同时记录 provider-native input/output tokens 与统一 tokenizer 计算的 visible transcript tokens；不将 cache 状态或 monetary fee混入逻辑 token 指标。
- [ ] 将 P0.1 `ProviderClient` 接入 orchestration，初期保持 non-streaming、single candidate 和 no fallback。
- [ ] 完成 DeepSeek V4 Flash/Pro agent manifests 与 exploratory study eligibility；服务 alias 在 study gate 中产生 provenance warning，不改变严格 `pilot_ready`。
- [ ] 在每个 randomized execution batch 前后运行 conformance，记录 requested/returned ID、UTC、provider request ID 和 config hash。
- [ ] 建立 scripted/fake Agent integration tests，验证预算 reservation/停止、duplicate call、ledger hash chain、crash recovery、错误恢复、artifact sealing、secret redaction 和一请求一记录。
- [ ] 在 Phase 2 single cell 上运行 unsealed engineering shakeout；这些轨迹不得进入 Gate R 数据。
- [ ] 将 `qualification candidate` 明确定义为 development-visible gate；Agent stop/finish 永远不能读取 sealed qualification。

#### Deliverables

- Model-independent orchestration loop、tool protocol 和 sandbox boundary。
- `trajectory-ledger.v1`、budget event schema 和 immutable run bundle。
- DeepSeek Flash/Pro agent manifests、prompt/documentation packs 和 conformance records。
- Scripted Agent、failure injection 和 budget enforcement tests。

#### Exit Criteria

- [ ] 替换 base model 不改变 orchestration skeleton、tools、feedback visibility 或 stop semantics。
- [ ] 每次外部调用、代码修改、Compiler/test/profile action 和 budget consumption 都可从 ledger 重建。
- [ ] Worker crash、controller restart 和重复提交不会触发自动 duplicate provider request；ambiguous boundary 记录 `possibly_charged`，terminal artifacts 不被覆盖。
- [ ] Agent 看不到 representation oracle source、production winner、sealed shapes/results 或其他 holdout trajectories。
- [ ] 工程 shakeout 可完成一次从 specification 到 candidate qualification 的闭环，但不计入正式结果。

**Status:** `pending`

### Phase 5: Calibration, Budget Dry Run, and P1 Freeze

**Objective:** 用与 holdout 无 lineage 关系的少量 probes 建立可审计 capability profile，并在看到 crossover 结果前冻结预算和分析代码。

#### Tasks

- [ ] 运行 baseline-only feasibility dry run，测量 compile/test/profile latency、GPU 占用、Agent turn 长度和常见失败恢复成本。
- [ ] 冻结多维 `B_L` 与 `B_H`；`B_H` 初始目标为 `B_L` 的三倍，但最终值由 dry run 决定。
- [ ] 在 DeepSeek bring-up 后，从 GLM/Qwen 中选择一个不同 family，完成 P0.1 conformance、exploratory eligibility 和相同 orchestration dry run；该 external family 必须进入正式 P1。
- [ ] 为每个 model–target node 选择四个初始 calibration probes：每个 stratum 两个 semantically distinct、lineage-isolated members；最多八个 probes 的增量规则预注册，且不复用 P1/P2 semantic members。
- [ ] 审核 calibration、development、screening holdout 和 sealed qualification 的 family/source/derivation lineage 隔离。
- [ ] 冻结 target-specific documentation packs、非解答式 examples、prompt template、tools、retrieval-off policy 和 context budget。
- [ ] 实现简单可审计 profile：预注册 prior/partial pooling 的 Beta-Binomial 或 logistic model，加 censored cost estimator；定义 prediction instance、leave-one-semantic-member-out validation 和 minimum sample size。
- [ ] 实现 workload+hardware feature-only、model identity metadata 和 static capability rules，作为 profile 增量价值的对照。
- [ ] 在 P1 前冻结全部 baseline policy：fixed、static rules、Compiler/library selector、feature-only、model-ID、profile one-shot、top-2、successive halving、idealized parallel 和 oracle。
- [ ] 明确哪些 baseline 从独立 event ledger 做无泄漏 replay，哪些需要 fresh runs；idealized equal-deadline parallel 以隔离 trajectories 的最大完成时间作为 counterfactual lower bound，同时报告资源总和，不声称实测并发 latency。
- [ ] 冻结 blocked-randomized execution schedule：按 semantic task/hardware/time block 分层，交错 target/Agent/replicate；final GPU timing 串行执行并隔离 compile/autotune cache、固定 concurrency/thermal policy。
- [ ] 冻结 transient infrastructure policy：未提交 provider request 或 worker preflight failure 可按相同 replicate ID 重调度；可能已计费/已生成 action 的失败进入 ledger，不静默 retry。
- [ ] 冻结 Brier/ECE、censored cost error、clustered-by-semantic-task analysis、actionable utility margin、calibration break-even 和 routing-headroom 脚本 hash。
- [ ] 用 synthetic fixtures 覆盖 qualification/tie/ambiguity、censoring、missing cells、replicate disagreement、crossover、Pareto-incomparable 和每个 No-Go trigger。
- [ ] 生成 P1 study manifest；在 manifest freeze 后任何 protocol 修改必须递增 study version。

#### Deliverables

- `calibration-pack.v1`、lineage audit 和 frozen documentation packs。
- `budget.v1` 的 `B_L/B_H` 实例与 baseline-only feasibility report。
- Simple capability profile、feature-only/rule/model-ID baselines 和 calibration report。
- Frozen baseline definitions/replay code、randomized schedule、P1 manifest、analysis code hash 和 pre-run checklist。

#### Exit Criteria

- [ ] Calibration probes 不包含 holdout solution lineage，oracle source 不进入 Agent-visible corpus。
- [ ] `B_L/B_H` 在 turns、tokens、compile/test/profile、GPU time、wall time 和 human intervention 上均有明确 cap。
- [ ] Profile 和所有 baseline 在 P1 解盲前冻结，且可从 calibration artifacts 重建。
- [ ] P1 所有 retained cells 的 provider、worker、target、task、budget 和 qualifier conformance 均通过。
- [ ] Pilot 的 Brier/ECE 与 uncertainty 仅作为小样本 exploratory diagnostics，并按 semantic task 聚类；不作论文级显著性结论。
- [ ] Formal autonomous trajectories 的 human intervention cap 为零；任何人工 hint/patch 只存在于 excluded shakeout ledger。

**Status:** `pending`

### Phase 6: P1 Crossover Screening on the First Hardware

**Objective:** 以最小 matched matrix 判断是否存在超过噪声的 target frontier 差异；无信号则尽早停止。

**P1 matrix:**

```text
3 targets
× 2 cross-family Agents (one DeepSeek + one GLM/Qwen)
× 2 workload families
× 2 lineage-distinct semantic packs per family
× A100
× 2 independent Agent replicates
= 48 high-budget trajectories + 48 nested anytime checkpoints
```

#### Tasks

- [ ] 按 frozen blocked-randomized schedule 对每个 Agent–task-pack–target–replicate cell 从 clean workspace 独立运行，不共享代码、trajectory 或 held-out evidence。
- [ ] 按 frozen event-ledger semantics 原子保存 `B_L` 只读 workspace checkpoint，再继续同一 trajectory 到 `B_H`。
- [ ] Trajectory 结束后，对 `B_L` snapshot 的 best development-visible candidate 独立运行 sealed qualification；没有 candidate 时记录 `no_candidate`。结果不回传 Agent。
- [ ] 对 `B_H` development-selected artifact 运行 sealed qualification 和独立 performance rerun；不向 Agent 回传 sealed 结果。
- [ ] 保存所有 success、failure、timeout、abstention 和 budget exhaustion artifacts。
- [ ] 每个 execution/time block 前后运行 provider sentinel conformance，并记录 worker thermal/clock/cache fingerprint。
- [ ] 所有三个 target 完成后，才按 frozen replicate aggregation 构造每个 Agent/semantic-task/hardware/checkpoint cell 的 ex-post target oracle 与 tie/ambiguous label。
- [ ] 以 semantic task 为 cluster，计算 qualification frontier、replicate stability、utility gap、timing-noise sensitivity、exploration cost share 和初步 Agent-conditioned crossover。
- [ ] 执行预注册 early-stop：完全无 target frontier 差异或全部 target 均不可达时直接 No-Go，不扩展 P2。
- [ ] 检查结果是否由 target-specific examples、compile cache、并发、服务 alias 变化或单一 shape 驱动。

#### Deliverables

- 48 条 sealed high-budget trajectories 与对应 anytime checkpoints。
- P1 oracle/crossover matrix、failure attribution 和 resource ledger。
- P1 early-stop report：continue、repair-and-repeat-as-new-study 或 No-Go。

#### Exit Criteria

- [ ] 每个预注册 P1 replicate 都有 terminal artifact，不删除失败样本。
- [ ] Oracle winner 只由独立 target runs 和独立 timing reruns构造，未进入 profile 或 prompt。
- [ ] 至少存在超过 timing noise/utility margin、且不是单一 task/replicate outlier 的 target frontier signal才进入 P2。
- [ ] 若信号只来自缺失 backend、unequal assets 或 evaluator bug，修复后必须使用新 study version 重跑。

**Status:** `pending`

### Phase 7: P2 Confirmation, Strong Baselines, and Gate R Decision

**Objective:** 扩展到第二 hardware 和第三 capability tier，运行强 portfolio baselines，并作出是否建设 progressive realization 的最终 pilot 决策。

**Full matrix after P2:**

```text
3 targets
× 3 Agent/model capability tiers
× 2 workload families
× 2 lineage-distinct semantic packs per family
× 2 hardware
× 2 independent Agent replicates
= 144 high-budget trajectories + 144 nested anytime checkpoints
```

#### Tasks

- [ ] 为第三个 capability tier 完成 P0.1 conformance、agent manifest、calibration 和 frozen profile；P1 已包含 cross-family pair，第三 tier 可优先使用 DeepSeek V4 Pro 或剩余 GLM/Qwen endpoint。
- [ ] 补齐 RTX 4090 上所有 Agent–semantic-task–target–replicate cells，并补齐第三 Agent 在 A100 上的 cells。
- [ ] 保留全部四个 workload–hardware blocks，包括没有 crossover 的 negative blocks。
- [ ] 运行 development-selected fixed target、static rules、feature-only router、model-ID router 和 capability-profile one-shot router。
- [ ] 按 Phase 5 frozen semantics 运行或 replay capability-filtered top-2 race、successive halving、Compiler/library selector、equal-resource all-target 和 idealized equal-deadline all-target lower bound。
- [ ] 构造 budget-matched per-Agent target oracle，只用于 ex-post regret 和 headroom 评价。
- [ ] 报告 qualification、relative-to-`B*`、worst-shape regression、actionable crossover、Brier/ECE、censored cost error 和分项 cost-to-qualified。
- [ ] 做 leave-one-block-out、timing/shape rerun、target asset audit，以及 hardware/Agent/budget 分轴分析。
- [ ] 以 semantic member 为 cluster 做 leave-one-member-out；hardware 作为 repeated measure，四个 blocks 只用于 pragmatic screen。
- [ ] 计算 calibration cold-start、break-even family size、failed exploration share 和 top-2/parallel 增量成本。
- [ ] 根据冻结条件输出 Gate R Go/No-Go；Gate R Go 只批准下一阶段 progressive policy，不等于论文 claim 已成立。

#### Gate R Go Conditions

- [ ] 四个 workload–hardware screening blocks 中至少两个出现由多个独立 semantic members 支撑的 actionable crossover，且不是 unsupported target 的静态过滤结果；replicates 只验证 member 内稳定性，不增加支持计数。
- [ ] 固定 Agent 时存在 workload/hardware frontier change；固定 workload/hardware 时至少一个 matched set 随 Agent 改变。
- [ ] Behavioral profile 相对 workload+hardware feature-only/rules 改善 qualification/cost prediction，超过预注册最小效应量。
- [ ] Calibration 在预注册 family size 内 break even。
- [ ] Fixed/rules/top-2/parallel 未共同支配复杂 policy 的可行空间。
- [ ] Target exploration 和失败尝试占 cost-to-qualified 的比例足以提供可测 headroom。

#### Gate R No-Go Triggers

- [ ] Winner 几乎固定，翻转只来自 timing noise。
- [ ] Agent profile 在 workload+hardware rules 外没有额外决策价值。
- [ ] Top-2 race 或 parallel all-target 的增量成本很低。
- [ ] 审计或平衡 primitives、examples、tools 和预算后 target-stack crossover 消失，无法支撑 target-selection headroom。
- [ ] 所有 Agent 在全部 targets 上均无法达到合理门槛，主要瓶颈是 generation capability。
- [ ] 主要失败来自缺失 backend/primitive，无法归因于 routing。
- [ ] Alias/version 波动使 ranking 剧烈变化，且重复 conformance/少量 probes 无法恢复。

#### Deliverables

- Full 144-trajectory dataset、nested anytime checkpoints 和 sealed result matrix。
- Fixed/rule/profile/top-2/successive-halving/parallel/oracle baseline report。
- Crossover、calibration、failure attribution、resource frontier 和 robustness analysis。
- `gate-r-decision.v1`：Go/No-Go、触发阈值、被排除机制、剩余风险和下一阶段边界。

#### Exit Criteria

- [ ] 所有 retained cells 和 baselines 已按 frozen protocol 完成或保留为 censored failure。
- [ ] Gate decision 可从 immutable artifacts 与 frozen analysis code 重建。
- [ ] Go 时创建新的 Gate P plan；No-Go 时收缩为 fixed/rule/top-2、Kernel generation 或 Compiler/backend project。
- [ ] 在 Gate R 决策前不实现完整 progressive switching policy。

**Status:** `pending`

## Target Schedule

当前仓库只有 provider harness，完整 pilot 的现实目标为 **10–16 周**。日历只用于资源协调，phase exit criteria 优先于周数：

- Weeks 1–2：Phase 1，protocol/manifests、worker/job/artifact contracts、inventory、support 与 security audit。
- Weeks 3–4：Phase 2，single-cell qualifier、Triton minimal adapter、sandbox recovery 和 generalized artifact bundle。
- Weeks 5–10：Phase 3 的 3–6 周 oracle/toolchain risk window；逐 target/workload/hardware 完成 known paths。
- Weeks 7–10：Phase 4 可在 Phase 2 contracts 稳定后并行开发 Agent runtime，但不得采集正式 trajectories。
- Weeks 9–11：Phase 5 cross-family conformance、calibration、budget dry run、baseline freeze 和 randomized schedule。
- Week 12：Phase 6 P1 的 48 条 trajectories、sealed qualification 和 early-stop decision。
- Weeks 13–14：Phase 7 P2 扩展至 144 条总 trajectories。
- Week 15：Frozen baselines/replays、independent reruns、leakage/asset/robustness audit。
- Week 16：Gate R decision、artifact freeze 和下一阶段边界。

若 Phase 3 发现 target/backend 不成熟，应延长 oracle readiness 或在解盲前收缩 targets，而不是并行采集不可解释的 Agent 数据。若已有可复现 oracle/toolchain assets，可压缩到 10–12 周，但不能跳过 exit criteria。

## Key Questions to Resolve During Execution

1. A100 与 RTX 4090 的 worker access、独占、时钟和 profiler 策略如何冻结？
2. CuTeDSL 是否能在 `sm80/sm89` 和两个 workload families 上形成公平、可复现的 third layer？
3. W4A8 的 packing/layout、scale granularity、accumulation dtype、epilogue 和 `B*` 应冻结为何种 contract？
4. RMSNorm-family 的两个独立 semantic members 具体选择 RMSNorm、residual-RMSNorm 还是其他变体，怎样证明不是轻微派生？
5. Qualification threshold、timing noise margin 和 actionable utility margin 的具体数值是什么？
6. `B_L/B_H` 在各资源轴上的 cap 经 baseline dry run 后应如何设置？
7. P1 external family 选择 GLM 还是 Qwen，P2 第三 capability tier 选择哪个 endpoint，怎样保证相同 orchestration 与可比 tool use？
8. Calibration profile 的最小效应量和 break-even family size 如何预注册？

## Decisions Made

- Pilot 优先发现 insight，固定 model snapshot 暂不作为启动门槛；服务 alias 作为版本风险显式记录。
- DeepSeek API 是 bring-up endpoint；正式 P1 必须加入 GLM/Qwen external family，OpenAI 不作为 Gate R 必需依赖。
- API 货币费用不进入初期 headline metrics；逻辑 tokens 与其他机器/人工资源分项记录。
- 第一个 vertical slice 使用 `RMSNorm-family × Triton × A100`。
- Candidate targets 为 Triton、TileLang、CuTeDSL；CuTeDSL 只能在正式 Agent 运行前替换。
- Candidate workloads 为 RMSNorm-family 与 W4A8 fused-dequant GEMM；dense GEMM fallback 只验证 harness。
- A100/RTX 4090 构成首轮 hardware strata；不把缺失 backend 写成跨硬件 routing 失败。
- Pilot 使用每 family 两个 lineage-distinct semantic packs、每 cell 两个 Agent replicates；P1/P2 总 trajectories 为 48/144。
- `B_L` 是 `B_H` anytime policy 的观察 checkpoint，不代表明确被告知低预算的 Agent。
- Qualification-first、serial wall-clock primary cost、resource caps 和 Pareto reporting共同定义 target oracle；不使用未定义的多维 `argmin`。
- Primary claim 是 Agent–target-stack fit，不把 stack difference 直接因果归为 representation。
- 先完成 oracle/qualification，再运行 Agent；先通过 Gate R，再实现 progressive policy。

## Errors Encountered

- 2026-07-14：第二段 patch 使用了重复的 `Status: pending` 锚点，Phase 3 被追加到文件末尾。改用唯一的 `Phase 4` heading 锚点后恢复正确顺序。

## Planning File Protocol

- `task_plan.md`：phase、status、exit criteria、决策和错误；每个 phase 完成后更新。
- `findings.md`：实验发现、support audit、oracle availability、风险和新决策；每两次查询/查看后更新。
- `progress.md`：实际执行、文件变化、测试、artifact ID 和错误日志；工作过程中持续更新。
- 每次重大决策前重读本文件；任何 protocol 变更都记录 rationale，并在正式数据开始后递增 study version。
