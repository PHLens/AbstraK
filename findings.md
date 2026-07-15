# Findings and Decisions: AbstraK Gate R Pilot

## Requirements Captured

- 先写完整 pilot plan，再继续实现。
- 使用 Planning with Files，将长期状态保存在项目根目录的 `task_plan.md`、`findings.md` 和 `progress.md`。
- 内容较长时分段写入。
- Pilot 目标是尽快判断研究假设是否有 insight，不把固定 model checkpoint/ID 当作启动门槛。
- 初期使用 DeepSeek API；OpenAI 后续按需要加入。
- 初期不以 API 货币费用为评价指标，主要记录逻辑 token 和其他资源消耗。
- 可用 GPU 是 NVIDIA A100 和 RTX 4090，时间预算不构成硬限制。
- Pilot 必须依次回答 target selection 是否必要、Agent information 是否有 context-only 之外的增量价值，以及差异能否被低成本预测和利用。

## Current Repository State

- `main` 已推送并与 `origin/main` 对齐，开始本计划前工作区干净。
- P0.1 已实现 provider/model manifest、严格单次请求、LiteLLM transport、usage normalization、error taxonomy、artifact sealing、secret scanning 和 conformance CLI。
- 默认配置位于 `~/.abstrak/config.yaml`，认证位于 `~/.abstrak/auth.json`；本机文件不是仓库内容。
- DeepSeek endpoint 为 `https://api.deepseek.com/v1`。
- DeepSeek V4 Flash/Pro 已完成真实 smoke probe；provider input tokens 只比可见内容的本地估计多约 7–8 个 chat-template tokens，没有 TokenFlux 式固定 4.4k 隐藏上下文迹象。
- 当前仓库没有 workload contract、GPU worker protocol、target adapter、qualifier、Agent loop、trajectory ledger 或 crossover evaluator。

## Proposal-Derived Research Constraints

- R.1 是 selection premise：即使允许 hindsight 选择最强 fixed target，它仍必须在 qualification coverage 或 cost-to-qualified 上显著落后于 cell oracle；否则没有 adaptive target-selection 问题。
- R.2 是 Agent premise：Agent-conditioned selector 必须在固定 workload/hardware 后，继续优于 context-only selector；否则使用 deterministic workload/hardware rules。
- R.3 是 realization premise：behavioral profile 必须在计入 calibration 与 exploration cost 后优于 identity/rules/top-2/parallel baselines；只有 R.1–R.3 均通过才进入 progressive realization。
- 首轮 target 候选为 Triton、TileLang、CuTeDSL；如果第三层没有 representation oracle，必须在正式 Agent 结果前替换或收缩 claim。
- 首轮 workload 应包含一个 memory/data-movement family 和一个 compute/representation-dominated low-precision family。
- 首轮 hardware 应是已有 backend 支持且存在 known qualified path 的环境；缺失 intrinsic/lowering/backend 属于 capability extension，不属于 router failure。
- Qualification 必须同时满足 correctness、robustness、integration constraints 和 relative-to-`B*` performance threshold。
- Representation oracle、budget-matched per-Agent target oracle 与 production oracle `B*` 解决不同问题，不能互相替代。
- Calibration、development、screening holdout 与 sealed qualification 必须按 semantic family、source lineage、作者和派生关系隔离。
- Shape、timing seed、target realization 和同源派生实现是重复测量，不是独立样本。
- Pilot 的四个 workload–hardware blocks只支持 kill/continue 与效应量估计，不支持论文级总体显著性主张。

## Experimental Design Findings

### Immediate Vertical Slice

最小的可靠下一步是 `RMSNorm-family × Triton × A100` 的 qualifier/oracle vertical slice。它不调用 LLM，先证明共同评价基础可以区分：

- known correct and fast；
- numerically incorrect；
- shape/corner-case fragile；
- correct but below performance threshold。

这一步通过后，才能把相同 qualification contract 扩展到其他 targets、W4A8 和 RTX 4090。

### Candidate Workloads

- Memory family 候选：RMSNorm，必要时把 residual-RMSNorm 作为独立 semantic member，而不是仅增加 shape。
- Low-precision family 候选：W4A8 fused-dequant GEMM。
- W4A8 在冻结前需要确定 weight packing、scale granularity、activation/accumulation dtype、layout、epilogue、shape pack 和 production baseline set。
- 如果 W4A8 oracle audit 暂时失败，dense/fused GEMM control 只能验证 harness，不得支撑 emerging-format claim。

### Candidate Targets

- Triton：block/tile DSL。
- TileLang：更显式的 tile/data-movement DSL。
- CuTeDSL：更显式的 layout/instruction path。
- CuTeDSL 对 `sm80/sm89` 和两个 workload 的支持必须实测，不能从文档名称推断。
- CUDA C++ 只在 CuTeDSL 无法形成公平第三层且已有 known oracle 时替换，且替换发生在正式 Agent 数据之前。

### Hardware

- A100 (`sm80`) 与 RTX 4090 (`sm89`) 都是 NVIDIA GPU，减少跨 vendor toolchain 差异，但数据中心/消费卡、memory subsystem 和 profiler/clock controls 不同。
- Worker manifest 必须记录 driver、CUDA、Compiler、clock/power state、独占策略、container digest 和 profiler availability。
- Controller 不应假定本地存在 GPU；worker request/result 和 environment fingerprint 必须是显式接口。

### Agents and Model Identity

- DeepSeek V4 Flash/Pro 适合 infrastructure bring-up，但属于同一模型 family。
- 正式 P1 必须使用一个 DeepSeek tier 加一个 GLM/Qwen external family；P2 再加入 DeepSeek Pro 或另一 endpoint 作为第三 capability tier。OpenAI 不是必需依赖。
- Pilot 接受 mutable service alias，但每条 trajectory 必须记录 requested model、returned model、provider、UTC timestamp、agent/config hash 和相邻 conformance artifact。
- Alias 变化是 robustness/version-risk 变量。若 ranking 因 alias 漂移系统性改变且少量 probes 无法恢复，将触发 No-Go 或 claim shrink。

### Resource Accounting

- 初期不计算 API 货币费用。
- 同时记录 provider-native input/output tokens 和统一 tokenizer 对 visible transcript 的 token 估计。
- 单独记录 Agent turns、compile/test/profile runs、GPU time、wall time、最大并发、target switching 和 human intervention。
- 不把这些资源压成任意加权单分数；equal-resource 和 equal-deadline 结果分别报告。
- `B_L` 是 `B_H` trajectory 中的只读 checkpoint，避免两个预算档独立重跑引入额外随机性。

## Key Technical Decisions

- 使用 manifest-driven design，而不是在 Python 代码中硬编码实验矩阵。
- Manifest schemas 先覆盖 hardware、target、task、agent、budget 和 study，所有交叉引用与 hash 可离线验证。
- 先建立 single-cell qualifier，再扩展 adapters/oracles，最后接 Agent loop。
- Agent loop 初期使用共同 textual action/tool protocol、non-streaming、single candidate、no fallback。
- 正式 RQ1 数据中每个 target 独立运行；target oracle 仅在所有 runs 和独立 timing rerun 完成后 ex post 构造。
- Sealed qualification 运行在独立 process/workspace，不向 Agent 返回逐例结果。
- Failure taxonomy 至少区分 representation、compiler/backend、agent、profile、routing 和 environment/evaluator failure。
- Baseline 必须形成递进 ladder：global fixed、per-Agent fixed、workload/hardware-only、加 Agent identity、加 behavioral profile、top-2/parallel 和 oracle；每层只声明相对直接前级的增量价值。
- R.1 同时使用 global hindsight fixed 和更强的 per-Agent hindsight fixed。前者失败只证明 portfolio 有价值；后者仍失败才证明 workload/hardware/task-local selection 有价值。
- A100 P1 negative result 不能否定未测量的 hardware effect；两个 Agents 跨 A100/4090 的 mandatory 96-trajectory matrix 完成后才判定整体 R.1，第三 Agent 是可选外部有效性扩展。
- Full progressive switching policy 不在 Gate R pilot 前实现，避免先造复杂系统再寻找 headroom。

## Go/No-Go Logic

### R.1 Target-Selection Necessity

- 对相同 matched cells 比较 calibration fixed、global/per-Agent hindsight fixed 与 qualification-first cell oracle。
- Headline 使用 qualification coverage deficit、actionable opportunity rate 和 conditional serial-wall-clock regret；winner label diversity 只作支持证据。
- 至少两个 lineage-isolated semantic members、至少两个 non-tied winners 支撑超过 `delta_Q` 或 `delta_C` 的差异，才构成 Go。
- Workload 和 hardware matched contrasts 分别报告；只通过一个时收缩 claim。Unsupported backend、asset imbalance、tie 和 timing noise 不计作 evidence。

### R.2 Agent Incremental Value

- 固定 workload/hardware 后，Agent-dependent target frontier change 必须由至少两个独立 semantic contexts 支撑。
- Agent-conditioned selector 必须相对 workload/hardware-only selector 产生超过预注册 margin 的 qualification 或 cost 增益。
- 若不通过，保留 R.1 characterization，系统收缩为 deterministic context selector。

### R.3 Low-Cost Predict-and-Exploit

- Behavioral profile 必须优于 Agent identity 与 context-only baselines，且 calibration 在预注册 family size 内 break even。
- Top-2、successive halving 或 parallel all-target 若以等价或更低总成本获得结果，则不建设复杂 progressive policy。
- 三层均通过才创建 Gate P；否则分别落到 fixed、per-Agent default、context rule 或 direct exploration。

## Open Questions

- A100 和 RTX 4090 分别位于什么 worker，如何获得独占与稳定时钟？
- Triton、TileLang、CuTeDSL 的具体冻结版本和容器策略是什么？
- RMSNorm 是否加入 residual variant，还是保持单一 semantics 跨 shapes？
- W4A8 contract 和最强 applicable `B*` 如何选择？
- Qualification threshold、noise margin、utility margin 和 tolerance 的具体值是多少？
- `B_L/B_H` 各资源轴的 cap 应根据什么 baseline dry run 冻结？
- 可选 P2b 第三 capability tier 使用 DeepSeek Pro 还是剩余 GLM/Qwen endpoint？
- A100 还是 RTX 4090 作为 P1 first hardware；当前计划默认 A100。

## Risks and Mitigations

- **Target support 不完整**：先做 oracle audit；正式 Agent 结果前 replace/shrink，不把 backend 缺失归因于 Agent。
- **W4A8 scope 过重**：允许 dense GEMM 仅作 harness control，但量化 claim 保持关闭。
- **服务 alias 漂移**：记录完整 provenance、运行前后 conformance 与时间分层；不宣称 checkpoint reproducibility。
- **Target assets 不公平**：冻结 documentation/examples/primitives policy，记录 standard-library 和 autotuning capability。
- **Leakage**：按 lineage grouped split，oracle source 不进入 prompts、retrieval 或 calibration。
- **Pseudo-replication**：以 semantic task pack/workload–hardware block 为决策单位，shape/seed 仅作 repeated measurement。
- **基础设施先行失控**：每个 phase 设 exit criteria；Gate R No-Go 时停止 router/IR 扩张。

## Resources

- Repository overview: `README.md`
- Provider conformance: `docs/p0.1-provider-conformance.md`
- Runtime config contract: `configs/README.md`
- Proposal root: `/home/cambricon/Nutstore Files/LoreForgeWiki/Spaces/Research/Proposal/agentic-kernel-engineering/proposal.md`
- Research questions/system: `/home/cambricon/Nutstore Files/LoreForgeWiki/Spaces/Research/Proposal/agentic-kernel-engineering/04-research-questions-and-system.md`
- Evaluation contract: `/home/cambricon/Nutstore Files/LoreForgeWiki/Spaces/Research/Proposal/agentic-kernel-engineering/05-evaluation-and-baselines.md`
- Go/No-Go gates: `/home/cambricon/Nutstore Files/LoreForgeWiki/Spaces/Research/Proposal/agentic-kernel-engineering/06-reality-check-and-go-no-go.md`
- Roadmap: `/home/cambricon/Nutstore Files/LoreForgeWiki/Spaces/Research/Proposal/agentic-kernel-engineering/07-research-roadmap.md`
- Pilot protocol: `/home/cambricon/Nutstore Files/LoreForgeWiki/Spaces/Research/Proposal/agentic-kernel-engineering/08-agent-representation-fit-pilot.md`

## Issues Encountered

- 第二段 `task_plan.md` patch 使用了非唯一状态行作为锚点，Phase 3 被临时放到末尾；已改用唯一 heading 锚点修复，并记录在 plan/progress。

## Planning Verification

- Planning files 已按长内容分段写入；精确行数在每次大修后重新检查，不作为 protocol 内容。
- `task_plan.md` 恰好包含 Phase 1–7，顺序正确且只有 Phase 1 为 `in_progress`。
- 修订后的 P1 arithmetic 为 48 条 high-budget trajectories 加 48 个 nested anytime checkpoints；mandatory dual-hardware P2a 为 96 加 96，可选第三 Agent P2b 后为 144 加 144。
- Whitespace/EOF scan 通过，三份文件均以单个 newline 结束。
- 三份文件没有 Obsidian-style internal links、Markdown tables 或 Mermaid，避免已知渲染问题。
- 三个 planning files 已纳入版本控制；本轮仍只修改这些 planning files，没有代码、proposal 或本地认证配置变更。
- 两路独立审查的 critical/major findings 已转化为 plan contracts，而不是只保留为评论：replication、budget semantics、alias eligibility、decision utility、cross-family P1、target-stack claim、worker security、baseline freeze 和 realistic schedule。

## Independent Engineering Review Findings

以下问题必须在 plan 定稿前修正：

- 当前 12/36 matrix 等价于每个随机 cell 一次 trajectory，无法区分 model sampling/alias drift 与 representation effect，也不足以支撑 qualification probability、Brier/ECE 或层次模型。
- Pilot 应在每个 workload family 内加入至少两个 lineage-distinct semantic task packs，并为每个 Agent–task–target–hardware cell 运行至少两个独立 replicate。初版因此得到 P1 48、含第三 Agent 的 full matrix 144；后续分层 Gate 将双 Agent、双 hardware 的 96 条设为最低完整矩阵，第三 Agent 作为可选扩展。
- 从同一 `B_H` trajectory 截取的 `B_L` 只能称为 anytime/resource checkpoint，不能声称是“知道低预算”的 budget-conditioned Agent。若以后研究 budget-aware behavior，必须单独运行明确告知 `B_L` 的策略。
- 多轴预算必须定义 event-boundary reservation、in-flight overshoot、hard cap 和 terminal classification。
- P0.1 的严格 `pilot_ready` 会拒绝 mutable alias。Exploratory pilot 应新增 study-scoped eligibility：要求 transport/action protocol ready、完整 alias provenance、batch 前后 conformance 和显式 reproducibility warning；不能静默改变 `pilot_ready` 的含义。
- Target winner、tie、regret 和 break-even 需要冻结决策规则。建议 qualification 为硬约束，以 isolated serial wall-clock-to-qualified 为 primary objective，并对 tokens/GPU/human 等其他资源施加 cap、报告 Pareto；noise 内使用 worst-shape score 等预注册 tie-break。
- Phase 2 已需要 worker protocol、minimal adapter、failure taxonomy 和 generalized artifact bundle，因此这些 contract 必须前移到 Phase 1/2，Phase 3 只扩展到其他 targets/cells。
- Provider-specific `ProviderArtifactStore` 不适合承载 source trees、patches、Compiler logs/binaries 和 nested worker artifacts。需要新的 content-addressed `artifact-bundle.v1`，包含 atomic incomplete/terminal state 与 checksum。
- 运行模型生成代码前必须建立 worker isolation、network/filesystem/process/resource boundary、timeout/cancel、GPU OOM/hang health check、reset/quarantine、cleanup、idempotent job ID 和 artifact transfer checksum。
- Top-2、successive-halving、parallel 等 baseline 的 selection/allocation/replay semantics 必须在 P1 前冻结。Ideal equal-deadline parallel 可作为隔离 trajectories 的 counterfactual lower bound，但不能伪装成单 GPU 上的实测并行 latency。
- 执行顺序需要按 task/Agent/target blocking 与 randomization，配合 batch 前后 provider conformance、serial GPU measurement、compile/autotune cache isolation 和 transient infrastructure failure reschedule policy。
- `hardware.v1` 应只存稳定 capability；新增 `worker.v1` 存配置环境，每条 run 记录 observed environment fingerprint，避免把易变状态混入 frozen hardware identity。
- Workload contract 必须冻结 semantic member、all-shape correctness、per-shape `B*` envelope、performance aggregation、worst-shape floor 和 N/A rule 后才能计算样本量。
- 当前仓库从 provider harness 起步，8 周过于乐观；更现实的 pilot 为 10–16 周，其中 oracle/toolchain Phase 3 单独预留 3–6 周风险窗口。

## Independent Scientific-Rigor Review Findings

- P1 只使用 DeepSeek Flash/Pro 与 proposal 的 cross-family screening 要求冲突。正式 P1 应使用 DeepSeek 加一个 GLM/Qwen family；DeepSeek Pro 可作为 P2 第三 capability tier。Bring-up 仍可先只用 DeepSeek。
- “Representation effect”在当前设计中混合了 DSL、Compiler/backend maturity、standard primitives、autotuner、documentation 和 examples。Primary claim 应写成 Agent–target-stack fit；representation 只是 stack 的一个因素，除非额外做 asset factorial ablation。
- 四个 workload-family×hardware blocks 是 pragmatic Gate R screen，不是四个完全独立科学样本。同一 semantic member 的 A100/4090 结果应视为 hardware repeated measures；lineage-distinct semantic members 才是 confirmatory statistical units。
- 两个 calibration probes/node 不足以支持可靠 ECE 或复杂 hierarchical calibration。Pilot 应增加初始 probe 数、定义 prediction instances 和 cross-validation，并将 ECE 降为满足最小样本量时才报告的 exploratory metric。
- Leakage gate 需要 operational checks：冻结 oracle/docs/examples/retrieval hashes，做 exact/near-duplicate overlap scan，并定义 semantic hierarchy，避免 calibration task 虽不同 repository 但仍是 RMSNorm/W4A8 的近派生。
- Qualification 必须冻结一个 headline `tau` 和围绕 `B*` 的 indifference/ambiguity band；0.8/0.9/0.95 只作为 sensitivity。Timing/baseline uncertainty 进入 qualification 与 tie 判定。
- P1/P2 autonomous primary trajectories 禁止任何人工 hint、patch 或 takeover；人工工作只属于 excluded shakeout 或单独 intervention study。
- Agent stop condition 只能由 development-visible candidate gate 触发；sealed qualification 不能影响停止。
- 一个 family implementation 是否允许 shape dispatch/autotuning，以及其 hidden search cost 如何计费，必须在 task/target contract 中冻结。

## Final Re-Review Resolutions

- Target oracle 改为 Agent×semantic-member×hardware×checkpoint 的 cell-level aggregation，不再跨 targets 配对 arbitrary replicate IDs。
- 每个 target 的两个 replicates 先归类为 stable-qualified、unstable 或 failed；只有 stable-qualified targets 进入 primary oracle，单 run winner 仅作 debugging。
- 每个 family 的两个 members 必须语义不同且 lineage-isolated，并冻结 derivation graph；不同来源或 shapes 本身不产生独立单位。
- Gate R 只按独立 semantic members 计数，replicates 只检验 member 内稳定性；1/2 disagreement 保守标为 unstable。
- Wall-clock、worst-shape 和 token tie-break 各自都有 practical-equivalence/uncertainty margin，微小差异不会制造 categorical winner。
- CuTeDSL final retain/replace 从 Phase 1 移到 Phase 3 exit；Phase 1 只冻结决策协议。
- `B_L` snapshot 在 trajectory 结束后单独 sealed qualify；无 candidate 明确记录，不再只有未评测 checkpoint。
- Replicate ID 从 `agent.v1` 移入 study/run manifest，保证同一 Agent identity 跨 repeats 不变。
- Hostile-code termination、escape prevention 和 worker health/quarantine 成为 Phase 2 硬 exit gate。
- Ambiguous crash boundary 只保证不自动 duplicate request，并记录 `possibly_charged`，不承诺 exactly-once provider billing。

## Final Mechanical Verification

- `task_plan.md` 包含且仅包含七个有序 phases；Phase 1 是唯一 `in_progress`，其余六个为 `pending`。
- P1、mandatory P2a、optional P2b 公式分别得到 48/96/144 条 high-budget trajectories，并为每条保留一个 nested anytime checkpoint。
- Cell-level oracle、replicate stability、semantic-member independence、`B_L` sealed qualification 和 cross-family P1 均可从 plan 中直接定位。
- Whitespace/EOF scan 通过；planning files 不包含 Obsidian-style internal links、Markdown tables 或 Mermaid。
- 本轮工作区只修改 `task_plan.md`、`findings.md` 和 `progress.md`，没有代码、proposal、artifact 或认证文件变化。

## Naive KernelBench Screen Implementation

- AbstraK 与 pinned KernelBench 现在统一固定为 CPython `3.10.*`；项目已移除 `StrEnum`、`datetime.UTC`、stdlib `tomllib` 和 PEP 695 泛型等 3.11/3.12-only 用法，不再需要双 Python sidecar。
- 首个实现不建设通用 Agent loop，而是严格分离单次生成 controller 与 GPU evaluator。Controller 只读取 pinned KernelBench task/prompt，不导入 Torch 或执行候选代码。
- KernelBench 固定在 commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`；只接受官方 `zero_shot` component sequence，checkout commit 不符或有本地修改时 fail closed。
- Bring-up matrix 为 DeepSeek V4 Flash/Pro × KernelBench `triton`/`tilelang`/`cute`。Smoke 是 square GEMM 的 6 cells；screen 增加 batched GEMM、LayerNorm 和 GEMM+Add+ReLU，共 24 cells。
- 每 cell 恰好一个 user message、一个 provider call、`turn_index=0`、temperature 0、8192 completion tokens、plain-text output；不会修改 `~/.abstrak/config.yaml` 中 P0.1 的 128-token/plain-JSON profile。
- 生成与评价使用独立 sealed bundles。评价前验证生成 checksum，显式要求 generated-code execution acknowledgement，并从 worker 环境移除常见 credential variables。
- Worker 记录 device、Python、Torch/CUDA 和可发现的 Triton/TileLang/CuTe package version；GPU 环境和正式 hostile-code isolation 仍待后续提供与实现。
- 初步 summary 按 Agent profile/target 报告 compile、correctness、performance coverage、relative-to-PyTorch ratio 与 `fast_1/fast_2`。单 replicate 结果只作 descriptive screen；negative result 不属于 equivalence evidence。
- 离线 smoke/screen 均能在 pinned checkout 上稳定解析 task 和生成 prompt hashes；没有进行 billable API 请求或 GPU 执行。

## A100 Worker Environment Bring-Up

- 远端 worker 为 Ubuntu 24.04、两张 A100-SXM4-80GB（SM80）、NVIDIA driver `575.57.08` 与系统 CUDA `12.6`。
- KernelBench 明确要求 Python 3.10；实验环境因此统一为 CPython `3.10.20`，不使用镜像中的 Python 3.12/Torch development 环境。
- PyTorch 锁定为当前 stable `2.13.0+cu126`，同一环境中锁定 Triton `3.7.1`、TileLang `0.1.12`、CuTe DSL `4.6.1`、`cuda-python`/`cuda-bindings` `12.9.5`。
- 持久卷只保留 AbstraK/KernelBench source、Python 3.10 runtime、portable Git、lock 和经 SHA-256 校验的离线 wheel archive；GPU venv、wheel staging 和 uv cache 在每次容器重建后放到 `/tmp`。这个分层避免 NFS 上的大量 metadata/import/JIT 开销。
- `scripts/bootstrap-a100.sh` 校验 KernelBench commit、wheel archive hash、精确 package versions、driver 下界、SM80 设备和 Torch FP16 CUDA 运算，任一不符即 fail closed。
- KernelBench 自带的 trusted add examples 在 Triton、TileLang 和 CuTe DSL 上均已 compile 且 correctness 通过，并完成 candidate/reference timing。这些只证明 toolchain 可用，不作为 Agent 性能结果。
- Remote AbstraK test suite 为 68 passed，Ruff 通过。本轮没有 API 调用，也没有执行模型生成的代码。
- Bring-up 时远端尚无 `/root/.abstrak/config.yaml`；live smoke 期间已以 mode `0600` 临时部署本地 config/auth，不纳入仓库或持久 artifacts。worker 仍未安装 Docker，故 hostile generated-code isolation 仍是正式 pilot 的独立 exit gate。

## First Live Naive Interaction Smoke

- Frozen study hash 为 `d09216ec7035707c803e8cde90540f68dee2ac506123fb72010b3c281e5b4341`，有效 run 为 `20260715T110357.954897Z-e3afdf778b`。矩阵是 DeepSeek V4 Flash/Pro × Triton/TileLang/CuTe × KernelBench Level 1 square GEMM，每 cell 一次请求。
- 远端 worker 到 `api.deepseek.com` 的未认证 HTTP 请求也被 connection reset；failed run `20260715T110146.001822Z-c23e15442b` 的 6 个 provider errors 因此归为 infrastructure，不计入 Agent 结果。六次均已 `request_submitted=true`，所以保守记为 `possibly_charged`，没有在同一 run 中静默 retry。有效 run 由本地 controller 生成，经 SHA-256 校验的 archive handoff 到 A100 evaluator。
- 六个 candidates 均生成且 finish reason 为 `stop`。每次 input 为 409 tokens，CuTe prompt 为 427；output 范围为 2,515–6,500 tokens。
- Flash Triton 为 `0.620 ms`，PyTorch reference 为 `0.518 ms`，ratio `0.835x`；Pro Triton 为 `0.781 ms`，reference 为 `0.517 ms`，ratio `0.662x`。两者均 compile 且通过 5 次 correctness trials，但均未快于 reference。
- 两个 TileLang candidates 均缺少 `@T.prim_func`。关闭 static check 的独立诊断确认 Flash 使用了不存在的 `tilelang.kernel`，Pro 尝试导入不存在的 `tilelang.kernel`。
- 两个 CuTe candidates 均被 static checker 标记缺少 `cute::`/`cutlass::`。关闭 static check 后仍失败：Flash 使用 placeholder include/build path；Pro 的 extension 使用 C++14 编译当前 PyTorch headers，且 binding 缺少 declaration。因此不是单纯 static-check false positive。
- Known-good Triton/TileLang/CuTe examples 在同一 worker 上已通过，所以这是 naive prompt 下的 Agent–target-stack familiarity/API floor，不是 backend unavailable。
- 两个 profiles 都只有 Triton 达到 correctness，target ranking 没有反转；该 run 没有支持 R.2 Agent incremental value，也不能证明两个 Agents 等价。由于 2/3 targets 呈现 floor，直接扩展 24 cells 只会复制混杂因素。
- 下一个 study version 应先消除 floor：冻结且平衡 target-specific API documentation/non-solution examples，或替换为模型可稳定表达的 target stacks；若要评价 R.2，还必须加入 GLM/Qwen external family。

---

本文件在任何 support audit、oracle discovery、baseline comparison 或实验结果出现后更新。外部网页与工具输出只写入本文件，不写入会被频繁注入上下文的 `task_plan.md`。
