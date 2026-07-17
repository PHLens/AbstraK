# AbstraK A100 R1 快速验证计划

## Summary

目标仅验证 provisional R1：

> 在固定 A100、Agent 和预算下，按 workload 选择 target stack，是否优于全局固定 target 和 per-Agent 固定 target？

本阶段不研究 Agent-aware selector、4090、动态或 multi-shape、W4A8、MoE、target switching 或 profiler counters。结果只能是：

- `positive_signal`：存在值得继续研究的 A100 selection opportunity。
- `provisional_negative`：矩阵有效且稳定，但 A100 上未发现信号，停止本阶段。
- `invalid_floor`：target、文档或 harness floor 导致无法判断。
- `inconclusive_instability`：Agent replicates 或 timing 不稳定，无法判断。
- `inconclusive_infrastructure`：provider/worker failure 使关键 cells 缺失，无法判断。

## Experiment Contract

### Task Packs

每个任务使用一个公开给 Agent 的静态 shape，允许 shape-specialized kernel。

- `rmsnorm-static`：row-wise RMSNorm，shape `(4096,4096)`，FP16 I/O、FP32 accumulation、gamma、`eps=1e-5`。
- `layernorm-static`：row-wise LayerNorm，同 shape，FP16 I/O、FP32 mean/variance、gamma/beta。
- `gemm-static`：FP16 GEMM，`(M,N,K)=(1024,4096,4096)`，FP32 accumulation。
- `gemm-bias-relu-static`：相同 GEMM shape，加 FP16 bias 和 ReLU epilogue。

这四个 task packs 是四个固定实例，不代表 shape generalization 或完整 workload-family generality。所有输入由 task manifest 中的冻结 seed 生成：算子输入使用 `[-1,1]` 内的有限 FP16 bounded-uniform values，Norm 的 gamma 使用 `[0.5,1.5]` 内的正值、beta 使用 `[-1,1]` 内的 values；reference 使用 FP32 accumulation 后转换为约定的 FP16 输出。每个 task pack 还冻结一个零值或常量输入 corner case。

每轮 dev test 使用两个固定种子；sealed qualification 使用四个不同隐藏随机种子和一个隐藏 corner case。正确性要求五个 cases 全部通过、输出无意外 NaN/Inf、无输入修改，默认 `atol=rtol=1e-2`。Task oracle 建立前必须确认该容差能接受 known-correct implementation，并冻结在 `TaskPackSpec` 中。

### Target Stacks

初始为 Triton 3.7.1、TileLang 0.1.12、CuTe DSL 4.6.1。Target 定义包含 DSL、compiler、标准 primitives、autotuner 和冻结文档，不将结果单独归因于语言。

每个 target card 使用相同字段结构，最多一个与测试集无关的 VectorAdd 示例，包含版本、scaffold、import/build/launch、dtype/device、允许 primitives 和常见错误。

正式运行前，每个 `task × target` 必须有不可见的 expert oracle：

- 通过全部五个 sealed cases。
- 核心算子不得 fallback 到 PyTorch。
- 三次 clean-process timing，5 warmups、100 CUDA-event trials。
- timing CV 不超过 5%；超出时允许完整重测一次，仍超出则阻塞。

任一 CuTe scientific task 无法建立 oracle，整列在 Agent 数据生成前替换为预先准备的 CUDA target。CUDA replacement 必须先完成 compiler inventory、target card、adapter 和全部四个 task oracles，再重新运行 shakeout 并生成新的 study hash。Triton 或 TileLang 缺失 oracle 时直接阻塞，不使用稀疏矩阵。

共同基线 `B*` 为每个任务上 PyTorch eager、`torch.compile(mode="max-autotune-no-cudagraphs")` 和适用 ATen/vendor 路径的最快稳定结果。Expert target oracle 只用于计算 target realization，不进入 `B*`。

## Harness Changes

保留现有 naive screen，新增独立 `abstrak-canary-study.v1`，不原地扩展旧 schema。

新增接口：

- `TaskPackSpec`：语义、shape、dtype、参数、种子、容差和 fallback policy。
- `TargetStackSpec`：版本、card/hash、允许资产、adapter 和 oracle。
- `CanaryStudySpec`：Agents、targets、tasks、预算、replicates 和 schedule seed。
- `WorkerJob/WorkerResult`：幂等 job ID、输入 hash、compile/test/timing 结果。
- `TrajectoryEvent`：request、candidate、feedback、tokens、GPU/wall time 和 terminal state。

固定 Agent loop：

1. Agent 接收任务、精确 shape、A100 稳定规格和 target card。
2. 每轮输出完整 `ModelNew`，不使用 patch 或任意 shell。
3. Worker 返回结构化 compile error、dev correctness、最大数值误差和正确候选 latency。
4. 最多四次模型调用；Agent 可提前 `finish`。
5. 保存 `first_candidate` 和 `final_candidate`；提前结束时 final 是最后一个完整候选，无候选时显式记录 `no_candidate`。随后独立 sealed qualification，结果不反馈 Agent。

预算固定为 temperature 0、每轮最多 8192 output tokens、最多四次 dev evaluation、总墙钟 20 分钟。墙钟从第一次 provider request 提交前开始，到 final candidate 的 dev result 或 terminal failure 为止，包含 provider、SSH、compile/test 和 timing。只有 dev-correct 候选才 timing。未使用预算作为实际节省记录。

运行拓扑为本地 controller + SSH A100 worker。模型 API 和 canonical artifacts 留在本地；远端只执行 hash-verified job，并返回 sealed result。Worker 使用 `/tmp` ephemeral workspace、无凭据、无网络、非 root、无持久卷写权限、process-group timeout 和 GPU health check。完整 bundle 校验后镜像到持久 volume。

正式 study 前运行两个不计分且不与 target-card VectorAdd 示例重复的 canaries：一个小型 row-reduction+scale，一个小型 GEMM+bias，用于分别覆盖 reduction 和 matmul/epilogue 路径。每个 `canary × target` 必须先有 trusted expert path。

- `2 Agents × 3 targets × 2 canaries × 1 replicate = 12` 条 trajectories。
- 允许一次统一 target-card/harness 修订。
- 修订后重跑全部 12 条 shakeout，冻结全部 hashes。
- 如果某个 target 的 expert canaries 可用，但两个 Agents 在该 target 的两个 canaries 上全部失败，则记为 per-target Agent floor，不启动正式矩阵。

## Execution And Decision

### Test Procedure

1. 冻结四个 task packs、三个 target cards、Agent prompt/config、预算、worker environment 和全部 hashes。
2. 为每个 `task × target` 建立 expert oracle，并为每个 task 测量共同基线 `B*`。
3. 运行 12 条 shakeout trajectories；若统一修订 target card 或 harness，则完整重跑并重新冻结 study hash。
4. 按冻结 schedule 运行 48 条正式 trajectories，保存所有成功、失败和 censored terminal artifacts。
5. 对每条 trajectory 的 first/final candidate 独立执行 sealed correctness；对 qualified candidates 执行 clean-process timing。
6. 先聚合每个 target 的两个 replicates，再计算 hindsight fixed baselines、cell oracle、regret 和终局状态。

正式矩阵：

`DeepSeek Flash/Pro × 4 tasks × 3 targets × 2 replicates = 48 trajectories`

使用固定 schedule seed `20260717`，按 task/Agent/replicate block 随机化 target 顺序。Replicate 必须从 clean workspace 启动；不选择性补跑。请求提交后的异常保留为 terminal/censored，不静默 retry。

主要报告：

- `qualified@first`、`qualified@final`。
- 相对 `B*` 的每任务 runtime efficiency，定义为 `runtime(B*) / runtime(candidate)`，数值越高越好。
- target realization ratio，定义为 `runtime(expert target oracle) / runtime(candidate)`。
- calls、tokens、wall time、compile/test/timing GPU time。
- 2/2 stable、1/2 unstable、0/2 failed。
- `Fixed-Global-Hindsight`、`Fixed-Agent-Hindsight` 和 `(Agent, task)` cell oracle。
- Fixed-vs-oracle stable-qualified coverage gap，以及双方均 stable-qualified 时的 performance regret：`(efficiency_oracle - efficiency_fixed) / efficiency_oracle`。

`qualified` 表示五个 sealed cases 全部正确、核心算子无 PyTorch fallback、无输入修改且运行完成。Qualified candidate 使用与 expert oracle 相同的 clean-process timing protocol；timing CV 超过 5% 时允许完整重测一次，仍超出则该 performance result 记为 unstable。每个 target 的两个 replicates 聚合为 stable-qualified（2/2）、unstable（1/2）或 failed（0/2）。Cell oracle 只从 stable-qualified targets 中选择：median efficiency 相差超过 5% 时选更快者，差异不超过 5% 时记为 performance tie；performance tie 中只有 median calls 少至少一次时才记录 lower-effort preference，否则保持 tie。

`Fixed-Global-Hindsight` 在完整矩阵上事后选择覆盖 stable-qualified tasks 最多的单一 target；`Fixed-Agent-Hindsight` 对每个 Agent 分别做同样选择。Coverage 相同时先比较 median efficiency，再比较 median calls。二者只用于快速判断 task-local selection 是否存在，不解释为已经校准好的部署 policy。

`positive_signal` 必须同时满足：

- 三个 target 均通过 expert-oracle 和 shakeout gate，结果不是 target floor。
- 至少两个 task packs 出现稳定、非并列的 target frontier。
- 至少两个不同 target 分别在不同 task 上成为 stable、非并列 winner。
- Cell oracle 相对 `Fixed-Agent-Hindsight` 的实际收益覆盖至少两个不同 task IDs，不能只靠同一 task 在两个 Agents 上重复计数。实际收益指 fixed 未 stable-qualify 而 oracle stable-qualify、双方 stable-qualified 但 efficiency 相差超过 5%，或 performance tie 时 oracle 的 median calls 少至少一次。

只击败 `Fixed-Global-Hindsight`、但不能击败 `Fixed-Agent-Hindsight`，结论记为“per-Agent fixed upper bound 足够”，不支持 task-local selection；如何从新任务前的 calibration 选出该 default 不属于本快速计划。

如果完整矩阵可分析且结果稳定，但没有达到 `positive_signal`，则记录 `provisional_negative`。如果结论依赖 unstable cells 或 timing 复测仍不稳定，则记录 `inconclusive_instability`；如果缺失或 censored infrastructure cells 可能改变结论，则记录 `inconclusive_infrastructure`。这些终局都停止本快速阶段，不在同一 study 内追加任务、修改 target 或实现 selector。

### Expected Outcomes And Responses

| Observed phenomenon | Interpretation | Response |
| --- | --- | --- |
| Expert oracle 无法覆盖完整 `task × target` 矩阵 | target/toolchain 不可用 | 修复、按预注册规则替换 target，或记录 `invalid_floor`；不启动正式矩阵 |
| Expert canary 可用，但某 target 上两个 Agents 的两个 canaries 全部失败 | target documentation/Agent familiarity floor | 允许一次统一 card/harness 修订并完整重跑 shakeout；仍失败则 `invalid_floor` |
| 同一个 target 在所有 tasks 上稳定胜出 | workload selection 没有观察到额外价值 | `provisional_negative`，当前证据支持 fixed target |
| 每个 Agent 各有固定最优 target，但 Agent 内部不随 task 改变 | per-Agent fixed upper bound 足够 | `provisional_negative`，不支持 task-local selection |
| 不同 tasks 有不同 stable winners，且 cell oracle 在至少两个不同 task IDs 上击败 per-Agent fixed | 存在 workload-conditioned selection signal | `positive_signal`，由后续独立计划决定是否扩大 task/hardware/Agent 范围 |
| 关键 cells 为 1/2 unstable 或 timing 复测仍不稳定 | Agent/service 或 measurement variability 主导 | `inconclusive_instability`，不把未检出信号解释为 negative |
| Provider/worker failures 导致缺失 cells，且这些 cells 可能改变终局 | 基础设施不足 | `inconclusive_infrastructure`，修复后使用新 study version |
| 大部分 stable-qualified target 差异不超过 5% | 实际性能近似等价 | 记 tie；若 fixed 已覆盖则 `provisional_negative` |

## Implementation Batches

### Batch 1: Contracts And Fixtures

新增 schema、四个 task packs、target cards、oracle registry 和 cross-reference validation。

### Batch 2: Qualifier And Worker

实现 target adapters、dev/sealed evaluator、fallback 检查、`B*`、SSH job protocol 和隔离。Worker isolation 与 timeout/health recovery tests 通过后才能运行 shakeout。

### Batch 3: Agent Loop And Artifacts

实现四轮 state machine、first/final snapshots、budget ledger、hash sealing 和随机化调度。

### Batch 4: Analysis And Execution

完成 shakeout、冻结 study、运行 48 条矩阵并生成 R1 报告。

每批独立提交并通过后再进入下一批。

## Test Plan

- schema 未知字段、hash/cross-reference 错误和重复 cell 必须拒绝。
- fake provider/worker 覆盖 compile fail、wrong result、early finish、timeout、duplicate job 和 artifact tamper。
- correct、numerically wrong、PyTorch fallback、correct-but-slow 四类控制得到预期判定。
- Target oracle 和 `B*` 重复 timing 满足稳定性门槛。
- shakeout 可端到端重放，正式 study 使用冻结后的 card、prompt、task、target 和 worker hashes。
- Worker isolation tests 覆盖 network/filesystem escape、host timeout、GPU OOM 后 health check 和 quarantine。
- Hindsight fixed、oracle、5% tie、2/2 stability、distinct-task support 和五类终局判定均有合成 fixtures。

## Assumptions And Defaults

- 使用当前 A100 环境：CPython 3.10.20、PyTorch 2.13.0+cu126，以及上述 target 版本。
- 接受 mutable DeepSeek service aliases，但记录 requested/returned model、UTC、provider request ID 和配置 hash，并在正式 execution batch 前后运行一次 provider conformance sentinel。
- 只统计 tokens 和实际资源，不计算 API 美元费用。
- 本阶段只测固定静态 shape，不声称 shape generalization 或动态负载能力。
- 不使用 target-specific profiler counters、跨 trajectory memory 或人工干预。
- 本阶段结束后不自动实现 selector；后续工作由 R1 终局触发新的独立计划。
