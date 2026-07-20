# AbstraK A100 R1 实验结果

## 终局

预注册终局为 `inconclusive_infrastructure`，不是 `positive_signal` 或
`provisional_negative`。

- 48/48 formal trajectories 已运行并通过 artifact checksum 验证。
- 3 条 trajectories 被基础设施删失：2 次 provider 180 秒超时，1 次
  TileLang worker 1200 秒编译超时。
- `qualified@first = qualified@final = 19/48`。
- 19 个 qualified first/final pairs 在各 trajectory 内源码相同，因此实际执行
  19 份 clean-process timing；17 份稳定，2 份完整重测后仍不稳定。
- 24 个 `(Agent, task, target)` cells 中，6 个为 `stable_qualified`，5 个
  `unstable`，10 个 `failed`，3 个 `infrastructure_missing`。
- 所有 stable-qualified cells 都来自 Triton。Fixed-Global-Hindsight 和两个
  Fixed-Agent-Hindsight 均选择 Triton；cell oracle 对 Fixed-Agent-Hindsight 的
  coverage gap 为 0，actual-gain task 为 0。

因此，本矩阵没有观察到 workload-conditioned target selection signal；但删失和
qualification instability 足以改变负结论，不能将结果升级为
`provisional_negative`。

## 正确性结果

下表是 final-qualified trajectory 数，分母为每个格子的两个 replicates。

| Agent | Task | Triton | TileLang | CuTe |
| --- | --- | ---: | ---: | ---: |
| Flash | RMSNorm | 2/2 | 0/2 | 0/2 |
| Flash | LayerNorm | 1/2 | 0/2 | 0/2 |
| Flash | GEMM | 2/2 | 0/2 | 1/2 |
| Flash | GEMM+Bias+ReLU | 2/2 | 1/2* | 0/2 |
| Pro | RMSNorm | 1/2 | 0/2 | 0/2 |
| Pro | LayerNorm | 2/2 | 0/2 | 0/2 |
| Pro | GEMM | 2/2 | 1/2 | 0/2 |
| Pro | GEMM+Bias+ReLU | 2/2 | 1/2 | 1/2* |

`*` 表示同 cell 的另一个 replicate 是基础设施删失，不是普通 agent failure。

按轴汇总：

| 维度 | Qualified |
| --- | ---: |
| Flash | 9/24 |
| Pro | 10/24 |
| Triton | 14/16 |
| TileLang | 3/16 |
| CuTe | 2/16 |
| RMSNorm | 3/12 |
| LayerNorm | 3/12 |
| GEMM | 6/12 |
| GEMM+Bias+ReLU | 7/12 |

Shakeout 为 6/12；三个 targets 均至少有一条 Agent canary 成功，因此通过冻结的
target-floor gate。Expert oracle 为 12/12 stable；其中 fused Triton oracle 首轮
process CV 超过 5%，按协议完整重测后稳定。Baseline gate 为 7/12 stable，但每个
task 都存在至少一个 stable baseline，故 `B*` 完整。

## 性能结果

共同基线 `B*`：

| Task | B* variant | Median latency |
| --- | --- | ---: |
| RMSNorm | compile | 0.050176 ms |
| LayerNorm | vendor | 0.070656 ms |
| GEMM | vendor | 0.238592 ms |
| GEMM+Bias+ReLU | vendor | 0.324608 ms |

六个 stable-qualified cells 的预注册聚合结果：

| Agent | Task | Target | Candidate median | Median efficiency | Target realization |
| --- | --- | --- | ---: | ---: | ---: |
| Flash | RMSNorm | Triton | 0.069120 ms | 0.778x | 0.794x |
| Flash | GEMM | Triton | 0.248832 ms | 0.968x | 1.014x |
| Flash | GEMM+Bias+ReLU | Triton | 0.260096 ms | 1.254x | 0.914x |
| Pro | LayerNorm | Triton | 0.058112 ms | 1.216x | 0.899x |
| Pro | GEMM | Triton | 0.283136 ms | 0.844x | 0.884x |
| Pro | GEMM+Bias+ReLU | Triton | 7.392512 ms | 0.757x | 0.552x |

Efficiency 和 realization 按 trajectory ratio 后取 replicate median，因此不保证等于
`reference / candidate-median`。Pro fused/Triton 两条正确实现分别为 0.217600 ms
和 14.567424 ms，暴露出极大的 agent-sample 性能方差；预注册协议没有定义跨
replicate latency-CV gate，所以保留其 2/2 qualification 并报告 median，不在看到
结果后追加阈值。

两个 timing-unstable 的正确候选分别是 Flash GEMM/CuTe 和 Pro GEMM/TileLang。
其余低层 target 的单次正确实现也没有形成任何 2/2 stable-qualified cell。

## 能力与 DSL 假设

本实验不能验证“Agent 越强越适合低层 DSL，Agent 越弱越应选择高层 DSL”：

1. Flash/Pro 是同一服务 family 的 mutable aliases，本实验没有独立校准出可靠、固定的
   capability ordering。
2. Pro 仅比 Flash 多一条 qualified trajectory（10 vs 9）；两者在 Triton 上都是
   7/8，在 TileLang+CuTe 上分别是 3/16 和 2/16，没有出现稳定的能力依赖 reversal。
3. TileLang 和 CuTe 没有任何 2/2 stable-qualified cell，因而无法比较“强 Agent 的低层
   winner”和“弱 Agent 的高层 winner”。
4. Target 是完整 stack，包含 DSL、compiler/backend、primitives、文档和 API 熟悉度；
   当前差异不能因果归因于抽象层次本身。

现有数据支持的较弱结论是：在这组 A100、静态任务、文档和服务 aliases 上，Triton
对两个 Agents 都明显更可靠，当前部署式 default 应是 fixed Triton，而不是
agent-conditioned DSL hierarchy。低层 targets 的失败主要表现为 API、编译、layout 和
launch 错误；少数正确实现还出现 timing instability 或数量级性能退化。该现象是下一轮
改善 target assets、加入跨 family capability tiers 后需要检验的机制假设，不是本轮已
验证的定律。

## 成本与完整性

- 48 trajectories 共 81 次 provider calls。
- 已知 token usage 为 136,403 input 和 346,618 output；2 条 provider-error
  trajectories 的 usage 不完整。
- trajectory wall-time 总和为 7,469.59 秒。
- pre/post Flash 和 Pro provider sentinels 均通过；只有预期的 mutable-alias warning。
- Formal schedule SHA-256：
  `c5a761912cc4dd886ba8a49e672e5cd4321fcd9b5dd02592233fd6a3ebd7df35`。
- Formal controller/worker revision：`7dc3b548c7ee866ae032684d8d2f0c017f65d349`。

主要 artifacts：

- `artifacts/r1-a100/r1-a100-formal-v1/`
- `artifacts/r1-a100/r1-a100-formal-timing-v1/`
- `artifacts/r1-a100/r1-a100-oracle-gates-v1/`
- `artifacts/r1-a100/r1-a100-baseline-gates-v1/`
- `artifacts/r1-a100/r1-a100-analysis-v1/study-report/analysis-report.json`

## 协议偏差

- 远端容器无法使用 nested bubblewrap，本轮按用户决定使用 supervised `setpriv`：
  low privilege 为真，但 network isolation 和 read-only filesystem 为假。
- Provider 使用 mutable service aliases；requested/returned model 与 sentinel provenance
  已记录，但不能等价为固定 checkpoint。
- Candidate timing controller 在执行时尚未提交。每个 timing record 独立 checksum
  封存；事后 study manifest 明确记录 dirty worktree、当时 controller source SHA-256
  `c4a3e637772282f9d980eafe070183ca6bc4f61cb455ffba0ab36eacc23942ff`，并标记
  manifest 是 timing 完成后创建。后续实现已增加原子 staging、formal sealed
  job/result 交叉校验和严格 resume identity；这些修复不修改已测数值。
