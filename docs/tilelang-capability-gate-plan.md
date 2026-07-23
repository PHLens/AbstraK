# TileLang Capability Gate 最小实验计划

## Summary

目标是先验证 adaptive abstraction/control-surface policy 是否存在值得继续研究的
selection opportunity，而不是直接证明 adaptive policy 优于 fixed policy：

> 在固定 Agent、TileLang compiler、A100、任务语义、反馈和预算后，不同
> Agent-facing capability packs 是否产生 workload-dependent 的 qualification、性能和
> 搜索成本差异，并且 per-task hindsight oracle 是否优于任一全局 fixed pack？

第一阶段只运行 fixed capability packs，不实现动态切换。首轮只运行每个机制组的一个
core workload；完整且明确的 No-Go 直接停止，出现 provisional signal 或科学性不确定时，
一次性补齐四个 reserve workloads。只有完整 Gate 为 Go，才补充弱 Agent 和 adaptive
policy 实验。

本实验只能证明同一 TileLang substrate 内有限 capability portfolio 的 opportunity，不能
证明完整 DSL hierarchy、不同 compiler/backend、region-wise lowering、复杂 Kernel 或
Agent 能力有序关系。

## Experiment Contract

### Fixed Variables

- Agent：`deepseek-v4-pro` profile。运行时冻结 resolved provider/model metadata；若服务不
  暴露不可变 model revision，则在结果中记录为复现限制。
- Hardware：单张 NVIDIA A100，SM80；GPU jobs 串行执行。
- Compiler：TileLang 0.1.12；CUDA、driver、PyTorch 和 worker revision 写入 manifest。
- Submission：每轮返回一个完整、可导入且定义 `ModelNew` 的 Python source。
- Search policy：相同的 iterative repair loop、dev feedback、停止规则和 hard budget。
- 每条 trajectory 独立，无跨 trajectory memory；sealed cases 和结果不反馈 Agent。

### Capability Packs

四个 pack 使用相同 TileLang compiler/backend/runtime，只改变 Agent-visible card 和 worker
强制执行的 API capability contract：

| Pack | Target ID | 新增控制能力 |
| --- | --- | --- |
| `B` / `tileops-core` | `tilelang-a100-core` | 高层 tile ops；threads 固定 128；pipeline stage 固定 0；默认 GEMM policy |
| `B+S` / `tileops-sched` | `tilelang-a100-sched` | threads `{64,128,256}`、stages `{0,1,2,3}`、GEMM warp policy |
| `B+M` / `tileops-map` | `tilelang-a100-map` | 受限 thread/loop mapping、local storage、warp reduction、同步和自动 shared-memory layout |
| `B+S+M` / `tileops-full` | `tilelang-a100-full` | `S` 和 `M` 的并集，同时作为 flat-union baseline |

`B` 允许 `T.Kernel`、无 tuning kwargs 的 `T.Parallel`、`num_stages=0` 的
`T.Pipelined`、shared/fragment allocation、copy、default-policy GEMM、high-level reductions
和所需标量数学操作。tile size 和 grid size 在所有 pack 中都仍由 Agent 决定，因此本实验
不宣称隔离了全部 schedule decision。

`S` 只开放预注册 literal domains；禁止自定义 pipeline `order/stage/sync/group`。
`M` 首轮只开放：

- `get_thread_binding`，仅限 `threadIdx.x` 语义；
- bounded `serial/unroll/vectorized`，vector width 只允许 `{2,4,8}`，unroll factor 不超过
  16；
- `alloc_local`、`warp_reduce_*` 和无参数 block synchronization；
- `annotate_layout` 与自动 `make_swizzled_layout`。

首轮不开放 arbitrary `Fragment`/layout lambda、`assume`、CTA rasterization swizzle、raw
CUDA/PTX/ISA、custom extension、TVM escape、dynamic import、star import、`getattr`、
`eval/exec`、TileLang symbol rebinding 或 PyTorch core-op fallback。

Capability contract 使用 default-deny AST validator。调用参数必须是可静态 constant-fold
的 literal；worker 记录每个 candidate 的 `used_capabilities`、`minimum_pack_bitmask`、
validation errors 和 generated source hashes。必须满足验证器单调性：

$$
B\subseteq B+S,\qquad B\subseteq B+M,\qquad
(B+S)\cup(B+M)=B+S+M.
$$

富 pack 是 `B` 的超集，因此富 pack 的观测结果更差时，只能解释为 Agent proposal/search
burden、提示成本或低层动作错误增加，不能解释为其理论最优程序更差。

### Workloads

每个 workload 是一个固定 shape 的受控实例。分组表示预期激活的 mechanism，不预设对应
pack 是 winner；最终 winner 只能由冻结实验得到。

| Mechanism group | Core workload | Reserve workload |
| --- | --- | --- |
| Base/simple | exact GELU，`(8192,4096)` | Gated-SiLU，`(8192,4096)` |
| Schedule | GEMM，`(M,N,K)=(1024,4096,4096)` | GEMM+bias+ReLU，`(4096,1024,4096)` |
| Mapping | irregular small-K GEMM，`(8191,8179,80)` | row-sum，`(16384,4096)` |
| `S+M` interaction | row-softmax，`(8192,4096)` | RMSNorm，`(8192,4096)` |

语义和数值合同：

- 所有输入为 contiguous FP16 CUDA tensors；关键数学和 reduction 使用 FP32。
- GELU 使用 exact/erf 语义；Gated-SiLU 固定为 `silu(x) * gate`。
- GEMM 使用 FP32 accumulation；fused GEMM 在 FP32 中加 bias 和应用 ReLU，最后 cast。
- row-sum 沿最后一维 reduction 并输出 FP32。
- softmax 使用稳定的 FP32 `max -> exp -> sum -> divide`，输出 FP16。
- RMSNorm 使用 FP32 mean-square、FP16 gamma、`eps=1e-5`，输出 FP16。
- 默认 `atol=rtol=1e-2`；softmax 使用 `atol=1e-3, rtol=1e-2`；row-sum 使用
  `atol=1e-2, rtol=1e-3`。
- 每题两个 dev random cases；sealed 包含四个不同 random seeds 和一个 zero corner case。
  Sealed seeds、case IDs 和 inputs 不进入 Agent prompt。
- Qualification 还要求输出 finite、输入未修改、无 framework fallback。

全部八题及其 core/reserve 身份必须在第一次 Agent request 前冻结。Reserve 不能根据某个
有利 cell 单独选择；一旦触发，四个 reserve 必须全部运行。

### Baselines And Expert Floor

每题的共同参考 latency 记为 $L_i^*$，它是在查看 Agent/pack 结果前，从 PyTorch eager、
`torch.compile(mode="max-autotune-no-cudagraphs")` 和适用 ATen/vendor path 中选择的最快
稳定结果。不要使用 `B*` 记号，以免与 capability pack `B` 混淆。

Agent 数据生成前必须通过以下 floor：

1. 每题存在一个 `B`-legal expert implementation，通过所有 sealed cases 且无 fallback。
2. 同一 `B` expert source 在四个 nested validators 下都被接受，并生成相同 CUDA hash。
3. `S`、`M`、`S+M` 各有至少一个 capability canary，实际使用新增能力、正确运行并生成
   不同 codegen；`S+M` canary 必须覆盖同步合法性。
4. 八题的 eager/compile/vendor baselines 均完成，且每题至少存在一个稳定 $L_i^*$。
5. Core/reserve 和 capability canaries 的运行时间不得由 launch floor 主导；若 expert
   preflight 发现关键任务无法表达、计时不可用或 capability inert，则记为
   `invalid_floor`，禁止开始 Agent study。不得在看到 Agent 数据后修改 shape 或 pack。

正式 timing 使用 25 warmups、200 CUDA-event trials 和三个 clean processes。Process-level
CV 上限为 5%；超限允许完整重测一次，仍不稳定则该 performance measurement 缺失，但
已通过的 sealed correctness 不被改写。

## Agent Harness

### Loop And Budget

每条 trajectory 的 hard limits：

- 最多 3 次 model calls；
- 每次最多 8192 completion tokens；
- search wall time 最多 1200 秒；
- 每个 worker job 使用 process-group timeout 和独立 compiler/autotune cache；
- dev 最多三次 compile/evaluate，sealed 和正式 timing 在 search 结束后独立执行。

Agent response 只包含一个 fenced Python source，不再要求 `CONTINUE/FINISH`。Controller
按以下规则停止和选择候选：

1. Parse、capability validation、compile、dev correctness 和 dev latency 依次执行。
2. Pack violation、compile error、OOM、candidate-induced timeout 和 wrong result 都消耗一次
   call，并作为结构化 dev feedback 返回。
3. Dev correct 且 latency 不超过 $1.25L_i^*$ 时提前停止；否则继续到三次 call 用尽。
4. 从 dev-correct candidates 中选择 median dev latency 最低者作为 final best-so-far；如果
   没有 dev-correct candidate，则使用最后一个完整 candidate。
5. First candidate 和 final best-so-far 都保存并做不反馈 Agent 的 sealed correctness；主指标
   使用 final best-so-far。

Dev feedback 只包含 bounded compiler/runtime error、case status、max abs/rel error、dev
latency、相对 $L_i^*$ 的状态和 capability violation。不得包含 sealed 信息、expert source
或其他 pack 的结果。

### Artifacts And Cost

新实验复用并泛化现有 `abstrak.canary` harness，只使用独立 study IDs、manifest 和 report。
R1 v1 contracts、CLI 默认行为、artifact hashes 和分析结果保持不变。模块边界为：

```text
src/abstrak/canary/                reusable policies, adapters, matrix runner and artifacts
benchmarks/capability-gate-a100/   study manifest, tasks, cards, experts and canaries
artifacts/capability-gate-a100/    immutable trajectories, timing and reports
```

每个实验只提供冻结的 matrix spec、资产和 gate policy，不复制 loop、worker、transport、
artifact store 或 phase runner。四个 pack 编码为 `TargetStackSpec` 变体，通过通用 target
adapter dispatch 复用 v1 `WorkerJob/WorkerResult` wire schema；pack violation 使用
`static_check_failed` 和稳定 error code 表示。不要给现有 v1 hashed models 原地增加
optional fields。

每条 trajectory 保存：

- 完整 prompt/response、provider/model metadata 和 provider-reported input/output tokens；
- 每轮原始 source/hash、used capabilities、minimum pack 和 validation result；
- compile、correctness、dev timing、provider 和 search wall time；
- first/final sealed results、正式 timing、lowered source hashes 和完整环境指纹。

成本报告原始向量，不使用任意加权总分：

$$
\mathbf{C}=(N_{\mathrm{input\ tokens}},N_{\mathrm{output\ tokens}},N_{\mathrm{calls}},
N_{\mathrm{compile}},N_{\mathrm{exec}},T_{\mathrm{compile}},T_{\mathrm{eval}},T_{\mathrm{wall}}).
$$

输入 token 的 pack-card 长度差异属于 interface 的实际成本，不用无意义 padding 消除，但
必须显式报告。

## Metrics And Decision Rules

### Per-Replicate And Per-Cell Metrics

对 task $i$、pack $p$、replicate $r\in\{1,2,3\}$：

$$
z_{ipr}=1[\text{final candidate passes sealed verified correctness}],
$$

$$
u_{ipr}=z_{ipr}\cdot
1[\text{timing stable}\land L_{ipr}\le1.25L_i^*].
$$

Cell aggregation：

$$
s_{ip}=\sum_r z_{ipr},\qquad
c_{ip}=\sum_r u_{ipr},\qquad
a_{ip}=1[s_{ip}=3].
$$

- `3/3` 称 `stable`；`2/3` 只能称 `mixed`；`0/3` 或 `1/3` 称 failed/unstable。
- 报告 raw counts 和 exact binomial intervals，不用三次重复声明统计显著性。
- Qualification winner 使用 $(a_{ip},s_{ip},c_{ip})$ 的字典序，但保留所有并列 pack。
- Latency winner 只在 `a=1` 且 timing stable 的 cells 中比较；距最快者不足 10% 的 pack
  保留为 performance tie。
- Calls、tokens 和 compile/eval 成本不用于强行打破 qualification/latency tie，而是独立
  报告 Pareto relation。

分别报告 stable、rep-level correctness 和 Competitive opportunity gaps：

$$
G_A=\frac{\sum_i\max_p a_{ip}-\max_p\sum_i a_{ip}}{|I|},
$$

$$
G_S=\frac{\sum_i\max_p s_{ip}-\max_p\sum_i s_{ip}}{3|I|},
$$

$$
G_C=\frac{\sum_i\max_p c_{ip}-\max_p\sum_i c_{ip}}{3|I|}.
$$

Latency opportunity $G_L$ 是 best-fixed latency pack 相对 per-task fastest stable pack 的
geometric-mean regret，只在双方均 `3/3` stable 且 timing 可用的任务交集上计算，同时报告
交集大小和逐任务 regret。不同指标分别选择自己的 hindsight oracle 和 best-fixed，不构造
混合 oracle。

### Core Gate

Core schedule 为：

```text
4 tasks x 4 packs x 3 replicates = 48 trajectories
maximum model calls = 144
```

使用冻结 seed 的 balanced blocked schedule，使每个 task/replicate 内四个 pack 的执行顺序
均衡；GPU jobs 串行运行。

`Provisional-Go` 需要同时满足：

- Core matrix 完整，至少 3/4 tasks 存在一个 `3/3` stable pack；
- 至少两个 mechanism groups 出现不同的 unique robust pack winners；
- per-task oracle 相对 best-fixed 的 Competitive gain 至少一个完整 cell equivalent
  （$3/12$），或者至少两个 tasks 存在 material latency gain 且 $G_L\ge10\%$。

`Core-No-Go` 需要完整且稳定的 core matrix，同时满足：

- 同一个 fixed pack 属于全部四题的 winner set；
- 没有其他 pack 获得 unique robust win；
- $G_A=0$、$G_C\le1/12$ 且 $G_L<5\%$。

`Core-No-Go` 直接停止。`Provisional-Go` 或未达到上述两端条件的科学性
`Inconclusive` 均一次性运行全部四个 reserve。Provider/API/GPU/harness infrastructure
failure 不使用 reserve 补偿：修复后以同一 cell identity 和 prompt/seed 最多重跑一次；仍
缺失则 core 为 `inconclusive_infrastructure`。

### Full Gate

Core 与 reserve 合并后为 8 tasks、96 trajectories、最多 288 model calls。每个机制组最多
贡献一次 winner-diversity evidence，避免把结构相似的两个 task 当作独立机制证据。

完整 Gate 为 `Go` 需要：

- Matrix 完整，至少 6/8 tasks 存在一个 `3/3` stable pack；
- $G_C\ge3/24=12.5\%$，或 $G_L\ge10\%$；
- opportunity gain 分布在至少两个 tasks 和至少两个 mechanism groups；
- 至少两个不同 packs 分别拥有 unique robust winner；
- flat-union `B+S+M` 或其他单一 fixed pack 没有在 qualification、latency 和实际成本上
  Pareto dominate portfolio。

完整 Gate 为 `No-Go` 的充分现象包括：

- 一个 fixed pack 在至少 7/8 tasks 属于 winner set，其他 pack 没有 unique robust win；
- $G_A=0$、$G_C\le1/24$、$G_L<5\%$；或
- 一个 fixed pack 在 qualification、latency 和成本上 Pareto dominate 其余 portfolio。

Go/No-Go 条件均不满足、关键 winner 只依赖 mixed cells、收益只来自一个 mechanism group、
timing 交集不足或任何 pack-specific scientific cell 缺失时，结论为 `Inconclusive`。不得
事后删除不利 task、增加 replicate 或调整 tie margin。

`Go` 只表示值得继续研究 adaptive policy，不表示 adaptive 方法已经成立。下一阶段顺序为：

1. 在相同冻结 matrix 补跑 `deepseek-v4-flash`，检验 Agent capability interaction；
2. 使用 core `B` trajectory 的早期状态设计并预注册 feedback-conditioned selector；
3. 比较 adaptive、所有 fixed packs、static cascade、flat union、feedback-blind router 和
   equal-budget portfolio；
4. 普通 workload 上成立后，再单独扩展量化、MoE、dynamic attention 等复杂 Kernel。

## Implementation Milestones

### M1: Reusable Harness Extension Points

- 在现有 `abstrak.canary` 中增加 hash-bound loop policy、candidate-only protocol、
  controller stop 和 best-correct-latency selection；R1 default policy 保持原行为。
- 增加 manifest-driven matrix/phase spec、balanced schedule 和动态 request ceiling，不修改
  `R1StudySchedule` 或其 hash。
- 在 evaluator 中增加 fail-closed target-adapter dispatch，同时保持 R1 `kernelbench`
  adapter 的结果、错误和 metadata 不变。

Exit：generic policy、matrix 和 adapter tests 通过；现有 R1 tests 和 artifact contracts
全部通过。

Commit：`feat: generalize the canary harness`

### M2: Workloads, Expert Floor And Baselines

- 在通用 target-adapter 接口上实现四个 TileLang capability specs、strict AST validator、
  argument-domain checks、minimum-pack inference 和 machine-rendered target cards。
- 实现并冻结八个 task packs、输入 cases、容差、B-legal expert sources 和三个 capability
  canaries。
- 扩展 eager/compile/vendor baseline registry，运行 A100 correctness/timing floor。
- 生成 task/card/source/environment hashes 和预注册 study manifest。

Exit：八题 expert floor、S/M/S+M canaries 和全部 $L_i^*$ 完整稳定；否则停止在
`invalid_floor`。

Commit：`feat: add capability-gate tasks and floor checks`

### M3: Fixed-Pack Agent Study Runner

- 用通用 matrix runner 装载 capability-gate manifest，实现 48-cell core schedule、resume、
  single infra retry、independent caches 和完整 cost ledger。
- 在现有 `abstrak-canary` CLI 增加 manifest-driven
  `validate-study|preflight-study|run-study|time-study` 入口；R1 入口保持不变。

Exit：fake provider/worker 覆盖全部 loop 状态；一个不计分的 live vertical slice 在 A100 上
完成并验证 artifact provenance。

Commit：`feat: add fixed capability-pack study runner`

### M4: Analysis And Conditional Reserve

- 实现 $G_A/G_S/G_C/G_L$、qualification/latency winner sets、best-fixed、hindsight oracle、
  Pareto cost 和 core/full Gate。
- 实现 `analyze-study` 和 reserve launch guard；只有 core report 允许时才能启动 reserve。
- 使用 synthetic fixtures 覆盖 Go、No-Go、Inconclusive、tie、mixed、missing、timing
  instability 和 pack dominance。

Exit：分析结果可完全从 sealed artifacts 重建；report 和 manifest hash 校验通过。

Commit：`feat: add capability-gate analysis and reporting`

### M5: Formal Run And Result Record

1. 在本地 clean `main` 完成 M1-M4，每个 milestone 单独 commit 并运行完整 test suite。
2. Push `origin/main`；A100 host 的 `lipenghui` volume checkout 同一 commit。
3. 使用 host 上的 `setpriv-supervised` worker，不进入容器；冻结环境和 cache policy。
4. 运行 preflight、core matrix、candidate timing 和 core analysis。
5. 仅按冻结规则决定停止或运行全部 reserve，不在中途修改任务、pack、prompt 或阈值。
6. 将 raw artifacts 留在 artifact tree，向仓库结果文档写入 manifest hashes、完整状态计数、
   opportunity gaps、成本、失败分析和结论。

Exit：得到 `Core-No-Go`，或得到包含 reserve 的最终 `Go/No-Go/Inconclusive`；没有未处理的
运行 session 或未分类缺失 cell。

Commit：`docs: record tilelang capability-gate results`

## Verification Checklist

- Pack registry、cards、adapter implementation 和 capability specs 全部 hash-bound。
- Validator 覆盖合法 source、forbidden symbol、alias/rebinding、dynamic lookup、非法 literal
  domain、fallback、raw CUDA/TVM escape 和 monotonicity。
- Task tests 覆盖 shapes、dtypes、dev/sealed isolation、zero cases、tolerances 和 pinned sources。
- Loop tests 覆盖 parse failure、pack violation、compile/runtime/wrong result、early stop、call
  exhaustion、best-so-far、provider failure、worker failure 和 retry policy。
- Schedule tests 精确覆盖 48 core cells、可选 48 reserve cells、balanced order、resume drift 和
  duplicate/missing identities。
- Analysis tests 覆盖全部 Gate outcomes、per-metric oracle、ties、group-level evidence 和成本
  Pareto。
- Artifact tests 覆盖 source/job/result linkage、tamper detection、secret rejection 和旧 R1
  artifact verification。
- Remote preflight 验证 A100/SM80、TileLang 0.1.12、CUDA/PyTorch versions、non-container
  worker、process isolation、cache isolation 和 GPU health。
- 全部现有 `tests/test_canary_*` 与新增 tests 通过后，才允许正式 provider calls。
