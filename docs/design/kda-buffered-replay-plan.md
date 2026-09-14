# Kimi-K3 Buffered Replay：统一 Decode 实现计划

状态：待实现。本文件是方案与验收计划，不代表当前运行时已支持这些能力。

## 1. 目标与范围

将 Kimi-K3 的 KDA decode 改为跨轮缓存历史输入、按需物化 recurrent state。
Standard decode 与 speculative decode 共用一套计算、metadata 和提交协议：
standard decode 是本轮输入宽度 `T=1` 的情况，speculative verify 是 `T>1`。

首阶段采用 **state-and-output route + 按需 flush**：在 kernel 内重建 state
并计算 output，但仅在需要 flush 或对外交付精确 state 时写回完整 state。
启动参数决定 history buffer 的容量。

本阶段不实现 output-only route，不扩展 GDN/Qwen3.5，不改变模型权重格式、
采样算法或 speculative acceptance 规则。共享基础设施的改动必须维持其他模型行为。
允许针对 `T=1` 做 kernel 编译特化，但不维护两套 Python 执行路径。

参考： [ReplaySSM: Cache SSM Inputs, Not State](https://dao-lab.ai/blog/2026/replayssm/)。
文章的性能数字不能直接作为 Kimi-K3 的预期收益；KDA 的 per-channel decay
需要单独推导和验证。

## 2. 当前实现与目标的区别

当前 KDA 普通 decode 每步更新并写回 state。Speculative verify 在 replay
启用时保存本轮 packed QKV、gate 相关输入与 beta，不保存每个 draft 的完整
state；acceptance 确定后重放 accepted tokens，并立即提交 recurrent/conv state。
现有 replay payload 是本轮工作区，不是跨轮 history buffer。

| 项目 | 当前 KDA | 本计划 |
| --- | --- | --- |
| 普通 decode | 每步写回 state | `T=1` 使用统一 buffered decode |
| Speculative commit | 重放 accepted prefix，写回 state | 提交候选 history entries，推进 GPU 指针 |
| 历史输入 | 仅保留本轮 replay payload | 保留 checkpoint 之后的 accepted history |
| State 写回 | 每步或每轮 | 容量触发 flush，或精确 state 交付边界 |
| Output 计算 | 显式计算 recurrent state | 首阶段仍显式重建 state，不增加 output-only |

`store_states=False` 只控制中间 state 是否写回，不等于 output-only。
State-and-output 与 flush 是两个独立维度：kernel 可以重建 state 用来计算
output，然后不写回它。Output-only 则是另一种代数计算方式，避免构建更新后的
完整 state，本阶段不实现。

## 3. 统一状态协议

每个 request 的每个 KDA layer 由以下内容共同表示当前状态：

- 已物化的 recurrent checkpoint `S_c` 及其绝对 token 位置 `c`。
- 从 `c` 到当前 accepted endpoint 的 history entries。
- 本轮尚未提交的 candidate entries。
- GPU metadata：ring 起点、已提交长度、本轮有效宽度和候选位置。
- 与绝对 token 位置一致的 conv history/窗口信息。

核心不变量：

> 当前逻辑 state = checkpoint + checkpoint 之后已提交的 history。

Rejected candidates 不属于逻辑 state。物理 checkpoint 可以落后于 accepted
endpoint，但任何消费者都不能把它误认成 endpoint 的完整 state。
请求位置 metadata 应尽量跨层共享；每层保存各自的 history 数据。

### 3.1 KDA history 存什么

先建立 FP32 reference，再确定 kernel 的缓存布局。优先验证缓存 normalized K、
delta correction U 与 decay 信息的方案；Q 用于本轮输出，不要求长期保存。
KDA 的 decay 按 key channel 变化，不能直接套用 Mamba-2 的标量 decay 公式。

Reference 需要证明：从 checkpoint 与历史重建得到的 state，和逐 token KDA
recurrence 一致；当前候选的 correction 只依赖已提交历史与同轮更早候选。
因此丢弃 rejected suffix 后，accepted prefix 的 history 仍然有效。

Conv 有独立的短窗口依赖，不能只用 recurrent history 替代。设计时明确保存
必要的卷积前输入或等价的 accepted window，并与 recurrent history 使用一致的
accepted endpoint。允许提交小型 conv window，但不保存每个 draft 的完整
recurrent state。最终字段、dtype 和精度策略在 reference 验证后固定。

### 3.2 Buffer 所有权

遵循 [cache-concepts.md](cache-concepts.md)：优先让 LCM cache group 管理
history storage 的分配、容量、回收及迁移，backend 只消费存储与 metadata。
不得先引入独立的 backend 私有长期 buffer，再补 request 生命周期。

先验证现有 sliding-history group 能否保护所有未被 checkpoint 覆盖的 entries。
保留范围必须覆盖历史和未提交窗口；若现有规则会提前回收，先扩展通用 cache
contract。Checkpoint 保持 state-family 语义，不给它虚构 row/page geometry。
Ring 是 kernel 的逻辑访问布局，不要求绕过 LCM 自建连续存储池。

## 4. 一套 Forward 与 Commit 路径

统一流程如下：

1. Refresh 固定地址 metadata，读取本轮有效宽度 `T`。
2. 从 checkpoint 与 committed history 重建本轮起点 state。
3. 按每个 request 的容量判断是否 flush 已提交历史。
4. 计算本轮 outputs 与 candidate history entries。
5. 获取本轮实际接受的输入 token 数 `a`。
6. 用同一个 GPU commit 操作推进 committed endpoint，丢弃 rejected suffix。

普通 decode 的有效行使用 `T=1, a=1`；speculative decode 的 `a` 来自验证。
Padding/idle 行不得推进状态。必须在调用边界统一 accepted-count 的定义，
不能把 draft matches 和实际推进 state 的输入 token 数混用，也不能盲目加一。

本轮 candidate 写入在下一轮读取前完成；commit 与后续 forward 必须有明确的
stream 顺序。普通 decode 可在无需等待 acceptance 时安排提交，但不得复制一套
计算或 metadata 实现。

共享 kernel 可以根据静态 `T` 选择优化配置。多 token correction 的并行计算是
后续 kernel 优化点，不能直接假定文章 GDN 的 triangular solve 已适用于 KDA。
第一版先以统一、可验证的 recurrence 保证语义，再减少串行计算成本。

## 5. Flush 与容量参数

新增启动参数，建议名称：

```bash
--ssm-replay-buffer-capacity 64
```

`64` 仅是配置示例，不是性能推荐值。容量 `L` 以每个 request、每层的 token
entries 计，包含 committed history 与本轮 candidates 占用的空间。
启动时固定，完成内存规划与分配后再捕获 CUDA graph；不支持运行中调整。

采用提前一个窗口 flush 的初始策略：已提交 history 长度为 `h`，服务配置的
最大执行窗口为 `T_max`，当 `h + 2 * T_max > L` 时 flush 旧的 committed history。
使用最大窗口可保证后续窗口不因剩余容量而缩短。启动校验 `L >= 2 * T_max`，
并单独检查 kernel 支持的容量/对齐约束；若需要额外物理 padding，日志分别报告
逻辑容量与实际分配。Standard decode 同样使用该规则，`T_max=1`。

Flush 只汇总本轮开始前已经接受的历史，不能把尚未验证的 candidates 写入
checkpoint。完成后通过 ring 指针推进回收历史空间，不搬移剩余 candidates。
所有判断使用 GPU 上的每请求 metadata，允许同一 batch 混合 flush/no-flush。

参数不绑定 `prefix_granularity`、state block granularity 或 KV kernel page size。
首阶段不定义 `0` 为关闭模式；默认容量在基准验证后决定。容量不足或硬件/kernel
不支持时启动报错，不静默退回另一套 decode 路径。

显存预算按最终布局计算：

```text
history bytes ≈ live requests × local KDA layers × L × bytes per entry
```

另外计入 checkpoint、conv、metadata、scratch、allocator packing 和 overlap
所需保护空间。容量增大可减少 flush，但也增加每步 history 读取/重建成本，
因此不保证越大越快。

## 6. 生命周期与 CUDA Graph

遵循 [unified_path.md](unified_path.md) 与 [scheduler.md](scheduler.md)：

- **Prefill → decode**：以 prefill final state 建立 checkpoint，history 清空。
- **Decode → incremental prefill**：先物化精确 accepted endpoint，复用现有
  prefill 输入协议；首阶段不要求 prefill kernel 直接读取 replay history。
- **Prefix cache**：只发布确实物化且位置匹配的 checkpoint。Flush 不自动赋予
  prefix 边界 provenance，不能发布候选 state 或落后的 checkpoint。
- **Retraction/恢复/迁移**：交付 checkpoint + history + metadata 的完整表示，
  或在交付边界统一物化 endpoint；首阶段优先采用后者，明确同步完成后再释放。
- **结束/取消/槽位复用**：不需要交付 state 时直接回收，不做无用 flush；在
  in-flight readers 完成前不能复用 buffer。Prefix 发布另按其精确位置处理。
- **Pool rebind**：清除旧 descriptors/views，重建存储绑定并按现有协议重新捕获。

Eager 与 CUDA graph 使用同一个 `refresh_decode_metadata` 和 GPU 操作序列。
Buffer 按最大运行 batch 容量分配，不以 graph capture ladder 上限代替运行容量。
避免每步 `.item()`、CPU flush 判断和临时 tensor 分配。Flush 标志、ring 指针、
accepted count 为设备数据，不因不同请求的决定重新捕获 graph。
Commit 如独立于模型 forward 捕获，也必须保持同一语义、固定 buffer 与明确依赖。

## 7. 工作拆分与交付顺序

| 阶段 | 主要工作 | 进入下一阶段的条件 |
| --- | --- | --- |
| A：语义与 reference | KDA 重建公式、缓存字段、conv 协议、acceptance 单位 | 多轮重建、拒绝、flush 与逐 token reference 一致 |
| B：Cache contract | LCM ownership、retention、容量参数、GPU metadata、生命周期 | 未 flush 数据不会被回收，内存预算与实际分配吻合 |
| C：统一 kernel/runtime | `T=1/T>1` 共享计算与 commit；按需 flush 替代每轮 eager replay | 两种 decode 均正确，非 flush 轮没有完整 recurrent state 写回 |
| D：系统集成 | CUDA graph、overlap、agentic prefill、prefix reuse、恢复路径 | Eager/graph 和混合 batch 边界测试通过 |
| E：性能与精度 | Real NVFP4 TP8 full model 对比、容量 sweep、NSYS、数据集验证 | 形成可复现报告后决定默认容量与是否启用 |

主要代码范围：

- `runtime/layers/attention/backends/state/kda.py`：统一 buffered decode 与 commit。
- `runtime/layers/attention/backends/state/mamba.py`：共享调用协议和 metadata 接口，
  不改变 GDN 的默认行为。
- `runtime/layers/attention/kv_cache/recipes/kimi_k3.py` 与 cache/LCM bridge：history
  group 声明、预算和状态生命周期；具体 C++ 变更由阶段 B 的缺口决定。
- Runtime 配置和 decode runner：参数校验、固定 buffer、acceptance 与 commit 顺序。
- `tokenspeed-kernel` 的 KDA ops 与 kernel：重建、候选计算、flush、GPU 指针提交。

上述 runtime 路径均相对于 `python/tokenspeed/`。Runtime 通过
`tokenspeed-kernel` 边界调用 kernel，不新增直接第三方依赖。
实现完成后更新相应 design 文档、模型 recipe 和 runbook；本计划不替代已有设计规范。

## 8. 正确性验收

以参数化测试覆盖组合，不为每个微小分支新增独立测试：

- `T=1` 与多 token 窗口，各种接受长度和 rejected suffix。
- 最小合法容量、较大容量、临界 flush、连续多次 flush、ring wraparound。
- 同 batch 的不同 history 长度与 flush 决策，padding/idle、请求重排与槽位复用。
- Prefill → decode → incremental prefill、prefix hit、取消、恢复、pool rebind。
- Eager/CUDA graph、overlap 开关、实际 batch 超过 capture ladder。

比较逐步 outputs、物化 endpoint state、conv window，以及长序列误差累积。
用 reference 先确定 dtype 对应容差并记录理由，不能为让新实现通过而临时放宽。
数学等价不代表浮点 bitwise 一致，也不要求随机采样文本逐字相同。

端到端使用真实 NVFP4 权重、TP8、完整 Kimi-K3；记录目标与 draft 模型版本、
代码 commit、kernel backend、软件环境、CUDA graph 配置和 sampling 参数。
固定 AIME 2026 题集版本、prompt、seed、生成预算与评分器，旧/新实现同条件运行，
报告分数及逐题结果。小型 agentic smoke test 只能验证执行，不能替代精度验证。
共享接口改变后运行相关 GDN/普通 decode 回归测试。

## 9. 性能验收与报告

Baseline 使用冻结的当前代码：standard decode 为逐步 state 更新，speculative
decode 为当前 eager replay-and-commit。两者分别和新实现对比，不能用不同
acceptance 或不同生成 token 数直接推断 kernel 加速。

固定 GPU、模型、输入、batch、后端、warmup 和 graph 设置；在持久分配的计算
节点上按 runbook 执行。覆盖 concurrency 1 与较大 batch，并测试多个合法容量。
分别测无 profiler 的重复端到端运行和短 NSYS，报告：

- 每步 decode/verify/commit 延迟，flush 与 no-flush 的延迟分布。
- 总 tokens/s、inter-token latency、峰值显存和最大可用 batch。
- 实际 acceptance、flush 频率、完整 state store 次数、kernel launch 数。
- Agentic incremental prefill 的边界物化开销。

提供包含 prefill、多个 decode/verify round，以及 flush/no-flush 的清晰命名
`.nsys-rep` 和 ZIP，附可复现命令。不同容量必须记录历史重建开销，不能只看
flush 次数下降。正式比较前约定可接受的性能波动与回归阈值；结果不足时不宣称
达到论文收益，不默认启用未经验证的配置。

## 10. 实施前必须关闭的问题

1. KDA 的缓存字段、dtype 与长历史重建误差是否满足精度要求？
2. LCM 能否完整表达 checkpoint 落后于 endpoint 时的 history retention？
3. Conv 与 recurrent history 如何保证拒绝、flush、迁移后的同位置一致性？
4. Endpoint 物化与 prefix publication 如何遵守现有 provenance 和 overlap 顺序？
5. 哪些容量与窗口由首版 kernel 支持，哪一档容量在 TP8 实测最合适？

这些问题属于阶段 A/B 的交付，不应通过引入 standard/speculative 两套实现绕过。
