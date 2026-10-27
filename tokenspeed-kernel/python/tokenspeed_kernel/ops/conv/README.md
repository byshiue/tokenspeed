# Causal conv1d prefill 候选实现

`_triton/prefill_candidate.py` 提供未注册的实验实现。
Serving 继续使用原有 kernel。公共 API、默认参数和运行时路由不变。

## 数据复用与状态

`window` 按 `[token, channel]` 划分 CTA。
每块读取最多 `M + width - 1` 行有效输入，再用 `tl.gather` 生成各个 tap 的窗口。
幂次扩展的行使用 mask。不同请求之间不共享窗口。
寄存器、线程间交换和 shared memory 的分配由 Triton 编译器决定。
源码中的一次加载不等于一次 HBM 事务。

`direct` 对每个 tap 分别加载移位输入，用于量化窗口复用的成本。
`serial` 调用未修改的旧 kernel，用于基线和 CTA 大小对照。
固定配置的实现名和 tile 参数都必须显式传入。没有新增服务端开关。

`_gluon/prefill_candidate.py` 使用同一 `tokenspeed-triton` 包的 Gluon 前端。
`gluon_serial` 和 `gluon_window` 使用同一个 Gluon kernel 入口。
显式编译参数 `TOKEN_PARALLEL` 选择两种 token 计算方式。
`False` 保留寄存器滑动窗口。每步只读取一个新的有效 token。
编译器按固定 `BLOCK_M` 展开这些步骤。每步保留有效长度判断。
短请求不读取无效 token。此改动减少循环控制和跨循环存活的寄存器。
`True` 并行计算二维窗口。编译器移除未选择的分支。
两者共用输入加载、tap 累加、SiLU、输出存储、请求边界和状态写回。
`gluon_window` 显式指定 channel 连续的线程布局。
当所有 warp 都沿 channel 排列时，各个 token 的窗口移位只在寄存器内完成。
更小的 channel tile 可能需要跨 warp 交换，必须检查生成代码和实测时间。
`gluon_direct` 使用相同布局，但分别加载四个 tap，作为对照。

`gluon_window_r64` 和 `gluon_window_r48` 分别显式设置 64 和 48 的寄存器上限。
这两个选择保留相同的数值运算。编译器可能为它们生成 spill。
必须同时检查时间和实际内存计数，不能只按 spill 数量选型。
扩展实验允许 `M=4` 和 `N=512/1024`，用于检查小 token tile 和宽 channel tile。

Gluon 实现将首个 tile 的历史读取和状态写回放入独立分支。
后续 tile 的编译体不读取状态。连续布局快路径使用模型的固定 channel 数计算地址。
快路径条件在 GPU 上判断。动态 stride 仍走同一个已编译的 kernel。

调用者为每个 forward 生成一次 `CausalConv1dPrefillMetadata`。
所有层共享此映射。映射中的 `block_m` 必须与候选配置一致。
输入和输出采用 `[channels, packed_tokens]`。调用者提供输出缓冲区。
输出缓冲区不能与输入或状态重叠。每个有效请求必须使用独立的状态 slot。

首个 token tile 读取旧历史，并写回最新的 `width - 1` 个原始输入。
短请求从“旧历史＋新输入”中取尾部。无历史时使用零。
写回前使用 CTA barrier，保证旧状态读取完成。
滑动窗口在历史值进入寄存器后更新状态，减少跨循环存活的状态变量。
后续 tile 只读取不可变的输入，不读取可被首块覆盖的状态。
PAD slot 和空请求不写输出，也不写状态。

候选支持宽度 2、3、4，以及 BF16、FP16、FP32。
它保留原 kernel 的乘法类型、tap 顺序、SiLU 和输出转换。
逐位比较以冻结的旧 kernel 和同一编译环境为准。
SM103 上的 FP32、宽度 3、bias 和 SiLU 分支显式保留首个乘法的舍入点。
此分支用于匹配旧 kernel 的生成代码。更换编译器后必须重新验证。
Gluon 的其他 FP32 tap 显式使用 FMA，避免首块和内部块的舍入方式不同。
bias 直接广播，保留负零的符号位。测试包含这两个数值边界。
统一 kernel 中，每批变化的 token 数、请求数、输入和输出 stride 不参与 constexpr 特化。
权重和状态布局的 stride 按固定模型布局编译。

## 两套 tactic 的自动选择

局部实验入口也接受显式的 `PrefillTacticPolicy`。
它选择 `gluon_serial, 8×256, 4 warps` 或
`gluon_window_r64, 8×512, 4 warps`。两者是同一 kernel 的编译变体。
两者共用 `block_m=8` 的映射。`serial` 仍调用冻结旧 kernel，供测试对照。
每次调用只选择并启动一个 kernel。选择器不修改卷积计算或状态更新。

GB300、BF16、36864 channels、宽度 4、SiLU 的保守校准配置为：

```python
from tokenspeed_kernel.ops.conv._triton.prefill_candidate import PrefillTacticPolicy

policy = PrefillTacticPolicy(
    min_parallel_tokens=2048,
    min_tokens_per_request=128,
    max_parallel_tokens=8192,
)
```

总 token 数在 2048～8192 之间，且 `总 tokens >= 128 * 请求数` 时选择 token 并行变体。
总量超出校准范围时也回退。只有精确 packed metadata 才可自动选择并行变体。
其他批次使用滑动窗口变体。总量相同的单个长请求和许多短请求可能选择不同 tactic。
首个 tile 的历史读取和状态更新成本使请求数成为必要的选择条件。
校准要求已选择的并行负载在每一轮中都比滑动窗口快至少 10%。
未选择的负载运行同一个 kernel 的滑动窗口 tactic；不调用原 kernel。
原 kernel 只作独立比较。滑动窗口 tactic 与原 kernel 数值一致，但生成代码和耗时可不同。
校准覆盖的形状之外不声称性能保证。
阈值必须显式传入。更换 GPU、模型布局、dtype 或编译器后应重新校准。

将 `policy` 传给 `launch_prefill_candidate(..., config=policy, ...)` 即可自动选择。
选择只读取输入 tensor 的 token 轴大小和 slots 的请求轴大小，不读取 CUDA tensor 内容。
也可在 forward 规划时调用
`policy.select(num_tokens, num_requests, exact_workload=True)`，让所有层复用返回的配置。
仅在这两个计数代表实际 packed 负载时传 `True`。容量估计必须传 `False`。
两套配置均应在启动时预热。运行中只做选择，不进行试跑或在线调优。
模型维度、dtype、权重和状态布局按固定配置编译。批次 token 数和请求数不参与编译特化。

普通 packed prefill 的轴大小表示本批实际负载。
容量 metadata 没有固定的 CPU 请求边界。自动入口对这种 metadata 一律选择滑动窗口。
因此，replay 将负载从大批次缩到小批次时，不会保留误选的并行 tactic。
精确 metadata 捕获的图固定本批长度；它可按阈值选择并行 tactic。
如需按 replay 实际长度选择并行 tactic，图管理器须选择已校准、预先捕获的图。
本轮不修改图管理器或 Serving 默认实现。

选择器测试覆盖短请求、阈值两侧、多个短请求合批、状态续接和固定图 replay：

```bash
PYTHONPATH=tokenspeed-kernel/python python -m pytest \
  tokenspeed-kernel/test/nvidia/ops/test_conv_prefill_tactics.py -q
```

将上述三个阈值保存为 JSON 文件后，可以运行自动选择入口的性能验收：

```bash
PYTHONPATH=tokenspeed-kernel/python:tokenspeed-kernel/test python \
  tokenspeed-kernel/test/nvidia/ops/bench_conv_prefill_tactics.py \
  --policy /tmp/k3-conv1d/policy.json --min-speedup 1.10 \
  --output /tmp/k3-conv1d/tactics.json
```

每组仍使用相同输入、初始状态和缓存条件，交替测量 5 轮、每轮 100 次。
结果同时记录原 kernel、滑动窗口 tactic、并行 tactic 和自动入口的时间。
每轮轮换测量顺序。报告分别列出自动入口相对滑动窗口和原 kernel 的加速比。
回退时检查自动入口与滑动窗口的 CUBIN 一致，避免把同一代码的计时波动当作回退。
已选择的并行负载若有一轮低于收益门槛，验收脚本失败。
计时范围不含 Python 选择成本。Graph replay 不重复执行 Python 选择。
四组 8K 负载另外测量 metadata＋kernel；两个代表负载验证 PDL 生产者链。

## 正确性测试

在已有 TokenSpeed GPU 环境中，从仓库根目录运行：

```bash
PYTHONPATH=tokenspeed-kernel/python python -m pytest \
  tokenspeed-kernel/test/nvidia/ops/test_conv_prefill_candidate.py \
  tokenspeed-kernel/test/nvidia/ops/test_conv_prefill_gluon.py \
  tokenspeed-kernel/test/nvidia/ops/test_conv_prefill_tactics.py -q
```

测试比较真实旧 kernel 和独立 FP32 PyTorch 参考。
旧 kernel 的输出与状态使用整数位模式比较。
测试覆盖短请求、tile 边界、零长度、PAD slot、乱序 slot、非连续视图、
多次 chunk 续接、CUDA Graph replay，以及超过 int32 的 token 地址。
编译守卫检查请求数、token 数和输入/输出 stride 变化不会触发候选重新编译。
检查同时覆盖 channel 连续布局和 token 连续布局。

这组固定数值契约的测试统一放在 `test/nvidia/ops/`。
显式布局测试位于 `test_conv_prefill_gluon.py`。
它复用上述行为检查，并单独覆盖自动选择使用的两套固定配置。
检查包括数值、chunk 续接、Graph replay 和动态形状编译守卫。

## B300/GB300 性能验收

在 B300 或 GB300 上运行。脚本只使用当前 CUDA device。
优先使用独占节点，也可按实验要求使用共享分配。
它检查 SM103 和 B300/GB300 型号，并验证原 prefill 函数及其装饰器的固定源码哈希。
同文件中 decode kernel 的改动不影响此基线校验。
基线和候选在同一张卡上重测。报告记录实际型号、时钟、GPU 利用率和显存用量。
先运行上面的正确性测试，再执行性能验收脚本。

```bash
PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/nvidia/ops/bench_conv_prefill_candidate.py \
  --mode sweep --output /tmp/k3-conv1d/results.json
```

`sweep` 保留最初的对照组和扫描范围。
扩展实验使用 `refine`，从 JSON 文件读取完整、显式的配置列表。例如：

```json
[
  {"implementation": "gluon_window_r64", "block_m": 8, "block_n": 512, "num_warps": 4},
  {"implementation": "gluon_window_r48", "block_m": 8, "block_n": 256, "num_warps": 4}
]
```

```bash
PYTHONPATH=tokenspeed-kernel/python python -m pytest \
  tokenspeed-kernel/test/nvidia/ops/test_conv_prefill_gluon.py -q
PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/nvidia/ops/bench_conv_prefill_candidate.py \
  --mode refine --configs /tmp/k3-conv1d/configs.json \
  --output /tmp/k3-conv1d/refine.json
```

`refine` 使用与 `sweep` 相同的正确性、计时、PDL 和 profiler 流程。
结果分别记录 Triton 调度文件和 Gluon kernel 文件的 SHA256。

主负载为 BF16、36864 channels、宽度 4、SiLU、已有历史。
四组长度为 `[8192]`、`[4096,4096]`、`[1330]*6+[212]`、
`[1330]*6+[106,106]`。这些组合用于固定总量、改变请求边界。
脚本先完成所有主配置的正确性检查，再测性能。

每次采样使用 CUDA Graph 内的 CUDA events。
状态复位在起始 event 之前。基线与候选使用相同输入和初始状态。
每组交替测量 5 轮，每轮 100 次。结果包含原始样本、中位数和 P95。
未显式刷新 L2。计时记录当前缓存条件，不能当作冷缓存 HBM 带宽测量。

`kernel` 范围只测卷积。
`metadata` 范围测 GPU 映射构建与卷积，不包含 Python 分配和规划时间。
实际服务每个 forward 只构建一次映射，不应将此成本重复算到每一层。
`chain` 范围测局部复制生产者与卷积，用于比较 PDL 开关。
生产者用于验证依赖和测量局部链，不代表 QKV GEMM 的实际耗时。
Gluon 的 PDL 汇编显式声明 memory clobber，防止依赖读取跨过 wait。

并行候选的入选条件为四组 kernel 中位数加速的几何平均至少 1.5 倍，
且任何一组的耗时不超过基线的 1.05 倍。目标为几何平均至少 2 倍。
串行大 tile 只参与消融对照，不作为并行候选入选。
小请求单独报告。脚本不据此修改运行时路由。

脚本保存 PTX、CUBIN、Triton GPU IR、寄存器、spill 和 shared memory 信息。
若环境有 `ncu`，它采集基线和选中候选的硬件计数器，输出报告及 CSV。
计数器不可用时记录失败原因，不尝试改变权限。
PTX 指令行数不是动态流量，源码字节估算不是实际 DRAM 字节。

单个配置可单独放入 profiler 范围：

```bash
PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/nvidia/ops/bench_conv_prefill_candidate.py \
  --mode profile --implementation window --block-m 64 --block-n 128 \
  --warps 4 --shape 0 --scope kernel --pdl off \
  --output /tmp/k3-conv1d/profile.json
```

外部 profiler 使用 `--profile-from-start off`，只收集 CUDA profiler API 范围。
离线编译只能验证编译结果，不能替代 GPU 正确性、PDL、Graph replay 或性能验收。
