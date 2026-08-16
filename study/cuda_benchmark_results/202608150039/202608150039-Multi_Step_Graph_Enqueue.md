# 202608150039 - Multi-Step Graph Enqueue

## 1. 结论

本轮只把 ignore-EOS 流水的 graph enqueue 深度从 1 增到 4。正确性通过，但三档
吞吐没有稳定改善：并发 1/4/8 分别变化 +0.38%/-0.23%/-1.70%。因此默认值保持
`STUDY_PIPELINE_STEPS=1`，step 4 不进入后续累计基线。

| 并发 | Step 1 (tok/s) | Step 4 (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 298.28 | 299.41 | +0.38% | 375.19 |
| 4 | 627.42 | 625.99 | -0.23% | 1305.21 |
| 8 | 953.05 | 936.88 | -1.70% | 2400.77 |

## 2. 单一迭代内容

Step 1 每轮执行以下顺序：

1. 当前 token 异步复制到 pinned CPU buffer；
2. 记录 copy event；
3. enqueue 下一张 CUDA Graph；
4. CPU 等待当前 copy，随后通知请求。

Step 4 改为使用 4 行 pinned ring buffer 和 4 个 CUDA event，先连续排队最多 4 组
`D2H copy + graph replay`，再依次等待 event 并交付 4 个 token。模型、KV Cache、
Attention kernel、graph shape 和调度 batch 均未改变。

## 3. 正确性验证

- 同一 batch 8 prompt 分别走原同步路径和 step 4 流水路径。
- 每请求 8 个输出 token 逐项完全一致。
- 每个 ring slot 使用独立 event，静态 graph 输出被覆盖前 D2H copy 已完成。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. 性能分析

Step 4 降低了 Python 返回 graph enqueue 的频率，但把流式 token 交付变成最多 4 个
token 一组。并发 8 平均 TPOT 从 3.923 ms 降到 3.713 ms（-5.36%），但平均
TTFT 从 250.58 ms 增到 256.91 ms，总测试时长从 6.57 s 增到 6.68 s。

当前 graph 计算本身约 3 ms，Step 1 已经能让 CPU token 处理与下一张 graph 重叠；
继续增加 enqueue 深度没有消除新的主要瓶颈，反而增加了流式交付抖动。

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`，Split-KV partition 64。
- `STUDY_PIPELINE_IGNORE_EOS=1`。
- 基线 `STUDY_PIPELINE_STEPS=1`，本轮候选为 4。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 基线：`cuda_benchmark_results/202608150031`。
- 本轮：`cuda_benchmark_results/202608150039`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.91 | 299.41 | 26.97 | 3.55 | 2.74 |
| 4 | 10.04 | 625.99 | 161.16 | 3.89 | 3.02 |
| 8 | 6.68 | 936.88 | 256.91 | 3.71 | 3.28 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 17.20% / 19% | 1425 / 1426 | 0.18% / 0.35% | 1061 / 1066 |
| 模型和 Graph 加载后 | 3.60% / 6% | 4469 / 4469 | 0.12% / 0.18% | 2151 / 2159 |
| 并发 1 | 68.57% / 93% | 4529 / 4574 | 3.77% / 8.97% | 2921 / 3054 |
| 并发 4 | 59.18% / 96% | 4661 / 4744 | 4.33% / 9.47% | 2916 / 3057 |
| 并发 8 | 51.21% / 98% | 4913 / 5086 | 4.94% / 8.94% | 2887 / 3039 |

## 8. 决策与下一步

- 保留 multi-step 实现作为可调实验能力。
- 默认 `STUDY_PIPELINE_STEPS=1`，后续基线使用 step 1。
- 下一项应分析 Prefill 与 Decode 的批次调度，而不是继续增加 graph enqueue 深度。
