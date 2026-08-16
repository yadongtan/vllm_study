# 202608150102 - Unrolled GQA Tile

## 1. 结论

本轮只把 GQA partial kernel 的 8-token 内层循环改为编译期固定 8 次并强制 unroll。
正确性通过，但 Decode profile 回退 2.07%，并发 1/4/8 吞吐分别下降
0.99%/5.75%/6.28%。该修改已恢复，不进入累计基线。

| 并发 | Parallel Merge 基线 (tok/s) | Unrolled Tile (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 314.69 | 311.56 | -0.99% | 375.19 |
| 4 | 675.49 | 636.63 | -5.75% | 1305.21 |
| 8 | 996.56 | 934.02 | -6.28% | 2400.77 |

## 2. 单一迭代内容

旧实现的 token 循环上界为运行时 `tile_tokens`，完整 tile 时值为 8，尾 tile 可以更短。
候选实现把循环写成固定 `kGqaTileTokens=8` 次，并加 `#pragma unroll`；尾 tile 使用
warp 一致的条件屏蔽。

目标是消除循环控制并让 NVCC 同时调度多个 QK、online softmax 和 PV 操作。共享
K/V 布局、tile 大小、partition、parallel merge 和数值公式均未修改。

## 3. 正确性验证

- 与逐 query-head Split-KV 参考实现对照通过。
- CUDA Graph capture/replay 对照通过。
- 与 SDPA 参考实现对照通过。
- partition 边界测试通过。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. Profile 变化

| batch 8，历史长度 1000，20 次 Decode | 运行时循环 | 强制 Unroll | 变化 |
|---|---:|---:|---:|
| GQA partial kernel | 19.215 ms | 19.382 ms | +0.87% |
| Split-KV merge kernel | 1.366 ms | 1.418 ms | +3.81% |
| Self CUDA 总时间 | 61.238 ms | 62.507 ms | +2.07% |

NVCC 对原本仅 8 次的短循环已经能做合理优化。强制完全展开增加指令代码和寄存器活跃
范围，可能降低 occupancy 或 instruction-cache 效率，最终没有获得计算重叠收益。

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`，Split-KV partition 64。
- parallel Split-KV merge。
- 4-step graph enqueue + batched token notification。
- `batch_wait_ms=5`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 基线：`cuda_benchmark_results/202608150058`。
- 本轮：`cuda_benchmark_results/202608150102`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.26 | 311.56 | 34.77 | 3.07 | 9.49 |
| 4 | 9.95 | 636.63 | 170.84 | 3.39 | 10.47 |
| 8 | 6.79 | 934.02 | 282.79 | 3.38 | 10.72 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 4.40% / 8% | 1455 / 1455 | 0.18% / 0.20% | 1088 / 1091 |
| 模型和 Graph 加载后 | 1.60% / 2% | 4498 / 4498 | 0.06% / 0.09% | 2127 / 2130 |
| 并发 1 | 68.30% / 92% | 4552 / 4568 | 3.54% / 9.12% | 2930 / 3070 |
| 并发 4 | 54.69% / 95% | 4710 / 4796 | 4.15% / 9.00% | 2913 / 3051 |
| 并发 8 | 46.50% / 98% | 4882 / 5002 | 4.50% / 9.20% | 2896 / 3033 |

## 8. 决策

恢复运行时短循环，保留 parallel merge。进一步优化 partial kernel 需要改变其并行
算法或使用更成熟的 attention 后端，简单展开和简单访存宽化均已被实测否定。
