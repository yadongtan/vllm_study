# Warp Reduction RMSNorm 性能测试报告

## 1. 结论

本轮只优化 Fused Add + RMSNorm 的 block reduction。算子 profile 从 8.574 ms
降到 8.416 ms，下降 1.84%；batch 8、1000-token、三轮 Prefill Self CUDA
总时间从 227.132 ms 降到 224.206 ms，下降 1.29%。

正式服务测试的收益集中在并发 8，达到 776.63 tok/s，相对上一有效版本提升
2.72%。并发 1 和 4 分别回退 1.48% 和 1.61%，说明该 kernel 优化在低并发下
容易被服务调度和测量波动覆盖。

| 并发 | 上一有效版本 (tok/s) | Warp RMS (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 273.47 | 269.43 | -1.48% | 375.19 |
| 4 | 527.09 | 518.61 | -1.61% | 1305.21 |
| 8 | 756.07 | 776.63 | +2.72% | 2400.77 |

## 2. 单一优化内容

旧 reduction 把 256 个线程的局部平方和全部写入 shared memory，然后进行 8 轮
树形规约。每一轮都需要一次 block 级同步。

新 reduction 分为两级：

1. 每个 32-thread warp 使用 `__shfl_down_sync` 在寄存器之间规约；
2. 8 个 warp 的 lane 0 仅把 8 个部分和写入 shared memory；
3. 第一个 warp 再规约这 8 个值；
4. 整个 block 只保留两次必要的 `__syncthreads()`。

RMSNorm 公式、BF16 residual add 的舍入位置、输入输出布局和 launch block 数均未
改变。本轮没有修改 SiLU、RoPE、Attention、KV Cache、CUDA Graph 或调度器。

## 3. 正确性验证

- RMSNorm 单算子与 PyTorch 对照通过。
- Fused Add + RMSNorm 单算子与 PyTorch 对照通过。
- CUDA Graph capture/replay 对照通过。
- 完整 24 层 Prefill logits、next token、KV Cache 对照通过。
- 完整 24 层 Decode logits、next token、KV Cache 对照通过。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. Profile 变化

| batch 8，1000 token，三轮 | 旧 shared reduction | Warp reduction | 变化 |
|---|---:|---:|---:|
| Fused Add + RMSNorm | 8.574 ms | 8.416 ms | -1.84% |
| Self CUDA 总时间 | 227.132 ms | 224.206 ms | -1.29% |

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 上一有效版本：`cuda_benchmark_results/202608142135`。
- 本轮：`cuda_benchmark_results/202608142151`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.24 | 269.43 | 26.59 | 4.04 | 3.11 |
| 4 | 12.11 | 518.61 | 190.11 | 4.84 | 3.74 |
| 8 | 8.07 | 776.63 | 275.76 | 5.54 | 4.38 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 13.00% / 13% | 968 / 969 | 0.14% / 0.21% | 1080 / 1085 |
| 模型和 Graph 加载后 | 2.40% / 3% | 4010 / 4011 | 0.11% / 0.21% | 2138 / 2140 |
| 并发 1 | 64.77% / 92% | 4066 / 4081 | 3.73% / 9.06% | 2931 / 3068 |
| 并发 4 | 52.63% / 82% | 4214 / 4295 | 4.27% / 9.57% | 2939 / 3067 |
| 并发 8 | 44.27% / 82% | 4477 / 4624 | 4.42% / 9.43% | 2928 / 3064 |

## 8. 后续决策

保留 Warp reduction。下一轮只处理 Prefill Attention 前的 Q layout copy：先确认
复制来自哪个 stride/layout，再决定能否通过生成兼容布局消除，而不更换 Attention
算法或改变数值口径。
