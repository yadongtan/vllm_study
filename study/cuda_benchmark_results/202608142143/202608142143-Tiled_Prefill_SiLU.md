# Tiled Prefill SiLU 性能测试报告

## 1. 结论

本轮只修改 Prefill `silu_and_mul` CUDA kernel 的线程工作划分。Profile 和正式
服务测试均未显示稳定收益，因此该方案判定为无效优化，不作为下一轮累计优化的基线。

batch 8、1000-token、三轮 Prefill profile 中，SiLU kernel 从 27.762 ms 增加到
28.249 ms，回退 1.75%；整轮 Self CUDA 时间从 227.132 ms 增加到 227.830 ms，
回退 0.31%。正式服务测试相对上一轮分别变化 -0.96% / +0.52% / -1.54%。

| 并发 | 上一版本 (tok/s) | Tiled SiLU (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 273.47 | 270.85 | -0.96% | 375.19 |
| 4 | 527.09 | 529.85 | +0.52% | 1305.21 |
| 8 | 756.07 | 744.44 | -1.54% | 2400.77 |

## 2. 单一优化内容

旧 kernel 使用一维全局元素编号。每个线程负责一个输出元素，并通过除法和取模从
全局编号恢复 row 与 column。

新 kernel 使用二维 grid：

- `grid.y` 选择 row；
- `grid.x` 选择 intermediate dimension 的 column tile；
- 每个线程处理 4 个 column 元素；
- 不再对每个输出元素执行 64 位除法和取模。

数学公式、输入输出布局和 BF16 舍入顺序均未修改。本轮没有修改 RMSNorm、RoPE、
Attention、KV Cache、Decode CUDA Graph 或调度器。

## 3. 正确性验证

- 小尺寸 SiLU 算子与 PyTorch 对照通过。
- `[8, 1000, 9728]` Prefill 输入对照通过。
- CUDA Graph capture/replay 对照通过。
- 完整 24 层 Prefill logits、next token、KV Cache 对照通过。
- 完整 24 层 Decode 对照通过。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. Profile 变化

| batch 8，1000 token，三轮 | 上一版本 | Tiled SiLU | 变化 |
|---|---:|---:|---:|
| `silu_and_mul` | 27.762 ms | 28.249 ms | +1.75% |
| Self CUDA 总时间 | 227.132 ms | 227.830 ms | +0.31% |

虽然新划分减少了整数索引计算，但该算子的主要成本仍是读取约两倍 intermediate
dimension 的 gate/up 数据并写回结果。额外的二维 block 数量和每线程循环没有抵消
显存带宽成本，因此不能仅凭减少除法推断性能会提升。

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 上一版本：`cuda_benchmark_results/202608142135`。
- 本轮：`cuda_benchmark_results/202608142143`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.12 | 270.85 | 26.31 | 4.02 | 3.10 |
| 4 | 11.87 | 529.85 | 177.50 | 4.90 | 3.79 |
| 8 | 8.44 | 744.44 | 297.96 | 5.38 | 4.29 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 12.60% / 14% | 967 / 968 | 0.15% / 0.27% | 1084 / 1091 |
| 模型和 Graph 加载后 | 3.20% / 6% | 4010 / 4011 | 0.17% / 0.30% | 2133 / 2139 |
| 并发 1 | 65.10% / 96% | 4067 / 4088 | 3.67% / 9.03% | 2936 / 3083 |
| 并发 4 | 53.11% / 84% | 4210 / 4278 | 4.17% / 9.11% | 2924 / 3045 |
| 并发 8 | 46.40% / 81% | 4476 / 4624 | 4.67% / 9.36% | 2926 / 3070 |

## 8. 后续决策

不继续沿用本轮 Tiled SiLU 作为累计基线。下一轮只优化 Prefill Fused Add + RMSNorm
kernel，目标是用 warp shuffle 完成规约，并减少 shared memory 与同步开销。
