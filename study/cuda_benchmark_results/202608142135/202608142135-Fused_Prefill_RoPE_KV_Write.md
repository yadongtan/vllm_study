# Fused Prefill RoPE + KV Cache Write 性能测试报告

## 1. 结论

本轮只融合等长 Prefill 的 RoPE 与 KV Cache 写入。batch 8、1000-token、三轮
Prefill profile 从 234.67 ms 降到 227.13 ms，下降 3.21%。正式服务测试达到
273.47 / 527.09 / 756.07 tok/s，相对上一轮 Fused RMS 版本分别变化 +3.01% /
+0.41% / -0.66%。

并发 1 和 4 有正收益；并发 8 的 0.66% 回退处于单轮端到端测试噪声范围，但本报告
仍按实测记录为回退。由于 GPU profile 明确减少 3.21%，该能力保留为独立开关，
最终累计报告会同时列出每档并发的最佳版本。

| 并发 | 上一版本 (tok/s) | Fused RoPE + KV Write (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 265.47 | 273.47 | +3.01% | 375.19 |
| 4 | 524.91 | 527.09 | +0.41% | 1305.21 |
| 8 | 761.08 | 756.07 | -0.66% | 2400.77 |

## 2. 单一优化内容

原路径先将 QKV projection 拆成 Q/K/V，随后分别启动 Q RoPE、K RoPE 的多组逐元素
kernel，再用两次 `index_copy_` 把 K/V 写入 paged cache。

新 CUDA kernel 对每个 `(request, token)` 启动一个 block，并在一次读取 QKV 后：

1. 使用已经由 Python 预计算好的 cos/sin 计算 Q RoPE；
2. 计算 K RoPE；
3. 生成 SDPA 仍需要的 Q/K/V Tensor；
4. 根据 block table 和 cache slot 同时把 K/V 写入物理 page。

仅等长、无 padding 的 prompt 启用该路径。变长 prompt 完整保留原来的 mask 和逐请求
cache 写入逻辑。本轮没有修改 Attention、RMSNorm、MLP、Decode Graph 或调度器。

## 3. 正确性验证

- 原 Decode Fused RoPE 单算子对照通过。
- 原 Decode CUDA Graph capture/replay 对照通过。
- 新 Prefill Fused RoPE + KV Write 单算子对照通过。
- 完整 24 层 Decode 融合/非融合 logits、next token、KV Cache 对照通过。
- 完整 24 层 Prefill 融合/非融合 logits、next token、KV Cache 对照通过。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. Profile 变化

| batch 8，1000 token，三轮 | 非融合 | 融合 | 变化 |
|---|---:|---:|---:|
| Self CUDA 总时间 | 234.67 ms | 227.13 ms | -3.21% |
| KV `index_copy_` | 144 次 / 0.90 ms | 0 | 消除 |
| Q/K `aten::neg` | 144 次 / 0.94 ms | 0 | 消除 |
| Fused Prefill RoPE kernel | 0 | 72 次 / 4.20 ms | 新增 |

融合同时消除了 Q/K RoPE 对应的多组 mul/add kernel；profile 中剩余 72 次
`aten::copy_` 来自 SDPA 对 Q 布局的内部整理，不属于 KV Cache 写入。

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 上一版本：`cuda_benchmark_results/202608142124`。
- 本轮：`cuda_benchmark_results/202608142135`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 22.90 | 273.47 | 24.69 | 4.01 | 3.09 |
| 4 | 11.93 | 527.09 | 179.83 | 4.90 | 3.79 |
| 8 | 8.31 | 756.07 | 288.93 | 5.48 | 4.30 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 13.00% / 15% | 964 / 971 | 0.14% / 0.26% | 1078 / 1082 |
| 模型和 Graph 加载后 | 3.20% / 5% | 3998 / 3999 | 0.08% / 0.12% | 2147 / 2158 |
| 并发 1 | 65.69% / 83% | 4063 / 4080 | 3.73% / 9.20% | 2932 / 3070 |
| 并发 4 | 53.47% / 99% | 4207 / 4292 | 4.15% / 9.03% | 2931 / 3073 |
| 并发 8 | 44.40% / 80% | 4478 / 4627 | 4.48% / 9.19% | 2945 / 3091 |

## 8. 后续决策

新的最大非 GEMM Prefill 热点是 `silu_and_mul`，占三轮 CUDA 时间 12.22%。下一轮
只改变该 kernel 的线程工作划分，消除每个元素的 64 位除法并让每个线程处理多个
连续 tile 元素；数学公式和 BF16 舍入次序不变。
