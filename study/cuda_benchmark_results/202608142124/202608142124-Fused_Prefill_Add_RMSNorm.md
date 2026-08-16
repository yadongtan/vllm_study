# Fused Prefill Add + RMSNorm 性能测试报告

## 1. 结论

本轮只优化 Prefill 的 residual add 和 RMSNorm。完整模型数值对照通过，batch 8、
1000-token、三轮 Prefill 的 CUDA profile 总时间从 266.58 ms 降到 239.57 ms，
下降 10.13%。正式服务测试达到 265.47 / 524.91 / 761.08 tok/s，相对上一轮
自适应 SiLU 版本提升 1.18% / 5.23% / 6.05%。

| 并发 | 上一版本 (tok/s) | Fused Add + RMSNorm (tok/s) | 改善 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 262.39 | 265.47 | +1.18% | 375.19 |
| 4 | 498.83 | 524.91 | +5.23% | 1305.21 |
| 8 | 717.67 | 761.08 | +6.05% | 2400.77 |

## 2. 单一优化内容

原路径在每个 Decoder Layer 中分别执行：

1. `residual + branch_output`；
2. 把结果保存为下一条 residual；
3. `pow -> mean -> add epsilon -> rsqrt -> mul -> weight mul`。

新路径携带 `hidden_states` 和 `residual` 两个状态，调用同一个 CUDA kernel：

1. 读取并相加 `hidden_states + residual`；
2. 将 BF16 舍入后的和写回 residual；
3. 在 block 内计算平方和与 inverse RMS；
4. 将归一化并乘权重后的结果写回 hidden_states。

本轮仅新增 `STUDY_USE_FUSED_PREFILL_RMS_NORM=1`。Attention、RoPE、KV Cache、
MLP GEMM、Decode CUDA Graph 和调度器均未修改。

## 3. 正确性验证

- RMSNorm 单算子与 PyTorch 参考实现通过。
- Fused Add + RMSNorm 单算子与 PyTorch 参考实现通过。
- CUDA Graph capture/replay 对照通过。
- 完整 24 层 Decode 融合/非融合 logits、next token、KV Cache 对照通过。
- 完整 24 层 Prefill 融合/非融合 logits、next token、KV Cache 对照通过。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

测试环境没有安装 pytest，因此使用 `.venv/bin/python3` 直接调用同一测试文件内的
五个测试函数和断言，全部通过。

## 4. Profile 变化

| batch 8，1000 token，三轮 | 非融合 | 融合 | 变化 |
|---|---:|---:|---:|
| Self CUDA 总时间 | 266.58 ms | 239.57 ms | -10.13% |
| 独立 residual `aten::add` | 144 次 / 7.21 ms | 0 | 消除 |
| RMSNorm `aten::pow` | 147 次 / 4.42 ms | 0 | 消除 |
| RMSNorm `aten::mean` | 147 次 / 1.87 ms | 0 | 消除 |
| 融合 kernel | 0 | 144 次 / 10.30 ms | 新增 |

Profile 原始文件：

- `prefill-profile-unfused-rms-b8-s1000.txt`
- `prefill-profile-fused-rms-b8-s1000.txt`

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 上一版本：`cuda_benchmark_results/202608142115`。
- 本轮：`cuda_benchmark_results/202608142124`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.59 | 265.47 | 29.75 | 4.02 | 3.10 |
| 4 | 11.97 | 524.91 | 181.32 | 4.86 | 3.78 |
| 8 | 8.26 | 761.08 | 281.89 | 5.53 | 4.38 |

并发 4 和 8 的平均 TTFT 相对上一轮分别从 198.96/309.03 ms 降到
181.32/281.89 ms，下降 8.87% 和 8.78%。

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 12.60% / 13% | 944 / 950 | 0.29% / 0.94% | 1062 / 1156 |
| 模型和 Graph 加载后 | 3.20% / 6% | 3977 / 3977 | 0.07% / 0.12% | 2132 / 2142 |
| 并发 1 | 61.16% / 82% | 4032 / 4056 | 3.57% / 8.94% | 2965 / 3105 |
| 并发 4 | 54.95% / 100% | 4200 / 4276 | 4.16% / 9.17% | 2992 / 3128 |
| 并发 8 | 45.47% / 82% | 4472 / 4622 | 4.46% / 9.17% | 2957 / 3093 |

## 8. 后续决策

该优化在 profile、吞吐和 TTFT 上方向一致，保留。新的 Prefill profile 显示除 GEMM
外，下一个独立热点是 RoPE：Q/K 的多个逐元素 mul、neg、add 以及每层 cache write。
下一轮只融合 Prefill RoPE 与 KV Cache 写入。
