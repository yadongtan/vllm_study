# 202608150058 - Parallel Split-KV Merge

## 1. 结论

本轮只优化 Split-KV 第二阶段 merge kernel。Merge profile 下降 28.37%，整段 Decode
CUDA 时间下降 2.63%；并发 1/4/8 端到端吞吐分别提升 1.20%/5.98%/1.15%。
该优化保留并进入累计基线。

| 并发 | 上一版本 (tok/s) | Parallel Merge (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 310.95 | 314.69 | +1.20% | 375.19 |
| 4 | 637.37 | 675.49 | +5.98% | 1305.21 |
| 8 | 985.26 | 996.56 | +1.15% | 2400.77 |

## 2. 单一迭代内容

旧 merge kernel 由 thread 0 串行遍历所有 partition 计算 `global_max` 和
`global_sum`。随后每个输出维度再次遍历 partition，并重复执行：

```text
exp(partial_max[partition] - global_max)
```

同一个 correction 在 `head_dim=64` 时最多重复计算 64 次。

新实现：

1. 一个 warp 的每个 lane 负责一个 partition；
2. 使用 warp shuffle 并行规约 `global_max` 和 `global_sum`；
3. 每个 partition 的 exponential correction 只计算一次；
4. correction 写入 32-float shared memory；
5. 所有输出维度复用该 correction。

当前 `max_model_len=2048`、partition 64，最大 partition 数正好为 32。wrapper 明确
校验 1 到 32 partitions。Partial kernel、online softmax 和 KV Cache 未修改。

## 3. 正确性验证

- 与逐 query-head Split-KV 参考实现对照通过。
- CUDA Graph capture/replay 对照通过。
- 与 SDPA 参考实现对照通过。
- partition 边界测试通过。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. Profile 变化

| batch 8，历史长度 1000，20 次 Decode | 旧 Merge | Parallel Merge | 变化 |
|---|---:|---:|---:|
| GQA partial kernel | 18.933 ms | 19.215 ms | +1.49% |
| Split-KV merge kernel | 1.907 ms | 1.366 ms | -28.37% |
| Self CUDA 总时间 | 62.894 ms | 61.238 ms | -2.63% |

Partial kernel 没有代码变化，1.49% 差异按 profile 波动处理。Merge 和整段时间均明确
下降，端到端三档并发也都为正收益。

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`，Split-KV partition 64。
- 4-step graph enqueue + batched token notification。
- `batch_wait_ms=5`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 基线：`cuda_benchmark_results/202608150052`。
- 本轮：`cuda_benchmark_results/202608150058`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.06 | 314.69 | 34.48 | 3.04 | 9.39 |
| 4 | 9.38 | 675.49 | 155.63 | 3.30 | 10.44 |
| 8 | 6.36 | 996.56 | 257.36 | 3.31 | 11.15 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 5.40% / 9% | 1458 / 1459 | 0.15% / 0.20% | 1095 / 1103 |
| 模型和 Graph 加载后 | 1.80% / 3% | 4498 / 4498 | 0.11% / 0.16% | 2145 / 2148 |
| 并发 1 | 68.30% / 93% | 4554 / 4578 | 3.54% / 8.97% | 2953 / 3110 |
| 并发 4 | 54.19% / 95% | 4688 / 4768 | 4.16% / 9.06% | 2916 / 3068 |
| 并发 8 | 45.08% / 98% | 4958 / 5160 | 4.61% / 8.94% | 2922 / 3098 |

## 8. 决策

保留 parallel merge。下一轮应优化仍占 Decode 约 31% 的 GQA partial kernel，或减少
占约 58% 的投影 GEMM 调用；所有后续版本以本目录为累计基线。
