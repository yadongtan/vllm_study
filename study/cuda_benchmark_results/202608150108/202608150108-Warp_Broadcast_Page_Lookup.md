# 202608150108 - Warp Broadcast Page Lookup

## 1. 结论

本轮只把 GQA partial 加载 K/V 时的 page-table 查询从每 lane 一次改为每 warp 的
lane 0 查询一次，再广播 physical block。正确性通过，但 Decode profile 回退
7.29%，并发 1/4/8 吞吐分别下降 0.75%/5.64%/3.47%。该修改已恢复。

| 并发 | Parallel Merge 基线 (tok/s) | Warp Lookup (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 314.69 | 312.35 | -0.75% | 375.19 |
| 4 | 675.49 | 637.39 | -5.64% | 1305.21 |
| 8 | 996.56 | 961.95 | -3.47% | 2400.77 |

## 2. 单一迭代内容

head_dim 64 时，一个 warp 的 32 lanes 处理同一 token 的连续 32 个维度。原实现中
每个 lane 都根据 token 读取相同的 physical page。候选版本仅让 lane 0 读取
`block_table`，再用 `__shfl_sync` 广播 physical block。

K/V 数据本身仍由全部 lanes 合并读取；online softmax、tile、partition、parallel
merge、KV Cache 和服务流水均未修改。

## 3. 正确性验证

- 与逐 query-head Split-KV 参考实现对照通过。
- CUDA Graph capture/replay 对照通过。
- 与 SDPA 参考实现对照通过。
- partition 边界测试通过。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. Profile 变化

| batch 8，历史长度 1000，20 次 Decode | 原查询 | Warp 广播 | 变化 |
|---|---:|---:|---:|
| GQA partial kernel | 19.215 ms | 21.436 ms | +11.56% |
| Split-KV merge kernel | 1.366 ms | 1.509 ms | +10.47% |
| Self CUDA 总时间 | 61.238 ms | 65.704 ms | +7.29% |

同地址的 page-table 读取很小且能命中缓存。增加 lane 分支、shuffle 和数据依赖比省掉
的缓存读取更贵，因此这一“减少逻辑读取次数”的改动并不等于减少实际显存瓶颈。

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
- 本轮：`cuda_benchmark_results/202608150108`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.21 | 312.35 | 34.54 | 3.07 | 9.47 |
| 4 | 9.94 | 637.39 | 169.88 | 3.41 | 10.54 |
| 8 | 6.56 | 961.95 | 264.52 | 3.49 | 10.90 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 5.40% / 8% | 1444 / 1444 | 0.15% / 0.23% | 1092 / 1094 |
| 模型和 Graph 加载后 | 1.60% / 4% | 4487 / 4487 | 0.07% / 0.09% | 2136 / 2138 |
| 并发 1 | 68.11% / 92% | 4542 / 4561 | 3.57% / 9.32% | 2908 / 3057 |
| 并发 4 | 56.53% / 95% | 4698 / 4791 | 3.98% / 8.95% | 2928 / 3076 |
| 并发 8 | 46.08% / 98% | 4960 / 5131 | 4.70% / 9.23% | 2907 / 3062 |

## 8. 决策

恢复逐 lane page-table 读取，保留 parallel merge。后续 partial kernel 的主要问题是
每个 query head 对 KV 的实际 QK/PV 与在线 softmax 工作，而不是 page-table 查询次数。
