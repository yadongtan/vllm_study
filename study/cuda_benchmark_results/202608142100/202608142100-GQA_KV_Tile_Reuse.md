# GQA KV Tile Reuse 性能测试报告

## 1. 结论

本轮只优化 Split-KV Paged Attention 的 K/V 读取。Qwen2-0.5B 的 14 个 query
heads 由 2 个 KV heads 服务，每个 KV head 对应 7 个 query heads。上一版本按
query head 启动 partial block，同一段 K/V 会被 7 个 block 重复从全局显存读取。

新 kernel 按 `(request, KV head, KV partition)` 启动 block，7 个 query-head
warps 共享加载到 shared memory 的 K/V tile。1、4、8 并发均完成 128/128 请求，
失败为 0。相对 Fused RoPE + Paged KV Write 基线，吞吐分别提升 1.53%、2.17% 和
1.44%，平均 TPOT 分别下降 1.76%、6.22% 和 5.99%。三档趋势一致，因此保留为
默认实现。

| 并发 | 上一版本 (tok/s) | GQA tile reuse (tok/s) | 提升 | 原生 vLLM (tok/s) | vLLM / 本轮 |
|---:|---:|---:|---:|---:|---:|
| 1 | 225.59 | 229.04 | +1.53% | 375.19 | 1.64x |
| 4 | 386.08 | 394.45 | +2.17% | 1305.21 | 3.31x |
| 8 | 502.03 | 509.26 | +1.44% | 2400.77 | 4.71x |

## 2. 实现方式

原 partial grid：

```text
[batch_size, num_query_heads=14, ceil(max_model_len / 256)]
```

新 partial grid：

```text
[batch_size, num_kv_heads=2, ceil(max_model_len / 64)]
```

每个 block 使用 8 个 warps：

- warp 0 到 warp 6 分别负责同一 GQA group 的 7 个 query heads。
- 全部 256 个线程合作把 8-token K/V tile 加载到 shared memory。
- 7 个 query warps 对共享 tile 分别完成 QK、online softmax 和 V 累加。
- partial max、sum 和未归一化输出仍写入原 Split-KV workspace。
- 第二阶段 merge kernel 不变。

按 KV head 合并 block 后，batch 1 的 block 数会下降。为了恢复 SM 并行度，本路径
将 partition 从 256 token 缩小为 64 token。2048 上下文因此有 32 个 partitions，
batch 1 时共有 `2 * 32 = 64` 个 partial blocks，避免只有 16 个 blocks。

## 3. 正确性和 Graph 验证

以下验证全部通过：

- batch 1、4、8 与原 Split-KV kernel 输出对比。
- position 0、63、127、511、777、999、1500、2047。
- 空 partition 中性 max/sum/output 写入。
- CUDA Graph 捕获和 replay。
- BF16 输出容差 `atol=2e-2, rtol=2e-2`。
- Python 语法和扩展重新编译。

## 4. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`。
- CUDA Graph batch sizes：1、2、4、8、16、32。
- `STUDY_USE_SPLIT_KV_ATTENTION=1`。
- `STUDY_USE_GQA_KV_TILE_REUSE=1`。
- `STUDY_USE_FUSED_RMS_NORM=1`。
- `STUDY_USE_FUSED_SILU_MUL=0`。
- `STUDY_USE_FUSED_ROPE_CACHE_WRITE=1`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 基线：`cuda_benchmark_results/202608142039`。
- 本轮：`cuda_benchmark_results/202608142100`。

## 5. 吞吐和延迟

| 版本 | 并发 | 成功/失败 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | P99 TTFT (ms) | 平均 TPOT (ms) | P99 TPOT (ms) | 平均 ITL (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 上一版本 | 1 | 128/0 | 27.84 | 225.59 | 60.56 | 59.21 | 4.06 | 22.59 | 3.15 | 4.00 |
| GQA reuse | 1 | 128/0 | 27.43 | 229.04 | 60.35 | 58.71 | 3.99 | 22.28 | 3.09 | 3.81 |
| 上一版本 | 4 | 128/0 | 16.28 | 386.08 | 302.77 | 439.69 | 5.17 | 28.01 | 3.99 | 5.19 |
| GQA reuse | 4 | 128/0 | 15.93 | 394.45 | 302.13 | 658.16 | 4.85 | 27.49 | 3.79 | 4.97 |
| 上一版本 | 8 | 128/0 | 12.52 | 502.03 | 546.01 | 849.69 | 5.87 | 37.77 | 4.46 | 6.30 |
| GQA reuse | 8 | 128/0 | 12.34 | 509.26 | 538.65 | 885.23 | 5.52 | 31.36 | 4.34 | 5.58 |

P99 TTFT 受固定批次排队边界影响，并发 4/8 有波动；TPOT、ITL 和总吞吐的方向
一致，表明 Decode GPU 路径确实得到小幅改善。

## 6. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 14.00% / 16% | 868 / 871 | 0.15% / 0.29% | 1061 / 1063 |
| 模型和 Graph 加载后 | 3.00% / 6% | 3906 / 3908 | 0.30% / 0.72% | 2121 / 2124 |
| 并发 1 | 67.47% / 99% | 4049 / 4082 | 3.58% / 9.09% | 3107 / 3262 |
| 并发 4 | 58.05% / 87% | 4769 / 5063 | 4.06% / 9.23% | 3132 / 3237 |
| 并发 8 | 53.95% / 93% | 6670 / 7839 | 4.22% / 9.23% | 3123 / 3226 |

64-token partition 将 workspace 从最多 8 个 partials 增加到 32 个 partials，
并发 8 的 CUDA Graph pool 和 workspace 峰值比上一轮更高。这是用少量显存换取
更多 parallel blocks 和 KV 读取复用。

## 7. 收益为什么有限

K/V 全局内存读取减少，但代价包括 shared-memory 装载、每 8 token 两次 block
同步、更多 64-token partials 和更重的 merge。该实现改善了内存流量，却还不是
Tensor Core/warp-specialized 的成熟 paged attention kernel，因此收益为 1% 到 2%。

下一步必须对完整 Decode CUDA Graph 做 kernel 时间排序。若 GEMM 已占绝对多数，
继续融合小逐元素算子的收益会快速递减；后续应优先处理实际第一热点。

## 8. 结果文件

- `gqa-reuse-concurrency-{1,4,8}.json`。
- `resources-gqa-reuse-concurrency-{1,4,8}.csv`。
- `resources-baseline-system-before.csv`。
- `resources-baseline-gqa-reuse-loaded.csv`。
- `resource-summary.json`。
- `gqa-reuse-server.log`。
