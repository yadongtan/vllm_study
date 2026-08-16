# CUDA Flash Attention V2 Packed Tile 优化报告

## 1. 优化结论

V2 已从全局 Q/KV tile 笛卡尔积改为按请求生成有效 tile 工作列表：

- 只启动同一请求内的 `(Q tile, KV tile, Q head)` CUDA block。
- 跨请求 tile 不再启动，也不再执行 QK、softmax 或 PV。
- `partial_max`、`partial_sum`、`partial_o` 改为 packed 布局，不再为跨请求
  组合预留空间。
- reduction kernel 通过每个 query token 的 `partial_start/count`，只归约该请求
  自己的 KV tiles。
- 一轮 scheduler batch 只在 Python 构造一次工作表，24 个 Decoder layer 复用。

优化后，1、4、8 并发均完成 128/128 请求，失败请求为 0，服务日志无 CUDA
异常。并发 4 输出吞吐提升 66.6%，并发 8 提升 185.0%；并发 8 峰值显存从
8819 MiB 降到 3091 MiB，降低 65.0%。

## 2. 测试口径

- 分支：`vllm_study/v2.0`。
- 基线提交：`a8a9fc1ae`，在其上实现本次 Packed Tile 优化。
- 测试目录：`cuda_benchmark_results/202608140213`。
- GPU：NVIDIA GeForce RTX 4080 SUPER，16 GiB。
- 模型：Qwen2-0.5B-Instruct，BF16。
- 环境变量：`STUDY_USE_CUDA_FLASH_ATTENTION_V2=1`。
- Q/KV block size：32/32。
- 并发：1、4、8。
- 每档：128 请求。
- 每请求：固定 1000 token 输入、50 token 输出。
- 无限请求速率、忽略 EOS、温度 0。
- `max_num_seqs=32`。
- `max_batch_num_tokens=2048`。
- `chunk_prefill_tokens=2048`。
- 优化前对照：`cuda_benchmark_results/202608140145`。

## 3. 性能结果

| 版本 | 并发 | 成功/失败 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | P99 TTFT (ms) | 平均 TPOT (ms) | P99 TPOT (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 优化前 V2 | 1 | 128/0 | 133.57 | 47.91 | 156.62 | 152.57 | 18.10 | 18.66 |
| Packed V2 | 1 | 128/0 | 133.63 | 47.89 | 157.18 | 153.73 | 18.09 | 18.39 |
| 优化前 V2 | 4 | 128/0 | 119.78 | 53.43 | 811.75 | 1301.49 | 59.78 | 66.80 |
| Packed V2 | 4 | 128/0 | 71.89 | 89.03 | 247.45 | 641.03 | 40.76 | 45.79 |
| 优化前 V2 | 8 | 128/0 | 172.78 | 37.04 | 1902.66 | 4514.35 | 181.44 | 204.49 |
| Packed V2 | 8 | 128/0 | 60.63 | 105.55 | 341.45 | 1335.67 | 70.26 | 74.08 |

### 相对优化前 V2

| 并发 | 输出吞吐变化 | 平均 TTFT 变化 | 平均 TPOT 变化 | 测试耗时变化 |
|---:|---:|---:|---:|---:|
| 1 | -0.04% | +0.36% | -0.06% | +0.04% |
| 4 | +66.6% | -69.5% | -31.8% | -40.0% |
| 8 | +185.0% | -82.1% | -61.3% | -64.9% |

并发 1 几乎不变是预期结果：单请求不存在跨请求 tile。并发越高，原实现的
跨请求组合越多，因此 packed 工作表收益越明显。

## 4. 与 V1 和早期 CUDA 自定义算子对比

这里的 `custom-cuda` 指 `ragged_gqa_attention`：一个 CUDA block 对应一个
query token 和一个 query head，block 内线程并行完成 QK、softmax 和 PV。它是
早期直接公式实现，不是 Flash Attention V1。

三者使用相同测试口径：1/4/8 并发，每档 128 请求，每请求固定 1000 token 输入、
50 token 输出，BF16，忽略 EOS。

### 吞吐和延迟原始结果

| 后端 | 并发 | 输出吞吐 (tok/s) | 平均 TTFT (ms) | P99 TTFT (ms) | 平均 TPOT (ms) | P99 TPOT (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Packed V2 | 1 | 47.89 | 157.18 | 153.73 | 18.09 | 18.39 |
| Flash V1 | 1 | 13.90 | 400.76 | 420.45 | 65.24 | 68.51 |
| custom-cuda | 1 | 29.70 | 233.41 | 265.48 | 29.59 | 32.29 |
| Packed V2 | 4 | 89.03 | 247.45 | 641.03 | 40.76 | 45.79 |
| Flash V1 | 4 | 11.06 | 2298.40 | 3121.57 | 321.87 | 350.03 |
| custom-cuda | 4 | 65.73 | 362.69 | 970.93 | 54.67 | 59.00 |
| Packed V2 | 8 | 105.55 | 341.45 | 1335.67 | 70.26 | 74.08 |
| Flash V1 | 8 | 8.19 | 5279.98 | 12840.69 | 888.96 | 976.67 |
| custom-cuda | 8 | 81.80 | 476.85 | 1742.16 | 89.90 | 95.29 |

### Packed V2 相对 Flash V1

| 并发 | 输出吞吐变化 | 平均 TTFT 变化 | 平均 TPOT 变化 |
|---:|---:|---:|---:|
| 1 | +244.6% | -60.8% | -72.3% |
| 4 | +704.7% | -89.2% | -87.3% |
| 8 | +1189.4% | -93.5% | -92.1% |

V1 由每个线程独自负责一个 query，并让该线程串行遍历完整打包 KV。并发增加时，
大量跨请求 QK 会先完成计算、再被 mask 丢弃。Packed V2 同时获得了 tile/warp
并行和请求内工作表两方面收益，所以并发越高，相对 V1 的提升越明显。

### Packed V2 相对早期 custom-cuda

| 并发 | 输出吞吐变化 | 平均 TTFT 变化 | 平均 TPOT 变化 |
|---:|---:|---:|---:|
| 1 | +61.2% | -32.7% | -38.8% |
| 4 | +35.5% | -31.8% | -25.4% |
| 8 | +29.0% | -28.4% | -21.9% |

早期 custom-cuda 已经按请求边界读取 KV，不会显式复制 GQA K/V，且没有 V2
优化前巨大的全局 partial 布局。因此它远快于 V1。但它仍由一个 block 负责一个
query/head，并沿较长 KV 做计算；Packed V2 使用 Q/KV tile、warp reduction 和
packed partial，在三个并发档都获得更高吞吐和更低延迟。

## 5. 三种实现的 GPU 与显存对比

以下是各轮测试期间的原始采样值。由于三轮不是同一时间启动，GPU 空闲基线和模型
加载基线略有差异；因此原始显存适合观察总体规模，严格增量应结合各自加载后基线。

| 后端 | 并发 | 平均 GPU 利用率 | 平均显存 (MiB) | 峰值显存 (MiB) |
|---|---:|---:|---:|---:|
| Packed V2 | 1 | 60.79% | 2650 | 2656 |
| Flash V1 | 1 | 83.57% | 2811 | 2955 |
| custom-cuda | 1 | 60.72% | 3402 | 3423 |
| Packed V2 | 4 | 39.94% | 2940 | 3010 |
| Flash V1 | 4 | 92.19% | 3004 | 3194 |
| custom-cuda | 4 | 56.34% | 3472 | 3515 |
| Packed V2 | 8 | 42.00% | 3069 | 3091 |
| Flash V1 | 8 | 95.04% | 3120 | 3241 |
| custom-cuda | 8 | 53.11% | 3573 | 3601 |

Packed V2 的一个关键现象是：并发 8 平均 GPU 利用率只有 42.00%，低于 V1 的
95.04% 和 custom-cuda 的 53.11%，但输出吞吐反而最高。这说明 GPU 利用率不能
单独代表有效性能：V1 的高利用率主要包含大量无效跨请求计算；Packed V2 用更少
GPU 工作完成了更多有效 token。

原始峰值显存方面，Packed V2 在 1/4/8 并发下均低于另外两种实现。相对
custom-cuda，Packed V2 峰值显存分别减少 767、505、510 MiB；不过其中还包括
不同测试轮次加载基线的差异。

## 6. Packed V2 本轮资源与显存

### Packed V2 测试期间指标

| 并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值显存 | 平均/峰值 CPU | 平均/峰值系统内存 |
|---:|---:|---:|---:|---:|---:|
| 1 | 60.79% / 79% | 2650 / 2656 | 16.18% / 16.22% | 3.37% / 9.11% | 10.13% / 10.47% |
| 4 | 39.94% / 98% | 2940 / 3010 | 17.95% / 18.38% | 3.55% / 9.25% | 10.29% / 10.47% |
| 8 | 42.00% / 99% | 3069 / 3091 | 18.74% / 18.88% | 3.62% / 9.00% | 10.30% / 10.48% |

模型加载后平均显存基线为 2616 MiB，因此 Packed V2 的测试期平均显存增量为：

- 并发 1：约 34 MiB。
- 并发 4：约 324 MiB。
- 并发 8：约 453 MiB。

优化前 V2 并发 8 平均/峰值显存为 8528/8819 MiB，优化后为 3069/3091
MiB：

- 平均显存降低约 5459 MiB（64.0%）。
- 峰值显存降低 5728 MiB（65.0%）。

GPU 平均利用率从优化前并发 8 的 79.35% 降到 42.00%，但输出吞吐提升
185.0%。这直接说明原先大量 GPU 工作是跨请求无效计算；现在 GPU 执行的总工作
更少，但有效 token 吞吐更高。

## 7. 实现方式

每个有效工作项包含 8 个 `int64` 字段：

```text
request_idx
q_tile_start
q_tile_length
kv_tile_start
kv_tile_length
partial_tile_base
kv_tile_index_in_request
request_num_kv_tiles
```

原来的 grid 是：

```text
global_num_q_tiles × global_num_kv_tiles × q_heads
```

现在改为：

```text
num_valid_request_local_tile_pairs × q_heads
```

主 kernel 使用 `blockIdx.x` 读取一个有效工作项，使用 `blockIdx.y` 选择 Q head。
因此请求 A 的 Q tile 永远不会与请求 B 的 KV tile 组成工作项。

packed partial 的单 head 大小从：

```text
total_q_tokens × ceil(total_kv_tokens / kv_block_size)
```

改为：

```text
Σ request_q_len × ceil(request_kv_len / kv_block_size)
```

reduction kernel 对每个 query row读取：

```text
partial_start[q_token]
partial_count[q_token]
```

只合并当前请求的 KV tile partial。

## 8. 验证

完成了三层验证：

1. Python/C++/CUDA 静态编译和扩展重编译通过。
2. 构造 3 个不同 `q_len/past_len` 的请求，工作表由原全局笛卡尔积的 54 个
   tile 减为 31 个有效 tile；逐项断言 Q/KV tile 均位于同一请求边界内。
3. BF16 CUDA 输出与逐请求 PyTorch SDPA 对齐，`atol=2e-2`、`rtol=2e-2`。
4. 端到端 4 并发、8 请求、100/10 token 冒烟测试：8/8 成功。
5. 正式 1/4/8 并发测试：每档 128/128 成功。

## 9. 剩余优化方向

跨请求 tile 已彻底取消，但同一请求内仍有两类可继续优化的工作：

1. Prefill 时完全位于 query 未来的 causal KV tile 仍会创建，tile 内 score 最终
   被 mask；可以在构造工作表时跳过整个未来 tile，只保留对角 tile 逐元素 mask。
2. `partial_o` 仍需写入并由第二 kernel 重新读取；可研究在一个 cooperative
   kernel 内完成跨 KV tile online softmax 合并，进一步减少全局显存流量。

## 10. 结果文件

- `cuda-flash-v2-concurrency-{1,4,8}.json`：完整 benchmark 数据。
- `resources-cuda-flash-v2-concurrency-{1,4,8}.csv`：逐秒资源采样。
- `resources-baseline-system-before.csv`：服务启动前基线。
- `resources-baseline-cuda-flash-v2-loaded.csv`：模型加载后基线。
- `resource-summary.json`：资源均值和峰值汇总。
- `cuda-flash-v2-server.log`：约 84 MB，仅保留本地，不建议提交 Git。
