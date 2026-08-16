# Split-KV Partition 128 性能测试报告

## 1. 结论

本轮只把 GQA Split-KV partition size 从 64 增加到 128。正确性通过，但 partial
kernel 和三档端到端吞吐显著回退，因此 partition 128 被拒绝。

| 并发 | Partition 64 基线 (tok/s) | Partition 128 (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 269.43 | 230.17 | -14.57% | 375.19 |
| 4 | 518.61 | 430.77 | -16.94% | 1305.21 |
| 8 | 776.63 | 691.58 | -10.95% | 2400.77 |

## 2. 单一优化内容

partition size 从 64 改为 128 后，`max_model_len=2048` 下每个 request/head 的
最大 partition 数从 32 降为 16，workspace 和 merge 工作量减半；与此同时，每个
partial block 需要顺序处理的 KV token 数翻倍。

grid/block 组织、8-token shared tile、online softmax、KV Cache 和 CUDA Graph 均未
改变。

## 3. 正确性验证

- Partition 128 GQA kernel 与逐 query-head Split-KV 参考实现对照通过。
- 覆盖位置 0、63、127、511、777、999、1500、2047。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. Profile 变化

| batch 8，历史长度 1000，20 次 Decode | Partition 64 | Partition 128 | 变化 |
|---|---:|---:|---:|
| GQA partial kernel | 18.933 ms | 35.186 ms | +85.85% |
| Split-KV merge kernel | 1.907 ms | 1.322 ms | -30.68% |
| Self CUDA 总时间 | 62.894 ms | 79.232 ms | +25.98% |

减少 merge 远不足以抵消 partial block 内串行 KV 工作翻倍，且 block 并行度下降。

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 基线：`cuda_benchmark_results/202608142151`。
- 本轮：`cuda_benchmark_results/202608142206`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.21 | 230.17 | 26.77 | 4.85 | 3.73 |
| 4 | 14.59 | 430.77 | 233.16 | 5.54 | 4.33 |
| 8 | 9.05 | 691.58 | 308.60 | 6.70 | 5.15 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 2.80% / 6% | 932 / 937 | 0.11% / 0.18% | 1079 / 1090 |
| 模型和 Graph 加载后 | 5.40% / 6% | 4083 / 4113 | 0.10% / 0.18% | 2137 / 2142 |
| 并发 1 | 67.44% / 87% | 4212 / 4266 | 3.64% / 9.14% | 2922 / 3035 |
| 并发 4 | 57.24% / 96% | 4440 / 4550 | 4.03% / 8.97% | 2955 / 3063 |
| 并发 8 | 44.88% / 82% | 4736 / 4899 | 4.19% / 9.09% | 2922 / 3065 |

## 8. 后续决策

拒绝 partition 128。继续测试 partition 32，以确认增大 partial block 并行度是否能
抵消 workspace 和 merge 开销；最终在 32/64/128 中选择实测最优值。
