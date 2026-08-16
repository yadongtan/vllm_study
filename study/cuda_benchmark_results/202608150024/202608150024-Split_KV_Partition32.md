# Split-KV Partition 32 性能测试报告

## 1. 结论

本轮只把 GQA Split-KV partition size 从 64 降到 32。并发 1/4 的服务吞吐分别
提高 4.06%/0.70%，但并发 8 下降 1.36%；同时 Decode profile 回退 9.15%，CUDA
Graph workspace 与加载后显存明显增加。以高并发吞吐为目标，最终保留 partition 64。

| 并发 | Partition 64 基线 (tok/s) | Partition 32 (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 269.43 | 280.38 | +4.06% | 375.19 |
| 4 | 518.61 | 522.27 | +0.70% | 1305.21 |
| 8 | 776.63 | 766.03 | -1.36% | 2400.77 |

## 2. 单一优化内容

partition size 从 64 改为 32 后，`max_model_len=2048` 下每个 request/head 的最大
partition 数从 32 增为 64。partial block 并行度翻倍，每个 block 的 KV 循环减半；
代价是 workspace、空 partition 工作和 merge 遍历均翻倍。

grid/block 以外的 kernel 公式、8-token shared tile、KV Cache 和 CUDA Graph 都未
修改。

## 3. 正确性验证

- Partition 32 GQA kernel 与逐 query-head Split-KV 参考实现对照通过。
- 覆盖位置 0、63、127、511、777、999、1500、2047。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. Profile 变化

| batch 8，历史长度 1000，20 次 Decode | Partition 64 | Partition 32 | 变化 |
|---|---:|---:|---:|
| GQA partial kernel | 18.933 ms | 20.377 ms | +7.63% |
| Split-KV merge kernel | 1.907 ms | 4.208 ms | +120.66% |
| Self CUDA 总时间 | 62.894 ms | 68.651 ms | +9.15% |

并行度增加没有抵消 merge 与固定 graph 中空 partition 的额外成本。并发 1 的端到端
提升更可能来自运行波动或服务层时序，不足以推翻稳定的 kernel profile 结果。

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 基线：`cuda_benchmark_results/202608142151`。
- 本轮：`cuda_benchmark_results/202608150024`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 22.34 | 280.38 | 26.77 | 3.84 | 2.96 |
| 4 | 12.04 | 522.27 | 186.74 | 4.86 | 3.73 |
| 8 | 8.20 | 766.03 | 282.78 | 5.39 | 4.31 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 19.80% / 23% | 1399 / 1408 | 0.14% / 0.23% | 1084 / 1085 |
| 模型和 Graph 加载后 | 3.80% / 7% | 4378 / 4379 | 0.14% / 0.28% | 2133 / 2144 |
| 并发 1 | 57.68% / 81% | 4431 / 4449 | 3.54% / 9.03% | 2906 / 3059 |
| 并发 4 | 50.89% / 78% | 4591 / 4669 | 4.27% / 9.21% | 2937 / 3067 |
| 并发 8 | 46.00% / 99% | 4853 / 5015 | 4.72% / 9.18% | 2951 / 3106 |

## 8. 三档选择

| Partition | Decode profile | 并发 8 | 结论 |
|---:|---:|---:|---|
| 32 | 68.651 ms | 766.03 tok/s | 低并发可选，高并发和显存较差 |
| 64 | 62.894 ms | 776.63 tok/s | 保留 |
| 128 | 79.232 ms | 691.58 tok/s | 拒绝 |

下一轮转向调度和 graph replay 之间的 CPU/GPU 同步，因为它比继续微调 partition 更有
可能缩小与原生 vLLM 的吞吐差距。
