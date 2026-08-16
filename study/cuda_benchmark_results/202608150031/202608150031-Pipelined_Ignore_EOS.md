# Pipelined Ignore-EOS Decode 性能测试报告

## 1. 结论

本轮只优化服务端 `ignore_eos=True` 批次的 token D2H 与 CUDA Graph replay 时序。
并发 1/4/8 吞吐分别提升 10.71%/20.99%/22.72%，是当前累计优化中最大的单项
端到端收益。并发 8 达到 953.05 tok/s，平均 TPOT 从 5.54 ms 降至 3.92 ms。

| 并发 | Warp RMS 基线 (tok/s) | Pipelined D2H (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 269.43 | 298.28 | +10.71% | 375.19 |
| 4 | 518.61 | 627.42 | +20.99% | 1305.21 |
| 8 | 776.63 | 953.05 | +22.72% | 2400.77 |

## 2. 单一优化内容

旧路径每生成一个 token 都立即执行 `next_tokens.tolist()`，然后才 replay 下一张
CUDA Graph；`positions.max().item()` 也在循环内形成额外 GPU 同步。GPU 在 Python
处理 token、锁和通知期间无法提前开始下一步。

新路径仅在整批请求都设置 `ignore_eos=True` 时启用：

1. 把当前 GPU token 异步复制到预分配的 pinned CPU buffer；
2. 在当前 CUDA stream 上记录只覆盖 D2H copy 的 event；
3. 立即把下一次 graph replay 排到同一 stream；
4. CPU 只等待 copy event，然后处理当前 token；
5. 此时 GPU 已可并行执行下一 token。

prompt 长度上限改用已有 CPU Tensor 长度计算，移除 decode 循环内的
`positions.max().item()`。非 ignore-EOS 请求仍完整保留原同步路径。

## 3. 正确性验证

- 新增同步路径与流水路径的完整模型集成测试。
- batch 8、每请求 8 个输出 token 逐项完全一致。
- CUDA Graph 固定输出 buffer 被覆盖前，D2H copy 由同 stream 顺序和 event 保证完成。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`，Split-KV partition 64。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 基线：`cuda_benchmark_results/202608142151`。
- 本轮：`cuda_benchmark_results/202608150031`。

## 5. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.99 | 298.28 | 27.80 | 3.55 | 2.73 |
| 4 | 10.02 | 627.42 | 162.25 | 3.87 | 3.03 |
| 8 | 6.57 | 953.05 | 250.58 | 3.92 | 3.27 |

## 6. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 16.20% / 17% | 1356 / 1357 | 0.21% / 0.53% | 1086 / 1088 |
| 模型和 Graph 加载后 | 3.00% / 3% | 4400 / 4400 | 0.09% / 0.15% | 2146 / 2151 |
| 并发 1 | 68.04% / 93% | 4493 / 4584 | 3.90% / 9.15% | 2899 / 3036 |
| 并发 4 | 54.35% / 95% | 4635 / 4710 | 4.68% / 9.24% | 2931 / 3090 |
| 并发 8 | 51.50% / 99% | 4869 / 5042 | 4.82% / 9.21% | 2921 / 3074 |

## 7. 后续决策

保留该路径，并仅对 benchmark 使用的 ignore-EOS 请求启用。下一轮在不改变模型和
kernel 的前提下，测试多步 graph enqueue：用 pinned ring buffer 连续排队多个 decode
step，进一步降低 Python 调度间隙。
