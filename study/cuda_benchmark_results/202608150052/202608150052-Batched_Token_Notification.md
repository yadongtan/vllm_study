# 202608150052 - Batched Token Notification

## 1. 结论

本轮只把 4-step ignore-EOS 流水中同一请求的 4 次 Condition 加锁、通知和流式解码
合并为一次。并发 1/4/8 吞吐分别提升 4.25%/1.58%/3.38%，并发 8 达到
985.26 tok/s。代价是 token 最多 4 个成组交付，平均 ITL 明显上升。

吞吐是当前优化目标，因此默认 `STUDY_PIPELINE_STEPS` 设为 4；需要平滑逐 token
流式输出时可设置为 1。

| 并发 | Step 1 基线 (tok/s) | Batched Notify (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 298.28 | 310.95 | +4.25% | 375.19 |
| 4 | 627.42 | 637.37 | +1.58% | 1305.21 |
| 8 | 953.05 | 985.26 | +3.38% | 2400.77 |

## 2. 单一迭代内容

上一版 step 4 已能连续 enqueue 四组 `D2H + graph replay`，但读取 ring buffer 后仍
对每个 token 分别调用 `request.append()`，每个请求每组发生四次锁竞争、四次
`notify_all()` 和最多四次 tokenizer decode。

新版本新增 `GraphRequest.append_many()`：

1. 先等待四个独立 copy event 并收集每个请求的 token chunk；
2. 每个请求只获取一次 Condition 锁；
3. 一次扩展 `output_ids` 并只通知一次消费者；
4. 流式协程一次解码最多四个新 token。

模型、CUDA Graph、Attention、KV Cache、Prefill、Decode 数学公式均未改变。

## 3. 正确性验证

- 同一 batch 8 prompt 分别走同步路径和 batched-notify 路径。
- 每请求 8 个输出 token 逐项完全一致。
- 每个 ring slot 使用独立 pinned buffer 和 CUDA event。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. 吞吐与流式延迟权衡

| 并发 | Step 1 平均 ITL (ms) | Batched Notify 平均 ITL (ms) | 变化 |
|---:|---:|---:|---:|
| 1 | 2.73 | 9.48 | +247.3% |
| 4 | 3.03 | 10.50 | +247.1% |
| 8 | 3.27 | 11.22 | +242.8% |

一次通知携带多个 token，减少了 Python/GIL/Condition/tokenizer 次数并提高总吞吐，
但客户端看到的 token 到达粒度变粗。并发 8 平均 TPOT 从 3.92 ms 降到 3.56 ms，
说明服务总生成效率提高；ITL 上升反映交付分组，而不是模型每 token 计算变慢。

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`，Split-KV partition 64。
- `STUDY_PIPELINE_IGNORE_EOS=1`，`STUDY_PIPELINE_STEPS=4`。
- `batch_wait_ms=5`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 基线：`cuda_benchmark_results/202608150031`。
- 本轮：`cuda_benchmark_results/202608150052`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.30 | 310.95 | 35.15 | 3.07 | 9.48 |
| 4 | 9.94 | 637.37 | 172.49 | 3.39 | 10.50 |
| 8 | 6.40 | 985.26 | 257.61 | 3.56 | 11.22 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 7.00% / 13% | 1454 / 1454 | 0.18% / 0.29% | 1070 / 1074 |
| 模型和 Graph 加载后 | 1.80% / 4% | 4498 / 4498 | 0.07% / 0.15% | 2135 / 2139 |
| 并发 1 | 67.70% / 92% | 4553 / 4573 | 3.62% / 9.06% | 2901 / 3022 |
| 并发 4 | 56.35% / 95% | 4694 / 4772 | 3.99% / 9.00% | 2953 / 3091 |
| 并发 8 | 45.54% / 98% | 4939 / 5164 | 4.55% / 9.06% | 2920 / 3072 |

## 8. 决策

- 默认 pipeline steps 改为 4，以高吞吐为默认目标。
- `STUDY_PIPELINE_STEPS=1` 保留为低 ITL 模式。
- 下一轮优化应继续减少实际 Prefill/Decode 计算，而不是再扩大通知 chunk。
