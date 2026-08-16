# 202608150750 - Async Mixed Output Copy

## 迭代目标

在 Mixed Prefill/Decode 调度路径中，将上一轮采样 token 的 GPU 到 CPU
复制放到独立 CUDA stream，并尝试与下一轮 mixed forward 重叠。该优化不改变
请求准入、batch 组成、生成 token 数或调度时机，也不等待凑批。

实验开关：

```bash
STUDY_USE_ASYNC_MIXED_OUTPUT_COPY=1
```

## 实现内容

1. 在默认 stream 中保存上一轮 `next_tokens` 的 GPU snapshot，避免下一轮执行
   覆盖复用的输出缓冲区。
2. 在独立 copy stream 上把 snapshot 异步复制到 pinned CPU buffer。
3. 默认 stream 不等待 D2H 完成，立即提交本轮 Mixed Prefill/Decode forward。
4. mixed forward 提交后等待 copy event，再发布上一轮 token。
5. Mixed metadata 中不再通过 `decode_cache_slots.tolist()` 和
   `decode_positions.tolist()` 读取 GPU Tensor，而是直接使用 scheduler 已有的
   CPU 请求状态，避免两次隐式 GPU 同步。

## 固定测试口径

- 模型：Qwen2-0.5B-Instruct，BF16
- 输入长度：1000 token
- 输出长度：50 token
- 请求数：128
- 并发：16、32
- `request-rate=inf`
- `ignore-eos`
- `max_model_len=2048`
- `max_num_batched_tokens=2048`
- `max_num_seqs=32`
- 启用 Mixed Prefill/Decode、vLLM FlashAttention、vLLM fused ops 和 fused mixed
  RoPE/cache write

返回文本重新 tokenize 会产生 0 到 100 的统计偏差，因此固定按
`128 * 50 = 6400` 个输出 token 除以 benchmark duration 计算吞吐。

## 性能结果

| 并发 | 本轮耗时 | 本轮校正吞吐 | 同机关闭开关基线 | 变化 |
|---:|---:|---:|---:|---:|
| 16 | 2.827459 s | 2263.52 tok/s | 2281.44 tok/s | -0.79% |
| 32 | 2.304772 s | 2776.85 tok/s | 2777.02 tok/s | -0.01% |

正确性：两档均为 128/128 请求成功，0 请求失败；服务日志没有
Traceback、CUDA exception 或 scheduler exception。

## 资源统计

模型加载后的空闲基线：GPU 平均/峰值利用率 8.2%/13.0%，显存
4287.2/4288.0 MiB，CPU 平均/峰值 0.061%/0.121%，进程内存
2440.4/2444.8 MiB。

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 16 | 33.0% / 93.0% | 4291.5 / 4298.0 MiB | 5.04% / 8.83% | 2960.7 / 3185.3 MiB |
| 32 | 26.2% / 94.0% | 4297.9 / 4298.0 MiB | 5.03% / 9.12% | 3000.2 / 3221.6 MiB |

资源采样包含 benchmark 启动和结束阶段，测试持续时间较短，因此平均 GPU
利用率只能用于同口径参考，峰值更能说明模型执行期间 GPU 已被充分拉起。

## 结论

异步复制没有形成稳定收益。上一轮 token 只有几十个 `int64`，D2H 数据量很小；
而 copy event、snapshot copy 和同步管理抵消了可重叠的收益。32 并发与基线完全
处于噪声范围，16 并发反而略退化。

因此 `STUDY_USE_ASYNC_MIXED_OUTPUT_COPY` 继续默认关闭，代码仅作为可选实验路径
保留。下一步应优先减少 mixed forward 的 GEMM/attention 实际耗时，而不是继续
优化这条很小的 token D2H 路径。
