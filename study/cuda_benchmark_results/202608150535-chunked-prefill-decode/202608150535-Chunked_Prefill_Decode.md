# 202608150535 - Chunked Prefill + Decode 同轮调度

## 1. 本轮只优化了什么

本轮只实现真正的 Chunked Prefill 调度，没有增加等待凑批、quiet window 或多 token 聚合：

- 新请求立即获得空闲 KV cache slot。
- prompt 可以从非零 offset 继续写入同一份 Paged KV Cache。
- 每轮先保留所有活跃 Decode 请求，每个 Decode 请求占用 1 个 token budget。
- 用 `max_num_batched_tokens=2048` 的剩余预算选择 prompt chunk。
- Decode token 与 Prefill chunk 通过 packed 布局进入同一个 `model.mixed()` forward。
- prompt 最后一个 chunk 完成后，请求才转入 Decode 状态。
- 没有 Prefill 工作时仍回到原来的固定 batch CUDA Graph Decode 路径。

开关为：

```bash
STUDY_USE_CHUNKED_PREFILL_DECODE=1
```

该开关当前不作为默认值。

## 2. 正确性验证

- CPU 假模型测试覆盖了 8-token budget 下两个 5/7-token prompt 的 chunk 边界。
- 第一轮 query lengths 为 `[5, 3]`，第二轮为 `[1, 4]`；第二轮中的 `1` 是已有请求的
  Decode token，`4` 是第二个 prompt 的剩余 chunk。
- 验证了 sequence lengths 从 `[5, 3]` 正确演进为 `[6, 7]`。
- 验证了 KV cache slots、下一 token position 以及 Prefill 到 Decode 的状态迁移。
- 真实 Qwen2 BF16 GPU 测试将同一 32-token prompt 分为 17+15 两个 chunk，与完整 Prefill
  对比：最终 argmax 完全一致，最大 logits 误差为 0.203125。
- 1/4/8/16/32 并发每档均完成 128 请求，失败数为 0。

## 3. 性能结果

测试口径仍为 1000 input、50 output、128 requests、`request-rate=inf`、
`max_model_len=2048`、`max_num_batched_tokens=2048`、`max_num_seqs=32`。

基线使用当前默认最佳版本。16/32 基线来自紧邻本轮之前的
`202608142355-high-concurrency`；1/4/8 基线来自 `202608150420-vllm-fused-ops`。

| 并发 | 默认最佳 tok/s | Chunked tok/s | 变化 | Chunked TTFT | Chunked TPOT |
|---:|---:|---:|---:|---:|---:|
| 1 | 381.00 | 378.58 | -0.6% | 22.89 ms | 2.76 ms |
| 4 | 1115.46 | 1074.22 | -3.7% | 50.47 ms | 3.41 ms |
| 8 | 1685.51 | 1608.77 | -4.6% | 74.46 ms | 4.34 ms |
| 16 | 1818.92 | 2113.70 | **+16.2%** | 106.49 ms | 6.80 ms |
| 32 | 2233.84 | 2240.75 | +0.3% | 192.79 ms | 13.26 ms |

16 并发 TTFT 从基线的 153.68 ms 降至 106.49 ms，改善约 30.7%；P99 TTFT 从
806.53 ms 降至 379.75 ms。该档位证明“Prefill 不再完全暂停 Decode”在存在空闲 slot 和
请求补位时有效。

## 4. 为什么 32 并发几乎没有改善

本测试的客户端最大并发正好是 32，服务端也只有 32 个真实 cache slots。所有请求都固定
1000-token 输入和 50-token 输出，因而请求会以非常接近的节奏成组完成：

1. 第一批 32 个请求占满全部 slots。
2. 在这些请求完成前，客户端不会发送超过并发限制的下一批请求。
3. 没有空 slot，就没有新的 Prefill 请求可以与当前 Decode 混合。
4. 第一批基本同时结束后，下一批又以 Prefill 为主重新开始。

因此，32 并发下 Chunked Prefill 正确地工作了，但该固定长度基准没有持续产生
“一部分请求 Decode、另一部分请求 Prefill”的稳态。32 并发吞吐仍由 batch-32 Decode graph
自身以及整模型 GEMM 效率决定。

vLLM 的 32 并发优势也不只来自 chunked prefill。它还使用编译后的持久 GPU runner、完整/
分段 CUDA Graph、预分配 metadata buffer、异步调度以及更高效的大 batch 模型执行路径。

## 5. 为什么低并发略有下降

当 prompt 本身能完整放进 2048-token budget 时，Chunked 路径仍要构造 packed input、positions、
token slot mapping、query start locations 和 sequence lengths，并调用 eager mixed forward。
默认路径对等长 prompt 使用更直接的 batched Prefill。1/4/8 并发中可供重叠的 Decode 工作较少，
metadata 构造和 mixed forward 固定开销大于调度收益，因此吞吐下降 0.6%~4.6%。

这与之前完整 prompt mixed 实验的结论一致：只实现调度公式、但没有 vLLM 的持久 runner 和
预分配 metadata buffer，收益会被 Python/Tensor 构造部分抵消。

## 6. 资源指标

模型加载后显存约 3820 MiB。测试期间：

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | WSL 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 1 | 67.5% / 98% | 3857 / 3870 MiB | 3.85% / 9.11% | 2943 / 3110 MiB |
| 4 | 48.1% / 95% | 3901 / 3928 MiB | 4.31% / 9.11% | 2859 / 3025 MiB |
| 8 | 38.2% / 93% | 3928 / 3930 MiB | 4.75% / 9.30% | 2878 / 3082 MiB |
| 16 | 32.7% / 92% | 3930 / 3930 MiB | 5.05% / 8.94% | 2870 / 3087 MiB |
| 32 | 29.8% / 84% | 3930 / 3932 MiB | 4.93% / 9.14% | 2884 / 3077 MiB |

短测试的 GPU 平均值包含 benchmark 客户端准备和收尾空闲时间，峰值与吞吐/延迟更适合判断。

## 7. 决策与下一步

该实现保留为可选功能，因为它是正确的 vLLM 式调度能力，并在 16 并发显著改善吞吐和 TTFT；
但暂不默认开启，因为 1/4/8 略有退化，32 并发没有解决主要瓶颈。

下一步转向 batch-32 Decode graph 本身：先测量整张 graph 的 GPU 时间和 GEMM/Attention 占比，
再只选择一个占比最高、可与 vLLM 实现对齐的执行优化进行迭代。
