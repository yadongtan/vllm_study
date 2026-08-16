# 202608150653 - Mixed Fused RoPE + KV Write

## 1. 本轮优化

packed mixed attention 原路径每层依次执行：

1. QKV projection；
2. 拆分 Q/K/V；
3. Q 和 K 的 RoPE elementwise 运算；
4. 两次 `index_copy_` 将 K/V 写入 paged cache；
5. FlashAttention 读取 paged cache。

本轮复用纯 Decode 已验证的 `fused_rope_cache_write` CUDA kernel。所有 packed token 被视为
`sequence_len=1` 的 token 行，每行仍使用自己的 `positions` 和 `token_cache_slots` 找到物理
page。kernel 一次完成 Q/K RoPE 与 K/V cache 写入，并返回 FlashAttention 所需 Q。

所有 mixed layer 都启用该路径时，model step 不再构造 cos/sin Tensor。Attention 公式、
token budget、请求边界和 paged cache 结构不变。

## 2. 性能结果

基线为 `202608150645-mixed-vllm-rms`。

| 并发 | Mixed vLLM RMSNorm | Fused RoPE + KV Write | 变化 |
|---:|---:|---:|---:|
| 16 | 2192.30 | 2254.40 | +2.8% |
| 32 | 2544.37 | 2546.50 | +0.1% |

均按服务端实际生成 6400 模型 token 校正。benchmark 对文本重新分词得到的 10160/9257
不是模型 token 数，不能直接用于 mixed 版本之间的吞吐比较。

| 并发 | 时长 (s) | 校正 output tok/s | TTFT (ms) | benchmark TPOT (ms) |
|---:|---:|---:|---:|---:|
| 16 | 2.839 | 2254.40 | 105.90 | 4.46 |
| 32 | 2.513 | 2546.50 | 196.46 | 7.51 |

## 3. 资源结果

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 16 | 29.6% / 92% | 4317 / 4324 MiB | 4.56% / 9.83% | 2981 / 3204 MiB |
| 32 | 35.9% / 99% | 4323 / 4324 MiB | 4.85% / 9.14% | 3012 / 3215 MiB |

## 4. 决策

16 并发有明确正收益，32 并发保持不回退，因此：

- `STUDY_USE_FUSED_MIXED_ROPE_CACHE_WRITE` 默认开启；
- `STUDY_USE_MIXED_PREFILL_DECODE` 默认开启；
- 仍可通过环境变量设为 0 做数值或性能对照。

这是本轮最后一个有正收益的独立 CUDA/调度优化。剩余 32 并发差距需要持久化、预分配的
mixed model runner 和 API/engine 专用共享内存协议，无法再通过替换一枚孤立算子解决。

