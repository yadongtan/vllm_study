# 202608150649 - Mixed Write Index 跨层复用

## 1. 本轮优化

mixed attention 原先每层调用一次：

```python
cache.slot_mapping(token_cache_slots, positions)
```

本轮在 model step 开头构造一次 KV 物理写入索引，供 24 层 `index_copy_` 复用。上一轮
block table 修改已撤销，因此只有 write index 生命周期发生变化。

## 2. 性能结果

| 并发 | Mixed vLLM RMSNorm | Write index 复用 | 变化 |
|---:|---:|---:|---:|
| 16 | 2192.30 | 2140.88 | -2.3% |
| 32 | 2544.37 | 2538.77 | -0.2% |

均按实际 6400 模型 token 校正。16 并发回退明确，32 并发无有效改善，因此该修改撤销。

| 并发 | 时长 (s) | TTFT (ms) | benchmark TPOT (ms) |
|---:|---:|---:|---:|
| 16 | 2.989 | 113.73 | 4.67 |
| 32 | 2.521 | 169.45 | 8.57 |

## 3. 资源结果

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 16 | 37.0% / 100% | 4316 / 4324 MiB | 4.81% / 8.95% | 2947 / 3184 MiB |
| 32 | 28.3% / 93% | 4323 / 4324 MiB | 4.73% / 9.30% | 3004 / 3236 MiB |

## 4. 决策

metadata 微算子不是 mixed forward 的主要开销。下一轮不再调整临时张量生命周期，而是
融合实际 GPU 工作：把 mixed Q/K RoPE 和 K/V paged cache 写入合并到现有 fused CUDA
kernel，并取消 mixed model 的 cos/sin 表构造。

