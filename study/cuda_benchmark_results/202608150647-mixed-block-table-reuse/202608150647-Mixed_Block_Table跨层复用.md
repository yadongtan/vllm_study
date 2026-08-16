# 202608150647 - Mixed Block Table 跨层复用

## 1. 本轮优化

packed mixed attention 原先在每个 Transformer layer 中执行：

```python
cache.block_table_int32.index_select(0, request_cache_slots)
```

本轮尝试在 model step 开头构造一次 block table，供 24 层 FlashAttention 复用。其他 mixed
计算、scheduler 和输出路径不变。

## 2. 性能结果

| 并发 | Mixed vLLM RMSNorm | Block table 复用 | 变化 |
|---:|---:|---:|---:|
| 16 | 2192.30 | 2088.04 | -4.8% |
| 32 | 2544.37 | 2434.95 | -4.3% |

均按服务实际生成 6400 个模型 token 校正。

| 并发 | 时长 (s) | TTFT (ms) | benchmark TPOT (ms) |
|---:|---:|---:|---:|
| 16 | 3.065 | 120.18 | 4.72 |
| 32 | 2.628 | 189.85 | 8.69 |

## 3. 资源结果

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 16 | 36.4% / 92% | 4309 / 4317 MiB | 4.63% / 9.70% | 2964 / 3179 MiB |
| 32 | 34.7% / 92% | 4316 / 4317 MiB | 5.17% / 9.52% | 3040 / 3293 MiB |

## 4. 决策

减少 23 次很小的 `index_select` 没有抵消长生命周期临时 block table 对 allocator/调用时序
的影响，端到端结果明确回退，因此该修改已撤销。

下一轮复用另一份 metadata：KV 写入物理 page 的 `write_indices`。它当前由
`slot_mapping(token_cache_slots, positions)` 在每层重复计算，下一版改为每个 mixed model
step 计算一次，block table 恢复每层构造，保持单变量测试。

