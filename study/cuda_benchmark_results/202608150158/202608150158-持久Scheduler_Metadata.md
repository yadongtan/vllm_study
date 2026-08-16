# 202608150158 - 持久 Scheduler Metadata

## 1. 本轮只优化什么

本轮只修改 continuous scheduler 每步准备 Decode metadata 的方式：

- active request 的 positions/cache slots 作为 GPU Tensor 持续存在；
- 每次 graph replay 后 positions 原地加一；
- 所有请求都继续时，直接把 `next_tokens/positions/cache_slots` 交给 graph；
- 只有请求结束、active set 收缩时才构造 indices 并执行 `index_select`。

请求接纳时机、Prefill、Decode、cache slot 和输出交付策略都未改变。

## 2. 正确性验证

- CPU continuous batching 测试通过，动态加入请求的输出仍为 `[10,11,12,13]`。
- `py_compile` 和 `git diff --check` 通过。
- CUDA 1/4/8 并发各完成 128 请求，失败请求为 0。

## 3. 性能结果

| 并发 | Continuous V2 | 持久 Metadata | 变化 | 相对无等待基线 | 原生 vLLM | 达到 vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 273.14 | 280.89 tok/s | +2.84% | -10.43% | 375.19 | 74.87% |
| 4 | 740.07 | 741.75 tok/s | +0.23% | +29.13% | 1305.21 | 56.83% |
| 8 | 1061.22 | 1103.82 tok/s | +4.02% | +15.58% | 2400.77 | 45.98% |

| 并发 | V2 ITL | 本轮 ITL | V2 TTFT | 本轮 TTFT |
|---:|---:|---:|---:|---:|
| 1 | 3.16 ms | 3.06 ms | 21.29 ms | 21.78 ms |
| 4 | 4.29 ms | 4.24 ms | 51.20 ms | 53.47 ms |
| 8 | 5.34 ms | 5.42 ms | 103.49 ms | 83.86 ms |

吞吐三档均未回退，并发 1/8 的改善分别为 2.84%/4.02%，说明每步临时 Tensor 和
无条件 gather 确实是固定开销。并发 8 P99 ITL 仍受整段 Prefill 插入影响，不能由
metadata 优化解决。

## 4. 资源指标

加载后基线：GPU 3.8%，显存 4517 MiB，CPU 0.115%，系统内存 2148.51 MiB。

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 1 | 67.97% / 92% | 4571.66 / 4588 MiB | 3.71% / 8.97% | 2946.41 / 3069.07 MiB |
| 4 | 47.00% / 93% | 4665.33 / 4734 MiB | 4.43% / 9.06% | 2938.16 / 3085.99 MiB |
| 8 | 40.00% / 85% | 4842.85 / 4964 MiB | 4.52% / 9.13% | 2973.58 / 3137.71 MiB |

## 5. 结论与下一步

本轮保留。它采用持久 metadata，不等待请求、不依赖固定测试并发，也不聚合输出。

下一轮只消除 CUDA Graph replay 的冗余 padding 写入。当前即使
`actual_batch_size == captured_batch_size`，仍会先 zero input/position、填充全部 scratch
slots，再用真实数据覆盖。完整 batch 不需要这些操作；非完整 batch 也只需重置尾部。
