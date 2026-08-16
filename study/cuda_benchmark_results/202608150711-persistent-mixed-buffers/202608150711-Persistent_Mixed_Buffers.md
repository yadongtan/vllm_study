# 202608150711 - Persistent Mixed Buffers

## 1. 迭代目标

减少 `_execute_mixed_admission()` 每轮产生的临时 Tensor 和 GPU allocator 开销，
同时保持请求立即调度、token budget、请求边界和模型计算不变。

## 2. 代码修改

新增 `PersistentMixedStepBuffers`，在 scheduler 初始化时一次性分配：

- packed `input_ids`、`positions`、`token_cache_slots`；
- `request_cache_slots`、`query_start_loc`、`sequence_lengths`；
- 复用的 position 模板；
- pinned CPU metadata buffer 和 H2D copy event。

运行时在有效 slice 上原位 `copy_` / `fill_`，不再为每次 mixed admission 调用
`torch.arange`、`torch.full`、`torch.tensor` 和 `torch.cat`。开关为：

```bash
STUDY_USE_PERSISTENT_MIXED_BUFFERS=1
```

该开关默认保持关闭，只有 A/B 结果稳定改善后才会改为默认。

## 3. 测试口径

- Qwen2-0.5B-Instruct，BF16；
- 1000 input tokens，50 output tokens；
- 每档 128 requests；
- 16 / 32 并发，`request-rate=inf`，`ignore-eos`；
- `max_model_len=2048`；
- `max_num_batched_tokens=2048`；
- `max_num_seqs=32`；
- 其余优化与 `202608150653-mixed-fused-rope` 一致。

benchmark 会对返回文本重新 tokenize，导致 `total_output_tokens` 失真。本报告统一使用
服务端真实生成量计算：`6400 / duration`。

## 4. 性能结果

| 版本 | 16 并发 tok/s | 32 并发 tok/s |
|---|---:|---:|
| 上一最佳 `202608150653` | 2254.40 | 2546.50 |
| Persistent 第一次 | 2380.47 | 2655.35 |
| 同机近邻基线（关闭开关） | 2281.44 | **2777.02** |
| Persistent 第二次 | 2342.32 | 2640.45 |
| Persistent 两次平均 | **2361.39** | 2647.90 |

相对同机近邻基线：

- 16 并发：`+3.50%`；
- 32 并发：`-4.65%`。

## 5. 资源数据（第一次 Persistent）

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 16 | 34.0% / 96.0% | 4246 / 4252 MiB | 4.86% / 9.17% | 2923 / 3150 MiB |
| 32 | 26.6% / 94.0% | 4251 / 4252 MiB | 5.25% / 9.15% | 2979 / 3214 MiB |

加载后空闲基线为 GPU 4.4%、显存 4242 MiB、CPU 0.07%、内存 2406 MiB。

## 6. 结论

该修改减少了临时分配，但新增的 slice copy、逐项 CPU metadata 写入和 pinned H2D
同步没有改善 32 并发吞吐。第一次相对旧结果的正收益不能在近邻 A/B 中复现。

因此本轮代码保留为可选实验路径，默认关闭，不纳入当前最佳配置。下一步应分析 mixed
Prefill 本身的 GPU/CPU 执行时间，而不是继续微调 metadata 分配。

## 7. 验证

```text
py_compile: passed
git diff --check: passed
Chunked Prefill check passed; max logits error: 0.25
benchmark: 16/32 并发均 128/128 成功
```
