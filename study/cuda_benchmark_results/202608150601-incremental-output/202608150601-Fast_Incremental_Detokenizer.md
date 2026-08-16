# 202608150601 - Fast Incremental Detokenizer

## 1. 本轮优化

上一版批量输出处理器虽然把每个 GPU step 的输出合并为一次 event-loop
callback，但仍会对每个新 token 重复解码该请求的全部累计输出。输出越长，重复工作越多。

本轮改为使用 `tokenizers.decoders.DecodeStream`：

- 每个请求维护独立的增量解码状态；
- 每个 token 只进入解码器一次；
- 正常路径只产生本次新增的文本片段；
- 增量解码异常时，回退到完整 tokenizer decode，保证输出正确性；
- 没有等待凑批，也没有改变每次模型 Decode 只生成一个 token 的语义。

测试开关：

```bash
STUDY_USE_ASYNC_OUTPUT_COPY=1
STUDY_USE_ASYNCIO_OUTPUT_QUEUE=1
STUDY_USE_BATCH_OUTPUT_PROCESSOR=1
STUDY_USE_VLLM_FLASH_ATTN=1
STUDY_USE_VLLM_FUSED_OPS=1
STUDY_USE_MIXED_PREFILL_DECODE=0
STUDY_USE_CHUNKED_PREFILL_DECODE=0
```

## 2. 测试口径

- 模型：Qwen2-0.5B-Instruct，BF16；
- 输入 1000 token，输出 50 token；
- 每档 128 请求，`request-rate=inf`，`ignore-eos`；
- `max_model_len=2048`；
- `max_num_batched_tokens=2048`；
- `max_num_seqs=32`；
- 并发：1 / 4 / 8 / 16 / 32。

## 3. 性能结果

客户端会把流式文本再次 tokenize，因此 JSON 中记录的 `total_output_tokens`
为 6322--6324，而服务端按请求上限实际生成了固定的 6400 token。为避免 tokenizer
文本往返造成约 1% 的计数误差，下表同时给出原始值和按 `6400 / duration` 校正后的值。

| 并发 | 时长 (s) | 客户端统计 tok/s | 6400-token 校正 tok/s | 平均 TTFT (ms) | 平均 TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.469 | 384.00 | 388.62 | 25.22 | 2.54 |
| 4 | 5.999 | 1054.08 | 1066.92 | 51.29 | 3.30 |
| 8 | 4.010 | 1576.54 | 1595.99 | 76.04 | 4.26 |
| 16 | 2.970 | 2128.81 | 2155.08 | 89.26 | 6.89 |
| 32 | 2.523 | 2506.11 | 2536.63 | 162.73 | 11.46 |

## 4. 与上一版 Asyncio Queue 对比

为了保持同一计数口径，上一版也按每档实际生成 6400 token 校正。

| 并发 | Asyncio Queue | Fast Incremental | 改善 |
|---:|---:|---:|---:|
| 16 | 2009.00 | 2155.08 | +7.3% |
| 32 | 2485.76 | 2536.63 | +2.0% |

本轮对 16/32 并发均有稳定正收益，因此保留批量输出处理器和增量解码路径。

## 5. 资源观察

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 16 | 29.2% / 94% | 4041 / 4041 MiB | 4.57% / 9.18% | 3016 / 3229 MiB |
| 32 | 22.2% / 92% | 4041 / 4041 MiB | 4.82% / 9.22% | 3010 / 3250 MiB |

加载完成、测试前的 GPU 利用率均值为 1.8%，CPU 利用率均值为 0.06%。短测试中的
按时间采样 GPU 均值会受到启动和结束空闲区间影响，因此主要结合峰值、请求时延和
CUDA profiler 判断。显存保持稳定，说明本轮没有额外 KV Cache 泄漏。

## 6. 结论和下一步

增量解码消除了完整输出的重复 tokenizer decode，但 32 并发 TPOT 仍为 11.46 ms；
同一模型的 batch-32 CUDA Graph 单步约 3.01 ms。这说明剩余主要差距位于 GPU replay
之间的 Python scheduler、输出发布和 API 线程 GIL 竞争，而不是模型 kernel 本身。

