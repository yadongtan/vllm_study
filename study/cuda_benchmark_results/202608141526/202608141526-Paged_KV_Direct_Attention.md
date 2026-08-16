# Paged KV Direct Attention 性能测试报告

## 1. 结论

本轮只优化 `qwen2_demo_v1.py` 的 Paged KV Cache Decode 读取路径，其他部分
保持不变：Prefill 仍使用 PyTorch SDPA，RoPE、MLP、采样、固定 batch CUDA
Graph 和服务调度器均未优化。

原 CUDA Graph V1 在每层 Decode 时先通过 `index_select` 把固定 2048 长度的
离散 KV 数据整理成连续 Tensor，再调用 PyTorch SDPA。本轮改为 16-token pages
和固定 `block_table`，自定义 CUDA Attention 直接通过 block table 读取有效
历史 KV，不再生成连续 history K/V，也不再创建固定长度 attention mask。

优化结果与 batch size 明显相关：

- 并发 1：输出吞吐下降 14.8%，平均 ITL 增加 20.4%。
- 并发 4：输出吞吐提升 2.2%，平均 ITL 降低 13.0%。
- 并发 8：输出吞吐提升 21.6%，平均 ITL 降低 28.1%。
- 服务加载后显存从约 5301 MiB 降至 4036 MiB。
- 并发 8 峰值显存从 7836 MiB 降至 6786 MiB。

结论是：删除 `index_select gather` 的方向有效，随着 batch 增大收益明显；但
当前教学版 8-warp CUDA kernel 的固定成本较高，单 batch 性能仍不如高度优化的
PyTorch SDPA。它还不是 vLLM Paged Attention 的完整性能水平。

| 并发 | CUDA Graph V1 (tok/s) | Direct Paged KV (tok/s) | 变化 | 原生 vLLM (tok/s) | vLLM / Direct |
|---:|---:|---:|---:|---:|---:|
| 1 | 162.75 | 138.70 | -14.8% | 375.19 | 2.71x |
| 4 | 230.61 | 235.71 | +2.2% | 1305.21 | 5.54x |
| 8 | 277.39 | 337.21 | +21.6% | 2400.77 | 7.12x |

## 2. 什么是 index_select gather

Paged KV Cache 的物理内存由多个 pages/blocks 组成。一个请求的逻辑历史可能
位于物理 block 2、19、7 中，而不是连续内存：

```text
请求逻辑 blocks:  [0, 1, 2]
block_table:      [2, 19, 7]
物理 KV Cache:    ... block 2 ... block 7 ... block 19 ...
```

原来的 `append_and_get()` 或 CUDA Graph Decode 路径会执行类似：

```python
block_ids = block_table[request]
packed_key = key_cache.index_select(0, block_ids)
packed_value = value_cache.index_select(0, block_ids)
```

`index_select` 根据 `block_ids` 从源 Tensor 的第 0 维取出指定位置。这里的
`gather` 不是 Python 函数名，而是指一种数据收集过程：GPU 从多个离散地址读取
数据，再复制到一个新的连续 Tensor。

```text
离散 pages                    新连续 Tensor
block 2  ───────────────┐
block 19 ────────────┐  ├──> [block 2 | block 19 | block 7]
block 7  ─────────┐  │  │
                  └──┴──┘
```

它有四项成本：

1. 完整历史 K 和 V 都要额外从显存读取一次。
2. 需要为连续 K/V 分配临时 Tensor。
3. 收集完成后 Attention 又要重新读取这份连续 K/V。
4. 原 Graph V1 固定收集 2048 个位置，即使实际历史只有约 1000 token。

以本模型为例，这个操作发生在每一个 Decoder 层、每一个 Decode step。历史越
长、batch 越大，重复复制的数据越多。Paged KV Cache 虽然避免了缓存增长时的
重新分配，但如果 Attention 之前仍执行 gather，就没有获得真正的零整理读取。

本轮的新路径是：

```text
Query + block_table + sequence position
                  |
                  v
Paged GQA CUDA kernel
                  |
                  +--> 直接读取物理 KV pages
                  +--> QK
                  +--> online softmax
                  +--> PV
                  v
Attention output
```

中间不再产生 packed history K/V。

## 3. 实现范围

### Paged KV Cache 布局

- KV block size：16 token。
- KV shape：`[layer, physical_block, block_offset, kv_head, head_dim]`。
- 每个真实 slot 和 padding scratch slot 都有固定 block table。
- 固定地址 pages 可以被 CUDA Graph 捕获并重复 replay。
- Prefill 写入时通过 slot、逻辑 block 和 block offset 直接计算物理地址。

### Direct Paged GQA CUDA kernel

- 一个 CUDA block 处理一个 `(请求, query head)`。
- block 内使用 8 个 warp 分摊有效历史 KV。
- 每个 warp 独立计算局部 online softmax。
- 8 个 warp 在同一个 block 的共享内存中合并局部 max、sum 和输出。
- GQA 映射保持 `kv_head = q_head / group_size`，不复制 K/V heads。
- 只遍历 `positions[request] + 1` 个有效 token，不计算未来位置。

### 未修改部分

- Prefill Attention。
- Chunked Prefill。
- RoPE 计算。
- RMSNorm、Residual 和 MLP。
- 每 token `.tolist()` 同步。
- 固定 batch 调度策略。
- CUDA Graph batch sizes：`1/2/4/8/16/32`。

## 4. 正确性验证

使用随机 BF16 Q/K/V 和固定分页 block table 验证：

- batch size：3。
- Q heads：14。
- KV heads：2。
- head dim：64。
- 历史位置：0、1000、2047。
- 相同输入连续执行两次，要求逐元素完全一致。
- 与原 PyTorch BF16 `scaled_dot_product_attention` 比较。
- 容差：`atol=2e-2, rtol=2e-2`。

最终单-kernel 8-warp 实现通过以上验证。测试过程中曾尝试双 kernel split-K
版本，但发现临时 partition buffer 存在数据竞争；相关无效性能结果已删除，没有
纳入本报告。

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`。
- Graph batch sizes：`1/2/4/8/16/32`。
- 并发：1、4、8。
- 每档：128 请求。
- 每请求：固定 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 原 CUDA Graph V1：`cuda_benchmark_results/202608141432`。
- Direct Paged KV：`cuda_benchmark_results/202608141526`。
- 原 Packed V2：`cuda_benchmark_results/202608140213`。
- 原生 vLLM：`cuda_benchmark_results/202608140242`。

## 6. 吞吐和延迟

| 版本 | 并发 | 成功/失败 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUDA Graph V1 | 1 | 128/0 | 38.60 | 162.75 | 57.07 | 6.34 | 4.91 |
| Direct Paged KV | 1 | 128/0 | 45.31 | 138.70 | 59.53 | 7.63 | 5.91 |
| CUDA Graph V1 | 4 | 128/0 | 27.25 | 230.61 | 404.90 | 11.78 | 8.84 |
| Direct Paged KV | 4 | 128/0 | 26.65 | 235.71 | 445.64 | 9.99 | 7.69 |
| CUDA Graph V1 | 8 | 128/0 | 22.65 | 277.39 | 824.36 | 14.30 | 11.28 |
| Direct Paged KV | 8 | 128/0 | 18.63 | 337.21 | 728.91 | 10.50 | 8.11 |

客户端分别统计到 6284、6282 和 6283 个文本 token；服务端实际生成量均为
6400 token。按服务端 6400 token 和总耗时归一化，吞吐分别为 141.26、
240.14 和 343.49 tok/s，与客户端统计趋势一致。

## 7. GPU、显存、CPU 和系统内存

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 服务加载后 | 3.00% / 6% | 4036 / 4038 | 0.12% / 0.15% | 3073 / 3091 |
| 并发 1 | 77.35% / 99% | 4189 / 4212 | 3.59% / 9.26% | 3869 / 3961 |
| 并发 4 | 71.55% / 99% | 4968 / 5182 | 3.87% / 9.18% | 3902 / 4017 |
| 并发 8 | 64.60% / 91% | 6301 / 6786 | 4.01% / 9.10% | 3876 / 3989 |

显存下降来自删除每层固定 2048 长度的 `history_key/history_value` 和 SDPA
相关 Graph workspace。GPU 利用率并没有提高，但并发 8 的完成时间明显缩短，
说明 GPU 执行的冗余显存复制减少了。单看利用率无法判断有效计算比例。

## 8. 为什么仍远慢于 vLLM

本轮只解决了 Paged KV 的直接读取。当前 CUDA kernel 仍是教学实现：

- 每个 Q head 独立读取共享 KV head，没有跨 GQA heads 复用 KV tile。
- QK 和 PV 使用标量 BF16 转 FP32 运算，没有向量化加载。
- 没有使用 Tensor Core、warp-level MMA 或专门的 Paged Attention 模板。
- 8 个 warp 固定启动，batch 1 和较短历史时固定成本偏高。
- Prefill、模型融合、异步 token 回传和 continuous batching 都未优化。

因此并发 8 虽然比原 CUDA Graph V1 快 21.6%，但 vLLM 仍快 7.12 倍。这次
结果只证明删除 `index_select gather` 的方向正确，不能证明当前教学 kernel 已
达到成熟 Paged Attention kernel 的性能。

## 9. 稳定性记录

正式 benchmark 和服务均正常退出，结果文件完整。释放六张 CUDA Graph 后，
WSL 服务再次进入不可响应状态；本次需要强制结束卡住的 `WslService` 进程后
重新启动服务。该问题与上一轮可重复，建议后续单独调查 WSL、CUDA 驱动和
PyTorch CUDA Graph 资源释放的兼容性。

## 10. 结果文件

- `cuda-graph-v1-concurrency-{1,4,8}.json`：完整 benchmark 数据。
- `resources-cuda-graph-v1-concurrency-{1,4,8}.csv`：资源逐秒采样。
- `resources-baseline-system-before.csv`：测试前系统基线。
- `resources-baseline-cuda-graph-v1-loaded.csv`：模型与 Graph 加载后基线。
- `resource-summary.json`：资源均值和峰值汇总。
- `cuda-graph-v1-server.log`：服务启动、请求与退出日志。
