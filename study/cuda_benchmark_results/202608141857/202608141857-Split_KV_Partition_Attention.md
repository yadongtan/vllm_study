# Split-KV Partition Attention 性能测试报告

## 1. 结论

本轮在 `qwen2_demo_v1.py` 的 Direct Paged KV Decode Attention 基础上实现了
Split-KV Partition Attention。长 KV 历史不再由单个 CUDA block 串行处理，而是
按 256 token 划分 partition，由多个 block 并行计算局部 attention，再由第二个
kernel 使用稳定 softmax 公式合并结果。

1、4、8 并发均完成 128/128 请求，失败请求为 0。相对上一版 Direct Paged KV，
输出吞吐分别提升 34.0%、41.2% 和 31.2%，平均 ITL 分别下降 30.5%、34.1% 和
31.2%。这说明当前约 1000 token 历史已经足以让 Split-KV 的额外并行度抵消
workspace 写入和第二阶段归并成本。

| 并发 | CUDA Graph V1 | Direct Paged KV | Split-KV | 相对 Direct | 原生 vLLM | vLLM / Split-KV |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 162.75 | 138.70 | 185.88 | +34.0% | 375.19 | 2.02x |
| 4 | 230.61 | 235.71 | 332.75 | +41.2% | 1305.21 | 3.92x |
| 8 | 277.39 | 337.21 | 442.56 | +31.2% | 2400.77 | 5.42x |

单位均为客户端统计的输出 token/s。Split-KV 已显著改善长历史 Decode，但仍未
达到原生 vLLM，尤其在并发增加后差距继续扩大。下一阶段瓶颈已经不只是单个
Attention kernel，还包括固定批次调度、GQA KV 重复读取、非 Attention 算子、
采样与结果回传同步等完整执行链。

## 2. 实现参数

- 被测实现：`study/inference_engine/qwen2_demo_v1.py`。
- Decode Attention：两阶段 Direct Paged Split-KV CUDA kernel。
- Partition size：256 token。
- 最大上下文：2048 token，因此每个 `(request, q_head)` 最多 8 个 partition。
- 第一阶段 grid：`[graph_batch_size, num_q_heads, max_partitions]`。
- 第一阶段只读取该请求 block table 指向的有效物理 KV pages。
- 每个 partition 输出 FP32 记录：`[partial_max, partial_sum, partial_output]`。
- Workspace shape：
  `[max_graph_batch_size, num_q_heads, max_partitions, head_dim + 2]`。
- `head_dim + 2` 中的两个额外元素分别保存局部 softmax 最大值和指数和。
- 第二阶段按全局最大值重新缩放各 partition，合并为最终 attention output。
- Workspace 在 KV Cache 初始化时一次性分配，各层顺序复用，地址固定，兼容
  CUDA Graph replay。
- 环境变量：`STUDY_USE_SPLIT_KV_ATTENTION=1`。
- 旧 Direct Paged KV kernel 保留，可用环境变量值 `0` 回退。

## 3. 正确性验证

正式性能测试前完成以下验证：

- CUDA 扩展在 RTX 4080 SUPER 的 `sm_89` 架构上成功编译。
- BF16 输出与 PyTorch SDPA 对比通过，容差为 `atol=2e-2, rtol=2e-2`。
- 覆盖位置边界：0、15、16、255、256、511、1000、2047。
- 相同输入重复执行结果逐元素一致。
- CUDA Graph batch 1 回放通过。
- CUDA Graph batch 8 及 padding scratch rows 回放通过。
- Python 语法检查和 `git diff --check` 通过。

## 4. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- WSL：Ubuntu 24.04。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`。
- `max_num_seqs=32`。
- CUDA Graph batch sizes：1、2、4、8、16、32。
- Prefill：eager；Decode：固定 batch CUDA Graph replay。
- 并发：1、4、8。
- 每档：128 个请求。
- 每请求：固定 1000 token 输入，请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- Direct Paged KV 对照：`cuda_benchmark_results/202608141526`。
- CUDA Graph V1 对照：`cuda_benchmark_results/202608141432`。
- 原生 vLLM 对照：`cuda_benchmark_results/202608140242`。

## 5. 吞吐和延迟

| 版本 | 并发 | 成功/失败 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | P99 TTFT (ms) | 平均 TPOT (ms) | P99 TPOT (ms) | 平均 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct Paged KV | 1 | 128/0 | 45.31 | 138.70 | 59.53 | - | 7.63 | - | 5.91 |
| Split-KV | 1 | 128/0 | 33.80 | 185.88 | 59.16 | 60.33 | 5.31 | 29.26 | 4.11 |
| Direct Paged KV | 4 | 128/0 | 26.65 | 235.71 | 445.64 | - | 9.99 | - | 7.69 |
| Split-KV | 4 | 128/0 | 18.89 | 332.75 | 334.88 | 719.60 | 6.51 | 30.76 | 5.06 |
| Direct Paged KV | 8 | 128/0 | 18.63 | 337.21 | 728.91 | - | 10.50 | - | 8.11 |
| Split-KV | 8 | 128/0 | 14.20 | 442.56 | 589.00 | 905.79 | 7.15 | 40.73 | 5.58 |

相对 Direct Paged KV：

| 并发 | 输出吞吐变化 | 平均 TTFT 变化 | 平均 TPOT 变化 | 平均 ITL 变化 |
|---:|---:|---:|---:|---:|
| 1 | +34.0% | -0.6% | -30.4% | -30.5% |
| 4 | +41.2% | -24.9% | -34.8% | -34.1% |
| 8 | +31.2% | -19.2% | -31.9% | -31.2% |

客户端分别统计到 6282、6287 和 6283 个输出 token。由于服务输出文本后客户端
会重新 tokenize，个别随机文本不严格保持一个服务端 token 对应一个客户端
token。服务端每档实际执行 6400 个生成 token；上表继续使用 `vllm bench serve`
原始结果，以保持所有历史报告口径一致。

## 6. GPU、显存、CPU 和系统内存

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 13.00% / 14% | 799 / 802 | 0.12% / 0.21% | 1036 / 1043 |
| 模型和 Graph 加载后 | 3.60% / 7% | 3877 / 3878 | 0.07% / 0.12% | 2392 / 2396 |
| 并发 1 | 72.63% / 97% | 4033 / 4133 | 3.61% / 9.22% | 3198 / 3296 |
| 并发 4 | 62.60% / 90% | 4743 / 5025 | 3.99% / 9.53% | 3184 / 3281 |
| 并发 8 | 57.24% / 92% | 6510 / 7273 | 4.07% / 9.06% | 3168 / 3277 |

相对模型加载后空闲基线，测试期间的平均增量为：

| 并发 | GPU 平均增量 | 显存平均增量 | CPU 平均增量 | 系统内存平均增量 |
|---:|---:|---:|---:|---:|
| 1 | +69.03 个百分点 | +156 MiB | +3.55 个百分点 | +807 MiB |
| 4 | +59.00 个百分点 | +866 MiB | +3.92 个百分点 | +792 MiB |
| 8 | +53.64 个百分点 | +2633 MiB | +4.00 个百分点 | +777 MiB |

并发越高，完整测试越快，逐秒监控包含的启动和收尾空闲采样占比也越大，因此
平均 GPU 利用率从 72.63% 降到 57.24% 不能解释为 GPU 工作不足。三档峰值均
达到 90% 以上，而吞吐随并发持续提升。显存随并发增加主要来自更大 Graph batch
执行时的激活和临时 workspace；Split-KV workspace 本身是固定预分配并重复使用。

## 7. 与旧版本和原生 vLLM 的对比

相对没有 Direct Paged KV 的 CUDA Graph V1，Split-KV 吞吐提升分别为 14.2%、
44.3% 和 59.5%。并发越高，删除固定长度 KV gather 和使用 Paged KV direct read
的综合价值越明显。

相对原生 vLLM，Split-KV 吞吐仍低 50.5%、74.5% 和 81.6%，vLLM 分别快
2.02、3.92 和 5.42 倍。vLLM 的优势来自整套推理系统，而不是单独一个
Split-KV kernel：

1. 更成熟的 Paged Attention/FlashAttention kernel 和按硬件、shape 选择实现。
2. GQA query heads 间复用 KV 数据，减少同一 KV head 的重复显存读取。
3. continuous batching 可在每个 step 加入新请求、移除完成请求。
4. 更完善的 CUDA Graph batch 调度及低开销输入更新。
5. RMSNorm、RoPE、MLP、采样等 Attention 之外的融合和优化。
6. 更少的 Python 同步和 token 回传开销。

## 8. 算子微基准与端到端收益

在历史长度约 1001 token 的单算子微基准中：

| Batch | Direct kernel (ms) | Split-KV (ms) | 算子耗时下降 |
|---:|---:|---:|---:|
| 1 | 0.115 | 0.034 | 70.4% |
| 4 | 0.117 | 0.042 | 64.1% |
| 8 | 0.117 | 0.063 | 46.2% |

端到端吞吐提升为 31.2%--41.2%，小于单算子提升。这是预期结果：算子微基准只
测一次 Decode Attention，而完整服务还执行 24 层中的 QKV/O projection、RoPE、
RMSNorm、MLP、采样、调度、网络和 tokenizer 工作。Attention 变快后，其他部分
在总耗时中的占比自然上升。该结果证明 Split-KV kernel 已生效，同时说明下一步
需要优化完整模型执行链，而不是仅继续增加 KV partition 数量。

## 9. 结果文件

- `split-kv-concurrency-{1,4,8}.json`：完整 benchmark 数据。
- `resources-split-kv-concurrency-{1,4,8}.csv`：测试期间逐秒资源采样。
- `resources-baseline-system-before.csv`：服务启动前系统基线。
- `resources-baseline-split-kv-loaded.csv`：模型与 Graph 加载后基线。
- `resource-summary.json`：资源均值和峰值汇总。
- `split-kv-server.log`：服务启动、请求处理和退出日志。

