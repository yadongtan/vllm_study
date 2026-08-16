# 原生 vLLM 与 Packed CUDA Flash Attention V2 性能对比报告

## 1. 结论

在相同模型和相同请求口径下，原生 vLLM 在 1、4、8 并发下均完成
128/128 请求，失败请求为 0。相比当前最新版 Packed V2，原生 vLLM 的输出
吞吐分别达到 7.83、14.66、22.75 倍；并发越高，vLLM 的批处理、Paged KV
Cache、CUDA Graph 和成熟 FlashAttention 调度优势越明显。

| 并发 | Packed V2 (tok/s) | 原生 vLLM (tok/s) | vLLM / V2 | vLLM 吞吐提升 |
|---:|---:|---:|---:|---:|
| 1 | 47.89 | 375.19 | 7.83x | +683.5% |
| 4 | 89.03 | 1305.21 | 14.66x | +1366.0% |
| 8 | 105.55 | 2400.77 | 22.75x | +2174.5% |

## 2. 测试口径

- 分支：`vllm_study/v2.0`。
- 基线提交：`a8a9fc1ae`，在其上实现本次 Packed V2 优化并完成测试。
- GPU：NVIDIA GeForce RTX 4080 SUPER，16 GiB。
- 模型：同一份 `/opt/models/Qwen2-0.5B-Instruct`，BF16。
- 并发：1、4、8。
- 每档：128 请求。
- 每请求：固定 1000 token 输入、50 token 输出。
- 无限请求速率、忽略 EOS、温度 0。
- `max_model_len=2048`。
- `max_num_seqs=32`。
- `max_num_batched_tokens=2048`。
- 启用 Chunked Prefill。
- vLLM 参数：`gpu_memory_utilization=0.80`。
- vLLM 版本：`0.1.dev17269+g60f957ac8.d20260808`。
- vLLM 自动选择 `FLASH_ATTN` 后端，并明确记录使用 FlashAttention 2。
- 服务日志中的 `deep_gemm` 导入 Traceback 是可选后端探测警告；Qwen2 本轮
  使用 FlashAttention 2，三档请求均成功，因此该警告不影响测试结果。
- Packed V2 结果目录：`cuda_benchmark_results/202608140213`。
- 本轮原生 vLLM 结果目录：`cuda_benchmark_results/202608140242`。

## 3. 吞吐与延迟

| 后端 | 并发 | 成功/失败 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | P99 TTFT (ms) | 平均 TPOT (ms) | P99 TPOT (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Packed V2 | 1 | 128/0 | 133.63 | 47.89 | 157.18 | 153.73 | 18.09 | 18.39 |
| 原生 vLLM | 1 | 128/0 | 17.06 | 375.19 | 23.38 | 24.35 | 2.24 | 2.27 |
| Packed V2 | 4 | 128/0 | 71.89 | 89.03 | 247.45 | 641.03 | 40.76 | 45.79 |
| 原生 vLLM | 4 | 128/0 | 4.90 | 1305.21 | 26.60 | 206.10 | 2.55 | 6.09 |
| Packed V2 | 8 | 128/0 | 60.63 | 105.55 | 341.45 | 1335.67 | 70.26 | 74.08 |
| 原生 vLLM | 8 | 128/0 | 2.67 | 2400.77 | 39.44 | 250.88 | 2.53 | 2.58 |

原生 vLLM 相对 Packed V2：

| 并发 | 测试耗时变化 | 平均 TTFT 变化 | 平均 TPOT 变化 |
|---:|---:|---:|---:|
| 1 | -87.2% | -85.1% | -87.6% |
| 4 | -93.2% | -89.3% | -93.7% |
| 8 | -95.6% | -88.5% | -96.4% |

## 4. GPU、显存、CPU 和内存

### 原生 vLLM 本轮实测

| 阶段 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 |
|---|---:|---:|---:|---:|
| 服务前基线 | 9.20% / 13% | 1060 / 1061 | 0.45% / 1.50% | 3.72% / 3.75% |
| vLLM 加载后 | 10.60% / 13% | 14517 / 14518 | 0.11% / 0.18% | 11.58% / 11.60% |
| 并发 1 | 68.63% / 100% | 14518 / 14521 | 4.24% / 9.07% | 13.65% / 14.00% |
| 并发 4 | 44.83% / 99% | 14522 / 14523 | 4.95% / 9.68% | 13.39% / 13.96% |
| 并发 8 | 30.67% / 98% | 14522 / 14523 | 5.33% / 9.59% | 13.30% / 13.94% |

vLLM 使用 `gpu_memory_utilization=0.80` 预分配 KV Cache，因此服务加载后显存
已达到约 14.5 GiB。测试期间峰值仅比加载后增加约 3--5 MiB，这正是预分配
Paged KV Cache 的预期行为。Packed V2 的模型加载后平均显存约 2616 MiB，峰值
显存分别为 2656、3010、3091 MiB；它的保留显存明显更少，但吞吐也远低于
vLLM。

并发 4 和 8 的完整测试仅持续 4.90 秒和 2.67 秒，而资源监控按秒采样，监控
窗口还包含客户端初始化与收尾，所以平均 GPU 利用率被空闲采样稀释。峰值 GPU
利用率 98%--100% 更能说明请求执行时 GPU 已被充分使用。该采样限制不影响由
benchmark 客户端精确计时的吞吐、TTFT 和 TPOT。

## 5. 为什么 vLLM 仍快很多

Packed V2 已取消跨请求无效 Q/KV tile，并把 partial buffer 改为 packed 布局，
但目前主要只优化了 Attention 中的一个环节。vLLM 还同时具备：

1. 成熟的 FlashAttention 2 kernel，包含更细致的 warp/block 划分、访存合并、
   shared memory 和寄存器优化。
2. Paged KV Cache，避免连续大 Tensor 的频繁拼接、复制和重新分配。
3. CUDA Graph，降低每个 decode step 的 Python 和 kernel launch 开销。
4. 更成熟的 continuous batching，可更充分地把多个请求的 prefill/decode 合并。
5. 经过优化的 RMSNorm、RoPE、矩阵乘、采样和模型执行流水线，而不只是 Attention。

并发 1 时差距为 7.83 倍，说明即使没有跨请求 tile，单请求 kernel 和整个模型
执行链仍有较大优化空间。并发从 1 增加到 8 时，Packed V2 吞吐只提高约 2.20
倍，而 vLLM 提高约 6.40 倍，因此下一阶段的重点应是减少每个 decode step 的
固定开销、改进批处理和 KV Cache，而不仅是继续消除跨请求 Attention tile。

## 6. 结果文件

- `vllm-native-concurrency-{1,4,8}.json`：完整 benchmark 明细。
- `resources-vllm-native-concurrency-{1,4,8}.csv`：测试期间逐秒资源采样。
- `resources-baseline-system-before.csv`：服务启动前资源基线。
- `resources-baseline-vllm-native-loaded.csv`：模型和 KV Cache 加载后基线。
- `resource-summary.json`：资源均值和峰值汇总。
- `vllm-native-server.log`：服务日志，本地保留但不建议提交 Git。
