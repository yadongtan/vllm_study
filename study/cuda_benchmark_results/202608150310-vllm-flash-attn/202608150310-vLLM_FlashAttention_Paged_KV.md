# 202608150310 - vLLM FlashAttention Paged KV

## 1. 本轮优化

此前 Decode 使用项目自定义的 Paged GQA Attention CUDA kernel。该实现已经支持
Paged KV Cache、Split-KV 和 GQA KV tile 复用，但在 block 内数据搬运、warp 级归约、
长序列分区和算子调度方面仍明显落后于 vLLM 的 FlashAttention 后端。

本轮增加 `STUDY_USE_VLLM_FLASH_ATTN=1`，在保持现有 KV Cache 物理布局
`[block, token, kv_head, head_dim]` 和现有调度器不变的前提下，Decode Attention 改为调用
vLLM 已编译的 `vllm.vllm_flash_attn.flash_attn_varlen_func`：

- 增加 CUDA Graph 地址稳定的 `int32` block table。
- 增加固定地址的 Decode `query_start_loc`。
- 每轮只更新有效长度、slot 和 block table 内容，不重新分配 Tensor。
- Attention 元数据在一次模型 Decode 中构造一次，并由全部 Transformer 层复用。
- 保留 CUDA Graph batch size 1/2/4/8/16/32、Paged KV Cache 和连续批处理语义。
- 与上一轮异步输出复制叠加使用，没有增加等待窗口、最小批量或多 token 聚合。

测试环境变量：

```bash
STUDY_USE_ASYNC_OUTPUT_COPY=1
STUDY_USE_VLLM_FLASH_ATTN=1
```

## 2. 正确性与微基准

- 原始 Attention 输出与参考实现最大绝对误差：`0.00390625`。
- 全模型 logits 最大绝对差：`0.21875`，处于 BF16 验证容差内。
- CUDA Graph capture 和 replay 均通过。
- 1/4/8 并发各完成 128 请求，失败数均为 0。

batch 8、历史长度约 1000、连续 100 次 Decode 的 profiler 结果：

| 项目 | 自定义 Paged Attention | vLLM FlashAttention | 变化 |
|---|---:|---:|---:|
| Attention kernel 总耗时 | 105.2 ms | 36.3 ms | -65.5% |
| 完整 GPU 总耗时 | 314.8 ms | 250.8 ms | -20.3% |

这说明本轮收益主要来自 Attention kernel 本身，而不是 scheduler 或测试请求组织方式。

## 3. 端到端性能

测试口径：1000 input tokens、50 output tokens、128 requests、`request-rate=inf`，并发
1/4/8，`max_model_len=2048`。基线为 `202608150300-async-output-copy`，即只启用异步
输出复制的版本。

| 并发 | 基线输出吞吐 | vLLM FlashAttention | 变化 | 基线平均 ITL | 新平均 ITL | 新平均 TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 315.48 tok/s | 363.65 tok/s | +15.27% | 2.59 ms | 2.20 ms | 25.41 ms |
| 4 | 878.96 tok/s | 982.74 tok/s | +11.81% | 3.45 ms | 2.93 ms | 53.95 ms |
| 8 | 1268.37 tok/s | 1471.26 tok/s | +16.00% | 4.70 ms | 3.84 ms | 75.24 ms |

总 token 吞吐分别达到 7767.32、20994.05 和 31449.22 tok/s。三个并发档位的输出
吞吐和平均 ITL 都稳定改善，因此保留该优化。

## 4. 资源指标

系统测试前基线：GPU 12.60%，显存 1357.40 MiB，CPU 0.159%，系统内存
1104.28 MiB。

模型加载后基线：GPU 3.00%，显存 4418.00 MiB，CPU 0.115%，系统内存
2193.99 MiB。

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 1 | 68.21% / 98% | 4469.88 / 4488 MiB | 3.91% / 9.36% | 3006.46 / 3167.29 MiB |
| 4 | 48.00% / 96% | 4507.23 / 4526 MiB | 4.64% / 9.00% | 3019.53 / 3187.22 MiB |
| 8 | 38.45% / 94% | 4525.09 / 4526 MiB | 4.77% / 9.53% | 3015.52 / 3201.05 MiB |

GPU 平均利用率会受测试时长和 1 秒采样粒度影响：优化后测试更快，短暂的 CPU 准备、
Prefill 和收尾阶段占样本比例更高，因此不能用平均利用率下降推断 GPU kernel 变慢。
端到端吞吐、ITL 和 profiler kernel 时间共同证明实际计算效率提升。

## 5. 结论与下一步

直接复用 vLLM FlashAttention Paged-KV 后端是当前最显著的单项 Decode 优化之一，且没有
改变精度、请求语义或测试负载。当前与 vLLM 的主要剩余架构差距不再是 Decode Attention，
而是：

1. 真正的 Chunked Prefill，以及 Prefill/Decode 在同一 token budget 中混合执行。
2. 普通 EOS 请求的 GPU-side finished 状态，使异步输出路径不局限于 `ignore_eos=true`。
3. 更完整的 GPU Model Runner 输入准备和采样流水线，继续减少逐步 CPU 同步。

Prefix cache 对本次随机 prompt 基准无收益；量化和 speculative decoding 会改变精度或工作量，
不纳入同口径优化。
