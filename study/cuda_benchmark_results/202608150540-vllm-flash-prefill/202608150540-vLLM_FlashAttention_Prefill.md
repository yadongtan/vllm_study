# 202608150540 - vLLM FlashAttention Prefill

## 1. 本轮优化

此前 `STUDY_USE_VLLM_FLASH_ATTN=1` 只覆盖 Decode，等长 Prefill 仍调用 PyTorch SDPA。
本轮增加 `STUDY_USE_VLLM_FLASH_PREFILL=1`：

- 将等长 prompt 的 Q/K/V 转为 token-major packed 布局。
- 使用一次 `cu_seqlens_q/cu_seqlens_k` 保存请求边界，并在全部层复用。
- 调用 `vllm.vllm_flash_attn.flash_attn_varlen_func` 执行 causal GQA Prefill。
- KV Cache 写入、Decode CUDA Graph、调度器和 fused ops 不变。
- 变长 prompt 暂时回退原 PyTorch SDPA 路径。

## 2. 正确性

新增全模型测试，使用 2 个等长 32-token prompt 比较 PyTorch SDPA 和 vLLM
FlashAttention Prefill：

- 最终 logits 最大绝对误差 `0.234375`，通过 BF16 `atol=0.5, rtol=2e-2`。
- 两条路径的 next-token argmax 完全一致。
- 完整 1/4/8 并发各完成 128 请求，失败数为 0。

## 3. 性能

基线为 `202608150420-vllm-fused-ops`，两边都启用异步输出复制、vLLM Decode
FlashAttention 和 vLLM fused ops。

| 并发 | 基线输出吞吐 | Flash Prefill | 变化 | 基线/新 TTFT | 基线/新 ITL |
|---:|---:|---:|---:|---:|---:|
| 1 | 381.00 tok/s | 381.64 tok/s | +0.17% | 20.94 / 20.66 ms | 2.16 / 2.16 ms |
| 4 | 1115.46 tok/s | 1096.59 tok/s | -1.69% | 41.79 / 43.02 ms | 2.69 / 2.72 ms |
| 8 | 1685.51 tok/s | 1669.15 tok/s | -0.97% | 71.19 / 58.41 ms | 3.24 / 3.53 ms |

并发 8 的平均 TTFT 降低 17.95%，说明 Prefill 延迟明显改善；但输出吞吐轻微下降约 1%，
可能来自 packed Q/K/V contiguous 转换和短 Decode 主导负载下的固定开销。该优化适合关注
TTFT 的服务配置，但不作为当前“最大输出吞吐”默认值。

## 4. 资源指标

系统测试前基线：GPU 12.60%，显存 1393.60 MiB，CPU 0.159%，系统内存
1082.50 MiB。模型加载后基线：GPU 2.60%，显存 4447.00 MiB，CPU 0.091%，系统内存
2173.00 MiB。

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 1 | 69.74% / 99% | 4497.57 / 4517 MiB | 3.89% / 9.01% | 2924.44 / 3087.88 MiB |
| 4 | 44.38% / 98% | 4537.92 / 4557 MiB | 4.59% / 9.09% | 2921.37 / 3096.43 MiB |
| 8 | 40.00% / 98% | 4556.18 / 4557 MiB | 4.81% / 9.14% | 2922.74 / 3111.55 MiB |

## 5. 决策

- 保留 `STUDY_USE_VLLM_FLASH_PREFILL` 可选开关。
- 最大输出吞吐基线仍使用 vLLM fused ops，但不启用 mixed 和 Flash Prefill。
- 关注 TTFT 时可启用 Flash Prefill，尤其在并发 8 的本轮负载下收益明显。
