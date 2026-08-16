# 202608150624 - 高并发 FlashAttention Prefill

## 1. 本轮验证

已有 `STUDY_USE_VLLM_FLASH_PREFILL=1` 在 1/4/8 并发下改善过 TTFT，但输出吞吐略降。
本轮在当前 Fast Incremental 输出基线上补测 16/32 并发，判断 1000-token prompt 的
大批 Prefill 是否能从 vLLM varlen FlashAttention 获得更明显收益。

## 2. 性能结果

| 并发 | PyTorch causal SDPA 基线 | vLLM Flash Prefill | 变化 |
|---:|---:|---:|---:|
| 16 | 2155.08 | 1807.35 | -16.1% |
| 32 | 2536.63 | 2292.46 | -9.6% |

表中为按服务实际生成 6400 token 校正后的 output tok/s。

| 并发 | 时长 (s) | 客户端 tok/s | 校正 tok/s | TTFT (ms) | TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.541 | 1786.74 | 1807.35 | 144.32 | 7.27 |
| 32 | 2.792 | 2264.52 | 2292.46 | 203.67 | 12.11 |

## 3. 资源结果

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 16 | 36.0% / 95% | 4062 / 4121 MiB | 4.46% / 8.89% | 2878 / 3226 MiB |
| 32 | 34.7% / 95% | 4080 / 4082 MiB | 4.60% / 9.20% | 3018 / 3247 MiB |

## 4. 结论

当前等长 causal Prefill 的 PyTorch SDPA 已经选择高效 Flash kernel。vLLM varlen 路径
还需要将 Q/K/V 转为 packed contiguous 布局，这些布局转换在当前等长请求上没有被
Attention 收益抵消。因此保持 `STUDY_USE_VLLM_FLASH_PREFILL=0`。

这一结果排除了“换一个 Prefill Attention 调用”作为高并发主要突破口。下一项优化转向
vLLM model runner 的 metadata 原则：prompt 长度与等长判断留在 CPU，移除每个 Prefill
小批次的 GPU scalar `.item()` 同步。

