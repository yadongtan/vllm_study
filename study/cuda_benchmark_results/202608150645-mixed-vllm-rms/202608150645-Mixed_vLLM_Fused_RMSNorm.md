# 202608150645 - Mixed vLLM Fused RMSNorm

## 1. 本轮优化

packed mixed Prefill/Decode 已支持在一个 forward 中处理新 prompt 与 active Decode token，
但 mixed layer 的两处 RMSNorm 仍固定调用自研 CUDA 算子，没有使用纯 Prefill/Decode 路径
已经默认启用的 vLLM fused ops。

本轮只修改 mixed layer 的 RMSNorm 选择：

- 无 residual 时使用 `vllm_rms_norm`；
- 有 residual 时使用 `vllm_fused_add_rms_norm`；
- Attention、MLP、scheduler、token budget 和输出链路不变。

测试启用 `STUDY_USE_MIXED_PREFILL_DECODE=1`，关闭交错双 forward 实验。

## 2. Token 计数说明

服务端调度器严格生成 `128 requests × 50 tokens = 6400` 个模型 token。benchmark JSON
对返回文本重新 tokenize，mixed 路径的任意 token 序列经过 decode/encode 不是严格双射，
因此误记为 10156/9302。报告只使用 `6400 / duration` 计算真实模型输出吞吐，不使用 JSON
中的文本重分词吞吐。

## 3. 性能结果

基线为 `202608150639-terminal-replay-skip`。

| 并发 | 非 mixed 基线 | Mixed + vLLM RMSNorm | 变化 |
|---:|---:|---:|---:|
| 16 | 1954.60 | 2192.30 | +12.2% |
| 32 | 2434.27 | 2544.37 | +4.5% |

| 并发 | 时长 (s) | 校正 output tok/s | TTFT (ms) | benchmark TPOT (ms) |
|---:|---:|---:|---:|---:|
| 16 | 2.919 | 2192.30 | 102.85 | 4.70 |
| 32 | 2.515 | 2544.37 | 161.61 | 8.69 |

TPOT 也受到文本 chunk/token 重分词口径影响，因此仅作为客户端观测值，核心结论以固定
6400 模型 token 的总时长为准。

## 4. 资源结果

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 16 | 36.3% / 92% | 4299 / 4306 MiB | 4.73% / 9.11% | 2952 / 3181 MiB |
| 32 | 30.6% / 92% | 4305 / 4306 MiB | 4.71% / 9.12% | 2998 / 3214 MiB |

## 5. 决策

补齐 vLLM fused RMSNorm 后，packed mixed forward 在 16/32 并发均超过非 mixed 基线，
证明正确方向是“同一 model runner 中混合 Prefill/Decode”，而不是两个独立 forward 交错。

下一轮把 mixed attention 使用的 block table 从“每层 `index_select` 一次”改成“每个 model
step 构造一次、24 层复用”，继续对齐 vLLM 持久化 input/metadata 设计。

