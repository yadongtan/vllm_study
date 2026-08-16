# 202608150642 - 交错 Prefill/Decode

## 1. 本轮优化

非 mixed scheduler 在 2048 token budget 下每次只能 Prefill 两个 1000-token 请求，但会
连续完成全部 admission，直到 32 个 cache slot 填满后才开始 Decode。

本轮新增 `STUDY_USE_INTERLEAVED_PREFILL_DECODE=1`：每次新 prompt admission 前，先用
已有 Decode CUDA Graph 推进 active requests 一步，再执行新 prompt Prefill。没有等待
凑批，总调度 token budget 仍不超过 2048。

## 2. 性能结果

| 并发 | 预热+末尾跳过基线 | 交错执行 | 变化 |
|---:|---:|---:|---:|
| 16 | 1954.60 | 1935.75 | -1.0% |
| 32 | 2434.27 | 2250.08 | -7.6% |

表中为按实际生成 6400 token 校正后的 output tok/s。

| 并发 | TTFT (ms) | TPOT (ms) | 客户端 output tok/s |
|---:|---:|---:|---:|
| 16 | 112.79 | 7.31 | 1912.16 |
| 32 | 209.93 | 12.14 | 2223.01 |

## 3. 原因分析

vLLM 的混合调度会把 Decode token 与 Prefill token 放入同一个 model-runner forward，
共享 QKV/GEMM kernel launch 和 packed FlashAttention metadata。本实验则是：

```text
小 batch Decode CUDA Graph -> Prefill forward -> 下一轮
```

虽然计算顺序交错了，但新增了很多 batch 2/4/8 的 Decode Graph replay，早到请求的 Decode
没有与 Prefill GEMM 合并，反而损失 batch-32 的矩阵乘效率。32 并发退化最明显。

## 4. 决策

`STUDY_USE_INTERLEAVED_PREFILL_DECODE` 保留为默认关闭的调度实验。后续不再用两个独立
forward 模拟 vLLM mixed execution，而是直接优化已经存在的 packed mixed forward。

检查发现 mixed layer 的 RMSNorm 仍固定调用自研算子，未使用已在纯 Prefill/Decode 路径
验证有效的 vLLM fused RMSNorm。下一轮只补齐这一处，然后重新测试 mixed 调度。

