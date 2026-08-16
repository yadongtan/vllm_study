# Causal Prefill Fast Path 性能测试报告

## 1. 结论

本轮只优化等长 prompt 的 Prefill Attention。原实现即使所有请求均为 1000
token，也会在每一层创建显式 `[batch, 1, 1000, 1000]` 布尔 mask，并以
`is_causal=False` 调用 SDPA。新实现检测到整个 batch 没有 padding 后，不再创建
二次方 mask，改为 `attn_mask=None, is_causal=True`，让 PyTorch 直接选择优化后的
causal SDPA 后端。变长请求仍走原有 mask 路径。

相对上一版本，1、4、8 并发输出吞吐分别提升 12.51%、20.98% 和 35.13%。并发 8
平均 TTFT 从 538.65 ms 降到 348.35 ms，下降 35.33%。三档均稳定获益，因此该
fast path 默认开启。

| 并发 | 上一版本 (tok/s) | 本轮 (tok/s) | 提升 | 原生 vLLM (tok/s) | vLLM / 本轮 |
|---:|---:|---:|---:|---:|---:|
| 1 | 229.04 | 257.69 | +12.51% | 375.19 | 1.46x |
| 4 | 394.45 | 477.20 | +20.98% | 1305.21 | 2.74x |
| 8 | 509.26 | 688.15 | +35.13% | 2400.77 | 3.49x |

## 2. 单一优化内容

默认环境变量：

```text
STUDY_USE_CAUSAL_PREFILL_FAST_PATH=1
```

模型进入 prefill 时只检查一次：

```text
所有 prompt_lens == padded sequence length
```

成立时，24 层 Attention 均使用无显式 mask 的 causal SDPA；不成立时继续构造
causal 与 valid-key 联合 mask。该优化没有改变 Decode、Paged KV Cache、CUDA
Graph、调度器或采样路径。

## 3. 正确性验证

- 单层随机 BF16 GQA：causal SDPA 与显式 causal mask 对照通过。
- 完整 Qwen2 24 层 Prefill：最终 logits 在 BF16 后端误差范围内一致。
- 两条路径的首个 next token 完全一致。
- 两条路径写入的 K/V Cache 在 BF16 累积误差范围内一致。
- Python 语法检查通过。
- 1、4、8 并发均成功完成 128/128 请求，失败为 0。

## 4. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`。
- CUDA Graph batch sizes：1、2、4、8、16、32。
- 继续开启 Split-KV、GQA KV tile reuse、Fused RMSNorm 和 Fused RoPE Cache Write。
- Fused SiLU + Multiply 继续关闭。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 上一版本：`cuda_benchmark_results/202608142100`。
- 本轮：`cuda_benchmark_results/202608142059`。
- 原生 vLLM：`cuda_benchmark_results/202608140242`。

## 5. 吞吐和延迟

| 版本 | 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 上一版本 | 1 | 27.43 | 229.04 | 60.35 | 3.99 | 3.09 |
| 本轮 | 1 | 24.40 | 257.69 | 36.20 | 3.99 | 3.10 |
| 上一版本 | 4 | 15.93 | 394.45 | 302.13 | 4.85 | 3.79 |
| 本轮 | 4 | 13.17 | 477.20 | 220.37 | 4.85 | 3.75 |
| 上一版本 | 8 | 12.34 | 509.26 | 538.65 | 5.52 | 4.34 |
| 本轮 | 8 | 9.13 | 688.15 | 348.35 | 5.50 | 4.25 |

TPOT 基本不变，而 TTFT 和总耗时明显下降，符合“只优化 Prefill”的预期。

## 6. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 12.40% / 13% | 840 / 841 | 0.15% / 0.32% | 1091 / 1093 |
| 模型和 Graph 加载后 | 2.60% / 4% | 3883 / 3884 | 0.09% / 0.16% | 2153 / 2154 |
| 并发 1 | 62.94% / 81% | 3939 / 3954 | 3.59% / 9.08% | 3001 / 3144 |
| 并发 4 | 52.05% / 81% | 4091 / 4165 | 4.05% / 9.24% | 3029 / 3153 |
| 并发 8 | 45.56% / 77% | 4255 / 4495 | 4.36% / 9.14% | 3006 / 3131 |

显存峰值明显低于显式 mask 版本，尤其是 batch 8，不再需要每层临时持有大型布尔
attention mask。

## 7. 下一步定位

当前并发 8 TTFT 仍为原生 vLLM 的约 8.8 倍。下一轮应单独 profile Prefill，区分
SDPA、QKV/MLP GEMM、逐层 KV Cache 写入和 Python 同步各自耗时，再只处理排名第一
的剩余瓶颈。
