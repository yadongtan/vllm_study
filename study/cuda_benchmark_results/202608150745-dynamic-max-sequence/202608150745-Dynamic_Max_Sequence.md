# 202608150745 - Dynamic Max Sequence For Mixed Attention

## 1. 迭代目标

mixed FlashAttention 当前固定使用 `max_seqlen_k=2048`。本轮尝试把当前 admission
batch 的最大真实 sequence length 传给 attention，减少不必要的 KV 工作范围。

## 2. 实现

`max_sequence_len` 作为可选参数贯穿 `Qwen2ForCausalLMGraph.mixed()`、GraphModel、
DecoderLayer 和 Attention。未传入时仍使用原来的 `cache.max_model_len`，因此默认行为
完全不变。profiler 可用 `--dynamic-max-sequence` 开启。

## 3. Profiler 结果

配置为 2 个 1000-token Prefill、16 个 Decode、3 次 mixed forward：

| 配置 | Self CUDA 总时间 |
|---|---:|
| 固定 `max_seqlen_k=2048` | 约 70.47 ms |
| 传入真实最大 sequence length | 约 71.00 ms |

动态值没有减少 FlashAttention kernel 时间，反而因 kernel 配置/运行时分支产生轻微回退。

## 4. 结论

该方向不接入 scheduler 默认路径，开关保持关闭。vLLM 的收益来自完整 runner 和 attention
后端协同，而不是简单把一个 max length 参数替换成 batch max。

## 5. 验证

```text
py_compile: passed
git diff --check: passed
profiler completed without numerical/runtime errors
```
