# PyTorch FlashAttention v1 MPS 端到端吞吐测试

测试口径：16 个请求，每请求 1000 输入 token、10 输出 token。

| 并发 | 成功 | 失败 | 耗时(s) | 请求/s | 输出 token/s | 总 token/s | 平均 TTFT(ms) | 平均 TPOT(ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 10 | 0 | 72.27 | 0.1384 | 1.3836 | 139.75 | 12242.86 | 4512.67 |

服务端启用了 `STUDY_USE_PYTORCH_FLASH_ATTENTION_V1=1`，并禁用了自定义 CUDA Attention。
