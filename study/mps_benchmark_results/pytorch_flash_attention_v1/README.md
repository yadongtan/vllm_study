# PyTorch FlashAttention v1 MPS 吞吐测试

本目录测试
[`pytorch_flash_attention_v1.py`](../../inference_engine/pytorch_attention/pytorch_flash_attention_v1.py)
的吞吐量，分为算子和端到端两层。

## 1. Attention 算子测试

从仓库根目录运行：

```bash
.venv/bin/python \
  study/mps_benchmark_results/pytorch_flash_attention_v1/benchmark_kernel.py
```

默认使用 Qwen2-0.5B 的 14 个 Q heads、2 个 KV heads 和 64 head dim，
测试以下场景：

- Decode：`Q=1, K=1000`；
- 短 Prefill：`Q=K=250`；
- 正式 Prefill：`Q=K=1000`。

每个场景会先与 PyTorch SDPA 对照正确性，再分别测量：

- 单次平均延迟；
- 每秒调用数；
- Query token/s；
- Attention score 元素/s；
- 估算 TFLOPS；
- 相对 SDPA 的延迟倍数。

默认结果写入 `kernel-results.json`。可调整参数：

```bash
.venv/bin/python \
  study/mps_benchmark_results/pytorch_flash_attention_v1/benchmark_kernel.py \
  --block-size 16 \
  --warmups 2 \
  --scenario prefill-1000 \
  --output /tmp/pytorch-flash-attention-v1.json
```

## 2. 端到端模型吞吐测试

端到端测试沿用 `mps_benchmark_results` 既有口径：

- 模型：Qwen2-0.5B-Instruct；
- 每请求输入 1000 token；
- 每请求输出 10 token；
- 每组 16 个请求；
- 并发数 1、2、4、8；
- `ignore_eos=true`；
- `temperature=0`。

运行：

```bash
study/mps_benchmark_results/pytorch_flash_attention_v1/run_e2e_benchmark.sh
```

脚本会自动启动服务、等待模型加载、依次运行三组 benchmark，并在
`results/<时间戳>/` 下保存：

- `concurrency-1.json`；
- `concurrency-2.json`；
- `concurrency-4.json`；
- `summary.md`；
- `server.log`。

默认模型目录是：

```text
models/Qwen/Qwen2-0.5B-Instruct
```

也可以通过环境变量覆盖：

```bash
STUDY_MODEL_PATH=/path/to/Qwen2-0.5B-Instruct \
  study/mps_benchmark_results/pytorch_flash_attention_v1/run_e2e_benchmark.sh
```

算子 benchmark 的 Query token/s 只统计 Query token，不乘 attention head
数量；端到端的输出 token/s 使用 vLLM benchmark 的定义。两者不是同一个指标，
不能直接进行数值比较。
