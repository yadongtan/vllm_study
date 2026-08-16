# 202608150400 - 混合 Prefill 与 Decode

## 1. 本轮优化

vLLM 的调度器不是先完整执行一批 Prefill、再单独执行一批 Decode。它会在同一个
`max_num_batched_tokens` 预算内，把新请求的 Prefill token 和已有请求的 Decode token
打包成一次模型前向。这样每个 Transformer 层的权重只读取一次，Attention 通过
`cu_seqlens_q` 保留每个请求的变长边界。

本轮在 `qwen2_demo_v1.py` 增加 packed mixed forward，并在服务端通过
`STUDY_USE_MIXED_PREFILL_DECODE=1` 启用：

- 新请求贡献完整 prompt token，已有请求贡献一个 Decode token。
- Q/K/V 在一维 packed token 布局中计算，使用 vLLM FlashAttention 的 varlen paged-KV
  接口完成一次 Attention。
- 每个请求的 `query_start_loc`、`sequence_lengths` 和 `block_table` 独立描述边界，
  不用 padding 其他请求的 Q/K tile。
- 当前轮的 K/V 先写入固定地址 Paged KV Cache，再由同一个 FlashAttention 调用读取。
- 无新请求加入时继续使用已验证的固定 batch CUDA Graph Decode；没有等待窗口、最小批量、
  多 token 聚合或人为延迟。
- 本轮保留原有 eager Prefill MLP 公式，避免把 MLP 融合和混合调度两个变量混在同一次
  性能结论中。

## 2. 正确性与运行验证

- `qwen2_demo_v1.py` 和 `v3_openai_server.py` 通过 `.venv/bin/python3 -m py_compile`。
- `git diff --check` 通过。
- 1/4/8 并发均完成 128 请求，失败请求数为 0。
- 服务日志无 CUDA、FlashAttention 或 scheduler 异常，服务正常关闭。
- 吞吐测试仍使用 1000 input tokens、50 output tokens、`max_model_len=2048`，并发
  由客户端即时发出；没有增加等待。

## 3. 性能结果

基线为 `202608150310-vllm-flash-attn`，即异步输出复制 + vLLM FlashAttention Paged KV；
本轮在此基础上只增加混合 Prefill/Decode。

| 并发 | 基线输出吞吐 | 混合前向输出吞吐 | 变化 | 基线平均 ITL | 新平均 ITL | 新平均 TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 363.65 tok/s | 361.22 tok/s | -0.67% | 2.20 ms | 2.22 ms | 25.09 ms |
| 4 | 982.74 tok/s | 982.61 tok/s | -0.01% | 2.93 ms | 2.83 ms | 58.89 ms |
| 8 | 1471.26 tok/s | 1502.34 tok/s | +2.11% | 3.84 ms | 3.52 ms | 85.88 ms |

并发 8 的改善来自请求持续进入时合并了部分 Prefill 与 Decode 权重读取；并发 1/4 的
收益被 eager packed 元数据、变长 RoPE 和一次性 token 拼接成本抵消。该优化没有损害
请求完成率，但收益小且对负载敏感，因此保留为可选开关，不宣称它单独解决了与 vLLM
的全部差距。

## 4. 资源指标

系统测试前基线：GPU 12.40% / 13%，显存 4426.8 / 4446.0 MiB，CPU 4.14% / 7.51%，
系统内存 2775.04 / 3002.98 MiB（平均 / 峰值）。模型加载后基线：GPU 3.60% / 7%，
显存 2731.2 / 4492.0 MiB，CPU 6.01% / 12.12%，系统内存 2023.29 / 2718.70 MiB。

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 1 | 69.04% / 99% | 4470.92 / 4489 MiB | 4.00% / 9.59% | 3001.46 / 3144.35 MiB |
| 4 | 47.15% / 94% | 4524.62 / 4558 MiB | 4.54% / 9.15% | 2963.53 / 3115.50 MiB |
| 8 | 37.09% / 93% | 4551.36 / 4559 MiB | 4.87% / 9.14% | 2944.49 / 3120.57 MiB |

GPU 采样按秒进行，短测试的平均值会被启动和收尾阶段稀释；端到端 throughput、ITL
和成功率以 benchmark 精确计时为准。

## 5. 结论

混合 Prefill/Decode 是符合 vLLM 架构的真实优化，但在当前 Qwen2-0.5B、固定 1000/50
负载上只带来 8 并发约 2.1% 的额外吞吐。主要原因是该基准的大部分时间已经处于纯
Decode，且 vLLM FlashAttention 已经消除了最大的 Attention kernel 差距。后续更值得
实现的是完整的 chunked prefill 状态（允许超出单轮 token budget 的 prompt 分块，并在
每轮与 Decode 合并）以及 GPU-side finished mask；它们是调度器结构升级，不能用等待凑批
替代。
