# Fused RoPE + Paged KV Cache Write 性能测试报告

## 1. 结论

本轮只优化 Decode 中 Q/K RoPE 和 K/V Paged Cache 写入。新增 CUDA kernel 直接
读取 QKV 线性层输出，在同一个 kernel 中计算旋转后的 Q/K、返回 Q，并将 K/V 写入
block table 指向的物理 page。Prefill、Attention、MLP、RMSNorm 和调度均不改变。

1、4、8 并发均完成 128/128 请求，失败请求为 0。相对上一轮最佳累计版本 Fused
Add + RMSNorm，输出吞吐分别提升 9.79%、7.18% 和 7.55%，三档平均 TPOT 与平均
ITL 也一致下降，因此保留该实现为默认 Decode 路径。

| 并发 | 上一最佳版本 (tok/s) | 本轮 (tok/s) | 提升 | 原生 vLLM (tok/s) | vLLM / 本轮 |
|---:|---:|---:|---:|---:|---:|
| 1 | 205.47 | 225.59 | +9.79% | 375.19 | 1.66x |
| 4 | 360.24 | 386.08 | +7.18% | 1305.21 | 3.38x |
| 8 | 466.78 | 502.03 | +7.55% | 2400.77 | 4.78x |

## 2. 单项优化内容

原 Decode 每层依次执行：

1. 由位置 Tensor 创建 inverse frequency、frequency、cos 和 sin Tensor。
2. Q 和 K 分别执行乘法、`rotate_half`、第二次乘法和相加。
3. 计算 cache slot 对应的物理写入索引。
4. 两次 `index_copy_` 分别写 K Cache 和 V Cache。

新路径先保留原 QKV GEMM，然后只启动一个自定义 CUDA kernel：

```text
QKV GEMM
  -> fused_rope_cache_write
       -> rotated Q output
       -> rotated K 写入物理 page
       -> V 直接写入物理 page
  -> Split-KV Paged Attention
```

kernel 的 grid 为 `[batch_size]`，每个 block 使用 256 线程处理一个请求当前 token。
线程遍历所有 Q head 和 KV head 的维度，在寄存器中根据 position 和 rope theta 计算
角度；物理 page 由 `block_table[cache_slot, logical_block]` 找到。这样取消了 Decode
中的 cos/sin Tensor 构造、Q/K 多个逐元素 kernel、临时旋转 Tensor和两次
`index_copy_`。

Prefill 继续使用原 PyTorch RoPE 和变长 Cache 写入逻辑。这使本轮变化严格限制在
固定 shape 的 CUDA Graph Decode 路径。

## 3. 数值顺序和验证

为了匹配原 BF16 路径，kernel 没有简单地以 FP32 完成整个公式再一次性转回 BF16。
它先把 cos/sin 转为 BF16，两个乘法结果各自转为 BF16，最后相加再转为 BF16，
保持 PyTorch eager 路径的分步舍入顺序。

以下验证全部通过：

- batch 1、4 的旋转 Q、写入 K 和写入 V 与 PyTorch 参考路径对比。
- 非顺序 cache slot、跨 page position 的物理 page 写入检查。
- 算子 CUDA Graph 捕获和 replay，replay 前替换 QKV 输入。
- 完整 Qwen2-0.5B 24 层融合/未融合 Decode logits 对比。
- 两条路径生成相同 next token。
- 24 层完整 K/V Cache 对比。
- Python 语法、shell 语法和 `git diff --check`。

当前环境没有 pytest，测试通过 `.venv/bin/python3` 直接调用测试函数完成。

## 4. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- WSL：Ubuntu 24.04。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- 被测实现：`study/inference_engine/qwen2_demo_v1.py`。
- `STUDY_USE_SPLIT_KV_ATTENTION=1`。
- `STUDY_USE_FUSED_RMS_NORM=1`。
- `STUDY_USE_FUSED_SILU_MUL=0`。
- `STUDY_USE_FUSED_ROPE_CACHE_WRITE=1`。
- `max_model_len=2048`，`max_num_seqs=32`。
- CUDA Graph batch sizes：1、2、4、8、16、32。
- 并发：1、4、8。
- 每档 128 请求，每请求固定 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 上一轮最佳基线：`cuda_benchmark_results/202608141935`。
- 本轮结果：`cuda_benchmark_results/202608142039`。
- 原生 vLLM：`cuda_benchmark_results/202608140242`。

## 5. 吞吐和延迟

| 版本 | 并发 | 成功/失败 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | P99 TTFT (ms) | 平均 TPOT (ms) | P99 TPOT (ms) | 平均 ITL (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 上一最佳版本 | 1 | 128/0 | 30.57 | 205.47 | 59.21 | 58.34 | 4.65 | 25.73 | 3.60 | 4.48 |
| Fused RoPE/KV Write | 1 | 128/0 | 27.84 | 225.59 | 60.56 | 59.21 | 4.06 | 22.59 | 3.15 | 4.00 |
| 上一最佳版本 | 4 | 128/0 | 17.45 | 360.24 | 311.72 | 470.27 | 5.66 | 31.55 | 4.53 | 5.67 |
| Fused RoPE/KV Write | 4 | 128/0 | 16.28 | 386.08 | 302.77 | 439.69 | 5.17 | 28.01 | 3.99 | 5.19 |
| 上一最佳版本 | 8 | 128/0 | 13.46 | 466.78 | 585.54 | 951.54 | 6.34 | 34.30 | 4.91 | 6.55 |
| Fused RoPE/KV Write | 8 | 128/0 | 12.52 | 502.03 | 546.01 | 849.69 | 5.87 | 37.77 | 4.46 | 6.30 |

| 并发 | 吞吐变化 | 平均 TTFT 变化 | 平均 TPOT 变化 | 平均 ITL 变化 |
|---:|---:|---:|---:|---:|
| 1 | +9.79% | +2.29% | -12.81% | -12.64% |
| 4 | +7.18% | -2.87% | -8.63% | -11.94% |
| 8 | +7.55% | -6.75% | -7.38% | -9.17% |

并发 1 的 TTFT 小幅增加但 TPOT、ITL 和总吞吐明显改善；并发 4/8 四项主要指标
方向一致。客户端分别统计到 6281、6286、6287 个输出 token。

## 6. GPU、显存、CPU 和系统内存

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 14.40% / 20% | 843 / 844 | 0.17% / 0.34% | 1052 / 1054 |
| 模型和 Graph 加载后 | 5.20% / 10% | 3885 / 3887 | 0.10% / 0.33% | 2135 / 2141 |
| 并发 1 | 63.66% / 86% | 4027 / 4063 | 3.56% / 9.20% | 3103 / 3277 |
| 并发 4 | 58.57% / 90% | 4582 / 4822 | 3.92% / 9.07% | 3162 / 3279 |
| 并发 8 | 54.58% / 93% | 5972 / 7066 | 4.17% / 8.97% | 3126 / 3240 |

融合算子没有持久 workspace；模型加载后的显存与上一轮相近。测试时间变短会提高
逐秒监控中启动和收尾样本占比，因此不能用不同轮次的平均 GPU 利用率单独推断
kernel 效率，吞吐与 TPOT 更直接。

## 7. 剩余差距和下一项

相对原生 vLLM，当前版本仍慢 1.66、3.38、4.78 倍。并发越高差距越大，说明只
减少逐元素 kernel 还不足以解决 batch 扩展效率。下一项单独实验 Split-KV
Attention 中 GQA 的 KV tile 复用：当前 7 个 query heads 共用一个 KV head，但每个
query-head block 会重复从全局显存读取同一段 K/V。

## 8. 结果文件

- `fused-rope-concurrency-{1,4,8}.json`：完整 benchmark 数据。
- `resources-fused-rope-concurrency-{1,4,8}.csv`：逐秒资源采样。
- `resources-baseline-system-before.csv`：服务启动前系统基线。
- `resources-baseline-fused-rope-loaded.csv`：模型和 Graph 加载后基线。
- `resource-summary.json`：资源均值和峰值汇总。
- `fused-rope-server.log`：服务启动、Graph 捕获、请求和退出日志。
