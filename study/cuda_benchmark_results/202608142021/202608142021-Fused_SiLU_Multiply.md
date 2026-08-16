# Fused SiLU + Multiply 性能测试报告

## 1. 结论

本轮只优化 SwiGLU 激活路径，将 Decode MLP 中的
`gate, up = gate_up.chunk(2)`、`F.silu(gate)` 和 `* up` 合并为一个自定义
CUDA kernel。其他已启用优化保持不变，包括 CUDA Graph、Paged KV Cache、
Split-KV Partition Attention 和 Fused Add + RMSNorm。

1、4、8 并发均完成 128/128 请求，失败请求为 0。相对上一轮 Fused Add +
RMSNorm 累计基线，输出吞吐变化分别为 -0.56%、-1.14% 和 +0.38%。变化幅度小且
方向不一致，延迟指标也没有形成一致改善，因此不能判定该 kernel 带来稳定性能
收益。

| 并发 | Fused RMSNorm 基线 (tok/s) | Fused SiLU (tok/s) | 变化 |
|---:|---:|---:|---:|
| 1 | 205.47 | 204.32 | -0.56% |
| 4 | 360.24 | 356.13 | -1.14% |
| 8 | 466.78 | 468.54 | +0.38% |

结论是保留实现和 `STUDY_USE_FUSED_SILU_MUL=1` 显式实验开关，但默认值改为
`0`。后续累计优化和对比均从上一轮 Fused RMSNorm 版本继续，避免把没有稳定收益
的实现叠加进最佳基线。

## 2. 单项优化内容

原 Decode MLP 路径为：

```python
gate_up = gate_up_proj(hidden_states)
gate, up = gate_up.chunk(2, dim=-1)
activated = F.silu(gate)
output = activated * up
return down_proj(output)
```

新路径启动一个 CUDA kernel。每个线程负责一个输出元素，从连续的 `gate_up`
Tensor 中读取同一行对应的 gate 和 up 元素，计算 SiLU 后立即完成乘法并写出结果。
它避免单独保存 `F.silu(gate)` 的完整中间 Tensor，也避免额外的逐元素 multiply
kernel 启动。

为了匹配原 BF16 计算顺序，kernel 会先把 FP32 SiLU 结果转换回 BF16，再与 up
相乘。该实现支持 FP32、FP64、FP16 和 BF16，并使用 PyTorch 当前 CUDA device
与 stream，可以被 CUDA Graph 捕获和 replay。

## 3. 正确性和兼容性验证

以下验证全部通过：

- batch 1、4、8 的 BF16 输出与 PyTorch `F.silu(gate) * up` 对比。
- 自定义 SiLU + Multiply 算子的 CUDA Graph 捕获和 replay。
- 完整 Qwen2-0.5B 24 层 Decode 的融合/未融合 logits 对比。
- 两条路径生成相同的 next token。
- 两条路径写入相同的 Paged KV Cache。
- Python 语法、shell 语法和 `git diff --check`。

当前虚拟环境没有安装 pytest，因此测试文件通过 `.venv/bin/python3` 直接导入并
执行测试函数，未使用系统 Python。

## 4. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- WSL：Ubuntu 24.04。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- 被测实现：`study/inference_engine/qwen2_demo_v1.py`。
- `STUDY_USE_SPLIT_KV_ATTENTION=1`。
- `STUDY_USE_FUSED_RMS_NORM=1`。
- `STUDY_USE_FUSED_SILU_MUL=1`。
- `max_model_len=2048`，`max_num_seqs=32`。
- CUDA Graph batch sizes：1、2、4、8、16、32。
- 并发：1、4、8。
- 每档 128 请求，每请求固定 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 上一轮累计基线：`cuda_benchmark_results/202608141935`。
- 本轮结果：`cuda_benchmark_results/202608142021`。

## 5. 吞吐和延迟

| 版本 | 并发 | 成功/失败 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | P99 TTFT (ms) | 平均 TPOT (ms) | P99 TPOT (ms) | 平均 ITL (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fused RMSNorm | 1 | 128/0 | 30.57 | 205.47 | 59.21 | 58.34 | 4.65 | 25.73 | 3.60 | 4.48 |
| Fused SiLU | 1 | 128/0 | 30.75 | 204.32 | 59.36 | 59.89 | 4.69 | 26.12 | 3.63 | 4.62 |
| Fused RMSNorm | 4 | 128/0 | 17.45 | 360.24 | 311.72 | 470.27 | 5.66 | 31.55 | 4.53 | 5.67 |
| Fused SiLU | 4 | 128/0 | 17.65 | 356.13 | 321.86 | 592.37 | 5.87 | 33.30 | 4.50 | 5.81 |
| Fused RMSNorm | 8 | 128/0 | 13.46 | 466.78 | 585.54 | 951.54 | 6.34 | 34.30 | 4.91 | 6.55 |
| Fused SiLU | 8 | 128/0 | 13.41 | 468.54 | 572.54 | 938.63 | 6.61 | 42.14 | 5.03 | 6.70 |

客户端统计到 6282、6286、6285 个输出 token，各轮输出 token 数和调度边界略有
差异。并发 8 吞吐提升 0.38%，但平均 TPOT 增加 4.37%、平均 ITL 增加 2.48%，
不能视为 Decode 得到稳定加速。

## 6. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 16.80% / 18% | 832 / 835 | 0.17% / 0.27% | 1047 / 1050 |
| 模型和 Graph 加载后 | 5.00% / 10% | 3895 / 3895 | 0.08% / 0.15% | 2347 / 2353 |
| 并发 1 | 66.61% / 85% | 4043 / 4071 | 3.61% / 9.15% | 3196 / 3344 |
| 并发 4 | 61.88% / 87% | 4779 / 5039 | 3.90% / 9.01% | 3196 / 3296 |
| 并发 8 | 57.70% / 92% | 6081 / 6643 | 4.09% / 9.24% | 3175 / 3290 |

资源数据包含逐秒采样，短测试中的启动、客户端初始化和收尾空闲样本会影响平均
GPU 利用率。自定义 kernel 没有增加持久 workspace，显存变化主要来自不同并发
下的请求 KV Cache、CUDA Graph pool 和采样时刻。

## 7. 为什么没有稳定加速

PyTorch 对 `F.silu` 和 multiply 已能使用高带宽逐元素 kernel；当前自定义 kernel
只是普通标量线程实现，没有向量化加载、专门的 BF16 指令优化或与前后 GEMM
epilogue 融合。虽然少了一次中间结果的全局内存往返和一次 kernel launch，但在
CUDA Graph 中 kernel launch 的 CPU 开销本来已经很低，节省不足以形成稳定收益。

真正更有价值的 SwiGLU 优化通常是把激活和乘法放进 GEMM epilogue，或使用经过
针对 shape、dtype 和 GPU 架构调优的融合实现。那会改变 GEMM 路径，超出了本轮
“只优化一个方向”的范围。

## 8. 结果文件

- `fused-silu-concurrency-{1,4,8}.json`：完整 benchmark 数据。
- `resources-fused-silu-concurrency-{1,4,8}.csv`：逐秒资源采样。
- `resources-baseline-system-before.csv`：服务启动前系统基线。
- `resources-baseline-fused-silu-loaded.csv`：模型和 Graph 加载后基线。
- `resource-summary.json`：资源均值和峰值汇总。
- `fused-silu-server.log`：模型加载、Graph 捕获、请求和退出日志。
