# Fused Add + RMSNorm 性能测试报告

## 1. 结论

本轮在 Split-KV Partition Attention 版本上增加 vLLM 风格的 CUDA
Fused Add + RMSNorm。Decode 不再在每一层分别执行残差相加、FP32 转换、平方、
均值归约、`rsqrt`、归一化和权重缩放，而是用一个 CUDA kernel 完成残差更新与
RMSNorm，并在层间保留独立的 `hidden_states` 和 `residual` 状态。

1、4、8 并发均完成 128/128 请求，失败请求为 0。相对上一轮 Split-KV 基线：

- 输出吞吐分别提升 10.5%、8.3% 和 5.5%。
- 平均 TPOT 分别降低 12.4%、13.1% 和 11.4%。
- 平均 ITL 分别降低 12.3%、10.6% 和 12.0%。

| 并发 | Split-KV (tok/s) | Fused RMSNorm (tok/s) | 提升 | 原生 vLLM (tok/s) | vLLM / 本轮 |
|---:|---:|---:|---:|---:|---:|
| 1 | 185.88 | 205.47 | +10.5% | 375.19 | 1.83x |
| 4 | 332.75 | 360.24 | +8.3% | 1305.21 | 3.62x |
| 8 | 442.56 | 466.78 | +5.5% | 2400.77 | 5.14x |

本轮优化有效，但并发越高吞吐收益越小。原因是 RMSNorm 的固定 kernel 和显存
往返成本在小 batch 中占比更高；batch 增大后，GEMM、Attention、RoPE、KV 写入、
采样和调度占据更多总耗时。下一步应基于完整 Decode trace 继续优化，而不是假设
RMSNorm 仍是最大瓶颈。

## 2. 实现方式

### 普通 RMSNorm kernel

首层 Decode 输入没有待合并的上一分支，因此使用单独的 `rms_norm_kernel`：

1. 一个 CUDA block 处理一行 `[hidden_size]`。
2. 256 个线程分摊 896 个 hidden elements。
3. 每个线程用 FP32 累加平方和。
4. block 内共享内存归约得到均方值。
5. 计算 `rsqrt(mean_square + epsilon)` 并乘 RMSNorm weight。

### Fused Add + RMSNorm kernel

其余残差边界使用 `fused_add_rms_norm_kernel`：

```text
residual = BF16(input + residual)
input = BF16(BF16(residual * inverse_rms) * weight)
```

它原地更新两个固定地址 Tensor：

- `residual` 保存相加后的残差，供下一分支继续累加。
- `input` 保存归一化结果，作为 Attention 或 MLP 输入。

BF16 舍入顺序与原 PyTorch 实现保持一致：残差相加后先舍入为 BF16，再以 FP32
计算平方和；归一化结果先舍入为 BF16，再乘 BF16 weight。这减少了 24 层累计
数值差异。

### 层间状态重排

原路径每层直接构造完整隐藏状态：

```text
h = h + Attention(RMSNorm(h))
h = h + MLP(RMSNorm(h))
```

融合路径保持 `(hidden_states, residual)`：

```text
normalized, residual = FusedAddRMSNorm(branch_output, residual)
```

24 层中，第一处使用普通 RMSNorm；之后的 Attention/MLP 残差边界以及最终模型
RMSNorm 使用融合 kernel。Prefill 保持原实现不变。可通过
`STUDY_USE_FUSED_RMS_NORM=0` 回退到旧 Decode 路径。

## 3. 正确性与兼容性验证

以下验证全部通过：

- BF16 普通 RMSNorm 与 PyTorch 参考实现对比，batch 1/4/8。
- BF16 Fused Add + RMSNorm 的 output 和 residual 分别与参考实现对比。
- 原地残差结果逐元素一致。
- 自定义算子 CUDA Graph 捕获和 replay。
- 完整 Qwen2-0.5B 24 层 Decode 与未融合路径 logits 对比。
- 融合和未融合路径得到相同的下一个 token。
- 两条路径写入的 Paged KV Cache 一致。
- Python 语法检查、shell 语法检查和 `git diff --check`。

当前虚拟环境未安装 pytest，因此测试文件通过 `.venv/bin/python3` 直接调用测试
函数执行，没有使用系统 Python，也没有改变项目依赖环境。

## 4. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- WSL：Ubuntu 24.04。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- 被测实现：`study/inference_engine/qwen2_demo_v1.py`。
- `STUDY_USE_SPLIT_KV_ATTENTION=1`。
- `STUDY_USE_FUSED_RMS_NORM=1`。
- `max_model_len=2048`，`max_num_seqs=32`。
- CUDA Graph batch sizes：1、2、4、8、16、32。
- 并发：1、4、8。
- 每档：128 请求。
- 每请求：固定 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- Split-KV 基线：`cuda_benchmark_results/202608141857`。
- 原生 vLLM 对照：`cuda_benchmark_results/202608140242`。

## 5. 吞吐与延迟

| 版本 | 并发 | 成功/失败 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | P99 TTFT (ms) | 平均 TPOT (ms) | P99 TPOT (ms) | 平均 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Split-KV | 1 | 128/0 | 33.80 | 185.88 | 59.16 | 60.33 | 5.31 | 29.26 | 4.11 |
| Fused RMSNorm | 1 | 128/0 | 30.57 | 205.47 | 59.21 | 58.34 | 4.65 | 25.73 | 3.60 |
| Split-KV | 4 | 128/0 | 18.89 | 332.75 | 334.88 | 719.60 | 6.51 | 30.76 | 5.06 |
| Fused RMSNorm | 4 | 128/0 | 17.45 | 360.24 | 311.72 | 470.27 | 5.66 | 31.55 | 4.53 |
| Split-KV | 8 | 128/0 | 14.20 | 442.56 | 589.00 | 905.79 | 7.15 | 40.73 | 5.58 |
| Fused RMSNorm | 8 | 128/0 | 13.46 | 466.78 | 585.54 | 951.54 | 6.34 | 34.30 | 4.91 |

相对 Split-KV：

| 并发 | 吞吐变化 | 平均 TTFT 变化 | 平均 TPOT 变化 | 平均 ITL 变化 |
|---:|---:|---:|---:|---:|
| 1 | +10.5% | +0.1% | -12.4% | -12.3% |
| 4 | +8.3% | -6.9% | -13.1% | -10.6% |
| 8 | +5.5% | -0.6% | -11.4% | -12.0% |

TTFT 包含固定批次排队时间，尤其 P99 会受批次边界影响；判断 Decode 优化时，
TPOT、ITL 和总吞吐更直接。本轮三项指标趋势一致，说明提升来自 Decode 路径，
而不是客户端计时波动。

客户端分别统计到 6282、6287、6284 个输出 token；服务端每档仍实际生成 6400
token。报告沿用 `vllm bench serve` 客户端输出吞吐，以保持历史结果口径一致。

## 6. GPU、显存、CPU 和系统内存

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 14.80% / 19% | 846 / 847 | 0.16% / 0.24% | 1061 / 1072 |
| 模型和 Graph 加载后 | 3.40% / 6% | 3913 / 3916 | 0.07% / 0.15% | 2338 / 2340 |
| 并发 1 | 69.92% / 99% | 4062 / 4093 | 3.61% / 9.11% | 3167 / 3279 |
| 并发 4 | 61.88% / 89% | 4635 / 4844 | 3.94% / 9.14% | 3195 / 3298 |
| 并发 8 | 55.70% / 89% | 6107 / 7107 | 4.19% / 9.19% | 3156 / 3273 |

相对模型加载后基线，测试期间平均显存增量约为 148、722、2194 MiB。Fused
RMSNorm 原地复用 input 和 residual，没有为每层保留新的持久 workspace。不同
轮次显存峰值还会受到 CUDA Graph pool 和逐秒采样时刻影响，因此不把小幅显存
差异解释为稳定收益。

并发越高测试持续时间越短，逐秒监控中的客户端初始化和收尾空闲样本占比越高，
所以平均 GPU 利用率下降不能直接表示有效 GPU 工作减少。三档 GPU 峰值均达到
89% 以上。

## 7. 与原生 vLLM 的剩余差距

融合后，原生 vLLM 分别仍快 1.83、3.62、5.14 倍。本轮将一个高频非 Attention
算子改成了 vLLM 式实现，但剩余差距仍来自完整系统：

1. Q/K RoPE 和 Paged KV Cache 写入仍是多个独立操作。
2. SwiGLU 的 SiLU 与 gate/up multiply 尚未融合。
3. 当前 Split-KV 没有在 7 个 GQA query heads 之间复用同一 KV tile。
4. 小 batch GEMM 的 shape、调度和 epilogue 尚未做专项优化。
5. 固定批次调度还不是 continuous batching。
6. Python token 回传和部分同步仍存在。

下一步应先采集一次 batch 8 的完整 CUDA Graph replay trace，按 GPU 时间排序剩余
kernel。若继续按当前代码结构推进，候选项是 Fused RoPE + Paged KV Write 或
Fused SiLU + Multiply；最终选择应由 trace 中的累计耗时决定。

## 8. 结果文件

- `fused-rms-concurrency-{1,4,8}.json`：完整 benchmark 数据。
- `resources-fused-rms-concurrency-{1,4,8}.csv`：逐秒资源采样。
- `resources-baseline-system-before.csv`：服务启动前系统基线。
- `resources-baseline-fused-rms-loaded.csv`：模型和 Graph 加载后基线。
- `resource-summary.json`：资源均值和峰值汇总。
- `fused-rms-server.log`：服务启动、Graph 捕获、请求和退出日志。

