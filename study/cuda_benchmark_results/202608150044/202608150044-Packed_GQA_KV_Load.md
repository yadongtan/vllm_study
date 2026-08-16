# 202608150044 - Packed GQA KV Load

## 1. 结论

本轮只把 GQA Split-KV kernel 从 BF16/FP16 标量 K/V load 改为每条 32-bit 指令搬运
两个相邻元素。正确性通过，但 Decode profile 回退 4.43%，并发 4/8 服务吞吐分别
下降 0.97%/2.79%。该实现已恢复为原标量加载，不进入累计基线。

| 并发 | Pipelined 基线 (tok/s) | Packed KV Load (tok/s) | 变化 | 原生 vLLM (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 298.28 | 305.80 | +2.52% | 375.19 |
| 4 | 627.42 | 621.34 | -0.97% | 1305.21 |
| 8 | 953.05 | 926.47 | -2.79% | 2400.77 |

## 2. 单一迭代内容

原 kernel 中 256 个线程以标量方式把一个 8-token tile 的 K/V 从 paged cache 搬到
shared memory。候选版本对 BF16/FP16 使用 32-bit load/store，每次搬两个相邻元素，
试图减少全局/共享访存指令及地址计算。

online softmax、8-token tile、partition 64、grid/block、KV Cache、CUDA Graph 和服务
流水均未修改。

## 3. 正确性验证

- 与逐 query-head Split-KV 参考实现对照通过。
- CUDA Graph capture/replay 对照通过。
- 与 SDPA 参考实现对照通过。
- partition 边界测试通过。
- 1、4、8 并发均完成 128/128 请求，失败请求为 0。

## 4. Profile 变化

| batch 8，历史长度 1000，20 次 Decode | 标量 Load | Packed Load | 变化 |
|---|---:|---:|---:|
| GQA partial kernel | 18.933 ms | 19.222 ms | +1.53% |
| Split-KV merge kernel | 1.907 ms | 2.108 ms | +10.54% |
| Self CUDA 总时间 | 62.894 ms | 65.683 ms | +4.43% |

当前标量循环本身已经形成合并访问。Packed 版本仍需为每对元素计算 token、page 和
dimension，且 reinterpret 访问没有产生足以抵消索引成本的访存收益。并发 1 的正值
与稳定的 kernel 回退相冲突，按运行波动处理。

## 5. 测试口径

- GPU：NVIDIA GeForce RTX 4080 SUPER，16376 MiB。
- 模型：`/opt/models/Qwen2-0.5B-Instruct`，BF16。
- `max_model_len=2048`，`max_num_seqs=32`，Split-KV partition 64。
- `STUDY_PIPELINE_IGNORE_EOS=1`，`STUDY_PIPELINE_STEPS=1`。
- 并发 1、4、8；每档 128 请求。
- 每请求 1000 token 输入、请求生成 50 token。
- 无限请求速率、忽略 EOS、温度 0。
- 基线：`cuda_benchmark_results/202608150031`。
- 本轮：`cuda_benchmark_results/202608150044`。

## 6. 吞吐和延迟

| 并发 | 耗时 (s) | 输出吞吐 (tok/s) | 平均 TTFT (ms) | 平均 TPOT (ms) | 平均 ITL (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.48 | 305.80 | 27.14 | 3.46 | 2.66 |
| 4 | 10.12 | 621.34 | 166.13 | 3.80 | 2.93 |
| 8 | 6.78 | 926.47 | 261.70 | 4.07 | 3.19 |

## 7. 资源指标

| 阶段/并发 | 平均/峰值 GPU | 平均/峰值显存 MiB | 平均/峰值 CPU | 平均/峰值系统内存 MiB |
|---|---:|---:|---:|---:|
| 测试前系统基线 | 16.00% / 16% | 1455 / 1470 | 0.34% / 0.81% | 1075 / 1078 |
| 模型和 Graph 加载后 | 3.20% / 4% | 4479 / 4480 | 0.08% / 0.15% | 2148 / 2151 |
| 并发 1 | 68.07% / 93% | 4532 / 4550 | 3.67% / 9.39% | 2898 / 3040 |
| 并发 4 | 59.53% / 95% | 4676 / 4755 | 4.39% / 9.03% | 2917 / 3058 |
| 并发 8 | 52.00% / 99% | 4922 / 5096 | 4.77% / 9.15% | 2939 / 3107 |

## 8. 决策

恢复标量 K/V shared-memory load。下一轮不再做简单 load 宽化，而应选择能减少实际
计算或同步次数的优化。
