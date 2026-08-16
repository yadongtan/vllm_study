# 202608150154 - Continuous Batching V2 Inference Mode

## 1. 本轮只修复什么

本轮只给 continuous scheduler 的 `_run()` 恢复 `@torch.inference_mode()`。调度顺序、
active set、cache slot、模型、CUDA kernel 和逐 token 输出均未改变。

## 2. 正确性验证

- CPU 动态加入测试继续通过：后到请求在首请求完成前加入 batch。
- `py_compile` 和 `git diff --check` 通过。
- 真实 CUDA 1/4/8 并发各完成 128 请求，失败请求为 0。

## 3. 显存结果

V1 的显存峰值为 `5746/8496/14762 MiB`；V2 为：

| 并发 | 加载后基线 | 平均显存 | 峰值显存 |
|---:|---:|---:|---:|
| 1 | 4515 MiB | 4577.77 MiB | 4633 MiB |
| 4 | 4515 MiB | 4652.69 MiB | 4702 MiB |
| 8 | 4515 MiB | 4832.92 MiB | 4954 MiB |

并发 8 峰值下降 9808 MiB，证明 V1 的异常不是 KV cache slot 泄漏，而是调度线程
遗漏 inference mode 后的 eager Prefill/autograd 分配。V2 的显存重新回到固定 cache
加正常工作区的范围。

## 4. 性能结果

直接正式基线是 `202608150145-无等待逐Token真实基线`。

| 并发 | Run-to-completion | Continuous V2 | 变化 | TTFT 变化 | ITL 变化 |
|---:|---:|---:|---:|---:|---:|
| 1 | 313.61 tok/s | 273.14 tok/s | -12.90% | 21.96→21.29 ms | 2.69→3.16 ms |
| 4 | 574.43 tok/s | 740.07 tok/s | +28.84% | 193.58→51.20 ms | 2.96→4.29 ms |
| 8 | 955.00 tok/s | 1061.22 tok/s | +11.12% | 245.89→103.49 ms | 3.28→5.34 ms |

相对原生 vLLM，V2 达到 `72.80%/56.70%/44.20%`。它的主要价值是消除高并发
队头阻塞，而不是靠等待形成大 batch；并发 1 的回退说明 scheduler step 本身仍有
明显固定开销。

## 5. 资源指标

| 并发 | GPU 平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|
| 1 | 66.40% / 90% | 3.67% / 9.13% | 2911.10 / 3039.13 MiB |
| 4 | 46.81% / 86% | 4.29% / 9.45% | 2892.21 / 3069.53 MiB |
| 8 | 41.00% / 85% | 4.58% / 9.16% | 2945.08 / 3098.53 MiB |

## 6. 结论与下一步

V2 保留，是第一版内存行为正确的 continuous batching。它不等待请求，也不聚合多个
输出 token。

下一轮只优化每步调度 metadata。V2 即使所有请求都继续，也会创建三个新的 CUDA
Tensor（indices、positions、cache slots）并对 `next_tokens` 执行 `index_select`。
vLLM 使用持久 input buffers/metadata，并只更新内容。下一版维护与 active set 对齐的
GPU position/slot 向量；没有请求结束时直接 replay，只有 active set 收缩时才 gather。
