# 202608150151 - Continuous Batching V1

## 1. 本轮只实现什么

本轮只把 scheduler 从整批 run-to-completion 改为 continuous active set：

- 没有 active request 时立即处理第一个请求，不等待凑批；
- 每个 Decode step 后 drain 已经到达的请求；
- 新请求获得独立 cache slot，完成 Prefill 后加入现有 Decode batch；
- 请求结束立即释放 cache slot；
- 默认逐 token 交付，不使用多 token 聚合。

没有实现 chunked prefill、token budget 或调度等待。

## 2. 正确性验证

- 新增纯 CPU 确定性调度测试。
- 第二个请求在第一个请求尚未结束时加入，graph replay 观察到 batch size=2。
- 两个请求均生成 `[10, 11, 12, 13]`。
- 真实 CUDA 1/4/8 并发各完成 128 请求，失败请求为 0。
- `py_compile` 和 `git diff --check` 通过。

## 3. 性能结果

直接基线为 `202608150145-无等待逐Token真实基线`。

| 并发 | Run-to-completion | Continuous V1 | 变化 | TTFT 变化 | ITL 变化 |
|---:|---:|---:|---:|---:|---:|
| 1 | 313.61 tok/s | 271.61 tok/s | -13.39% | 21.96→22.44 ms | 2.69→3.16 ms |
| 4 | 574.43 tok/s | 719.57 tok/s | +25.27% | 193.58→56.14 ms | 2.96→4.35 ms |
| 8 | 955.00 tok/s | 1063.06 tok/s | +11.31% | 245.89→109.16 ms | 3.28→5.19 ms |

Continuous admission 明显消除了高并发队头阻塞，但新 Prefill 会暂停已有 Decode，且
当前实现每步构造动态 metadata Tensor，因此 ITL 上升。第一版没有 chunked prefill，
这一代价符合预期。

## 4. 显存异常

加载后基线显存为 4504 MiB；测试期间：

| 并发 | 显存平均 | 显存峰值 |
|---:|---:|---:|
| 1 | 5497.60 MiB | 5746 MiB |
| 4 | 7456.06 MiB | 8496 MiB |
| 8 | 11588.46 MiB | 14762 MiB |

这不是 KV Cache slot 增长：cache 仍在启动时固定分配，slot 只复用已有物理 pages。
代码审计发现，新 scheduler 直接在 `_run()` 中调用 `model.prefill()`，而 `_run()` 没有
继承旧 `_execute_batch()` 的 `@torch.inference_mode()`。因此调度线程中的 Prefill 按
普通 autograd 模式执行，产生额外 autograd/activation 分配并抬高 CUDA allocator
保留水位，也解释了并发 1 的明显回退。

## 5. 资源指标

| 并发 | GPU 平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|
| 1 | 64.73% / 83% | 3.65% / 9.03% | 2948.78 / 3072.13 MiB |
| 4 | 48.81% / 95% | 4.26% / 9.08% | 2990.69 / 3145.41 MiB |
| 8 | 41.92% / 99% | 4.61% / 9.11% | 2925.09 / 3085.82 MiB |

## 6. 结论

V1 不作为有效性能版本，但 continuous batching 架构方向保留。下一版只补回
`torch.inference_mode()`，不改变调度顺序、cache slot、Prefill/Decode 算法或测试参数，
然后重新执行完整 1/4/8 测试。
