# 202608150203 - CUDA Graph 仅重置 Padding 尾部

## 1. 本轮只优化什么

`FixedBatchCUDAGraph.replay()` 旧路径每步执行：清零全部 input ids、清零全部 positions、
把全部 slots 写成 scratch slots，再复制三组真实输入。

本轮改为先复制真实输入；仅当 actual batch 小于 captured graph batch 时，才清零尾部
input/position 并恢复尾部 scratch slots。full batch 从 6 次输入准备操作降为 3 次。

调度、active set、模型、kernel 和输出方式均未改变。

## 2. 正确性验证

- 新增静态缓冲测试：先 full batch=4 replay，再 partial batch=2 replay。
- 验证前两行更新为真实输入，后两行 input/position 为 0，slots 恢复为 12/13。
- Continuous scheduler 动态加入测试继续通过。
- `py_compile`、`git diff --check` 通过。
- CUDA 1/4/8 并发各完成 128 请求，失败请求为 0。

## 3. 性能结果

| 并发 | 持久 Metadata | 尾部重置 | 变化 | 平均 TTFT | 平均 ITL |
|---:|---:|---:|---:|---:|---:|
| 1 | 280.89 | 280.72 tok/s | -0.06% | 21.32 ms | 3.07 ms |
| 4 | 741.75 | 753.45 tok/s | +1.58% | 52.72 ms | 4.17 ms |
| 8 | 1103.82 | 1104.93 tok/s | +0.10% | 92.46 ms | 5.24 ms |

收益较小，但三档没有实质性回退；并发 4 提升 1.58%。改动减少的是所有负载都会执行
的冗余写入，不依赖请求到达形态，因此保留。

## 4. 资源指标

加载后基线：GPU 3.0%，显存 4529 MiB，CPU 0.104%，系统内存 2164.71 MiB。

| 并发 | GPU 平均/峰值 | 显存平均/峰值 | CPU 平均/峰值 | 内存平均/峰值 |
|---:|---:|---:|---:|---:|
| 1 | 66.52% / 85% | 4584.59 / 4600 MiB | 3.74% / 9.24% | 2948.22 / 3074.42 MiB |
| 4 | 48.27% / 99% | 4667.13 / 4714 MiB | 4.50% / 9.24% | 2925.85 / 3092.79 MiB |
| 8 | 39.23% / 82% | 4859.62 / 4999 MiB | 4.54% / 8.94% | 2925.73 / 3097.77 MiB |

## 5. 下一步

下一轮只增加 vLLM 风格 token budget，不等待请求。每个 scheduler iteration 的已有
Decode token 与新 Prefill token 合计不得超过 `max_num_batched_tokens=2048`；能立即
接纳的请求马上 Prefill，超出预算的留到下一 iteration。当前 prompt 长度 1000，
因此单步最多接纳 2 个新请求，可避免一次完整 Prefill 长时间阻塞 active Decode。
