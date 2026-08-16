# 202608130210 - 中断的 CUDA Flash Attention V1 测试

## 1. 状态

该目录不是一个完成的性能版本。测试在正式 benchmark 完成前被外层进程终止，不能
用于吞吐、延迟或不同实现之间的性能比较。

## 2. 已生成文件

- `resources-baseline-system-before.csv`：启动前资源基线。
- `resources-baseline-cuda-flash-v1-loaded.csv`：模型加载后资源基线。
- `resources-cuda-flash-v1-concurrency-1.csv`：被中断的并发 1 采样片段。
- `cuda-flash-v1-server.log`：被中断服务的日志。

## 3. 缺失数据

- 没有任何并发档位的 benchmark JSON。
- 没有完成 128 请求的证据。
- 没有可用的吞吐、TTFT、TPOT 或 ITL 汇总。
- 没有完整的 1/4/8 并发资源汇总。

## 4. 后续正式结果

CUDA Flash Attention V1 的完整正式测试位于
`cuda_benchmark_results/202608130151`，对应报告为
`202608130151-CUDA_Flash_Attention_V1.md`。该正式轮次完成 1/4/8 并发、每档
128 请求，并包含性能、资源数据和分析。

## 5. 结论

保留本目录仅用于说明测试历史。累计对比和优化决策必须排除本目录，避免把残缺采样
误认为有效结果。
