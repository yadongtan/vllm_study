"""运行最小 CUDA 自定义算子示例。

执行：

    cd /opt/vllm_study
    .venv/bin/python3 study/cuda_demo/demo.py

第一次运行会在 ``study/cuda_demo/build`` 编译扩展。
"""

import torch

try:
    # 推荐方式：从仓库根目录以模块运行 ``python -m study.cuda_demo.demo``。
    from study.cuda_demo import add_one
except ModuleNotFoundError:
    # 兼容直接运行 ``python study/cuda_demo/demo.py``：此时 Python 会把
    # study/cuda_demo 放入 sys.path，需要把仓库根目录临时加入搜索路径。
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from study.cuda_demo import add_one


def main() -> None:
    """创建输入、调用 CUDA 算子，并与 PyTorch 参考结果比较。"""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # 使用 BF16 模拟当前 Qwen2 推理的数据类型。
    input_tensor = torch.arange(
        1024,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # 调用自定义 CUDA 算子，并用普通 PyTorch 表达式生成参考结果。
    actual = add_one(input_tensor)
    expected = input_tensor + 1

    # max().item() 需要取回 CPU 标量，也会等待前面的 CUDA 工作完成。
    max_error = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"input[:8]:  {input_tensor[:8].tolist()}")
    print(f"output[:8]: {actual[:8].tolist()}")
    print(f"max error:  {max_error}")
    print("CUDA custom operator demo passed.")


if __name__ == "__main__":
    main()
