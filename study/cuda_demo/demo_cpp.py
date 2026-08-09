"""绕过 Python 包装函数，直接调用已注册的 C++/CUDA 算子。

运行方式（在仓库根目录）：

    .venv/bin/python3 study/cuda_demo/demo_cpp.py

这里的“直接”是指直接调用 PyTorch dispatcher 中由 ``binding.cpp`` 注册的
``torch.ops.study_cuda_demo.add_one``，而不是调用 Python 包装函数
``study.cuda_demo.add_one``。

Python 无法直接用函数名调用 `.cu` 中的 ``__global__`` kernel；kernel 必须由
C++/CUDA 启动函数用 ``<<<grid, block, ...>>>`` 启动，再通过 PyTorch 的
``torch.ops`` 暴露给 Python。
"""

import sys
from pathlib import Path

import torch


try:
    # 推荐的模块导入方式：仓库根目录在 sys.path 中。
    from study.cuda_demo import _load_extension
except ModuleNotFoundError:
    # 兼容直接执行此文件：此时 sys.path 默认只有 cuda_demo 目录，
    # 因此把仓库根目录（当前文件向上两级）加入模块搜索路径。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from study.cuda_demo import _load_extension


def main() -> None:
    """加载扩展后直接调用 torch.ops 中的 C++/CUDA 注册入口。"""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # _load_extension() 负责调用 torch.utils.cpp_extension.load：
    # 它会编译 binding.cpp 和 add_one.cu，并执行 TORCH_LIBRARY 注册。
    _load_extension()

    # 生成位于 GPU 上的 BF16 输入，和 demo.py 使用相同的数据类型。
    input_tensor = torch.arange(
        1024,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # 这里没有调用 study.cuda_demo.add_one(input_tensor)。
    # 这一行直接进入 binding.cpp 中注册的：
    #   TORCH_LIBRARY_IMPL(study_cuda_demo, CUDA, m)
    #   m.impl("add_one", &add_one_cuda)
    # PyTorch dispatcher 根据 input_tensor.device 选择 CUDA 实现。
    actual = torch.ops.study_cuda_demo.add_one(input_tensor)

    # 用 PyTorch 原生表达式计算参考结果。
    expected = input_tensor + 1

    # 取标量会同步当前 CUDA 工作，随后检查两个结果是否完全一致。
    max_error = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"input[:8]:  {input_tensor[:8].tolist()}")
    print(f"output[:8]: {actual[:8].tolist()}")
    print(f"max error:  {max_error}")
    print("Direct torch.ops C++/CUDA call passed.")


if __name__ == "__main__":
    main()
