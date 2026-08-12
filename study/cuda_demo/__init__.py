"""一个最小、独立的 PyTorch CUDA 自定义算子示例。

调用链：add_one() -> torch.ops -> C++ 注册 -> CUDA kernel。
第一次调用 ``add_one`` 时编译，缓存位于本目录的 ``build/``。
"""

from pathlib import Path
import os
import sys

import torch


# PyTorch wheel 自带 CUDA toolkit。由 torch.py 反推出 site-packages，
# 避免错误使用 Windows 主机上的 CUDA_PATH。
_site_packages = Path(torch.__file__).resolve().parents[1]
_cuda_root = _site_packages / "nvidia" / "cu13"
os.environ.setdefault("CUDA_HOME", str(_cuda_root))

# 将虚拟环境和配套 CUDA 工具放在 PATH 前面，确保找到正确的 nvcc。
_venv_bin = Path(sys.prefix) / "bin"
os.environ["PATH"] = ":".join(
    (str(_venv_bin), str(_cuda_root / "bin"), os.environ.get("PATH", ""))
)

# cpp_extension 会在导入时读取 CUDA_HOME 并缓存结果，因此必须在导入
# ``load`` 之前设置环境变量；否则后面即使 os.environ 已有 CUDA_HOME，
# PyTorch 仍可能报 “CUDA_HOME environment variable is not set”。
import torch.utils.cpp_extension as _cpp_extension

_cpp_extension.CUDA_HOME = str(_cuda_root)
load = _cpp_extension.load

_loaded = False


def _load_extension() -> None:
    """编译并加载本目录中的 C++/CUDA 扩展，且只执行一次。"""

    global _loaded
    if _loaded:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_demo requires a CUDA-capable device")

    # __file__ 是本文件路径；parent 就是 study/cuda_demo。
    root = Path(__file__).resolve().parent
    build_directory = root / "build"
    build_directory.mkdir(exist_ok=True)

    # RTX 4080 SUPER 的 Compute Capability 是 8.9。
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    load(
        # 扩展逻辑名称，会出现在缓存、目标文件和诊断信息中。
        name="study_cuda_demo",
        # 所有 .o、.so 和 Ninja 文件都限定在本 Demo 的 build 目录。
        build_directory=str(build_directory),
        # binding.cpp 注册算子；add_one.cu 实现启动函数和 GPU kernel。
        sources=[
            str(root / "binding.cpp"),
            str(root / "add_one.cu"),
        ],
        # 传给宿主 C++ 编译器（通常为 g++）的优化参数。
        extra_cflags=["-O3"],
        # 传给 nvcc：优化、快速数学近似和性能分析所需源码行号。
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        # 最终链接时加入 CUDA Runtime 动态库。
        extra_ldflags=[str(_cuda_root / "lib" / "libcudart.so.13")],
        # 明确启用 CUDA/nvcc 编译流程。
        with_cuda=True,
        # 不提供 PyInit 模块入口；通过 TORCH_LIBRARY 和 torch.ops 调用。
        is_python_module=False,
        # 打印实际 Ninja、g++、nvcc 和链接命令，便于学习。
        verbose=True,
    )
    _loaded = True


def add_one(input_tensor: torch.Tensor) -> torch.Tensor:
    """调用 CUDA 算子，返回逐元素 ``input_tensor + 1``。"""

    _load_extension()
    return torch.ops.study_cuda_demo.add_one(input_tensor)


__all__ = ["add_one"]
