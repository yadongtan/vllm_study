# CUDA 自定义算子最小 Demo

此目录完全独立，只实现 `output[i] = input[i] + 1`，用于学习 PyTorch 调用
CUDA 算子的完整链路。

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 编译/加载扩展，提供 Python 函数 `add_one` |
| `binding.cpp` | 注册算子 schema 及其 CUDA 实现 |
| `add_one.cu` | GPU kernel、C++ 启动函数、检查、dtype 分发与 stream |
| `demo.py` | 调用算子，并与 PyTorch 参考结果比较 |
| `build/` | 自动生成的独立编译缓存，不提交 Git |

## 调用链

```text
demo.py: add_one(input)
  -> __init__.py: torch.ops.study_cuda_demo.add_one(input)
  -> binding.cpp: PyTorch dispatcher 选择 CUDA 实现
  -> add_one.cu: add_one_cuda(input)
  -> add_one_kernel<<<blocks, threads, 0, stream>>>(...)
  -> 每个 GPU 线程处理一个元素
```

## 运行

```bash
cd /opt/vllm_study
.venv/bin/python3 study/cuda_demo/demo.py
```

## `load(...)` 参数详解

### `name="study_cuda_demo"`

扩展的逻辑名称，会进入编译缓存、目标文件和诊断信息。
`TORCH_LIBRARY(study_cuda_demo, ...)` 也采用这个名称是为了保持概念统一，
但二者不是通过 Python 变量自动关联的。

### `build_directory=str(build_directory)`

指定 Ninja 中间文件、`.o` 和最终 `.so` 的位置。`str(...)` 把 `Path` 转成
底层编译工具接受的字符串。本例固定为 `study/cuda_demo/build`。

### `sources=[...]`

需要编译的源文件列表：`.cpp` 由宿主 C++ 编译器处理，`.cu` 由 nvcc 处理。
`binding.cpp` 负责注册，`add_one.cu` 负责实现。

如果把新 kernel 写进已经列入 `sources` 的现有 `.cu`，不用增加 source；如果创建
新的 `.cu`，就必须把它加入列表。

### `extra_cflags=["-O3"]`

传给宿主 C++ 编译器（通常为 g++）的参数。`-O3` 表示高等级优化。

### `extra_cuda_cflags=[...]`

传给 nvcc：

- `-O3`：优化 CUDA/C++ 代码；
- `--use_fast_math`：使用较快数学近似，可能产生很小的浮点差异；
- `-lineinfo`：保留 CUDA 源码行号，方便 Nsight 定位 kernel。

### `extra_ldflags=[...]`

传给最终链接器。这里加入 PyTorch wheel 自带的 `libcudart.so.13`，使共享库能
解析 CUDA Runtime API。路径和版本必须与当前 PyTorch 环境一致。

### `with_cuda=True`

明确启用 CUDA 扩展流程，让 `.cu` 由 nvcc 编译。

### `is_python_module=False`

生成的 `.so` 不需要 `PyInit_...` Python 模块入口。我们通过 `TORCH_LIBRARY`
注册到 `torch.ops`，调用形式是：

```python
torch.ops.study_cuda_demo.add_one(input_tensor)
```

### `verbose=True`

输出 Ninja、g++、nvcc 与链接命令，便于学习和排错。

## `binding.cpp` 每一行来自哪里

### 头文件

```cpp
#include <torch/extension.h>
```

来自 PyTorch 安装包的 C++ 头文件，提供 `torch::Tensor`、注册宏等 API。
尖括号表示从编译器 include 搜索路径查找。

### 前置声明

```cpp
torch::Tensor add_one_cuda(const torch::Tensor& input);
```

函数体在 `add_one.cu`。这里只声明它，使本编译单元知道函数的返回类型和参数。

- `torch::Tensor`：PyTorch C++ Tensor 包装对象；
- `const`：函数不通过这个引用修改 Tensor 包装对象；
- `&`：按引用传递，避免复制包装对象；
- 末尾 `;`：这是声明而不是函数定义。

### 注册 schema

```cpp
TORCH_LIBRARY(study_cuda_demo, m) {
  m.def("add_one(Tensor input) -> Tensor");
}
```

`TORCH_LIBRARY` 是 PyTorch 提供的宏，加载 `.so` 时执行注册逻辑。

- `study_cuda_demo`：`torch.ops` 下的命名空间；
- `m`：宏创建的 Library builder 变量，名字可以换；
- `add_one`：算子名；
- `Tensor input`：一个名为 input 的 Tensor 参数；
- `-> Tensor`：返回一个 Tensor。

因此 Python 调用名成为 `torch.ops.study_cuda_demo.add_one`。

### 注册 CUDA 实现

```cpp
TORCH_LIBRARY_IMPL(study_cuda_demo, CUDA, m) {
  m.impl("add_one", &add_one_cuda);
}
```

- `TORCH_LIBRARY_IMPL`：为已有 schema 提供具体后端实现；
- `CUDA`：PyTorch dispatcher 的 CUDA dispatch key；
- `m.impl`：把算子名与实现函数连接；
- `&add_one_cuda`：取得函数地址，不是立即调用函数。

如果以后实现 CPU 版本，可另外注册 `TORCH_LIBRARY_IMPL(..., CPU, ...)`。

## `add_one.cu` 特殊语法

- `template <typename scalar_t>`：为不同 dtype 生成 kernel 实例；
- `__global__`：CPU 启动、GPU 执行的 CUDA kernel；
- `__restrict__`：承诺指针不重叠，帮助编译器优化；
- `blockIdx.x`：当前线程块编号；
- `threadIdx.x`：当前线程在块内编号；
- `blockDim.x`：每个线程块的线程数；
- `<<<blocks, threads, 0, stream>>>`：块数、每块线程数、动态共享内存字节数、stream；
- `AT_DISPATCH_FLOATING_TYPES_AND2`：依据 Tensor dtype 选择 `scalar_t`；
- `data_ptr<scalar_t>()`：取得 Tensor 连续存储的类型化设备指针；
- `CUDAGuard`：切换到输入所在 CUDA device；
- `getCurrentCUDAStream`：使用 PyTorch 当前 stream，保证执行顺序；
- `C10_CUDA_KERNEL_LAUNCH_CHECK`：把 CUDA 启动错误变成 Python 异常。

## 为什么不使用单独的 Python C 模块

传统扩展常用 `PYBIND11_MODULE`，然后直接 `import xxx`。这个 Demo 使用
`TORCH_LIBRARY + torch.ops`，与当前 attention 算子以及 vLLM 常见的 PyTorch
算子注册方式一致，并且能让 PyTorch dispatcher 根据 CPU/CUDA 后端选择实现。
