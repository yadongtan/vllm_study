# 从 Python 调用自定义 CUDA 算子：按实现顺序学习

这个 Demo 只完成一件事：输入一个 CUDA Tensor，为每个元素加 `1`。

```text
output[i] = input[i] + 1
```

重点不是加法本身，而是理解如何从零开始，按照正确顺序完成下面的调用链：

```text
demo.py
  -> __init__.py 中的 add_one()
  -> torch.ops.study_cuda_demo.add_one
  -> binding.cpp 注册的 CUDA 实现
  -> add_one.cu 中的 add_one_cuda()
  -> add_one_kernel<<<...>>>()
```

本文以现有 `demo.py` 为目标，严格按照实际开发顺序说明。每一步都会使用上一步
已经确定的函数名、参数和返回值。

## 第 0 步：先确定 Python 接口和计算目标

动手写 CUDA 之前，先明确最终希望怎样调用：

```python
output = add_one(input_tensor)
```

本 Demo 约定：

- 输入：一个位于 CUDA 上的连续 Tensor；
- 支持：`float32`、`float16`、`bfloat16`；
- 输出：与输入 shape、dtype、device 相同；
- 计算：每个元素加 `1`；
- 不原地修改输入，而是返回新 Tensor。

这个约定决定了后续所有层的接口：CUDA kernel 需要输入指针、输出指针和元素数；
C++ 启动函数需要接收一个 `torch::Tensor` 并返回一个 `torch::Tensor`；注册 schema
也必须是一个 Tensor 输入、一个 Tensor 输出。

## 第 1 步：在 `add_one.cu` 中编写 CUDA kernel

首先写真正运行在 GPU 上的函数：

```cpp
template <typename scalar_t>
__global__ void add_one_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (index < size) {
    output[index] = static_cast<scalar_t>(
        static_cast<float>(input[index]) + 1.0f);
  }
}
```

为什么先定义成这三个参数：

- `input`：Tensor 底层 CUDA 内存的输入指针；
- `output`：输出 Tensor 的 CUDA 内存指针；
- `size`：总元素数，用于防止线程越界。

特殊语法含义：

- `template <typename scalar_t>`：同一份 kernel 支持多种浮点 dtype；
- `__global__`：这个函数从 CPU 端启动，但由 GPU 线程执行；
- `__restrict__`：告诉编译器输入输出内存不重叠，便于优化；
- `blockIdx.x`：当前线程块编号；
- `blockDim.x`：每个线程块的线程数；
- `threadIdx.x`：当前线程在线程块内的编号。

这三个 CUDA 内建变量组成全局线程编号：

```text
index = blockIdx.x * blockDim.x + threadIdx.x
```

每个线程只处理一个元素。因为线程总数会向上取整，所以必须使用：

```cpp
if (index < size)
```

否则最后一个线程块中的多余线程可能访问 Tensor 边界之外的显存。

完成这一步后，我们有了 GPU kernel，但 Python 和普通 C++ 都还不能直接调用它。
下一步需要写一个运行在 CPU 侧的 C++ 启动函数。

## 第 2 步：在 `add_one.cu` 中编写 C++ 启动函数

根据第 0 步的接口约定，C++ 函数接收一个 Tensor，返回一个 Tensor：

```cpp
torch::Tensor add_one_cuda(const torch::Tensor& input)
```

这里的 `add_one_cuda` 是后面 `binding.cpp` 要绑定的函数，因此名字、参数类型、
`const` 和引用符号都必须保持一致。

### 2.1 检查输入

```cpp
TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
```

kernel 使用一维指针连续访问数据，所以要求输入位于 CUDA 且内存连续。检查失败时，
`TORCH_CHECK` 会抛出 Python 能看到的异常。

### 2.2 创建输出

```cpp
auto output = torch::empty_like(input);
const int64_t size = input.numel();
```

`empty_like` 创建与输入 shape、dtype、device 相同的未初始化 Tensor，正好满足第 0
步的输出约定。`numel()` 得到需要处理的元素总数。

### 2.3 计算 CUDA 启动规模

```cpp
constexpr int threads = 256;
const int64_t blocks = (size + threads - 1) / threads;
```

每个线程块使用 256 个线程。整数公式 `(size + threads - 1) / threads` 是向上取整，
确保即使 `size` 不是 256 的整数倍，也有足够线程覆盖全部元素。

### 2.4 使用 PyTorch 当前设备和 stream

```cpp
const c10::cuda::CUDAGuard device_guard(input.device());
const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
```

- `CUDAGuard`：把当前 CUDA 设备切换到输入所在设备；
- `getCurrentCUDAStream`：让自定义 kernel 加入 PyTorch 当前 stream。

使用当前 stream 很重要，它保证此前的 PyTorch CUDA 运算完成后，本 kernel 才读取
输入；后续 PyTorch 运算也会等待本 kernel 的结果。

### 2.5 根据 Tensor dtype 启动 kernel

```cpp
AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    input.scalar_type(),
    "add_one_cuda",
    [&] {
      add_one_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          size);
    });
```

`input.scalar_type()` 是运行时 dtype。分发宏会为它选择对应的 `scalar_t`，再实例化
第 1 步定义的模板 kernel。

kernel 启动配置：

```text
<<<blocks, threads, 0, stream>>>
    │       │       │    └─ CUDA stream
    │       │       └────── 动态共享内存字节数，本例不使用所以为 0
    │       └────────────── 每个线程块的线程数
    └────────────────────── 线程块数量
```

`data_ptr<scalar_t>()` 把 PyTorch Tensor 转换为 kernel 所需的设备内存指针。

最后检查启动错误并返回输出：

```cpp
C10_CUDA_KERNEL_LAUNCH_CHECK();
return output;
```

现在 `add_one.cu` 已经同时提供：

1. GPU 上执行的 `add_one_kernel`；
2. CPU 侧可调用的 `add_one_cuda`。

下一步需要把 `add_one_cuda` 注册成 Python 能通过 PyTorch dispatcher 找到的算子。

## 第 3 步：根据 `add_one_cuda` 编写 `binding.cpp`

首先引入 PyTorch C++ 扩展 API：

```cpp
#include <torch/extension.h>
```

然后按照第 2 步的函数签名进行声明：

```cpp
torch::Tensor add_one_cuda(const torch::Tensor& input);
```

这里仅有声明，没有函数体，这是正常的。`binding.cpp` 和 `add_one.cu` 会分别编译：

```text
binding.cpp  -> binding.o
add_one.cu   -> add_one.cuda.o
```

最终链接共享库时：

```text
binding.o 中需要 add_one_cuda
              +
add_one.cuda.o 中提供 add_one_cuda 的定义
              ↓
       study_cuda_demo.so
```

因此不要在 `binding.cpp` 中写 `#include "add_one.cu"`。包含 `.cu` 相当于复制整个
源码，容易产生重复定义，也破坏正常的分离编译结构。

### 3.1 定义算子 schema

```cpp
TORCH_LIBRARY(study_cuda_demo, m) {
  m.def("add_one(Tensor input) -> Tensor");
}
```

它来自第 0 步确定的接口：一个 Tensor 输入，一个 Tensor 输出。

- `study_cuda_demo`：`torch.ops` 下的命名空间；
- `m`：PyTorch 宏提供的 Library builder；
- `add_one`：Python 最终使用的算子名；
- `Tensor input`：一个 Tensor 参数；
- `-> Tensor`：返回一个 Tensor。

由此产生 Python 调用名：

```python
torch.ops.study_cuda_demo.add_one(input_tensor)
```

### 3.2 把 CUDA schema 绑定到第 2 步的函数

```cpp
TORCH_LIBRARY_IMPL(study_cuda_demo, CUDA, m) {
  m.impl("add_one", &add_one_cuda);
}
```

- `CUDA`：PyTorch dispatcher 的 CUDA 后端键；
- `m.impl("add_one", ...)`：为名为 `add_one` 的 schema 提供实现；
- `&add_one_cuda`：取得第 2 步函数的地址，并不是现在就调用它。

输入是 CUDA Tensor 时，dispatcher 会选择这个实现。到这一步，C++/CUDA 侧代码
已经完整，下一步要把两个源文件编译、链接并加载到当前 Python 进程。

## 第 4 步：在 `__init__.py` 中配置 CUDA 编译环境

本项目的 CUDA toolkit 随 PyTorch wheel 安装，路径是：

```text
.venv/lib/python3.12/site-packages/nvidia/cu13
```

代码从 `torch.__file__` 推导它，而不是依赖 Windows 的 `CUDA_PATH`：

```python
_site_packages = Path(torch.__file__).resolve().parents[1]
_cuda_root = _site_packages / "nvidia" / "cu13"
os.environ.setdefault("CUDA_HOME", str(_cuda_root))
```

然后把虚拟环境和 CUDA 编译器放到 `PATH` 前面：

```python
_venv_bin = Path(sys.prefix) / "bin"
os.environ["PATH"] = ":".join(
    (str(_venv_bin), str(_cuda_root / "bin"), os.environ.get("PATH", ""))
)
```

`cpp_extension` 在导入时缓存 `CUDA_HOME`，所以本 Demo 显式同步：

```python
import torch.utils.cpp_extension as _cpp_extension

_cpp_extension.CUDA_HOME = str(_cuda_root)
load = _cpp_extension.load
```

完成这一步后，Python 已经知道应该使用哪个 `nvcc`、CUDA 头文件和运行库。

## 第 5 步：在 `__init__.py` 中编译并加载扩展

先确定源文件和独立构建目录：

```python
root = Path(__file__).resolve().parent
build_directory = root / "build"
build_directory.mkdir(exist_ok=True)
```

然后调用：

```python
load(
    name="study_cuda_demo",
    build_directory=str(build_directory),
    sources=[str(root / "binding.cpp"), str(root / "add_one.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_ldflags=[str(_cuda_root / "lib" / "libcudart.so.13")],
    with_cuda=True,
    is_python_module=False,
    verbose=True,
)
```

各参数按编译发生的顺序理解：

1. `sources` 指定第 3 步的 `binding.cpp` 和第 1～2 步的 `add_one.cu`；
2. `.cpp` 使用宿主 C++ 编译器，附加 `extra_cflags`；
3. `.cu` 使用 `nvcc`，附加 `extra_cuda_cflags`；
4. 两个目标文件和 `extra_ldflags` 指定的 CUDA Runtime 一起链接为 `.so`；
5. `.so` 被加载到当前 Python 进程；
6. 加载时执行 `TORCH_LIBRARY`，于是算子出现在 `torch.ops` 中。

参数具体含义：

- `name`：扩展的编译逻辑名称；
- `build_directory`：Ninja、目标文件和 `.so` 的保存目录；
- `sources`：参与编译的所有源文件；
- `extra_cflags`：传给宿主 C++ 编译器；
- `extra_cuda_cflags`：传给 `nvcc`；
- `extra_ldflags`：传给最终链接器；
- `with_cuda=True`：启用 CUDA 编译流程；
- `is_python_module=False`：不使用 `PyInit_*`/pybind 模块入口，而使用
  `TORCH_LIBRARY + torch.ops`；
- `verbose=True`：打印真实编译和链接命令。

其中：

- `-O3`：高等级编译优化；
- `--use_fast_math`：使用更快的数学近似，可能有细微浮点差异；
- `-lineinfo`：保留 CUDA 源码行号，便于 Nsight 定位 kernel；
- `libcudart.so.13`：当前 PyTorch CUDA 13 环境的 CUDA Runtime。

如果后来把新 kernel 写入已经位于 `sources` 的 `add_one.cu`，列表无需改变；如果
创建另一个 `.cu` 文件，则必须把新文件加入 `sources`。

本 Demo 用 `_loaded` 防止在同一个 Python 进程中重复加载：

```python
_loaded = False

def _load_extension():
    global _loaded
    if _loaded:
        return
    load(...)
    _loaded = True
```

## 第 6 步：在 `__init__.py` 中提供 Python 包装函数

扩展加载后，`binding.cpp` 注册的算子已经存在。包装函数只做两件事：

```python
def add_one(input_tensor: torch.Tensor) -> torch.Tensor:
    _load_extension()
    return torch.ops.study_cuda_demo.add_one(input_tensor)
```

1. `_load_extension()`：确保第 5 步已经完成；
2. `torch.ops.study_cuda_demo.add_one(...)`：调用第 3 步注册的算子。

这里三个名称必须与前面对应：

```text
torch.ops.study_cuda_demo.add_one
          │               │
          │               └─ m.def("add_one(...)")
          └───────────────── TORCH_LIBRARY(study_cuda_demo, m)
```

dispatcher 看到输入是 CUDA Tensor，选择 `TORCH_LIBRARY_IMPL(..., CUDA, ...)`，进入
`add_one_cuda`，最后启动第 1 步的 kernel。

## 第 7 步：在 `demo.py` 中调用并验证

导入第 6 步的包装函数：

```python
from study.cuda_demo import add_one
```

创建 CUDA BF16 输入：

```python
input_tensor = torch.arange(
    1024,
    device="cuda",
    dtype=torch.bfloat16,
)
```

调用自定义算子，同时用 PyTorch 写参考结果：

```python
actual = add_one(input_tensor)
expected = input_tensor + 1
```

最后比较：

```python
torch.testing.assert_close(actual, expected, atol=0, rtol=0)
```

如果编译、注册、dtype 分发、指针访问或 kernel 计算任何一层有问题，这里都会失败。

## 第 8 步：运行完整 Demo

在仓库根目录运行：

```bash
cd /opt/vllm_study
.venv/bin/python3 study/cuda_demo/demo.py
```

第一次运行的实际过程：

```text
导入 study.cuda_demo
  -> 调用 add_one
  -> _load_extension 发现尚未加载
  -> 编译 binding.cpp
  -> nvcc 编译 add_one.cu
  -> 链接 study_cuda_demo.so
  -> 加载 .so 并执行算子注册
  -> torch.ops 调用 add_one_cuda
  -> add_one_cuda 启动 add_one_kernel
  -> demo.py 比较 CUDA 输出与参考输出
```

后续源代码没有变化时，Ninja 会复用 `study/cuda_demo/build` 中的编译缓存。

成功输出类似：

```text
input[:8]:  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
output[:8]: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
max error:  0.0
CUDA custom operator demo passed.
```

## 最终记忆顺序

以后添加新的 PyTorch CUDA 算子，可以沿用同一个顺序：

1. 先确定 Python 输入、输出和计算语义；
2. 写 `__global__` CUDA kernel；
3. 写接收/返回 Tensor 的 C++ 启动函数；
4. 在 `binding.cpp` 声明启动函数并定义 schema；
5. 用 `TORCH_LIBRARY_IMPL` 把 schema 绑定到 CUDA 函数；
6. 在 Python 中用 `load()` 编译、链接、加载所有源文件；
7. 提供调用 `torch.ops` 的 Python 包装函数；
8. 在 Demo/测试中与 PyTorch 参考实现比较。

`demo_cpp.cpp` 展示的是另一条“C++ 可执行文件直接链接 `.cu` 目标文件”的路径；
它不属于本文基于 `demo.py` 的 Python 调用流程。
