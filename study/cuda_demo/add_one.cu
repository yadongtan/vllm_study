// torch::Tensor、TORCH_CHECK 和 PyTorch 扩展类型。
#include <torch/extension.h>

// 获取 PyTorch 当前 CUDA stream。
#include <ATen/cuda/CUDAContext.h>
// CUDAGuard 让当前设备与输入 Tensor 所在设备一致。
#include <c10/cuda/CUDAGuard.h>
// kernel 启动错误检查宏。
#include <c10/cuda/CUDAException.h>
// CUDA 内建类型、线程索引及 kernel 启动语法。
#include <cuda_runtime.h>


// 模板让同一个 kernel 支持 float、half 和 bfloat16。
// __global__ 表示该函数由 CPU 启动、由 GPU 线程执行。
template <typename scalar_t>
__global__ void add_one_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t size) {
  // blockIdx.x 是线程块号；blockDim.x 是每块线程数；
  // threadIdx.x 是当前线程在块内的编号。
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  // blocks 向上取整，最后一块可能有多余线程，所以必须检查边界。
  if (index < size) {
    // 通过 float 做加法，再转回输入 dtype。
    output[index] = static_cast<scalar_t>(
        static_cast<float>(input[index]) + 1.0f);
  }
}


// binding.cpp 注册给 study_cuda_demo.add_one 的 C++ 实现。
torch::Tensor add_one_cuda(const torch::Tensor& input) {
  // TORCH_CHECK 失败时抛出 Python 能看到的 RuntimeError。
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat ||
          input.scalar_type() == torch::kFloat16 ||
          input.scalar_type() == torch::kBFloat16,
      "input must be float32, float16, or bfloat16");

  // 继承输入的 shape、dtype、device，但不初始化输出内容。
  auto output = torch::empty_like(input);
  const int64_t size = input.numel();
  if (size == 0) {
    return output;
  }

  // 一维 grid，每个 block 256 个线程；blocks 向上取整覆盖所有元素。
  constexpr int threads = 256;
  const int64_t blocks = (size + threads - 1) / threads;

  // 切换到输入所在 CUDA device，并复用 PyTorch 当前 stream。
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());

  // 根据输入运行时 dtype，把 scalar_t 替换成对应的 C++ 标量类型。
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "add_one_cuda",
      [&] {
        // <<<blocks, threads, shared_memory_bytes, stream>>> 是启动配置。
        add_one_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
      });

  // 将 CUDA kernel 启动错误转换为 PyTorch 异常。
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
