// 这是一个真正的 C++ 主程序：它直接调用 add_one.cu 中的 C++ 函数，
// 不使用 TORCH_LIBRARY、TORCH_LIBRARY_IMPL、torch::ops 或 Python。

#include <torch/torch.h>
#include <cuda_runtime.h>

#include <iostream>

// add_one_cuda 的函数体在同目录的 add_one.cu 中。
// 这里的声明必须与 .cu 中的函数签名完全一致，链接器才能找到它。
torch::Tensor add_one_cuda(const torch::Tensor& input);

int main() {
  // 这个检查调用的是 LibTorch 的 CUDA 查询 API；它不是自定义算子注册。
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available" << std::endl;
    return 1;
  }

  // 直接构造一个位于 GPU 上的 BF16 Tensor，作为 add_one_cuda 的参数。
  const auto options = torch::TensorOptions()
                            .device(torch::kCUDA)
                            .dtype(torch::kBFloat16);
  const auto input = torch::arange(1024, options);

  // 关键调用：直接进入 add_one.cu 中的 C++ 启动函数。
  // 该函数内部才会用 <<<blocks, threads, stream>>> 启动 GPU kernel。
  const auto actual = add_one_cuda(input);

  // 使用 LibTorch C++ API 计算参考值，便于验证 CUDA 结果。
  const auto expected = input + 1;
  const auto max_error = (actual - expected).abs().max().item<float>();

  // item<float>() 会等待异步 CUDA 工作完成，再把一个标量复制到 CPU。
  if (!torch::allclose(actual, expected, 0.0, 0.0)) {
    std::cerr << "CUDA result mismatch; max error = " << max_error
              << std::endl;
    return 1;
  }

  // CUDA Runtime 提供设备名称；这同样不经过 PyTorch 算子注册。
  int device_index = 0;
  cudaDeviceProp device_properties{};
  cudaGetDevice(&device_index);
  cudaGetDeviceProperties(&device_properties, device_index);
  std::cout << "device: " << device_properties.name << '\n';
  std::cout << "input[:8]:  " << input.slice(0, 0, 8) << '\n';
  std::cout << "output[:8]: " << actual.slice(0, 0, 8) << '\n';
  std::cout << "max error:  " << max_error << '\n';
  std::cout << "Direct C++ -> add_one.cu call passed." << std::endl;
  return 0;
}
