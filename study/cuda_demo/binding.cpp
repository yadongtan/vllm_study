// PyTorch C++ 扩展 API：Tensor 类型和 TORCH_LIBRARY 宏来自这里。
#include <torch/extension.h>

// 函数体定义在 add_one.cu；这里先声明，供算子注册使用。
torch::Tensor add_one_cuda(const torch::Tensor& input);

// 注册名为 study_cuda_demo 的算子命名空间。
// m 是宏提供的 Library 对象，用来保存算子 schema。
TORCH_LIBRARY(study_cuda_demo, m) {
  // 声明 Python 可见的签名：一个 Tensor 输入，一个 Tensor 输出。
  m.def("add_one(Tensor input) -> Tensor");
}

// 为上面的算子注册 CUDA 后端实现。
TORCH_LIBRARY_IMPL(study_cuda_demo, CUDA, m) {
  // 输入位于 CUDA 时，PyTorch dispatcher 调用 add_one_cuda。
  m.impl("add_one", &add_one_cuda);
}
