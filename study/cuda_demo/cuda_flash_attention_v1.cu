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
// 针对q很短的情况，大量decode
template <typename scalar_t>
__global__ void cuda_flash_attention_v1_kernel(
    const scalar_t* __restrict__ q,const scalar_t* __restrict__ k,const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ o,
    int64_t batch_size,int64_t q_num_heads,int64_t q_seq_len,int64_t head_dim,
    int64_t kv_num_heads,int64_t kv_seq_len,int64_t kv_block_size,
   ) {
  // blockIdx.x 是线程块号；blockDim.x 是每块线程数；
  // threadIdx.x 是当前线程在块内的编号。
  int64_t num_q_blocks_idx = blockIdx.x;
  int64_t batch_size_idx = blockIdx.y;
  int64_t num_heads_idx = blockIdx.z;

  int64_t q_block_size = blockDim.x;
  int64_t q_idx = num_q_blocks_idx * q_block_size + threadIdx.x;
  // 每个q的head_dim，qwen2有64个维度，所以这里是64个值的起始偏移量
  int64_t offset  =((batch_size_idx * q_num_heads + num_heads_idx) * q_seq_len + q_idx) * head_dim;
  // 1. 取q
  // 2. 取k并转置
  // 3. 注意力点积:x = q @ k.T
  // 4. 缩放 head_dim ** -0.5
  // 5. 求 max_new = max(max(x_i)  x in (x_n, ..., x_m), max_old)
  // 6. 求 exp_x = e**(x_i - max(x_new)) x in (x_n, ..., x_m)
  // 7. 求 sum_exp_new = Σ(exp_x)
  // 8. 求上一次记录缩放scale = (e ** (max_old - max_new))
  // 9. 求新的output_new = x @ v，其中x = e**(x_i - max(x_new))，未归一化除以 sum_exp_new
  // 10. 求未归一化之前的分子: output_temp = output_old * scale + output_new，记录到寄存器
  // 11. 求归一化分母: sum_exp_total = sum_exp_old * (e **(max_old - max_new)) + sum_exp_new
  // 10. 求新的总和: output = output_temp/ sum_exp_total
  // 11. 更新max_old, sum_exp_old，output_old(未归一化分子 output_temp)
}



// binding.cpp 注册给 study_cuda_demo.add_one 的 C++ 实现。
torch::Tensor cuda_flash_attention_v1_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    int64_t q_block_size,
    int64_t kv_block_size) {
  // TORCH_CHECK 失败时抛出 Python 能看到的 RuntimeError。
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(
      q.scalar_type() == torch::kFloat ||
          q.scalar_type() == torch::kFloat16 ||
          q.scalar_type() == torch::kBFloat16,
      "q must be float32, float16, or bfloat16");
  TORCH_CHECK(
      k.scalar_type() == torch::kFloat ||
          k.scalar_type() == torch::kFloat16 ||
          k.scalar_type() == torch::kBFloat16,
      "k must be float32, float16, or bfloat16");
  TORCH_CHECK(
      v.scalar_type() == torch::kFloat ||
          v.scalar_type() == torch::kFloat16 ||
          v.scalar_type() == torch::kBFloat16,
      "v must be float32, float16, or bfloat16");


    auto sizes = q.sizes();
    int64_t batch_size = sizes[0];
    int64_t num_heads = sizes[1];
    int64_t seq_len_q = sizes[2];
    int64_t head_dim = sizes[3];


  // 继承输入的 shape、dtype、device，但不初始化输出内容。
  auto o = torch::empty_like(q);
  if (seq_len_q == 0) {
    return o;
  }
  const int64_t k_seq_len = k.sizes()[2]; // k.shape = [batch, num_heads, seq_len, dim]
  const int64_t kv_num_heads = k.sizes()[1];
  if (k_seq_len == 0) {
    return o;
  }

  // q_block_size即每个q多少个块
  // num_Q_blocks要启动多少个块
  // grid定义多少个块，block定义每个块多少个线程，假设每个线程处理一个q
  int num_Q_blocks = (seq_len_q + q_block_size - 1) / q_block_size;
  // int num_K_blocks = (k_seq_len + kv_block_size - 1) / kv_block_size;
  dim3 grid(num_Q_blocks, batch_size, num_heads);
  dim3 block(q_block_size, 1, 1);

  // 切换到输入所在 CUDA device，并复用 PyTorch 当前 stream。
  const c10::cuda::CUDAGuard device_guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream(q.device().index());

  // 根据输入运行时 dtype，把 scalar_t 替换成对应的 C++ 标量类型。
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "cuda_flash_attention_v1_cuda",
      [&] {
        // <<<blocks, threads, shared_memory_bytes, stream>>> 是启动配置。
        cuda_flash_attention_v1_kernel<scalar_t><<<grid, block, 0, stream>>>(
            q,k,v,mask,o,
            batch_size, num_heads, seq_len_q, head_dim,
            kv_num_heads, k_seq_len, kv_block_size
            );
      });

  // 将 CUDA kernel 启动错误转换为 PyTorch 异常。
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}
