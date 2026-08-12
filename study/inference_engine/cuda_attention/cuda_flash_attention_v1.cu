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


// 支持 float、half 和 bfloat16。
template <typename scalar_t>
__global__ void cuda_flash_attention_v1_kernel(
    const scalar_t* __restrict__ q,const scalar_t* __restrict__ k,const scalar_t* __restrict__ v,
    const bool* __restrict__ mask,
    const int64_t* __restrict__ query_start,
    const int64_t* __restrict__ kv_start,
    int64_t num_requests,
    scalar_t* __restrict__ o,
    int64_t batch_size,int64_t q_num_heads,int64_t q_seq_len,int64_t head_dim,
    int64_t kv_num_heads,int64_t kv_seq_len,int64_t kv_block_size
   ) {
  int64_t num_q_blocks_idx = blockIdx.x;
  int64_t batch_size_idx = blockIdx.y;
  int64_t num_heads_idx = blockIdx.z;
  int64_t q_block_size = blockDim.x;
  // 计算block_offset
  int64_t block_offset = num_q_blocks_idx * q_block_size;
  // 先找线程块在q中的偏移量
  int64_t q_seq_len_offset = batch_size_idx * q_num_heads * q_seq_len * head_dim + num_heads_idx * q_seq_len * head_dim + block_offset * head_dim;
  // 再找这个线程块中这个线程的偏移量, shape = [1,head_dim] 在qwen2中head_dim=64，每个线程要计算全部的kv，kv=32k？
  const int64_t q_token_idx = block_offset + threadIdx.x;
  int64_t q_token_offset = q_seq_len_offset + threadIdx.x * head_dim;
  // threadIdx.x是这个block中的q_seq_len的索引
  if(q_token_idx>= q_seq_len){
      return;
  }
  int64_t request_idx = 0;
  for(int64_t i = 0; i < num_requests; i++){
        if (q_token_idx >= query_start[i] &&
            q_token_idx < query_start[i + 1]) {
            request_idx = i;
            break;
        }
  }
  const int64_t request_q_start = query_start[request_idx];
  const int64_t request_q_end = query_start[request_idx + 1];
  const int64_t request_kv_start = kv_start[request_idx];
  const int64_t request_kv_end = kv_start[request_idx + 1];
  const int64_t request_q_len = request_q_end - request_q_start;
  const int64_t request_kv_len = request_kv_end - request_kv_start;
  const int64_t request_past_kv_len = request_kv_len - request_q_len;
  // 1. 取q
  const scalar_t* q_ptr = q + q_token_offset;
  // 2. 取k并转置, k.shape = [batch_size, kv_num_heads, kv_seq_len, head_dim]，其实不用显式转置，乘的时候注意下位置就是了
  int64_t group_size = q_num_heads / kv_num_heads;
  int64_t kv_num_heads_idx = num_heads_idx / group_size;
  float max_old = -INFINITY;
  float sum_exp_old = 0.0f;
  // shared_memory shape：
  // scores:     [q_block_size, kv_block_size]
  // output_acc: [q_block_size, head_dim]
  extern __shared__ float shared_memory[];
  float* score_ptr = shared_memory + threadIdx.x * kv_block_size;
  float* output_acc_ptr = shared_memory + q_block_size * kv_block_size + threadIdx.x * head_dim;
  for(int i =0; i < head_dim; i++){
    output_acc_ptr[i] = 0.0f;
  }

  const float softmax_scale = rsqrtf(static_cast<float>(head_dim));
  for(int64_t kv_block_offset = 0; kv_block_offset < kv_seq_len; kv_block_offset += kv_block_size){
    int64_t kv_block_end = min(kv_seq_len, kv_block_offset + kv_block_size);
    float max_block = -INFINITY;
    for(int64_t kv_token_idx = kv_block_offset; kv_token_idx < kv_block_end; kv_token_idx++){
        int64_t kv_token_offset = batch_size_idx * kv_num_heads * kv_seq_len * head_dim
        + kv_num_heads_idx * kv_seq_len * head_dim + kv_token_idx * head_dim;
        const scalar_t* k_ptr = k + kv_token_offset; //找到与之相乘的k起点
        float score = 0.0f;
        for(int64_t d = 0;d < head_dim; d++){
            // 3. 注意力点积:x = q @ k.T
            score += static_cast<float>(q_ptr[d]) * static_cast<float>(k_ptr[d]);
        }
        // 4. 缩放 head_dim ** -0.5
        score *= softmax_scale;
        // 应用mask
        const int64_t local_kv_idx = kv_token_idx - kv_block_offset;
        bool is_visible = false;
        if (kv_token_idx >= request_kv_start && kv_token_idx < request_kv_end) {
            const int64_t request_local_kv_idx = kv_token_idx - request_kv_start;
            if (request_local_kv_idx < request_past_kv_len) {
                // 当前请求的历史 KV 全部可见。
                is_visible = true;
            } else {
                // 当前请求的本次新增 KV 读取块对角 mask。
                const int64_t current_k_local_idx = request_local_kv_idx - request_past_kv_len;
                const int64_t current_k_global_idx = request_q_start + current_k_local_idx;
                const int64_t mask_offset = q_token_idx * q_seq_len + current_k_global_idx;
                is_visible = mask[mask_offset];
            }
        }
        if (!is_visible) {
          score = -INFINITY;
        }
        score_ptr[local_kv_idx] = score; // 保存缩放后的局部attention score
        max_block = fmaxf(max_block, score);
    }
    // 5. 求 max_new = max(max(x_i)  x in (x_n, ..., x_m), max_old)
    const float max_new = fmaxf(max_block, max_old);
    // 6. 求 exp_x = e**(x_i - max(x_new)) x in (x_n, ..., x_m)
    float sum_exp_block = 0.0f;
    const int64_t kv_block_length = kv_block_end - kv_block_offset;
    for(int64_t local_kv_idx = 0;local_kv_idx < kv_block_length; local_kv_idx++){
        const float current_score = score_ptr[local_kv_idx];
        const float exp_score = current_score == -INFINITY
                                    ? 0.0f
                                    : expf(current_score - max_new);
        score_ptr[local_kv_idx] = exp_score;
        sum_exp_block += exp_score;
    }
    // 7. 求上一次记录缩放scale = (e ** (max_old - max_new))
    const float old_scale = max_old == -INFINITY ? 0.0f : expf(max_old - max_new);
    // 8. 求 sum_exp_new = exp_old * old_scale + Σ(exp_new_x)
    const float sum_exp_new = sum_exp_old * old_scale + sum_exp_block;
    for(int64_t i = 0; i < head_dim; i++){
        output_acc_ptr[i] *= old_scale;
    }
    for(int64_t local_kv_idx = 0; local_kv_idx < kv_block_length; local_kv_idx++){
        const int64_t kv_token_idx = kv_block_offset + local_kv_idx;
        const int64_t v_token_offset = batch_size_idx * kv_num_heads * kv_seq_len * head_dim +
                                     kv_num_heads_idx * kv_seq_len * head_dim +
                                     kv_token_idx * head_dim;
        const scalar_t* v_ptr = v + v_token_offset;
        const float exp_score = score_ptr[local_kv_idx];
        for(int64_t i = 0; i < head_dim; i++){
            // 9. 求新的output_new = x @ v，其中x = e**(x_i - max(x_new))，未归一化除以 sum_exp_new
            // 10. 求未归一化之前的分子: output_temp = output_old * scale + output_new，记录到shared_memory
            // 不过这里之前就已经完成了output_old * scale，避免在这里重复算
            output_acc_ptr[i] += exp_score * static_cast<float>(v_ptr[i]) ;
        }
    }
    // 11. 更新max_old, sum_exp_old
    max_old = max_new;
    sum_exp_old = sum_exp_new;
  }
  scalar_t* output_ptr = o + q_token_offset;
  for(int64_t i = 0; i < head_dim; i++){
        // 10. 求新的总和: output = output_temp/ sum_exp_total
        if (sum_exp_old > 0.0f) {
            output_ptr[i] = static_cast<scalar_t>(output_acc_ptr[i] / sum_exp_old);
        } else {
            output_ptr[i] = static_cast<scalar_t>(0.0f);
        }
  }
}



// binding.cpp 注册给 study_cuda.cuda_flash_attention_v1 的 C++ 实现。
torch::Tensor cuda_flash_attention_v1_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& query_start,
    const torch::Tensor& kv_start,
    int64_t q_block_size,
    int64_t kv_block_size) {
  // TORCH_CHECK 失败时抛出 Python 能看到的 RuntimeError。
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(q.scalar_type() == torch::kFloat || q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16, "q must be float32, float16, or bfloat16");
  TORCH_CHECK(k.scalar_type() == torch::kFloat || k.scalar_type() == torch::kFloat16 || k.scalar_type() == torch::kBFloat16, "k must be float32, float16, or bfloat16");
  TORCH_CHECK(v.scalar_type() == torch::kFloat || v.scalar_type() == torch::kFloat16 || v.scalar_type() == torch::kBFloat16, "v must be float32, float16, or bfloat16");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q, k, and v must have the same dtype");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, and v must be on the same CUDA device");
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
  const int64_t k_seq_len = k.sizes()[2]; // k.shape = [batch_size, kv_num_heads, kv_seq_len, head_dim]
  TORCH_CHECK(k_seq_len > 0,"Check failed: k_seq_len > 0 must be true ");
  const int64_t kv_num_heads = k.sizes()[1];
  const int64_t num_requests = query_start.numel() - 1;
  TORCH_CHECK(kv_num_heads > 0,"Check failed: kv_num_heads > 0 must be true");

  TORCH_CHECK(num_heads % kv_num_heads == 0,"Check failed: num_heads % kv_num_heads == 0 must be true");
  if (k_seq_len == 0) {
    return o;
  }

  // q_block_size即每个q多少个块
  // num_Q_blocks要启动多少个块
  // grid定义多少个块，block定义每个块多少个线程，假设每个线程处理一个q
  TORCH_CHECK(q_block_size > 0 && q_block_size <= 1024,"Check failed: q_block_size > 0 && q_block_size <= 1024 must be true");
  TORCH_CHECK(kv_block_size > 0,"Check failed: kv_block_size > 0 must be true");
  int num_Q_blocks = (seq_len_q + q_block_size - 1) / q_block_size;
  // int num_K_blocks = (k_seq_len + kv_block_size - 1) / kv_block_size;

  dim3 grid(num_Q_blocks, batch_size, num_heads);
  dim3 block(q_block_size, 1, 1);

  // 切换到输入所在 CUDA device，并复用 PyTorch 当前 stream。
  const c10::cuda::CUDAGuard device_guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream(q.device().index());
  const size_t shared_memory_bytes = static_cast<size_t>(q_block_size)
                                      * static_cast<size_t>(kv_block_size + head_dim)
                                      * sizeof(float);
  // 根据输入运行时 dtype，把 scalar_t 替换成对应的 C++ 标量类型。
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "cuda_flash_attention_v1_cuda",
      [&] {
        // <<<blocks, threads, shared_memory_bytes, stream>>> 是启动配置。
        cuda_flash_attention_v1_kernel<scalar_t><<<grid, block, shared_memory_bytes, stream>>>(
            q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            query_start.data_ptr<int64_t>(),
            kv_start.data_ptr<int64_t>(),
            num_requests,
            o.data_ptr<scalar_t>(),
            batch_size, num_heads, seq_len_q, head_dim,
            kv_num_heads, k_seq_len, kv_block_size
            );
      });

  // 将 CUDA kernel 启动错误转换为 PyTorch 异常。
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}
