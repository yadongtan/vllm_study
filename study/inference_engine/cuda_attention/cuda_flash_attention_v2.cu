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


__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(
        0xffffffff,
        value,
        offset);
  }

  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(
        value,
        __shfl_down_sync(0xffffffff, value, offset));
  }

  return value;
}

__device__ __forceinline__ bool is_attention_visible(
    int64_t request_idx,
    int64_t q_token_idx,
    int64_t kv_token_idx,
    int64_t q_seq_len,
    const bool* __restrict__ mask,
    const int64_t* __restrict__ query_start,
    const int64_t* __restrict__ kv_start) {
    if (request_idx < 0) {
        return false;
    }

    const int64_t request_q_start = query_start[request_idx];
    const int64_t request_q_end = query_start[request_idx + 1];
    const int64_t request_kv_start = kv_start[request_idx];
    const int64_t request_kv_end = kv_start[request_idx + 1];

    // 其他请求的 KV 不可见。
    if (kv_token_idx < request_kv_start || kv_token_idx >= request_kv_end) {
        return false;
    }

    const int64_t request_q_len = request_q_end - request_q_start;
    const int64_t request_kv_len = request_kv_end - request_kv_start;
    const int64_t request_past_kv_len = request_kv_len - request_q_len;
    const int64_t request_local_kv_idx = kv_token_idx - request_kv_start;

    // 当前请求的历史 KV 全部可见。
    if (request_local_kv_idx < request_past_kv_len) {
        return true;
    }

    // 当前请求本轮新增 K 在所有本轮 Q 中的全局列下标。
    const int64_t current_k_local_idx = request_local_kv_idx - request_past_kv_len;
    const int64_t current_k_global_idx = request_q_start + current_k_local_idx;

    // mask shape: [q_seq_len, q_seq_len]
    const int64_t mask_idx = q_token_idx * q_seq_len + current_k_global_idx;

    return mask[mask_idx];
}

// 支持 float、half 和 bfloat16。
template <typename scalar_t>
__global__ void cuda_flash_attention_v2_kernel(
    const scalar_t* __restrict__ q,const scalar_t* __restrict__ k,const scalar_t* __restrict__ v,
    const bool* __restrict__ mask,
    const int64_t* __restrict__ query_start,
    const int64_t* __restrict__ kv_start,
    // work_items shape = [num_valid_tiles, 8]。每个 block 直接取得一个
    // 同一请求内的 Q/KV tile，不再生成跨请求的笛卡尔积。
    const int64_t* __restrict__ work_items,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_o,
    int64_t partials_per_head,
    int64_t q_num_heads,int64_t q_seq_len,int64_t head_dim,
    int64_t kv_num_heads,int64_t kv_seq_len,int64_t kv_block_size
   ) {
  constexpr int64_t work_width = 8;
  const int64_t* work = work_items + blockIdx.x * work_width;
  const int64_t request_idx = work[0];
  const int64_t q_block_start = work[1];
  const int64_t current_q_block_size = work[2];
  const int64_t kv_block_start = work[3];
  const int64_t current_kv_block_size = work[4];
  const int64_t partial_tile_base = work[5];
  const int64_t kv_tile_idx = work[6];
  const int64_t request_num_kv_tiles = work[7];

  const int64_t q_head_idx = blockIdx.y;
  const int64_t q_tile_offset =
      q_head_idx * q_seq_len * head_dim + q_block_start * head_dim;
  const int64_t group_size = q_num_heads / kv_num_heads;
  const int64_t kv_head_idx = q_head_idx / group_size;
  const int64_t kv_tile_offset =
      kv_head_idx * kv_seq_len * head_dim + kv_block_start * head_dim;

  // 算出score的大小
  const int64_t score_size = current_q_block_size * current_kv_block_size;
  extern __shared__ float score_shared[];
  const float softmax_scale =
    rsqrtf(static_cast<float>(head_dim));
  // 每次跨越128个线程
  for(int64_t score_idx = threadIdx.x;score_idx < score_size; score_idx += blockDim.x){
      int64_t q_idx = score_idx / current_kv_block_size;
      int64_t k_idx = score_idx - q_idx * current_kv_block_size;
      const int64_t q_token_idx = q_block_start + q_idx;
      const int64_t kv_token_idx = kv_block_start + k_idx;
      // 需要知道这个block整体的起始位置
      const scalar_t* q_ptr = q + q_tile_offset + q_idx * head_dim;
      const scalar_t* k_ptr = k + kv_tile_offset + k_idx * head_dim;
      float score = 0.0f;
      for (int64_t d = 0; d < head_dim; ++d) {
          score += static_cast<float>(q_ptr[d]) * static_cast<float>(k_ptr[d]);
      }
      score *= softmax_scale;
      const bool is_visible = is_attention_visible(request_idx, q_token_idx, kv_token_idx, q_seq_len, mask, query_start, kv_start);
      if (!is_visible) {
          score = -INFINITY;
      }
      const int64_t score_shared_idx = q_idx * kv_block_size + k_idx;
      score_shared[score_shared_idx] = score;
  }
  __syncthreads(); //算出了所有的scores。
  const int lane_idx = threadIdx.x % 32;
  const int warp_idx = threadIdx.x / 32;
  const int num_warps = blockDim.x / 32;
  for (int64_t q_idx = warp_idx; q_idx < current_q_block_size; q_idx += num_warps) {
    float local_max = -INFINITY;
    for (int64_t k_idx = lane_idx; k_idx < current_kv_block_size; k_idx += 32) {
        const float score = score_shared[q_idx * kv_block_size + k_idx];
        local_max = fmaxf(local_max, score);
    }
    float row_max = warp_reduce_max(local_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0);
    // partial_tile_base 指向当前 Q tile 第一行在单个 head packed
    // partial 中的起点。不同请求不再预留任何空洞。
    const int64_t partial_idx =
        q_head_idx * partials_per_head
        + partial_tile_base
        + q_idx * request_num_kv_tiles
        + kv_tile_idx;
    if (lane_idx == 0) {
        partial_max[partial_idx] = row_max;
    }
    float local_sum = 0.0f;
    for (int64_t k_idx = lane_idx; k_idx < current_kv_block_size; k_idx += 32) {
        const int64_t score_idx = q_idx * kv_block_size + k_idx;
        const float score = score_shared[score_idx];
        const float exp_score = score == -INFINITY ? 0.0f : expf(score - row_max);
        score_shared[score_idx] = exp_score;
        local_sum += exp_score;
    }
    const float row_sum = warp_reduce_sum(local_sum);
    if (lane_idx == 0) {
        partial_sum[partial_idx] = row_sum;
    }
    __syncwarp();

    // 每个 lane 负责n个 head_dim，沿当前 KV tile 累加输出分子。
    const int64_t partial_o_offset = partial_idx * head_dim;
    for (int64_t d = lane_idx; d < head_dim; d += 32) {
        float output_acc = 0.0f;
        for (int64_t k_idx = 0; k_idx < current_kv_block_size; ++k_idx) {
            const float exp_score =
                score_shared[q_idx * kv_block_size + k_idx];
            const scalar_t* v_ptr =
                v + kv_tile_offset + k_idx * head_dim;
            output_acc +=
                exp_score * static_cast<float>(v_ptr[d]);
        }
        partial_o[partial_o_offset + d] = output_acc;
    }
  }
}


template <typename scalar_t>
__global__ void cuda_flash_attention_v2_reduce_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_o,
    const int64_t* __restrict__ partial_start,
    const int64_t* __restrict__ partial_count,
    scalar_t* __restrict__ o,
    int64_t q_seq_len,
    int64_t partials_per_head,
    int64_t head_dim) {
  const int64_t q_row = blockIdx.x;
  const int lane_idx = threadIdx.x;
  const int64_t q_token_idx = q_row % q_seq_len;
  const int64_t q_head_idx = q_row / q_seq_len;
  const int64_t num_kv_blocks = partial_count[q_token_idx];
  const int64_t partial_row_offset =
      q_head_idx * partials_per_head + partial_start[q_token_idx];

  float local_max = -INFINITY;
  for (int64_t kv_block_idx = lane_idx;
       kv_block_idx < num_kv_blocks;
       kv_block_idx += 32) {
    local_max = fmaxf(
        local_max,
        partial_max[partial_row_offset + kv_block_idx]);
  }

  float global_max = warp_reduce_max(local_max);
  global_max = __shfl_sync(0xffffffff, global_max, 0);

  float local_sum = 0.0f;
  for (int64_t kv_block_idx = lane_idx;
       kv_block_idx < num_kv_blocks;
       kv_block_idx += 32) {
    const int64_t partial_idx = partial_row_offset + kv_block_idx;
    const float split_max = partial_max[partial_idx];
    const float correction =
        split_max == -INFINITY || global_max == -INFINITY
            ? 0.0f
            : expf(split_max - global_max);
    local_sum += partial_sum[partial_idx] * correction;
  }

  float global_sum = warp_reduce_sum(local_sum);
  global_sum = __shfl_sync(0xffffffff, global_sum, 0);

  const int64_t output_offset = q_row * head_dim;
  for (int64_t d = lane_idx; d < head_dim; d += 32) {
    float output_acc = 0.0f;
    for (int64_t kv_block_idx = 0;
         kv_block_idx < num_kv_blocks;
         ++kv_block_idx) {
      const int64_t partial_idx = partial_row_offset + kv_block_idx;
      const float split_max = partial_max[partial_idx];
      const float correction =
          split_max == -INFINITY || global_max == -INFINITY
              ? 0.0f
              : expf(split_max - global_max);
      const int64_t partial_o_idx = partial_idx * head_dim + d;
      output_acc += partial_o[partial_o_idx] * correction;
    }

    o[output_offset + d] = global_sum > 0.0f
        ? static_cast<scalar_t>(output_acc / global_sum)
        : static_cast<scalar_t>(0.0f);
  }
}



// binding.cpp 注册给 study_cuda.cuda_flash_attention_v2 的 C++ 实现。
torch::Tensor cuda_flash_attention_v2_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& query_start,
    const torch::Tensor& kv_start,
    const torch::Tensor& work_items,
    const torch::Tensor& partial_start,
    const torch::Tensor& partial_count,
    int64_t partials_per_head,
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
  TORCH_CHECK(work_items.is_cuda(), "work_items must be a CUDA tensor");
  TORCH_CHECK(partial_start.is_cuda(), "partial_start must be a CUDA tensor");
  TORCH_CHECK(partial_count.is_cuda(), "partial_count must be a CUDA tensor");
  TORCH_CHECK(work_items.device() == q.device(),
              "work_items must be on the same CUDA device as q");
  TORCH_CHECK(partial_start.device() == q.device(),
              "partial_start must be on the same CUDA device as q");
  TORCH_CHECK(partial_count.device() == q.device(),
              "partial_count must be on the same CUDA device as q");
  TORCH_CHECK(work_items.scalar_type() == torch::kLong, "work_items must be int64");
  TORCH_CHECK(partial_start.scalar_type() == torch::kLong, "partial_start must be int64");
  TORCH_CHECK(partial_count.scalar_type() == torch::kLong, "partial_count must be int64");
  TORCH_CHECK(work_items.is_contiguous(), "work_items must be contiguous");
  TORCH_CHECK(partial_start.is_contiguous(), "partial_start must be contiguous");
  TORCH_CHECK(partial_count.is_contiguous(), "partial_count must be contiguous");
  auto sizes = q.sizes();
  int64_t num_heads = sizes[0];
  int64_t seq_len_q = sizes[1];
  int64_t head_dim = sizes[2];


  // 继承输入的 shape、dtype、device，但不初始化输出内容。
  auto o = torch::empty_like(q);
  if (seq_len_q == 0) {
    return o;
  }
  const int64_t k_seq_len = k.sizes()[1]; // k.shape = [kv_num_heads, kv_seq_len, head_dim]
  TORCH_CHECK(k_seq_len > 0,"Check failed: k_seq_len > 0 must be true ");
  const int64_t kv_num_heads = k.sizes()[0];
  const int64_t num_valid_tiles = work_items.size(0);
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
  TORCH_CHECK(work_items.dim() == 2 && work_items.size(1) == 8,
              "work_items must have shape [num_valid_tiles, 8]");
  TORCH_CHECK(partial_start.numel() == seq_len_q,
              "partial_start must have one entry per query token");
  TORCH_CHECK(partial_count.numel() == seq_len_q,
              "partial_count must have one entry per query token");
  TORCH_CHECK(partials_per_head > 0,
              "partials_per_head must be positive");

  const size_t threads = 128;
  dim3 grid(num_valid_tiles, num_heads, 1);
  dim3 block(threads, 1, 1);

  // 切换到输入所在 CUDA device，并复用 PyTorch 当前 stream。
  const c10::cuda::CUDAGuard device_guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream(q.device().index());
  const size_t shared_memory_bytes = static_cast<size_t>(q_block_size)
                                      * static_cast<size_t>(kv_block_size)
                                      * sizeof(float);
  auto float_options = q.options().dtype(torch::kFloat);
  auto partial_max = torch::empty({num_heads, partials_per_head}, float_options);
  auto partial_sum = torch::empty({num_heads, partials_per_head}, float_options);
  auto partial_o = torch::empty(
      {num_heads, partials_per_head, head_dim}, float_options);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "cuda_flash_attention_v2_cuda",
      [&] {
        // <<<blocks, threads, shared_memory_bytes, stream>>> 是启动配置。
        cuda_flash_attention_v2_kernel<scalar_t><<<grid, block, shared_memory_bytes, stream>>>(
            q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            query_start.data_ptr<int64_t>(),
            kv_start.data_ptr<int64_t>(),
            work_items.data_ptr<int64_t>(),
            partial_max.data_ptr<float>(),
            partial_sum.data_ptr<float>(),
            partial_o.data_ptr<float>(),
            partials_per_head,
            num_heads, seq_len_q, head_dim,
            kv_num_heads, k_seq_len, kv_block_size
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const int64_t num_q_rows = num_heads * seq_len_q;
        dim3 reduce_grid(num_q_rows, 1, 1);
        dim3 reduce_block(32, 1, 1);
        cuda_flash_attention_v2_reduce_kernel<scalar_t>
            <<<reduce_grid, reduce_block, 0, stream>>>(
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                partial_o.data_ptr<float>(),
                partial_start.data_ptr<int64_t>(),
                partial_count.data_ptr<int64_t>(),
                o.data_ptr<scalar_t>(),
                seq_len_q,
                partials_per_head,
                head_dim);
      });

  // 将 CUDA kernel 启动错误转换为 PyTorch 异常。
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}
