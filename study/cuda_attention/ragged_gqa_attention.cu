#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void ragged_gqa_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ query_start_loc,
    const int64_t* __restrict__ kv_start_loc,
    const int64_t* __restrict__ past_lens,
    const int64_t* __restrict__ q_lens,
    int64_t num_requests,
    int64_t total_q_tokens,
    int64_t total_kv_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t q_token = static_cast<int64_t>(blockIdx.x);
  const int64_t q_head = static_cast<int64_t>(blockIdx.y);
  const int tid = threadIdx.x;
  if (q_token >= total_q_tokens || q_head >= num_q_heads) return;

  int64_t request = 0;
  while (request + 1 < num_requests &&
         q_token >= query_start_loc[request + 1]) {
    ++request;
  }
  const int64_t q_local = q_token - query_start_loc[request];
  const int64_t visible_len = past_lens[request] + q_local + 1;
  if (q_local < 0 || q_local >= q_lens[request]) return;
  const int64_t kv_begin = kv_start_loc[request];
  const int64_t group_size = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t q_base = (q_head * total_q_tokens + q_token) * head_dim;
  const int64_t kv_stride = total_kv_tokens * head_dim;

  extern __shared__ float scores[];
  if (tid == 0) {
    float max_score = -INFINITY;
    for (int64_t k = 0; k < visible_len; ++k) {
      const int64_t k_base = kv_head * kv_stride + (kv_begin + k) * head_dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < head_dim; ++d) {
        dot += static_cast<float>(query[q_base + d]) *
               static_cast<float>(key[k_base + d]);
      }
      scores[k] = dot * scale;
      max_score = fmaxf(max_score, scores[k]);
    }
    float denom = 0.0f;
    for (int64_t k = 0; k < visible_len; ++k) {
      scores[k] = expf(scores[k] - max_score);
      denom += scores[k];
    }
    scores[visible_len] = denom;
  }
  __syncthreads();
  const float denom = scores[visible_len];
  for (int64_t d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t k = 0; k < visible_len; ++k) {
      const int64_t v_base = kv_head * kv_stride + (kv_begin + k) * head_dim;
      acc += (scores[k] / denom) * static_cast<float>(value[v_base + d]);
    }
    output[q_base + d] = static_cast<scalar_t>(acc);
  }
}

torch::Tensor ragged_gqa_attention_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& kv_start_loc,
    const torch::Tensor& past_lens,
    const torch::Tensor& q_lens,
    int64_t max_kv_len,
    double scale) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "query, key and value must be CUDA tensors");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type(),
              "query, key and value must have the same dtype");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "query, key and value must be [heads, tokens, head_dim]");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                  value.is_contiguous(),
              "query, key and value must be contiguous");
  TORCH_CHECK(query_start_loc.is_cuda() && kv_start_loc.is_cuda() &&
                  past_lens.is_cuda() && q_lens.is_cuda(),
              "attention metadata must be CUDA tensors");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt64 &&
                  kv_start_loc.scalar_type() == torch::kInt64 &&
                  past_lens.scalar_type() == torch::kInt64 &&
                  q_lens.scalar_type() == torch::kInt64,
              "attention metadata must be int64 tensors");
  TORCH_CHECK(query.size(0) % key.size(0) == 0,
              "num_q_heads must be divisible by num_kv_heads");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "key and value must have the same shape");
  TORCH_CHECK(query.size(2) == key.size(2),
              "query and key head_dim must match");
  const auto total_q_tokens = query.size(1);
  const auto total_kv_tokens = key.size(1);
  const auto num_requests = query_start_loc.numel() - 1;
  auto output = torch::empty_like(query);
  if (total_q_tokens == 0 || num_requests == 0) return output;
  TORCH_CHECK(max_kv_len > 0 && max_kv_len <= total_kv_tokens,
              "max_kv_len must be within the packed KV length");
  TORCH_CHECK(total_kv_tokens + 1 <= 16384,
              "naive CUDA attention supports at most 16383 packed KV tokens");
  const c10::cuda::CUDAGuard device_guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  const dim3 grid(total_q_tokens, query.size(0), 1);
  constexpr int threads = 128;
  const size_t shared_bytes = static_cast<size_t>(max_kv_len + 1) *
                              sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
      "ragged_gqa_attention_cuda", [&] {
        ragged_gqa_attention_kernel<scalar_t><<<grid, threads, shared_bytes,
                                                stream>>>(
            query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            query_start_loc.data_ptr<int64_t>(),
            kv_start_loc.data_ptr<int64_t>(), past_lens.data_ptr<int64_t>(),
            q_lens.data_ptr<int64_t>(), num_requests, total_q_tokens,
            total_kv_tokens, query.size(0), key.size(0), query.size(2),
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
