#include <torch/extension.h>

torch::Tensor ragged_gqa_attention_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& kv_start_loc,
    const torch::Tensor& past_lens,
    const torch::Tensor& q_lens,
    int64_t max_kv_len,
    double scale);

torch::Tensor cuda_flash_attention_v1_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& query_start,
    const torch::Tensor& kv_start,
    int64_t q_block_size,
    int64_t kv_block_size);

torch::Tensor cuda_flash_attention_v2_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& query_start,
    const torch::Tensor& kv_start,
    int64_t q_block_size,
    int64_t kv_block_size);

TORCH_LIBRARY(study_cuda, m) {
  m.def("ragged_gqa_attention(Tensor query, Tensor key, Tensor value, "
        "Tensor query_start_loc, Tensor kv_start_loc, Tensor past_lens, "
        "Tensor q_lens, int max_kv_len, float scale) -> Tensor");
  m.def(
      "cuda_flash_attention_v1(Tensor q, Tensor k, Tensor v, Tensor mask, "
      "Tensor query_start, Tensor kv_start, int q_block_size, "
      "int kv_block_size) -> Tensor");
  m.def(
      "cuda_flash_attention_v2(Tensor q, Tensor k, Tensor v, Tensor mask, "
      "Tensor query_start, Tensor kv_start, int q_block_size, "
      "int kv_block_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(study_cuda, CUDA, m) {
  m.impl("ragged_gqa_attention", &ragged_gqa_attention_cuda);
  m.impl("cuda_flash_attention_v1", &cuda_flash_attention_v1_cuda);
  m.impl("cuda_flash_attention_v2", &cuda_flash_attention_v2_cuda);
}
