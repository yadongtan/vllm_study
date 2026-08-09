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

TORCH_LIBRARY(study_cuda, m) {
  m.def("ragged_gqa_attention(Tensor query, Tensor key, Tensor value, "
        "Tensor query_start_loc, Tensor kv_start_loc, Tensor past_lens, "
        "Tensor q_lens, int max_kv_len, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(study_cuda, CUDA, m) {
  m.impl("ragged_gqa_attention", &ragged_gqa_attention_cuda);
}
