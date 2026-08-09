// PyTorch C++ 扩展的核心头文件：提供 torch::Tensor、TORCH_CHECK、
// AT_DISPATCH_* 等类型和宏。
#include <torch/extension.h>

// 提供 at::cuda::getCurrentCUDAStream，用于接入 PyTorch 当前执行流。
#include <ATen/cuda/CUDAContext.h>
// 提供 CUDAGuard；多 GPU 时确保 kernel 在输入所在设备上启动。
#include <c10/cuda/CUDAGuard.h>
// CUDA Driver API 的基础声明。
#include <cuda.h>
// CUDA Runtime API、dim3、blockIdx/threadIdx 和 kernel 启动支持。
#include <cuda_runtime.h>

// scalar_t 是模板类型占位符。启动函数根据 query 的运行时 dtype，
// 将它实例化为 float、double、at::Half 或 at::BFloat16。
//
// 每个 CUDA block 负责一个 (q_token, q_head)：
//   blockIdx.x -> packed query 中的 token 编号；
//   blockIdx.y -> query head 编号；
// block 内的线程协作计算该 query/head 对所有可见 KV 的注意力。
template <typename scalar_t>
__global__ void ragged_gqa_attention_kernel(
    // Q/K/V 和输出的布局分别是：
    // query/output: [num_q_heads, total_q_tokens, head_dim]
    // key/value:    [num_kv_heads, total_kv_tokens, head_dim]
    // __restrict__ 承诺这些指针访问的内存不互相重叠，允许编译器优化。
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    // 变长请求使用 packed 布局拼在一起。start_loc 是长度为
    // num_requests + 1 的前缀和边界。例如 [0, 3, 8] 表示两个请求
    // 分别占用 token 区间 [0, 3) 和 [3, 8)。
    const int64_t* __restrict__ query_start_loc,
    const int64_t* __restrict__ kv_start_loc,
    // past_lens[r]：请求 r 在本轮 query 之前已有的历史 KV 数量。
    // q_lens[r]：请求 r 本轮参与计算的 query token 数量。
    const int64_t* __restrict__ past_lens,
    const int64_t* __restrict__ q_lens,
    // 后续标量描述 packed Tensor 的整体形状与注意力缩放系数。
    int64_t num_requests,
    int64_t total_q_tokens,
    int64_t total_kv_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  // 二维 grid 中，一个 block 唯一对应一个 query token 和一个 query head。
  const int64_t q_token = static_cast<int64_t>(blockIdx.x);
  const int64_t q_head = static_cast<int64_t>(blockIdx.y);
  // tid 是当前线程在 block 内的一维编号，范围为 [0, blockDim.x)。
  const int tid = threadIdx.x;
  // 防御性边界检查。正常 grid 配置恰好覆盖范围，但保留检查更安全。
  if (q_token >= total_q_tokens || q_head >= num_q_heads) return;

  // query 是多个变长请求拼接成的一维 token 序列。扫描前缀和边界，
  // 找到全局 q_token 属于哪个请求。当前实现使用线性扫描，逻辑直观，
  // 请求很多时可以进一步优化为预先生成 token_to_request 映射。
  int64_t request = 0;
  while (request + 1 < num_requests &&
         q_token >= query_start_loc[request + 1]) {  // 8 [0,5,6,10]
    ++request;
  }
  // 当前 token 在所属请求内部的局部位置。若请求的 packed 区间从 10
  // 开始、q_token=12，则 q_local=2。
  const int64_t q_local = q_token - query_start_loc[request];
  // 因果注意力下，本轮第 q_local 个 token 能看到：全部历史 KV，
  // 加上本轮从第 0 个到自己（含自己）的 token。
  const int64_t visible_len = past_lens[request] + q_local + 1;
  // 校验该 token 确实处于请求声明的本轮 query 长度内。
  if (q_local < 0 || q_local >= q_lens[request]) return;
  // 请求 r 的 KV 在 packed K/V Tensor 中从 kv_begin 开始。
  const int64_t kv_begin = kv_start_loc[request];
  // GQA 中多个 Q head 共享一个 KV head。例如 14 个 Q head、2 个 KV
  // head 时 group_size=7，Q head [0..6] 使用 KV head 0，[7..13] 使用 1。
  const int64_t group_size = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / group_size;
  // 把三维 [head, token, dim] 下标转换为连续内存的一维起始偏移。
  const int64_t q_base = (q_head * total_q_tokens + q_token) * head_dim;
  // 相邻 KV head 在连续 K/V Tensor 中跨过的元素数量。
  const int64_t kv_stride = total_kv_tokens * head_dim;

  // 动态共享内存由 kernel 启动参数 shared_bytes 决定，并被手工分成：
  //   scores[visible_len]：保存该 query 对每个可见 key 的分数/指数值；
  //   reduction[blockDim.x]：保存每个线程的局部归约结果。
  // 共享内存位于一个 block 内，访问速度远快于全局显存。
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduction = scores + visible_len;
  // 静态共享标量保存整个 block 归约得到的最大 attention score。
  __shared__ float max_score_shared;

  // 第一阶段：QK^T。
  // 一个 block 固定负责一个 query/head，然后依次遍历所有可见 key token。
  // 对每个 key，block 内线程沿 head_dim 分工，并把局部点积归约成一个分数。
  for (int64_t k = 0; k < visible_len; ++k) {
    // 定位当前请求、当前共享 KV head、第 k 个可见 key 的起始地址。
    const int64_t k_base = kv_head * kv_stride + (kv_begin + k) * head_dim;
    // 每个线程累加自己负责的 d=tid, tid+blockDim.x, ... 这些维度。
    // 即使 head_dim 大于线程数，步进循环也能覆盖全部维度。
    float partial = 0.0f;
    for (int64_t d = tid; d < head_dim; d += blockDim.x) {
      partial += static_cast<float>(query[q_base + d]) *
                 static_cast<float>(key[k_base + d]);
    }
    // 每个线程将自己的局部点积写入共享内存，准备树形归约。
    reduction[tid] = partial;
    // 等待所有线程完成写入，否则某线程可能读取到其他线程的旧数据。
    __syncthreads();
    // 树形归约：第一轮前半线程加后半线程，之后每轮参与线程减半。
    // 最终 reduction[0] 是完整的 Q·K 点积。
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (tid < offset) reduction[tid] += reduction[tid + offset];
      __syncthreads();
    }
    // 只有线程 0 写最终分数，乘 scale（通常是 1/sqrt(head_dim)）。
    if (tid == 0) scores[k] = reduction[0] * scale;
    // 确保 scores[k] 已写好，才进入下一个 k；下一轮会复用 reduction。
    __syncthreads();
  }

  // 第二阶段：数值稳定的 Softmax。
  // 公式为 exp(score - max_score) / sum(exp(score - max_score))。
  // 减去最大值可避免 exp 对较大正数产生上溢。
  // 首先，每个线程跨步扫描自己负责的 key，求线程局部最大值。
  float local_max = -INFINITY;
  for (int64_t k = tid; k < visible_len; k += blockDim.x) {
    local_max = fmaxf(local_max, scores[k]);
  }
  // 把线程局部最大值写入共享内存，再做一次树形 max 归约。
  reduction[tid] = local_max;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + offset]);
    }
      __syncthreads();
  }
  // 线程 0 把 block 最大值写到共享标量；所有线程同步后读取。
  if (tid == 0) max_score_shared = reduction[0];
  __syncthreads();
  const float max_score = max_score_shared;
  // 每个线程计算自己负责位置的 exp(score-max)，就地覆盖 scores，
  // 同时累加自己的局部指数和。
  float local_sum = 0.0f;
  for (int64_t k = tid; k < visible_len; k += blockDim.x) {
    scores[k] = expf(scores[k] - max_score);
    local_sum += scores[k];
  }
  // 将所有线程的局部和归约成 softmax 分母。
  reduction[tid] = local_sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) reduction[tid] += reduction[tid + offset];
    __syncthreads();
  }
  // 前面的归约循环末尾已经同步；这里再次同步是冗余但无害的防御性同步。
  __syncthreads();
  const float denom = reduction[0];

  // 第三阶段：P*V。
  // 线程沿输出 head_dim 分工。对每个输出维度 d，遍历全部可见 value，
  // 累加 softmax_weight[k] * value[k,d]。
  for (int64_t d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t k = 0; k < visible_len; ++k) {
      // 找到当前 GQA KV head、当前请求、第 k 个 value 的起始偏移。
      const int64_t v_base = kv_head * kv_stride + (kv_begin + k) * head_dim;
      // scores[k] 当前保存未归一化指数；除以 denom 得到 softmax 权重。
      acc += (scores[k] / denom) * static_cast<float>(value[v_base + d]);
    }
    // 累加使用 float 提升数值稳定性，写回时转换成原 Tensor dtype。
    output[q_base + d] = static_cast<scalar_t>(acc);
  }
}

// CPU 侧的 C++ 启动函数。它接收 PyTorch Tensor，负责输入验证、
// 输出分配、grid/block/共享内存配置、dtype 分发和 CUDA kernel 启动。
// 真正在 GPU 上进行 QK/Softmax/PV 计算的是上面的 kernel。
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
  // 1. 验证 Q/K/V 全部位于 CUDA，否则不能把 data_ptr 交给 CUDA kernel。
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "query, key and value must be CUDA tensors");
  // 2. kernel 的同一个 scalar_t 同时解释 Q/K/V，所以 dtype 必须相同。
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type(),
              "query, key and value must have the same dtype");
  // 3. 当前索引公式只支持 [heads, tokens, head_dim] 三维布局。
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "query, key and value must be [heads, tokens, head_dim]");
  // 4. data_ptr 后使用手写连续偏移，非连续 stride 会导致错误寻址。
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                  value.is_contiguous(),
              "query, key and value must be contiguous");
  // 5. kernel 直接读取边界/长度元数据，因此它们也必须位于 CUDA。
  TORCH_CHECK(query_start_loc.is_cuda() && kv_start_loc.is_cuda() &&
                  past_lens.is_cuda() && q_lens.is_cuda(),
              "attention metadata must be CUDA tensors");
  // 6. 元数据指针在 kernel 中声明为 int64_t*，Tensor dtype 必须匹配。
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt64 &&
                  kv_start_loc.scalar_type() == torch::kInt64 &&
                  past_lens.scalar_type() == torch::kInt64 &&
                  q_lens.scalar_type() == torch::kInt64,
              "attention metadata must be int64 tensors");
  // 7. GQA 映射 q_head/group_size 要求 Q head 数整除 KV head 数。
  TORCH_CHECK(query.size(0) % key.size(0) == 0,
              "num_q_heads must be divisible by num_kv_heads");
  // 8. K/V 必须逐位置配对，因此 shape 完全相同。
  TORCH_CHECK(key.sizes() == value.sizes(),
              "key and value must have the same shape");
  // 9. Q·K 与 P·V 都要求相同的 head_dim。
  TORCH_CHECK(query.size(2) == key.size(2),
              "query and key head_dim must match");
  // 从 Tensor 和前缀和元数据推导 kernel 所需的全局尺寸。
  const auto total_q_tokens = query.size(1);
  const auto total_kv_tokens = key.size(1);
  // start_loc 对 N 个请求有 N+1 个边界，因此请求数要减 1。
  const auto num_requests = query_start_loc.numel() - 1;
  // 输出 shape/dtype/device 与 query 一致，内容由 kernel 写入。
  auto output = torch::empty_like(query);
  // 空 batch 无需启动 kernel，直接返回相应的空输出。
  if (total_q_tokens == 0 || num_requests == 0) return output;
  // max_kv_len 用于预先配置所有 block 的动态共享内存上限。
  TORCH_CHECK(max_kv_len > 0 && max_kv_len <= total_kv_tokens,
              "max_kv_len must be within the packed KV length");
  // 教学实现设置显式上限，避免为异常长序列申请过多 block 共享内存。
  TORCH_CHECK(max_kv_len <= 8192,
              "parallel CUDA attention supports at most 8192 KV tokens");
  // 多 GPU 时切换到 query 所在设备；离开作用域时自动恢复之前设备。
  const c10::cuda::CUDAGuard device_guard(query.device());
  // 使用 PyTorch 当前 stream，使同一 stream 内的前后 Tensor 操作保持顺序。
  const auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  // 二维 grid：[total_q_tokens, num_q_heads, 1]。每个 block 处理一个
  // (q_token, q_head)，对应 kernel 中 blockIdx.x/blockIdx.y。
  const dim3 grid(total_q_tokens, query.size(0), 1);
  // 每个 block 固定 128 个线程，协作完成点积、softmax 归约和 P*V。
  constexpr int threads = 128;
  // 动态共享内存布局是 max_kv_len 个 score 加 threads 个归约槽。
  // kernel 中实际 reduction 紧跟当前 visible_len；按最大长度申请可确保容量足够。
  const size_t shared_bytes = static_cast<size_t>(max_kv_len + threads) *
                              sizeof(float);
  // 根据 query 的运行时 dtype 决定模板 scalar_t。FLOATING_TYPES 提供
  // 常规浮点类型，AND2 再显式加入 FP16 和 BF16。
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
      // 这是 dispatch 错误信息中的标签，不是函数调用。
      "ragged_gqa_attention_cuda", [&] {
        // [&] 是按引用捕获外层局部变量的 Lambda，因此这里可以使用
        // grid、threads、shared_bytes、stream、query/output 等变量。
        // 第三个 <<<>>> 参数是每个 block 的动态共享内存字节数。
        ragged_gqa_attention_kernel<scalar_t><<<grid, threads, shared_bytes,
                                                stream>>>(
            // Tensor.data_ptr<T>() 取得类型化 CUDA 内存指针。
            query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            query_start_loc.data_ptr<int64_t>(),
            kv_start_loc.data_ptr<int64_t>(), past_lens.data_ptr<int64_t>(),
            q_lens.data_ptr<int64_t>(), num_requests, total_q_tokens,
            total_kv_tokens, query.size(0), key.size(0), query.size(2),
            static_cast<float>(scale));
      });
  // kernel 启动是异步的；这里检查启动配置等即时 CUDA 错误，并把错误
  // 转换为 PyTorch 异常。它不会为了正常执行而全局同步 GPU。
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // 返回的 Tensor 记录在当前 stream 上产生，后续同 stream 操作会正确排队。
  return output;
}
