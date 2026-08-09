"""验证自定义 CUDA ragged GQA attention 算子的正确性。

测试同时构造：

* `actual`：调用我们编写的 CUDA 算子；
* `expected`：使用 PyTorch SDPA 计算参考结果。

最后比较两者的数值误差。由于 CUDA 算子使用 BF16 和并行归约，
允许一个小的绝对/相对误差是正常的。
"""

import torch
import torch.nn.functional as F

from study.inference_engine.cuda_attention import ragged_gqa_attention


def test_ragged_gqa_attention():
    """测试多个变长请求、GQA 分组和因果注意力边界。"""

    # 固定随机种子，使每次测试生成完全相同的 Q/K/V 输入，便于复现。
    torch.manual_seed(0)

    # 测试 Qwen2 类 GQA 配置：14 个 query heads 共享 2 个 KV heads，
    # 因此每个 KV head 对应 7 个 query heads；每个 head 的维度为 64。
    q_heads, kv_heads, dim = 14, 2, 64

    # 三个请求本轮分别产生 1、4、7 个 query token，故 query 是变长的。
    # 这些长度存放在 GPU 上，CUDA 算子可直接读取，避免额外的 CPU/GPU 拷贝。
    q_lens = torch.tensor([1, 4, 7], device="cuda", dtype=torch.long)

    # 每个请求在本轮计算前已有的历史 KV 长度分别为 0、3、11。
    # 这同时覆盖首 token prefill、短历史 decode 和长历史 decode。
    past_lens = torch.tensor([0, 3, 11], device="cuda", dtype=torch.long)

    # 前缀和边界：q_start[r] 到 q_start[r+1] 是第 r 个请求在连续
    # query 缓冲区中的区间。这里得到 [0, 1, 5, 12]。
    q_start = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.long), q_lens)
    ).cumsum(0)

    # KV 缓冲区同时包含历史 token 和本轮 token；每个请求的 KV 长度为
    # past_lens + q_lens，即 [1, 7, 18]。kv_start 得到 [0, 1, 8, 26]，
    # 用于从拼接后的 key/value 中定位各请求的连续区间。
    kv_lens = past_lens + q_lens
    kv_start = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.long), kv_lens)
    ).cumsum(0)

    # 按 [heads, tokens, head_dim] 生成连续的 ragged query/KV 缓冲区。
    # 使用 BF16 与实际推理保持一致，并直接在 CUDA 上创建数据。
    query = torch.randn(
        q_heads,
        int(q_lens.sum()),
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        kv_heads,
        int(kv_lens.sum()),
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)

    # 调用自定义 CUDA 实现。算子内部根据每个请求的边界、历史长度和
    # query 长度执行 QK^T -> scale -> causal softmax -> P V。
    actual = ragged_gqa_attention(
        query,
        key,
        value,
        q_start,
        kv_start,
        past_lens,
        q_lens,
        int(kv_lens.max()),
        dim**-0.5,
    )

    # 分配参考输出；下面逐请求调用 PyTorch SDPA，避免把不同请求的
    # 变长序列错误地拼成一个固定长度 batch。
    expected = torch.empty_like(query)

    # GQA 的参考实现显式把每个 KV head 复制到对应的 query-head 分组。
    # 这只用于测试基准，不是自定义 CUDA 算子的实现方式。
    groups = q_heads // kv_heads

    for r in range(3):
        # 取出第 r 个请求在 query 和 KV 拼接缓冲区中的区间。
        qs, qe = q_start[r].item(), q_start[r + 1].item()
        ks, ke = kv_start[r].item(), kv_start[r + 1].item()

        # SDPA 需要带 batch 维度 [batch, heads, sequence, dim]，
        # 所以给当前请求增加一个 batch=1 的维度，并复制 GQA 的 KV head。
        q = query[:, qs:qe].unsqueeze(0)
        k = key[:, ks:ke].unsqueeze(0).repeat_interleave(groups, dim=1)
        v = value[:, ks:ke].unsqueeze(0).repeat_interleave(groups, dim=1)

        # 当前请求的 query/KV token 数。
        qn = qe - qs
        kn = ke - ks

        # 构造因果 mask：第 i 个 query 只能看到历史 token 以及本轮中
        # 不晚于它的 token。历史长度 past_lens[r] 使本轮第 i 个 token
        # 对应到完整序列中的位置 past_lens[r] + i。
        mask = torch.arange(kn, device="cuda")[None, :] <= (
            past_lens[r] + torch.arange(qn, device="cuda")[:, None])

        # 使用 PyTorch SDPA 生成参考结果。dropout 必须为 0，保证推理确定性；
        # scale 与 CUDA 算子保持相同，避免比较时引入额外差异。
        expected[:, qs:qe] = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            scale=dim**-0.5,
        ).squeeze(0)

    # BF16 并行计算与 SDPA 的实现细节不同，允许 2e-2 的误差范围。
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
