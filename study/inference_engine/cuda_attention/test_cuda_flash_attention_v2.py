"""CUDA Flash Attention V2 packed tile 工作表正确性测试。"""

import torch
import torch.nn.functional as F

from study.inference_engine.cuda_attention import cuda_flash_attention_v2
from study.inference_engine.qwen2_demo import (
    Request,
    build_cuda_flash_v2_work,
)


def test_cuda_flash_attention_v2_packed_tiles() -> None:
    """变长请求的 V2 结果应与逐请求 PyTorch SDPA 一致。"""
    torch.manual_seed(0)
    device = torch.device("cuda")
    q_heads, kv_heads, head_dim = 14, 2, 64
    q_lens = [1, 4, 7]
    past_lens = [0, 3, 11]
    kv_lens = [q + past for q, past in zip(q_lens, past_lens)]
    q_start = torch.tensor(
        [0, 1, 5, 12], device=device, dtype=torch.long
    )
    kv_start = torch.tensor(
        [0, 1, 8, 26], device=device, dtype=torch.long
    )

    query = torch.randn(
        q_heads, sum(q_lens), head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        kv_heads, sum(kv_lens), head_dim, device=device, dtype=torch.bfloat16
    )
    value = torch.randn_like(key)

    compact_mask = torch.zeros(
        sum(q_lens), sum(q_lens), device=device, dtype=torch.bool
    )
    q_offset = 0
    for q_len in q_lens:
        compact_mask[q_offset:q_offset + q_len, q_offset:q_offset + q_len] = (
            torch.ones(q_len, q_len, device=device, dtype=torch.bool).tril()
        )
        q_offset += q_len

    requests = [
        Request(torch.zeros(q_len, dtype=torch.long), 1)
        for q_len in q_lens
    ]
    for request, past_len in zip(requests, past_lens):
        request.start = past_len
    work_items, partial_start, partial_count, partials_per_head = (
        build_cuda_flash_v2_work(
            requests,
            q_start.tolist(),
            device,
            q_block_size=2,
            kv_block_size=3,
        )
    )

    # 验证工作表没有生成跨请求 tile。原来的全局笛卡尔积会产生
    # ceil(total_q/2) * ceil(total_kv/3) = 54 个 tile；packed 工作表只含
    # Σ ceil(q_i/2) * ceil(kv_i/3) = 1 + 6 + 24 = 31 个有效 tile。
    work_cpu = work_items.cpu()
    assert work_cpu.shape == (31, 8)
    assert work_cpu.shape[0] < 54
    for work in work_cpu:
        request = int(work[0])
        q_tile_start = int(work[1])
        q_tile_len = int(work[2])
        kv_tile_start = int(work[3])
        kv_tile_len = int(work[4])
        assert int(q_start[request]) <= q_tile_start
        assert q_tile_start + q_tile_len <= int(q_start[request + 1])
        assert int(kv_start[request]) <= kv_tile_start
        assert kv_tile_start + kv_tile_len <= int(kv_start[request + 1])

    actual = cuda_flash_attention_v2(
        query,
        key,
        value,
        compact_mask.contiguous(),
        q_start,
        kv_start,
        work_items,
        partial_start,
        partial_count,
        partials_per_head,
        2,
        3,
    )

    expected = torch.empty_like(query)
    groups = q_heads // kv_heads
    for request, (q_len, past_len) in enumerate(zip(q_lens, past_lens)):
        qs = int(q_start[request].item())
        qe = int(q_start[request + 1].item())
        ks = int(kv_start[request].item())
        ke = int(kv_start[request + 1].item())
        q = query[:, qs:qe].unsqueeze(0)
        k = key[:, ks:ke].unsqueeze(0).repeat_interleave(groups, dim=1)
        v = value[:, ks:ke].unsqueeze(0).repeat_interleave(groups, dim=1)
        mask = torch.arange(ke - ks, device=device)[None, :] <= (
            past_len + torch.arange(q_len, device=device)[:, None]
        )
        expected[:, qs:qe] = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            scale=head_dim**-0.5,
        ).squeeze(0)

    # BF16 和不同归约顺序会产生小量舍入差异。
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
