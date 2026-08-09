import torch
import torch.nn.functional as F

from study.cuda_attention import ragged_gqa_attention


def test_ragged_gqa_attention():
    torch.manual_seed(0)
    q_heads, kv_heads, dim = 14, 2, 64
    q_lens = torch.tensor([1, 4, 7], device="cuda", dtype=torch.long)
    past_lens = torch.tensor([0, 3, 11], device="cuda", dtype=torch.long)
    q_start = torch.cat((torch.zeros(1, device="cuda", dtype=torch.long), q_lens)).cumsum(0)
    kv_lens = past_lens + q_lens
    kv_start = torch.cat((torch.zeros(1, device="cuda", dtype=torch.long), kv_lens)).cumsum(0)
    query = torch.randn(q_heads, int(q_lens.sum()), dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(kv_heads, int(kv_lens.sum()), dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    actual = ragged_gqa_attention(
        query, key, value, q_start, kv_start, past_lens, q_lens,
        int(kv_lens.max()), dim**-0.5)
    expected = torch.empty_like(query)
    groups = q_heads // kv_heads
    for r in range(3):
        qs, qe = q_start[r].item(), q_start[r + 1].item()
        ks, ke = kv_start[r].item(), kv_start[r + 1].item()
        q = query[:, qs:qe].unsqueeze(0)
        k = key[:, ks:ke].unsqueeze(0).repeat_interleave(groups, dim=1)
        v = value[:, ks:ke].unsqueeze(0).repeat_interleave(groups, dim=1)
        qn = qe - qs
        kn = ke - ks
        mask = torch.arange(kn, device="cuda")[None, :] <= (
            past_lens[r] + torch.arange(qn, device="cuda")[:, None])
        expected[:, qs:qe] = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=dim**-0.5).squeeze(0)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
