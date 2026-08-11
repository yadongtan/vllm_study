import torch


def pytorch_flash_attention_v1(
    batch_q: torch.Tensor,
    batch_k: torch.Tensor,
    batch_v: torch.Tensor,
    mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """使用 PyTorch 演示支持 GQA 的分块在线 Softmax Attention。

    维度符号：
        Hq：Query head 数量，例如 Qwen2-0.5B 的 14。
        Hkv：Key/Value head 数量，例如 Qwen2-0.5B 的 2。
        G：每个 KV head 服务的 Query head 数量，G = Hq / Hkv。
        Q：本轮参与计算的 Query token 总数。
        K：包含 KV cache 后的 Key/Value token 总数。
        D：每个 attention head 的维度，例如 64。

    输入形状：
        batch_q: [Hq, Q, D]
        batch_k: [Hkv, K, D]
        batch_v: [Hkv, K, D]
        mask: [Q, K]，True 表示对应的 Query 可以看到该 Key。

    返回形状：
        [Hq, Q, D]

    实现不会把 K/V 真实复制 G 份，而是给 K/V 增加一个长度为 1
    的维度，让 ``torch.matmul`` 通过广播实现 GQA 的 KV 共享。
    """
    print(
        f"batch_q.shape:{batch_q.shape}, "
        f"batch_k.shape:{batch_k.shape}, "
        f"batch_v.shape:{batch_v.shape}"
    )

    # Tensor.shape 返回 torch.Size；这里解包三个维度。
    # batch_q: [Hq, Q, D]，例如 [14, 20, 64]。
    num_q_heads, q_len, head_dim = batch_q.shape

    # batch_k: [Hkv, K, D]。这里只需要取 Hkv 来计算 GQA 分组数。
    num_kv_heads, _, _ = batch_k.shape

    # G = Hq // Hkv。Qwen2 示例中 14 // 2 = 7，即每个 KV head
    # 被连续的 7 个 Query heads 共享。
    group_size = num_q_heads // num_kv_heads

    # Tensor.reshape 只改变张量的逻辑形状（必要时会复制），不改变元素顺序：
    # [Hq, Q, D] -> [Hkv, G, Q, D]
    # [14, 20, 64] -> [2, 7, 20, 64]。
    # 第一维让 Query 和对应的 KV head 对齐；第二维表示共享该 KV
    # head 的一组 Query heads。
    batch_q_grouped = batch_q.reshape(
        num_kv_heads,
        group_size,
        q_len,
        head_dim,
    )

    # 内部函数沿 K 维按 block_size 取块，并执行在线 Softmax。
    # 返回值仍保留分组布局：[Hkv, G, Q, D]。
    output = _pytorch_flash_attention_v1(
        batch_q_grouped,
        batch_k,
        batch_v,
        mask,
        block_size,
    )

    # 将分组布局恢复成调用方需要的标准多头布局：
    # [Hkv, G, Q, D] -> [Hq, Q, D]。
    # 这里不需要 permute，因为 reshape 前也是按每个 KV head 对应的
    # G 个连续 Query heads 进行分组。
    return output.reshape(
        num_q_heads,
        q_len,
        head_dim,
    )


def _pytorch_flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """对 K/V 分块并用在线 Softmax 累加每一块的 Attention 结果。

    输入形状：
        q: [Hkv, G, Q, D]
        k: [Hkv, K, D]
        v: [Hkv, K, D]
        mask: [Q, K]

    输出形状：
        [Hkv, G, Q, D]

    每个 Query head、每个 Query token 都独立保存三份在线状态：
        max_scores：到目前为止见过的最大 attention score，即 m。
        exp_sums：以 m 为基准的指数和，即 l = sum(exp(score - m))。
        output_acc：尚未除以 l 的 Value 加权和。
    """
    # q.shape 是 [Hkv, G, Q, D]，解包后用于创建同布局的累加张量。
    num_kv_heads, group_size, q_len, head_dim = q.shape

    # k.shape 是 [Hkv, K, D]，第 1 维是需要分块遍历的 KV 长度 K。
    kv_len = k.shape[1]

    # 保存调用方的输入 dtype。在线 Softmax 用 FP32 计算完成后，最终输出
    # 会通过 Tensor.to 转回该 dtype，例如 float16 或 bfloat16。
    input_dtype = q.dtype

    # Tensor.float() 将元素转换为 float32，shape 不变：
    # q: [Hkv, G, Q, D]
    # k/v: [Hkv, K, D]
    # FP32 可以减少半精度下 exp、求和和跨块累计的数值误差。
    q = q.float()
    k = k.float()
    v = v.float()

    # torch.zeros 创建全 0 的 FP32 张量，保存未归一化的输出分子。
    # shape: [Hkv, G, Q, D]。
    output_acc = torch.zeros(
        (num_kv_heads, group_size, q_len, head_dim),
        dtype=torch.float32,
        device=q.device,
    )

    # torch.full 创建所有元素都为 -inf 的 FP32 张量。
    # 最后一维保留为 1，便于之后和 scores 的 K-block 维广播。
    # shape: [Hkv, G, Q, 1]。
    max_scores = torch.full(
        (num_kv_heads, group_size, q_len, 1),
        -torch.inf,
        dtype=torch.float32,
        device=q.device,
    )

    # torch.zeros_like 创建与 max_scores 相同 shape、dtype 和 device 的
    # 全 0 张量，保存在线 Softmax 的分母。
    # shape: [Hkv, G, Q, 1]。
    exp_sums = torch.zeros_like(max_scores)

    # Scaled Dot-Product Attention 的缩放系数 1 / sqrt(D)。
    scale = head_dim ** -0.5

    # range 每次沿 K 维前进 block_size，只把一个 K/V block 放入本轮计算。
    for block_start in range(0, kv_len, block_size):
        # min 处理最后一个不足 block_size 的尾块。
        block_end = min(block_start + block_size, kv_len)

        # Tensor 切片沿 mask 的 K 维取当前块：
        # [Q, K] -> [Q, current_block_size]。
        mask_block = mask[:, block_start:block_end]
        # torch.any 对整个 mask block 做逻辑或，返回标量 bool Tensor。
        # 若整块对所有 Query 都不可见，则跳过对应的 K/V 计算。
        if not torch.any(mask_block):
            continue

        # 沿 token 维截取当前 K/V block：
        # [Hkv, K, D] -> [Hkv, current_block_size, D]。
        k_block = k[:, block_start:block_end, :]
        v_block = v[:, block_start:block_end, :]

        # k_block.unsqueeze(1) 在 group 位置插入长度为 1 的维度：
        # [Hkv, B, D] -> [Hkv, 1, B, D]。
        # transpose(-1, -2) 交换最后两个维度：
        # [Hkv, 1, B, D] -> [Hkv, 1, D, B]。
        #
        # torch.matmul 执行最后两维的批量矩阵乘法，并广播 group 维：
        # q:         [Hkv, G, Q, D]
        # k_block_T: [Hkv, 1, D, B]，其中 1 会广播成 G
        # scores:    [Hkv, G, Q, B]
        # 最后乘 scale 完成 score = QK^T / sqrt(D)，shape 不变。
        scores = torch.matmul(
            q,
            k_block.unsqueeze(1).transpose(-1, -2),
        ) * scale

        # 使用 None 索引等价于连续调用两次 unsqueeze(0)：
        # [Q, B] -> [1, 1, Q, B]。
        # 前两个长度为 1 的维度随后广播到 Hkv 和 G。
        mask_block_4d = mask_block[None, None, :, :]

        # ~ 对 bool mask 取反；Tensor.masked_fill 将不可见位置填为 -inf。
        # mask 从 [1, 1, Q, B] 广播到 scores 的 [Hkv, G, Q, B]，
        # scores 的 shape 不变。
        scores = scores.masked_fill(
            ~mask_block_4d,
            -torch.inf,
        )

        # Tensor.amax 沿当前 K block 的最后一维 B 求最大值。
        # keepdim=True 保留长度为 1 的末维，以便后续广播：
        # [Hkv, G, Q, B] -> [Hkv, G, Q, 1]。
        block_max = scores.amax(dim=-1, keepdim=True)

        # torch.maximum 逐元素选择历史最大值和当前块最大值中较大者。
        # 两个输入和输出 shape 都是 [Hkv, G, Q, 1]。
        new_max = torch.maximum(max_scores, block_max)

        # mask_block.any(dim=-1) 判断每个 Query 在当前 block 是否至少
        # 有一个有效 Key：[Q, B] -> [Q]。
        # 再通过 None 增加广播维：[Q] -> [1, 1, Q, 1]。
        active_queries = mask_block.any(dim=-1)[
            None,
            None,
            :,
            None,
        ]

        # torch.exp(max_scores - new_max) 把历史累计量换算到新的最大值
        # 基准下，shape 为 [Hkv, G, Q, 1]。
        # torch.where(condition, x, y) 逐元素选择：
        # - 当前块有有效 Key：使用 exp(old_max - new_max) 缩放历史状态；
        # - 当前块对该 Query 全 mask：使用 1，保持历史状态不变。
        # torch.ones_like 创建与 max_scores 相同的全 1 张量。
        # history_scale.shape = [Hkv, G, Q, 1]
        history_scale = torch.where(
            active_queries,
            torch.exp(max_scores - new_max),
            torch.ones_like(max_scores),
        )

        # 计算当前块的未归一化 Softmax 权重 exp(score - new_max)。
        # torch.where 把 mask 位置直接设为 0，避免某一行全被 mask 时
        # 出现 exp(-inf - -inf) = NaN。
        # condition [1, 1, Q, B] 会广播，输出 shape 为 [Hkv, G, Q, B]。
        block_exp = torch.where(
            mask_block_4d,
            torch.exp(scores - new_max),
            torch.zeros_like(scores),
        )

        # v_block.unsqueeze(1)：[Hkv, B, D] -> [Hkv, 1, B, D]。
        # torch.matmul 再次利用 group 维广播执行加权求和：
        # block_exp: [Hkv, G, Q, B]
        # v_block:   [Hkv, 1, B, D]
        # 输出:       [Hkv, G, Q, D]
        block_output = torch.matmul(
            block_exp,
            v_block.unsqueeze(1),
        )

        # 先用 history_scale 将历史输出分子换算到 new_max 基准，再加上
        # 当前块的分子。history_scale 的末维 1 会广播到 D。
        # 运算前后 shape 均为 [Hkv, G, Q, D]。
        output_acc = (
            output_acc * history_scale
            + block_output
        )

        # Tensor.sum 沿当前 block 的 B 维对指数权重求和：
        # [Hkv, G, Q, B] -> [Hkv, G, Q, 1]。
        # 历史分母同样先乘 history_scale，再加入当前块指数和。
        exp_sums = (
            exp_sums * history_scale
            + block_exp.sum(
                dim=-1,
                keepdim=True,
            )
        )

        # 保存本轮全局最大值，供下一个 K/V block 合并使用。
        # shape: [Hkv, G, Q, 1]。
        max_scores = new_max

    # torch.finfo(float32).tiny 是 float32 最小正规格化正数。
    # Tensor.clamp_min 将分母限制为至少 tiny，防止直接除以 0；
    # shape 仍为 [Hkv, G, Q, 1]。
    denominator = exp_sums.clamp_min(
        torch.finfo(torch.float32).tiny
    )

    # torch.where 对有有效 Key 的 Query 执行最终归一化：
    # output_acc [Hkv, G, Q, D] / denominator [Hkv, G, Q, 1]
    # denominator 的最后一维广播到 D，结果为 [Hkv, G, Q, D]。
    # 对整行 mask 全为 False 的 Query，输出显式保持为全 0。
    output = torch.where(
        exp_sums > 0,
        output_acc / denominator,
        torch.zeros_like(output_acc),
    )

    # Tensor.to 将 FP32 累加结果转回调用方原始 dtype；shape 不变：
    # [Hkv, G, Q, D]。
    return output.to(input_dtype)






# q.shape = [q_token, head_dim]
# k.shape = [seq_len, head_dim]
# v.shape = [seq_len, head_dim]
if __name__ == "__main__":
    # 请求1 本次 2个q，2个历史k 2 + 2 + 1 = 5
    # 请求2 本次 1个q，0个历史k
    mask = torch.tensor([
        [True, True,True,False,False],
        [True, True, True, True, False],
        [False, False, False, False, True],
    ])

    q = torch.tensor([[1, 2,  3,  4],
                      [5, 6,  7,  8],
                      [9, 10, 11, 12]], dtype=torch.float32).unsqueeze(0) # [1, 3, 4]
    q = q.repeat_interleave(14, dim=0)
    k = torch.tensor([[1,  2,  3,  4],
                      [5,  6,  7,  8],
                      [9,  10, 11, 12],
                      [13, 14, 15, 16],
                      [17, 18, 19, 20]], dtype=torch.float32).unsqueeze(0) # [1,5,4]
    k = k.repeat_interleave(2, dim=0)
    k_T = torch.tensor([[1, 5, 9,  13, 17],
                        [2, 6, 10, 14, 18],
                        [3, 7, 11, 15, 19],
                        [4, 8, 12, 16, 20]], dtype=torch.float32)  # [4,5]
    v = torch.tensor([[1,  2,  3,  4],
                      [5,  6,  7,  8],
                      [9,  10, 11, 12],
                      [13, 14, 15, 16],
                      [17, 18, 19, 20]], dtype=torch.float32).unsqueeze(0) # [1,5,4]
    v = v.repeat_interleave(2, dim=0)
    block_size = 2
    pytorch_flash_attention_v1(q, k, v, mask, 2)
