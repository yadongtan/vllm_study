import torch


# 先不考虑mask。
def test(q: torch.Tensor,
         k_T: torch.Tensor,
         v: torch.Tensor):
    # q.shape = [q_token, head_dim]
    # k.shape = [seq_len, head_dim]
    # v.shape = [seq_len, head_dim]
    o = torch.zeros_like(q)
    head_dim = q.shape[1]
    for q_idx in range(q.shape[0]):
        print("q_idx:", q_idx)
        q_token = q[q_idx].unsqueeze(0)
        accumulate_dtype = q_token.dtype
        max_score = torch.full((), -torch.inf, dtype=accumulate_dtype,device=q_token.device)
        sum_exp_scores = torch.full((),0, dtype=accumulate_dtype,device=q_token.device)
        for k_block_idx in range(k_T.shape[1]):
            k_block = k_T[:, k_block_idx:k_block_idx + 1]
            print("q_token: ", q_token.shape, "k_T: ", k_block.shape)
            # 得到部分Attention
            scores = q_token @ k_block * head_dim ** (-0.5)
            print(f"q_token {q_token} @ k_T: {k_block} = score: {scores}")
            # 对scores进行softmax，先不缩放了，反正逻辑是一样的
            print(f"socres softmax before: {scores}")
            scores,this_max_score,this_sum_exp_scores = softmax(scores, max_score)
            last_max_score = max_score #记录上一次最大值
            max_score = this_max_score #取最大值
            last_sum_exp_scores = sum_exp_scores # 上一次 sum(e**(x_i-max))

            sum_exp_scores = this_sum_exp_scores + last_sum_exp_scores *  torch.exp(last_max_score-max_score)   # 这一次的 sum(e**(x_i-max))
            # 拿到上一次的结果
            last_output = o[q_idx] # [4]
            new_output_1 = last_output * last_sum_exp_scores * torch.exp(last_max_score-max_score)
            print(f"socres softmax after: {scores}")
            # 拿到对应的v
            v_block = v[k_block_idx:k_block_idx + 1, :]
            print("scores.shpe: ", scores.shape, ", v.shape: ", v_block.shape)
            output = scores @ v_block
            print(f"scores {scores.shape} @ v_block {v_block.shape} = output {output.shape}")
            new_output_2 = output * this_sum_exp_scores
            output = (new_output_1 + new_output_2) / sum_exp_scores
            o[q_idx] = output[0] #假设只有一行结果
    print("output: ", o)

def softmax(scores: torch.Tensor,last_max_score: torch.Tensor) -> torch.Tensor:
    # 使用 PyTorch 内置函数
    max_score = torch.maximum(torch.max(scores), last_max_score) # max
    exp_scores = torch.exp(scores - max_score) # ℓ
    # e**(x_i-max) / sum(e**(x_i-max))
    sum_exp_scores = torch.sum(exp_scores, dim=1)
    return exp_scores / sum_exp_scores, max_score, sum_exp_scores


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
                      [9, 10, 11, 12]], dtype=torch.float32) # [1, 3, 4]
    k = torch.tensor([[1,  2,  3,  4],
                      [5,  6,  7,  8],
                      [9,  10, 11, 12],
                      [13, 14, 15, 16],
                      [17, 18, 19, 20]], dtype=torch.float32) # [1,5,4]
    k_T = torch.tensor([[1, 5, 9,  13, 17],
                        [2, 6, 10, 14, 18],
                        [3, 7, 11, 15, 19],
                        [4, 8, 12, 16, 20]], dtype=torch.float32)  # [4,5]
    v = torch.tensor([[1,  2,  3,  4],
                      [5,  6,  7,  8],
                      [9,  10, 11, 12],
                      [13, 14, 15, 16],
                      [17, 18, 19, 20]], dtype=torch.float32) # [1,5,4]
    test(q, k_T, v)


