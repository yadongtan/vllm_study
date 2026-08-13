# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""最小化的 vLLM 风格 Qwen2 模型加载示例。

这个文件不会调用 ``AutoModelForCausalLM.from_pretrained()``。
它会完成以下工作：

1. 使用普通 PyTorch 模块自行搭建 Qwen2 网络结构。
2. 像 vLLM 一样，把 Q/K/V 和 Gate/Up 定义成融合参数。
3. 逐个读取 Hugging Face Safetensors 中的张量。
4. 把分开的 checkpoint 权重写入融合参数的正确切片。
5. 使用一个最小化的自回归循环生成回答。

示例只关注“创建模型”和“加载权重”。
真正的 vLLM 还实现了张量并行、流水线并行、优化算子、
Paged KV Cache、
连续批处理和独立采样器等功能。
"""

# Iterator 用来标注上下文管理器 yield 的类型。
from collections.abc import Iterator
# contextmanager 可以把带有 yield 的函数转换为上下文管理器。
from contextlib import contextmanager
# Enum 用来为不同类型的 KV Cache 提供明确的键。
from enum import Enum
# Path 用面向对象的方式拼接和检查文件路径。
from pathlib import Path
import os
import queue
import threading
import traceback
from study.inference_engine.pytorch_attention.pytorch_flash_attention_v1 import (
    pytorch_flash_attention_v1,
)

# torch 提供张量、设备、数据类型及推理模式等基础能力。
import torch
# F 包含无状态算子；这里使用 SiLU、线性投影和 SDPA Attention。
import torch.nn.functional as F
from prometheus_client.decorator import append
try:
    from study.inference_engine.cuda_attention import (
        cuda_flash_attention_v1,
        cuda_flash_attention_v2,
        ragged_gqa_attention,
    )
except ModuleNotFoundError:
    # Support direct execution via ``python study/inference_engine/qwen2_demo.py``.
    from cuda_attention import (
        cuda_flash_attention_v1,
        cuda_flash_attention_v2,
        ragged_gqa_attention,
    )
# safe_open 可以按名称逐个读取 Safetensors 张量。
# 这样能够避免一次复制所有权重。
from safetensors import safe_open
# nn 提供 Module、Linear、Embedding、ModuleList 等神经网络组件。
from torch import nn
# Transformers 这里只负责配置和分词，不负责创建或加载模型。
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager

# __file__ 是当前脚本；parents[1] 是仓库根目录 vllm/。
# 后面的 / 运算符依次拼出本地 Hugging Face 模型目录。
MODEL_PATH = (
    Path(os.environ.get("STUDY_MODEL_PATH", "/opt/models/Qwen2-0.5B-Instruct"))
)


def get_rope_theta(config: PretrainedConfig) -> float:
    """Read RoPE theta across Transformers configuration versions."""
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return float(rope_theta)

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is not None:
            return float(rope_theta)

    config_dict = config.to_dict()
    rope_theta = config_dict.get("rope_theta")
    if rope_theta is not None:
        return float(rope_theta)

    raise AttributeError("Qwen2 configuration does not define rope_theta")

class Request:
    def __init__(
        self,
        input_ids,
        req_id,
        max_new_tokens: int = 128,
        ignore_eos: bool = False,
    ):
        self.input_ids = input_ids
        self.start = 0
        self.position = 0
        self.req_id = req_id
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = ignore_eos
        self.decode_tokens: list[torch.Tensor] = []
        self.prefill_seq_len = input_ids.shape[0]
        self.already_read_token_idx = 0
        self.finished = False
        self.error: BaseException | None = None
        self.output_condition = threading.Condition()

    def fail(self, error: BaseException) -> None:
        """Wake the client and propagate an exception from the scheduler."""
        with self.output_condition:
            self.error = error
            self.finished = True
            self.output_condition.notify_all()

    def append_decode_token(
        self,
        decode_token: torch.Tensor,
        finished: bool = False,
    ) -> None:
        with self.output_condition:
            self.decode_tokens.append(decode_token)
            self.start = self.position
            self.position += 1
            self.finished = finished
            self.output_condition.notify_all()

    def wait_for_new_tokens(self) -> tuple[list[int], bool]:
        """等待网络线程尚未读取的 token，并推进读取索引。"""
        with self.output_condition:
            while (
                self.already_read_token_idx == len(self.decode_tokens)
                and not self.finished
            ):
                self.output_condition.wait()

            if self.error is not None:
                raise RuntimeError(
                    "Scheduler failed during inference"
                ) from self.error

            start = self.already_read_token_idx
            end = len(self.decode_tokens)
            token_ids = [
                int(token.item())
                for token in self.decode_tokens[start:end]
            ]
            # 只由网络线程推进已经读取到的位置。
            self.already_read_token_idx = end
            return token_ids, self.finished

    def get_and_set_next_process(self, max_chunk_prefill_len):
        # decode请求
        if self.prefill_seq_len == self.position and len(self.decode_tokens) == 0:
            print("self.start: ", self.start, "self.position: ", self.position)
            return 1
        elif len(self.decode_tokens) > 0:
            print("self.start: ", self.start, "self.position: ", self.position)
            return 1
        # 剩余prefill长度大于分块长度
        elif self.prefill_seq_len - self.position > max_chunk_prefill_len:
            self.start = self.position
            self.position = self.position + max_chunk_prefill_len
            print("self.start: ", self.start, "self.position: ", self.position)
            return max_chunk_prefill_len
        # 剩余prefill长度小于分块长度
        else:
            self.start = self.position
            self.position =  self.prefill_seq_len
            print("self.start: ", self.start, "self.position: ", self.position)
            return self.position - self.start


    def get_next_process_len(self, max_chunk_prefill_len):
        # decode请求
        if self.prefill_seq_len < self.position:
            return 1
        # 剩余prefill长度大于分块长度
        elif self.prefill_seq_len - self.position > max_chunk_prefill_len:
            return max_chunk_prefill_len
        # 剩余prefill长度小于分块长度
        else:
            return self.prefill_seq_len - self.position

    def update_prefill_len(self):
        self.start = self.position

    # 返回当前prefill/decode关注的token
    def get_current_pd_token(self):
        if self.position <= self.prefill_seq_len:
            return self.input_ids[self.start:self.position]
        # decode，直接返回固定的
        else:
            return self.decode_tokens[self.position - self.prefill_seq_len - 1]

    def get_current_pd_token_pos(self):
        return torch.arange(self.start, self.position, dtype=torch.int64)



def get_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """选择可用的计算设备及适合该设备的模型参数类型。"""

    # 优先选择 NVIDIA/AMD CUDA-like GPU。
    # 这类设备通常具有最高推理吞吐量。
    if torch.cuda.is_available():
        # 支持 BF16 时优先使用 BF16；否则退回更通用的 FP16。
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # torch.device 描述参数和输入张量应放置的位置。
        return torch.device("cuda"), dtype

    # macOS 上的 Apple Silicon GPU 通过 MPS 后端使用。
    if torch.backends.mps.is_available():
        # 此模型在当前 MPS 环境使用 FP16，可以把权重内存减半。
        return torch.device("mps"), torch.float16

    # 没有加速设备时使用 CPU；FP32 的 CPU 算子兼容性最好。
    return torch.device("cpu"), torch.float32


# 这个装饰器把函数转换成可用于 with 的上下文管理器。
@contextmanager
def set_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """临时设置浮点参数的默认类型，并在离开 with 后恢复现场。"""

    # 保存原始默认类型，避免脚本修改全局 PyTorch 状态后不恢复。
    previous_dtype = torch.get_default_dtype()
    # 后续 Linear/Embedding 创建的浮点参数将直接使用目标类型。
    torch.set_default_dtype(dtype)
    try:
        # 暂停本函数，把控制权交给 with 代码块创建模型。
        yield
    finally:
        # 无论 with 中是否发生异常，都恢复原始默认类型。
        torch.set_default_dtype(previous_dtype)


class Qwen2RMSNorm(nn.Module):
    """Qwen2 使用的 Root Mean Square Layer Normalization。"""

    def __init__(self, hidden_size: int, eps: float) -> None:
        # 初始化 nn.Module 内部的参数和子模块管理机制。
        super().__init__()
        # weight 是形状为 [hidden_size] 的可训练缩放参数。
        # 它的初始值全为 1。
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # eps 防止均方值太小时除零；Qwen2 配置中通常是 1e-6。
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """把形状 [batch, sequence, hidden_size] 的隐藏状态归一化。"""

        # 保存输入类型，以便用 FP32 计算归一化后再转换回来。
        input_dtype = hidden_states.dtype
        # 在最后一维求平方均值。
        # keepdim=True 保留广播所需的长度为 1 的维度。
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        # RMSNorm 公式：normalized = x / sqrt(mean(x^2) + eps)。
        # 分母 sqrt(mean(x^2) + eps) 表示输入的“均方根”大小。
        # eps 用于避免分母为 0，提高数值稳定性。
        # torch.rsqrt(a) 等于 1/sqrt(a)。
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        # 转回原类型，并乘 checkpoint 中学习到的逐通道缩放权重。
        return self.weight * normalized.to(input_dtype)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """完成 RoPE 公式所需的 [-x2, x1] 半维旋转。"""

    # 沿 head_dim 把 [..., 64] 平分为两个 [..., 32] 张量。
    first, second = hidden_states.chunk(2, dim=-1)
    # 拼成 [-second, first]；它之后会与 sin 相乘。
    return torch.cat((-second, first), dim=-1)


def build_rope(
    positions: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """为当前位置构造 Rotary Position Embedding 的 cos 和 sin。"""

    # 只为偶数维构造频率。
    # Qwen2-0.5B 的 head_dim=64，所以这里有 32 项。
    dimensions = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    # 不同通道采用不同频率。
    # rope_theta 来自 config.json，本模型是 1,000,000。
    # dimensions = [0, 2, 4, 6, ..., 62]
    temp = dimensions / head_dim
    temp2 = rope_theta ** temp
    inv_freq = 1.0 / temp2
    # 外积得到 [sequence_length, head_dim/2] 的“位置 x 频率”矩阵。
    frequencies = torch.outer(positions, inv_freq)
    # 复制一次频率，使最后一维恢复为完整 head_dim。
    # 这样可以匹配 rotate_half 的布局。
    embeddings = torch.cat((frequencies, frequencies), dim=-1)
    # 增加 batch 和 heads 两个广播维。
    # 最终形状为 [1, 1, sequence, head_dim]。
    shape = (1, 1, positions.numel(), head_dim)
    # cos/sin 先以 FP32 计算，再转成模型类型。
    # view 只改变形状，不复制数据。
    return embeddings.cos().to(dtype).view(shape), embeddings.sin().to(dtype).view(
        shape
    )


class KVCacheType(Enum):
    K = 1
    V = 2

# request_id -> cache type -> layer index -> cached tensor。
KVCacheStore = dict[int, dict[KVCacheType, dict[int, torch.Tensor]]]

class Qwen2Attention(nn.Module):
    """带 Grouped Query Attention 和融合 QKV 参数的自注意力层。"""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        # Query 头数量；Qwen2-0.5B 为 14。
        self.num_heads = config.num_attention_heads
        # Key/Value 头数量；本模型为 2，这就是 GQA 而不是普通 MHA。
        self.num_kv_heads = config.num_key_value_heads
        # 每个头的维度：hidden_size 896 / 14 = 64。
        self.head_dim = config.hidden_size // self.num_heads
        # 所有 Query 头拼接后的宽度：14 * 64 = 896。
        self.q_size = self.num_heads * self.head_dim
        # 所有 Key 或 Value 头拼接后的宽度：2 * 64 = 128。
        self.kv_size = self.num_kv_heads * self.head_dim

        # Hugging Face 分别定义 q_proj、k_proj、v_proj。
        # vLLM 为减少算子启动和通信开销，把三者合成一个 qkv_proj。
        self.qkv_proj = nn.Linear(
            # 输入宽度是 hidden_size=896。
            config.hidden_size,
            # 输出宽度是 Q 896 + K 128 + V 128 = 1152。
            self.q_size + 2 * self.kv_size,
            # Qwen2 checkpoint 的 Q/K/V 都包含 bias。
            # 因此融合层也需要 bias。
            bias=True,
        )

        # o_proj 把多头注意力输出从 896 维投影回 hidden_size=896。
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)

    def pad_sequences_with_offset(self, tensors):
        """
        将多个不同长度的张量填充到相同长度，并在每行中保持相对偏移位置

        Args:
            tensors: list of torch.Tensor

        Returns:
            填充后的二维张量
        """
        if not tensors:
            return torch.tensor([])

        # 计算总长度（所有张量长度之和）
        total_length = sum(len(t) for t in tensors)

        # 创建结果张量
        result = torch.zeros(len(tensors), total_length)

        # 记录当前偏移位置
        current_pos = 0

        for i, tensor in enumerate(tensors):
            # 将当前张量放置在对应位置
            result[i, current_pos:current_pos + len(tensor)] = tensor
            current_pos += len(tensor)

        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ids: list[Request],
        query_start_loc: list[int],
        kv_cache_manager: KVCacheStore,
        layer_index: int,
    ) -> torch.Tensor:
        """计算一层因果自注意力，输入和输出均为 [B, S, 896]。"""

        # 从输入形状中取出批大小 B 和序列长度 S。
        # 最后一维无需单独保存。
        sequence_length, _ = hidden_states.shape
        # 一次矩阵乘法产生 [S, 1152] 的融合 QKV。
        qkv = self.qkv_proj(hidden_states)
        # 按 Q/K/V 的实际宽度把融合结果拆成三个逻辑张量。
        query, key, value = qkv.split(
            (self.q_size, self.kv_size, self.kv_size),
            dim=-1,
        )

        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        # Query 从 [B,S,896] 变成 [B,S,14,64]，再变成 [B,14,S,64]。
        query = query.view(
            sequence_length, self.num_heads, self.head_dim
        ).transpose(0, 1)
        # Key 从 [B,S,128] 变成 [B,S,2,64]，再变成 [B,2,S,64]。
        key = key.view(
            sequence_length, self.num_kv_heads, self.head_dim
        ).transpose(0, 1)
        # Value 使用与 Key 相同的 GQA 头数量和形状变换。
        value = value.view(
            sequence_length, self.num_kv_heads, self.head_dim
        ).transpose(0, 1)
        # 对 Query 应用 q*cos + rotate_half(q)*sin。
        # 这个操作会把位置信息编码进向量。

        query = query * cos + rotate_half(query) * sin
        # Key 必须使用相同的旋转位置编码；Value 不使用 RoPE。
        key = key * cos + rotate_half(key) * sin

        # Keep compact GQA K/V; the CUDA operator maps Q heads to KV heads.
        # PyTorch SDPA 执行缩放点积、因果遮罩、Softmax，
        # 然后把注意力概率与 Value 相乘。
        # if batch_size != 1 or len(ids) != 1 or len(query_start_loc) != 1:
        #     raise NotImplementedError(
        #         "This minimal KV cache demo supports batch_size=1"
        #     )
        full_key_chunks: list[torch.Tensor] = []
        full_value_chunks: list[torch.Tensor] = []
        past_lens: list[int] = []
        q_lens: list[int] = []

        # 先将当前的输入qkv拆分开，待会再跟缓存合并为一个超大矩阵
        k_split: list[torch.Tensor] = []
        v_split: list[torch.Tensor] = []

        # 拆分k_v
        for request_index in range(len(ids)):
            query_left = query_start_loc[request_index]
            query_right = query_start_loc[request_index + 1]
            k_split.append(key[:, query_left:query_right, :])
            v_split.append(value[:, query_left:query_right, :])
            q_lens.append(query_right - query_left)

        for request_index in range(len(ids)):
            request_cache = kv_cache_manager[ids[request_index].req_id]
            start_position = ids[request_index].start
            past_lens.append(start_position)
            # 非首次prefill则一定有缓存
            # print("layer[", layer_index, "], use cache before, key.shape: ", key.shape, ", value.shape: ", value.shape)
            if start_position > 0:
                k_cache = request_cache[KVCacheType.K][layer_index]
                v_cache = request_cache[KVCacheType.V][layer_index]
                cache_length = k_cache.shape[1]
                assert cache_length == start_position
                # key = torch.cat((k_cache, key), dim=1)
                # value = torch.cat((v_cache, value), dim=1)
                all_k = torch.cat((k_cache, k_split[request_index]), dim=1)
                all_v = torch.cat((v_cache, v_split[request_index]), dim=1)
                full_key_chunks.append(all_k)
                full_value_chunks.append(all_v)
                print("layer[", layer_index,"], use cache after, key.shape: ", key.shape, ", value.shape: ", value.shape)
                # 重新定义mask矩阵
                request_cache[KVCacheType.K][layer_index] = all_k
                request_cache[KVCacheType.V][layer_index] = all_v
            else:
                all_k = k_split[request_index]
                all_v = v_split[request_index]
                full_key_chunks.append(all_k)
                full_value_chunks.append(all_v)
                request_cache[KVCacheType.K][layer_index] = all_k
                request_cache[KVCacheType.V][layer_index] = all_v

        key = torch.cat(full_key_chunks, dim=1).contiguous()
        value = torch.cat(full_value_chunks, dim=1).contiguous()
        query = query.contiguous()

        # Ragged metadata replaces the dense cross-request causal mask.
        query_start = torch.tensor(
            query_start_loc, device=hidden_states.device, dtype=torch.long
        )
        q_lens_tensor = torch.tensor(
            q_lens, device=hidden_states.device, dtype=torch.long
        )
        past_lens_tensor = torch.tensor(
            past_lens, device=hidden_states.device, dtype=torch.long
        )
        kv_lens_tensor = past_lens_tensor + q_lens_tensor
        kv_start = torch.cat(
            (
                torch.zeros(1, device=hidden_states.device, dtype=torch.long),
                kv_lens_tensor,
            )
        ).cumsum(0)
        scale = self.head_dim**-0.5

        use_cuda_flash_attention_v1 = (
            os.environ.get("STUDY_USE_CUDA_FLASH_ATTENTION_V1", "0") == "1"
        )
        use_cuda_flash_attention_v2 = (
            os.environ.get("STUDY_USE_CUDA_FLASH_ATTENTION_V2", "0") == "1"
        )

        compact_mask = None
        if use_cuda_flash_attention_v1 or use_cuda_flash_attention_v2:
            compact_mask = torch.zeros(
                query.shape[1],
                query.shape[1],
                dtype=torch.bool,
                device=hidden_states.device,
            )
            for request_index in range(len(ids)):
                query_left = query_start_loc[request_index]
                query_right = query_start_loc[request_index + 1]
                query_len = query_right - query_left
                compact_mask[
                    query_left:query_right,
                    query_left:query_right,
                ] = torch.tril(
                    torch.ones(
                        query_len,
                        query_len,
                        dtype=torch.bool,
                        device=hidden_states.device,
                    )
                )

        # v2 直接接收打包后的三维 [H, S, D] Q/K/V。
        if use_cuda_flash_attention_v2:
            assert compact_mask is not None
            attention = cuda_flash_attention_v2(
                query,
                key,
                value,
                compact_mask.contiguous(),
                query_start.contiguous(),
                kv_start.contiguous(),
                32,
                32,
            )
        # v1 接收四维 [B, H, S, D]，这里临时增加大小为 1 的 batch 维。
        elif use_cuda_flash_attention_v1:
            assert compact_mask is not None
            attention = cuda_flash_attention_v1(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                compact_mask.contiguous(),
                query_start.contiguous(),
                kv_start.contiguous(),
                32,
                32,
            ).squeeze(0)
        # 使用自己python实现的flash attention
        elif os.environ.get("STUDY_USE_PYTORCH_FLASH_ATTENTION_V1", "0") == "1":
            attention_mask = torch.zeros(
                query.shape[1], key.shape[1], dtype=torch.bool,
                device=hidden_states.device,
            )
            q_offset = 0
            k_offset = 0
            for request_index in range(len(ids)):
                query_len = q_lens[request_index]
                kv_len = int(kv_lens_tensor[request_index].item())
                query_positions = torch.arange(
                    query_len, device=hidden_states.device
                )
                key_positions = torch.arange(
                    kv_len, device=hidden_states.device
                )
                local_mask = key_positions[None, :] <= (
                    past_lens[request_index] + query_positions[:, None]
                )
                attention_mask[
                    q_offset:q_offset + query_len,
                    k_offset:k_offset + kv_len,
                ] = local_mask
                q_offset += query_len
                k_offset += kv_len

            attention = pytorch_flash_attention_v1(
                query,
                key,
                value,
                attention_mask,
                128
            )
        # pytorch提供的attention
        elif os.environ.get("STUDY_USE_CUDA_ATTENTION", "0") == "1":
            # cuda实现的attention
            attention = ragged_gqa_attention(
                query, key, value, query_start, kv_start,
                past_lens_tensor, q_lens_tensor,
                int(kv_lens_tensor.max().item()), scale,
            )
        else:
            dense_key = key.repeat_interleave(
                self.num_heads // self.num_kv_heads, dim=0
            )
            dense_value = value.repeat_interleave(
                self.num_heads // self.num_kv_heads, dim=0
            )
            attention_mask = torch.zeros(
                query.shape[1], key.shape[1], dtype=torch.bool,
                device=hidden_states.device,
            )
            q_offset = 0
            k_offset = 0
            for request_index in range(len(ids)):
                query_len = q_lens[request_index]
                kv_len = int(kv_lens_tensor[request_index].item())
                query_positions = torch.arange(
                    query_len, device=hidden_states.device
                )
                key_positions = torch.arange(
                    kv_len, device=hidden_states.device
                )
                local_mask = key_positions[None, :] <= (
                        past_lens[request_index] + query_positions[:, None]
                )
                attention_mask[
                    q_offset:q_offset + query_len,
                    k_offset:k_offset + kv_len,
                ] = local_mask
                q_offset += query_len
                k_offset += kv_len
            attention = F.scaled_dot_product_attention(
                query,
                dense_key,
                dense_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )
        if os.environ.get("STUDY_DEBUG_FINITE") == "1" and not torch.isfinite(
            attention
        ).all():
            raise FloatingPointError(
                f"non-finite attention output at layer {layer_index}"
            )

        return self.o_proj(
            attention.transpose(0, 1).reshape(sequence_length, self.q_size)
        )


class Qwen2MLP(nn.Module):
    """使用融合 Gate/Up 投影的 SwiGLU 前馈网络。"""

    def __init__(self, config: PretrainedConfig) -> None:
        # 初始化 PyTorch 模块。
        super().__init__()
        # Hugging Face 把 gate_proj 和 up_proj 定义成两个 Linear。
        # vLLM 将它们融合成一个输出宽度加倍的 gate_up_proj。
        self.gate_up_proj = nn.Linear(
            # 输入是每个 token 的 hidden_size=896 隐藏向量。
            config.hidden_size,
            # 输出含两个 intermediate_size=4864 分区，总宽度为 9728。
            2 * config.intermediate_size,
            # Qwen2 的 MLP 投影没有 bias。
            bias=False,
        )
        # down_proj 把中间维度 4864 投影回 hidden_size=896。
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行 SwiGLU：down_proj(silu(gate) * up)。"""

        # 一次融合投影后，沿最后一维平分出 gate 和 up 两部分。
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        # gate 经过 SiLU 激活后逐元素控制 up，再由 down_proj 降维。
        return self.down_proj(F.silu(gate) * up)


class Qwen2DecoderLayer(nn.Module):
    """一个完整 Qwen2 Decoder 层：Attention + MLP + 两次残差连接。"""

    def __init__(self, config: PretrainedConfig, layer_index: int) -> None:
        # 初始化模块注册机制。
        super().__init__()
        # 创建本层的 GQA 自注意力子模块。
        self.self_attn = Qwen2Attention(config)
        # 创建本层的 SwiGLU 前馈网络。
        self.mlp = Qwen2MLP(config)
        # Attention 之前的 Pre-Norm。
        self.input_layernorm = Qwen2RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        # MLP 之前的第二个 Pre-Norm。
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.layer_index = layer_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ids: list[Request],
        query_start_loc: list[int],
        kv_cache_manager: KVCacheStore,
    ) -> torch.Tensor:
        """依次执行归一化、Attention、残差、归一化、MLP、残差。"""
        # 保存 Attention 分支的输入，用于第一次残差连接。
        residual = hidden_states
        # Qwen2 使用 Pre-Norm：先归一化，再进入 Attention。
        hidden_states = self.input_layernorm(hidden_states)
        # Attention 输出与未归一化的原输入相加。
        hidden_states = residual + self.self_attn(
            hidden_states,
            cos,
            sin,
            ids,
            query_start_loc,
            kv_cache_manager,
            self.layer_index,
        )

        # 保存 Attention 残差结果，作为 MLP 分支的残差。
        residual = hidden_states
        # MLP 前再次执行 RMSNorm。
        hidden_states = self.post_attention_layernorm(hidden_states)
        # MLP 输出与第二份 residual 相加。
        # 相加结果就是本 Decoder 层的最终输出。
        return residual + self.mlp(hidden_states)


class Qwen2Model(nn.Module):
    """Qwen2 主干：Token Embedding、24 个 Decoder 层和最终 RMSNorm。"""

    def __init__(self, config: PretrainedConfig) -> None:
        # 初始化父类。
        super().__init__()
        # 保存配置，forward 构造 RoPE 时还要读取头数和 rope_theta。
        self.config = config
        # 把 token id 映射成 896 维向量；权重形状为 [151936,896]。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # ModuleList 会正确注册所有 Decoder 层的参数。
        self.layers = nn.ModuleList(
            # 根据 config.json 创建 num_hidden_layers=24 个独立层。
            Qwen2DecoderLayer(config, layer_index)
            for layer_index in range(config.num_hidden_layers)
        )
        # 所有 Decoder 层之后还有一次最终 RMSNorm。
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        ids: list[Request],
        query_start_loc: list[int],
        kv_cache_manager: KVCacheStore,
    ) -> torch.Tensor:
        """把 [B,S] token id 转换为 [B,S,896] 上下文化隐藏状态。"""

        # Embedding 查表把整数 token id 变成浮点隐藏向量。
        hidden_states = self.embed_tokens(input_ids)
        # 同一层内所有注意力头共享当前序列的 RoPE cos/sin。
        # 构建同纬度的cos, sin =>   [batch, ...]
        # [0, 22, 44]
        #
        # [0,1,2,3,4,5,...,21,0,1,2,3,4,...,21]
        position_chunks: list[torch.Tensor] = []
        for req in ids:
            position_chunks.append(req.get_current_pd_token_pos().to(device=input_ids.device,dtype=torch.float32))
        rope_pos_index = torch.cat(position_chunks, dim=0)
        cos, sin = build_rope(
            rope_pos_index,
            # 本模型 head_dim = hidden_size 896 / num_heads 14 = 64。
            self.config.hidden_size // self.config.num_attention_heads,
            # RoPE 基数直接读取 config.json。
            get_rope_theta(self.config),
            # 在输入所在设备直接创建位置张量，避免跨设备复制。
            input_ids.device,
            # cos/sin 使用与隐藏状态一致的 FP16/BF16/FP32 类型。
            hidden_states.dtype,
        )
        # 隐藏状态依次通过 24 层；每层使用相同位置对应的 cos/sin。
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos,
                sin,
                ids,
                query_start_loc,
                kv_cache_manager,
            )
        # 返回最终归一化结果；此时还没有计算词表 logits。
        return self.norm(hidden_states)


class Qwen2ForCausalLM(nn.Module):
    """在 Qwen2 主干后增加词表投影，得到下一个 token 的 logits。"""

    def __init__(self, config: PretrainedConfig) -> None:
        # 初始化父类。
        super().__init__()
        # 本 demo 通过复用 Embedding 权重实现 lm_head。
        # 因此这里只支持权重绑定模型。
        if not config.tie_word_embeddings:
            raise NotImplementedError("This minimal demo requires tied embeddings")
        # 创建完整 Qwen2 主干。
        # 所有参数目前仍是未加载的初始化值。
        self.model = Qwen2Model(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        ids: list[Request],
        query_start_loc: list[int],
        kv_cache_manager: KVCacheStore,
    ) -> torch.Tensor:
        """返回最后一个输入位置对整个词表的未归一化分数。"""

        # 主干计算序列中所有 token 的上下文化隐藏状态。
        hidden_states = self.model(
            input_ids,
            ids,
            query_start_loc,
            kv_cache_manager,
        )
        # 自回归生成只需要最后位置；形状由 [B,S,896] 变为 [B,896]。
        last_indices = torch.tensor(
            [
                query_start_loc[i + 1] - 1
                for i in range(len(ids))
            ],
            dtype=torch.long,
            device=hidden_states.device,
        )

        last_hidden_state = hidden_states.index_select(
            0,
            last_indices,
        )


        # 使用 Embedding 矩阵作为 lm_head，输出 [B,vocab_size] logits。
        return F.linear(last_hidden_state, self.model.embed_tokens.weight)


class VllmStyleWeightLoader:
    """逐张量把 Hugging Face checkpoint 加载到融合模型参数中。"""

    # 键是 checkpoint 参数名片段；值是目标片段以及融合分片标识。
    # 例如 q_proj.weight 会映射到 qkv_proj.weight 的 Q 区域。
    QKV_PARTS = {
        ".q_proj.": (".qkv_proj.", "q"),
        ".k_proj.": (".qkv_proj.", "k"),
        ".v_proj.": (".qkv_proj.", "v"),
    }
    # gate_proj 和 up_proj 同理写入 gate_up_proj 的前后两个区域。
    GATE_UP_PARTS = {
        ".gate_proj.": (".gate_up_proj.", "gate"),
        ".up_proj.": (".gate_up_proj.", "up"),
    }

    def __init__(self, model: nn.Module, config: PretrainedConfig) -> None:
        # named_parameters 产生“完整参数名 -> Parameter”映射。
        # 这个映射方便按 checkpoint 名查找目标参数。
        self.params = dict(model.named_parameters())
        # Q 融合区宽度为 14 * 64 = 896。
        self.q_size = config.num_attention_heads * (
            config.hidden_size // config.num_attention_heads
        )
        # K 或 V 融合区宽度为 2 * 64 = 128。
        self.kv_size = config.num_key_value_heads * (
            config.hidden_size // config.num_attention_heads
        )
        # Gate 和 Up 各占 gate_up_proj 的 4864 行。
        self.intermediate_size = config.intermediate_size
        # 记录至少被写入过一次的目标参数名，用于最终检查漏载。
        self.loaded_params: set[str] = set()
        # 记录每个融合参数已经加载了哪些分片。
        # 这样可以发现只加载 Q 却漏掉 K/V 的错误。
        self.loaded_shards: dict[str, set[str]] = {}
        # 统计从 checkpoint 读取的源张量数量。
        self.source_tensor_count = 0

    @staticmethod
    def copy_tensor(target: torch.Tensor, source: torch.Tensor) -> None:
        """检查形状并把一个 CPU checkpoint 张量复制到目标参数。"""

        # 形状不一致通常表示模型结构或映射规则有错误。
        # 此时应立即失败。
        if target.shape != source.shape:
            raise ValueError(
                f"Shape mismatch: target={tuple(target.shape)}, "
                f"source={tuple(source.shape)}"
            )
        # 转换到目标设备和类型后执行原地 copy_。
        # 这不会替换已经注册的 Parameter 对象。
        target.copy_(source.to(device=target.device, dtype=target.dtype))

    def load_packed_tensor(
        self,
        source_name: str,
        source_tensor: torch.Tensor,
        source_fragment: str,
        target_fragment: str,
        shard_id: str,
    ) -> None:
        """把一个 HF 投影写入 vLLM 风格融合参数的指定切片。"""

        # 只替换投影名，保留 model.layers.N 等前缀和 weight/bias 后缀。
        target_name = source_name.replace(source_fragment, target_fragment)
        # 使用转换后的完整名字找到模型中已经注册的融合参数。
        target_param = self.params[target_name]

        # Q 位于 qkv_proj 输出维的起始位置，偏移为 0。
        if shard_id == "q":
            offset = 0
        # K 紧跟在 Q 后面，所以起点是 q_size=896。
        elif shard_id == "k":
            offset = self.q_size
        # V 位于 Q 和 K 之后，所以起点是 896+128=1024。
        elif shard_id == "v":
            offset = self.q_size + self.kv_size
        # Gate 是 gate_up_proj 的前半部分，偏移为 0。
        elif shard_id == "gate":
            offset = 0
        # Up 是后半部分，起点为 intermediate_size=4864。
        elif shard_id == "up":
            offset = self.intermediate_size
        # 映射表若传入未知标识，说明程序内部存在错误。
        else:
            raise ValueError(f"Unknown packed shard: {shard_id}")

        # Linear 权重第 0 维是输出通道。
        # narrow 取得目标参数的对应连续行。
        target_shard = target_param.data.narrow(0, offset, source_tensor.shape[0])
        # 把单独的 q/k/v 或 gate/up 权重写入刚取得的融合切片。
        self.copy_tensor(target_shard, source_tensor)
        # 标记这个融合目标参数已被写入。
        self.loaded_params.add(target_name)
        # 同时记录具体分片。
        # 最终必须看到 q+k+v 或 gate+up 的完整集合。
        self.loaded_shards.setdefault(target_name, set()).add(shard_id)

    # 加载权重不需要构建 autograd 计算图。
    # no_grad 可以减少内存和状态开销。
    @torch.no_grad()
    def load_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """处理 checkpoint 中的一个“参数名、CPU 张量”二元组。"""

        # 每调用一次就代表从 Safetensors 中读到了一个源张量。
        self.source_tensor_count += 1

        # 先检查当前张量是否为 q_proj、k_proj 或 v_proj。
        for source_fragment, (target_fragment, shard_id) in self.QKV_PARTS.items():
            # 参数名包含某个源片段时，使用对应融合规则。
            if source_fragment in name:
                # 写入 qkv_proj 的正确输出通道切片。
                self.load_packed_tensor(
                    name,
                    tensor,
                    source_fragment,
                    target_fragment,
                    shard_id,
                )
                # 一个参数只匹配一个 QKV 规则，处理后立即返回。
                return

        # 如果不是 QKV，再检查是否为 gate_proj 或 up_proj。
        for source_fragment, (target_fragment, shard_id) in self.GATE_UP_PARTS.items():
            # 匹配成功意味着它需要写进 gate_up_proj。
            if source_fragment in name:
                # 写入前半 Gate 或后半 Up 切片。
                self.load_packed_tensor(
                    name,
                    tensor,
                    source_fragment,
                    target_fragment,
                    shard_id,
                )
                # 融合张量处理结束，不再尝试普通同名加载。
                return

        # 非融合权重应当能按 checkpoint 名称直接找到同名目标参数。
        if name not in self.params:
            raise KeyError(f"Checkpoint tensor has no target parameter: {name}")
        # Embedding、Norm、o_proj、down_proj 等参数直接整块复制。
        self.copy_tensor(self.params[name].data, tensor)
        # 记录这个普通参数已经成功加载。
        self.loaded_params.add(name)

    def verify_complete(self) -> None:
        """验证参数和融合分片均已加载，采用 fail-closed 行为。"""

        # 遍历模型期望的全部目标参数。
        # 不能只相信 checkpoint 提供的名字。
        for name in self.params:
            # 每个 qkv_proj 的 weight 和 bias 都必须收到三个源分片。
            if ".qkv_proj." in name:
                expected = {"q", "k", "v"}
                # set 精确比较还能发现未知映射造成的异常集合。
                if self.loaded_shards.get(name) != expected:
                    raise ValueError(f"Incomplete QKV parameter: {name}")
            # 每个 gate_up_proj.weight 都必须同时收到 Gate 和 Up。
            if ".gate_up_proj." in name:
                expected = {"gate", "up"}
                if self.loaded_shards.get(name) != expected:
                    raise ValueError(f"Incomplete gate/up parameter: {name}")

        # 用集合差求出模型有、但 checkpoint 没有覆盖的参数。
        missing = set(self.params) - self.loaded_params
        # 任何漏载都会留下随机初始化值。
        # 所以必须报错而不能继续推理。
        if missing:
            raise ValueError(f"Model parameters not loaded: {sorted(missing)}")

    def load(self, model_path: Path) -> None:
        """发现全部 Safetensors 分片，逐文件、逐张量完成加载。"""

        # sorted 保证多分片 checkpoint 每次都按稳定顺序处理。
        weight_files = sorted(model_path.glob("*.safetensors"))
        # 没有权重文件时立即给出比后续 KeyError 更明确的错误。
        if not weight_files:
            raise FileNotFoundError(f"No Safetensors weights found in {model_path}")

        # 一个大型模型可能有多个 model-00001-of-XXXXX.safetensors 文件。
        for weight_file in weight_files:
            # 显示当前加载进度。
            print(f"Reading {weight_file.name}...")
            # 文件在 CPU 侧打开。
            # 单个张量随后才转换并复制到目标设备。
            with safe_open(weight_file, framework="pt", device="cpu") as weights:
                # keys() 给出当前文件中的全部 checkpoint 参数名。
                for name in weights.keys():
                    # get_tensor 读取当前张量。
                    # load_tensor 决定直接加载还是融合加载。
                    self.load_tensor(name, weights.get_tensor(name))

        # 所有文件处理完后，再统一检查是否漏了参数或融合分片。
        self.verify_complete()
        # 本模型会把 290 个 HF 张量装入 170 个 vLLM 风格目标参数。
        print(
            f"Loaded {self.source_tensor_count} checkpoint tensors into "
            f"{len(self.loaded_params)} fused model parameters."
        )




class Scheduler:
    def __init__(self, eos_token_ids: set[int], model: Qwen2ForCausalLM) -> None:
        self.running_list: list[Request] = []  # req_id
        self.waiting_list:list[Request] = []
        self.finished_list: list[Request] = []
        # Match vLLM's configured scheduler sequence limit for this test.
        self.max_num_seqs = 32
        # 最大单次batch处理长度
        # vLLM's current default max_num_batched_tokens is 2048.
        self.max_batch_num_tokens = 2048
        # 分批prefill长度
        # vLLM chunks prefill using the remaining batched-token budget.
        self.chunk_prefill_tokens = self.max_batch_num_tokens
        self.kv_cache_manager: KVCacheStore = {}
        self.eos_token_ids: set[int] = eos_token_ids
        self.max_new_tokens = 128
        self.model: Qwen2ForCausalLM = model
        self.incoming_requests: queue.Queue[Request] = queue.Queue()
        self.wakeup_event = threading.Event()
        self.stop_event = threading.Event()

        self.worker_thread = threading.Thread(target=self.start_scheduler, daemon=True)
        self.worker_thread.start()

    def add_request(self, request: Request) -> Request:
        self.incoming_requests.put(request)
        self.wakeup_event.set()
        return request

    def start_scheduler(self):
        try:
            while not self.stop_event.is_set():
                self._drain_incoming_requests()
                if not self.running_list and not self.waiting_list:
                    self.wakeup_event.wait()
                    self.wakeup_event.clear()
                    continue
                self.step()
        except BaseException as error:
            traceback.print_exc()
            self.stop_event.set()
            pending_requests = (
                self.running_list
                + self.waiting_list
                + list(self.incoming_requests.queue)
            )
            for request in pending_requests:
                request.fail(error)

    def _drain_incoming_requests(self) -> None:
        """把网络线程提交的请求转入 Scheduler 私有 waiting_list。"""
        while True:
            try:
                self.waiting_list.append(self.incoming_requests.get_nowait())
            except queue.Empty:
                return

    # inference_mode 比 no_grad 更彻底地关闭 autograd，适合纯推理函数。
    @torch.inference_mode()
    def step(
        self,
    ) -> list[list[int]]:
        if len(self.running_list) != 0 or len(self.waiting_list) != 0:
            # 先看还有没有空闲位置
            running_total_tokens = 0
            for running_req in self.running_list:
                # running_req_list 可能经过一轮model计算后，需要重新计算这一批次需要处理多少个token
                # 还需要继续处理请求
                next_precessor_len = running_req.get_next_process_len(self.chunk_prefill_tokens)
                next_batch_num_token = running_total_tokens + next_precessor_len
                # 没有更多空间了
                if next_batch_num_token > self.max_batch_num_tokens:
                    next_batch_num_token = (
                        self.max_batch_num_tokens - running_total_tokens
                    )
                    running_req.get_and_set_next_process(next_batch_num_token)
                    running_total_tokens += next_batch_num_token
                else:
                    tokens = running_req.get_and_set_next_process(
                        self.chunk_prefill_tokens
                    )
                    running_total_tokens += tokens

            # 还有空闲再处理waiting队列
            while (len(self.running_list) < self.max_num_seqs
                    and len(self.waiting_list) > 0
                    and self.max_batch_num_tokens > running_total_tokens):
                # 放请求到running队列
                # 计算能放多少个token进去，从waiting先取一个出来
                wait_req_head = self.waiting_list[0]
                # 先看需要添加多少个token
                # 看剩余空间
                left_token_len = self.max_batch_num_tokens - running_total_tokens
                next_process_len = wait_req_head.get_and_set_next_process(
                    left_token_len
                    if left_token_len < self.chunk_prefill_tokens
                    else self.chunk_prefill_tokens
                )
                self.running_list.append(wait_req_head)
                self.waiting_list.remove(wait_req_head)
                running_total_tokens += next_process_len


            # 申请kvcache
            for req in self.running_list:
                # 每一个请求的开始位置
                if self.kv_cache_manager.get(req.req_id) is None:
                    self.kv_cache_manager[req.req_id] = {
                        KVCacheType.K: {},
                        KVCacheType.V: {},
                    }

            # 模拟model
            for running_req in self.running_list:
                # 记录处理前状态
                print("before forward req[", running_req.req_id, "], prefill_len: ", running_req.prefill_seq_len, ", start: ", running_req.start, ", position: ", running_req.position, ", decode_list: ", running_req.decode_tokens)
                # 是prefill
            # shape = [batch, 1]
            next_token_list = self.batch_prefill_and_decode()
            for i in range(0, len(next_token_list)):
                running_req = self.running_list[i]
                next_token = next_token_list[i]
                if running_req.prefill_seq_len > running_req.position:
                    running_req.update_prefill_len()
                    continue
                if running_req.prefill_seq_len <= running_req.position:
                    token_count_after_append = (
                        len(running_req.decode_tokens) + 1
                    )
                    finished = (
                        (
                            next_token in self.eos_token_ids
                            and not running_req.ignore_eos
                        )
                        or token_count_after_append
                        >= running_req.max_new_tokens
                    )
                    running_req.append_decode_token(
                        torch.tensor(
                            [next_token],
                            device=running_req.input_ids.device,
                            dtype=running_req.input_ids.dtype,
                        ),
                        finished=finished,
                    )
            # 记录处理后状态
            for running_req in self.running_list:
                print("after forward req[", running_req.req_id, "], prefill_len: ", running_req.prefill_seq_len,
                    ", start: ", running_req.start, ", position: ", running_req.position, ", decode_list: ",
                    running_req.decode_tokens)
            # 重新管理调度队列
            new_running_req: list[Request] = []
            for running_req in self.running_list:
                if running_req.finished:
                    # The request will never use its per-layer K/V tensors
                    # again. Drop the entry immediately so CUDA memory can be
                    # reused by subsequent requests.
                    self.kv_cache_manager.pop(running_req.req_id, None)
                    self.finished_list.append(running_req)
                else:
                    new_running_req.append(running_req)
            self.running_list = new_running_req
            print("waiting_list.len: ", len(self.waiting_list), " running_list.len: ", len(self.running_list), " finished_list.len: ", len(self.finished_list))
            if len(self.waiting_list) == 2 and len(self.running_list) == 1 and len(self.finished_list) == 2:
                print("checkpoint")

    # inference_mode 比 no_grad 更彻底地关闭 autograd，适合纯推理函数。
    @torch.inference_mode()
    def batch_prefill_and_decode(
            self,
    ) -> list[int]:
        # 扁平化处理
        # [1, 2, 3, 4, 20, 30, 99]
        # [[1,2,3,4], [20, 30], [99]]
        # [0,4,6,7]
        # 表示(0, 4)一组，(4, 6)一组,(6,7)一组，这些是q的token
        # 生成这样的两个张量
        query_start_loc: list[int] = [0]
        input_id_chunks: list[torch.Tensor] = []
        for running_req in self.running_list:
            input_id_chunks.append(running_req.get_current_pd_token())
            query_start_loc.append(
                query_start_loc[-1] + running_req.get_current_pd_token().shape[0]
            )
        input_ids =  torch.cat(input_id_chunks, dim=0)
        print("[batch_prefill_and_decode] input_ids.shape: ", input_ids.shape)
        print("[batch_prefill_and_decode] query_start_loc: ", query_start_loc)
        # logits = model(
        #     input_ids,
        #     ids,
        #     query_start_loc,
        #     kv_cache_manager,
        # )

        # [batch, logits]
        batch_logits = self.model(
            input_ids=input_ids,
            ids=self.running_list,
            query_start_loc=query_start_loc,
            kv_cache_manager=self.kv_cache_manager,
        )
        generated: list[int] = []
        for index in range(batch_logits.shape[0]):
            # 取当前请求 logits 最大的 5 个候选 token。
            logit = batch_logits[index]
            top_k = min(10, logit.shape[-1])
            top_k_logits, top_k_token_ids = torch.topk(
                logit,
                k=top_k,
                dim=-1,
            )

            # 只在 Top-5 候选中归一化概率，避免从整个词表随机抽样。
            top_k_probs = F.softmax(top_k_logits, dim=-1)

            # 按 Top-5 概率分布随机选择一个候选 token。
            sampled_index = torch.multinomial(
                top_k_probs,
                num_samples=1,
            )
            next_token_id = int(
                top_k_token_ids[sampled_index].item()
            )
            # 遇到 <|im_end|> 或 pad/endoftext 时结束。
            # 不把这个特殊 token 加入正文。
            # if next_token_id in eos_token_ids:
            #     break
            # 保存新生成的普通 token id。
            generated.append(next_token_id)
        return generated


import threading

def main() -> None:
    """加载本地模型并启动 input() 驱动的多轮对话。"""

    # 在创建模型前确认固定的本地目录存在。
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

    # 根据当前机器选择 CUDA、MPS 或 CPU 以及相应参数类型。
    device, dtype = get_device_and_dtype()
    # AutoConfig 只读取 config.json。
    # 它不会创建 Transformers 模型或加载权重。
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    # Tokenizer 负责文本/token 转换和应用模型自带的 ChatML 模板。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    # 提示即将在哪个设备、以什么数据类型创建模型。
    print(f"Building vLLM-style Qwen2 on {device} with {dtype}...")
    # 与 vLLM initialize_model 类似：
    # 直接在目标设备和目标数据类型下创建参数。
    with set_default_dtype(dtype), device:
        # 这里只搭建网络结构，参数暂时仍是随机初始化值。
        model = Qwen2ForCausalLM(config)

    # 逐个读取 Safetensors，并原地写入普通参数或融合参数切片。
    VllmStyleWeightLoader(model, config).load(MODEL_PATH)
    # 切换到 eval 模式。
    # 本模型虽无 Dropout，显式设置仍是标准推理步骤。
    model.eval()
    # 告诉用户模型已经可以接收输入。
    print("Model loaded. Enter exit or quit to stop.")

    # messages 保存完整多轮历史；system 消息定义助手的基本行为。
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    # Qwen2 使用 im_end 结束回答。
    # 同时把 pad/endoftext 也视作停止标记。
    eos_token_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    scheduler = Scheduler(eos_token_ids, model)
    # 不断读取用户输入，直到显式退出、Ctrl+C 或输入流结束。
    req_id = 10000
    while True:
        try:
            # input 阻塞等待一行文本；strip 去除首尾空白。
            user_input = input("\nYou: ").strip()
        # Ctrl+D 触发 EOFError，Ctrl+C 触发 KeyboardInterrupt。
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        # 允许用户输入两个常见命令正常退出。
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        # 空输入不进入对话历史，直接等待下一次输入。
        if not user_input:
            continue

        # 把当前问题追加到历史，后续每一轮都会携带此前上下文。
        messages.append({"role": "user", "content": user_input})
        # 将 role/content 消息转换成 Qwen2 使用的 ChatML 文本格式。
        prompt = tokenizer.apply_chat_template(
            messages,
            # 返回字符串；下一步再统一 tokenize。
            # 这样便于观察和理解两个阶段。
            tokenize=False,
            # 在末尾增加 <|im_start|>assistant，提示模型开始回答。
            add_generation_prompt=True,
        )
        # 文本编码为 [1,S] token id，并移动到模型所在设备。
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids[0].to(device)

        # 提交后，Scheduler 后台线程继续计算；当前线程消费该请求的增量输出。
        request = Request(input_ids=input_ids, req_id=req_id)
        request_handle = scheduler.add_request(request)

        generated_ids: list[int] = []
        while True:
            token_ids, finished = request_handle.wait_for_new_tokens()
            generated_ids.extend(token_ids)

            if finished:
                break

        # 直接交互模式等待请求完全结束后一次性输出。
        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        print(f"Assistant: {response}")
        messages.append(
            {"role": "assistant", "content": response}
        )
        req_id += 1


# 只有直接执行本文件时才启动交互。
# 被 import 时不会自动加载约 1 GB 权重。
if __name__ == "__main__":
    # 调用程序入口。
    main()
