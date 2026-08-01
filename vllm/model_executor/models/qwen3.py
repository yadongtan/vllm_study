# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""只用于推理（inference-only）的 Qwen3 模型，并兼容 HuggingFace 权重。

这个文件可以从上到下理解成 4 个层次：

1. :class:`Qwen3Attention`：一层注意力，包括 Q/K/V 投影、QK-Norm、
   RoPE（旋转位置编码）和输出投影。
2. :class:`Qwen3DecoderLayer`：一个 Transformer block，即“注意力 + MLP”，
   两个子层周围都有 RMSNorm 和残差连接。
3. :class:`Qwen3Model`：把很多个 ``Qwen3DecoderLayer`` 叠起来。大部分通用逻辑
   直接复用 Qwen2 的实现，因为 Qwen2 与 Qwen3 的主体结构非常相似。
4. :class:`Qwen3ForCausalLM`：在主干模型外再包上语言模型输出头、logits 处理和
   HuggingFace 权重加载逻辑，形成 vLLM 最终使用的完整模型。

阅读张量形状时要注意：vLLM 为提高推理效率，经常把传统的
``[batch_size, sequence_length, hidden_size]`` 前两维展平。因此本文件里的
``hidden_states`` 常见形状是 ``[num_tokens, hidden_size]``。以下注释统一使用
Qwen3-0.6B 在两张 GPU 上做张量并行（TP=2）的例子。该模型有 28 层、
hidden_size=1024、intermediate_size=3072、16 个 Q head、8 个 KV head、
head_dim=128、词表大小 151936，公开说明的上下文长度为 32768。比如同时处理
2 条、每条 4 个 token 的输入时，hidden_states 可能是 ``[8, 1024]``，而不是
``[2, 4, 1024]``。代码使用 ``...`` 的地方也允许存在额外前导维度。

Qwen3-0.6B 在两卡 TP 下，每一层的注意力 head 分工如下（28 层都一样）：

* GPU 0：Q head 0~7，KV head 0~3；Q 0/1 共用 KV 0，Q 2/3 共用 KV 1，
  Q 4/5 共用 KV 2，Q 6/7 共用 KV 3。
* GPU 1：Q head 8~15，KV head 4~7；同样每两个相邻 Q head 共用一组 KV。

这里说“相关的 Q 和 KV”不是指它们运行时算出来的内容比较相似，而是指 GQA
结构预先规定了某个 Q head 必须使用哪个 KV head。Qwen3-0.6B 中每个 KV head
服务的 Q head 数是 ``16 // 8 = 2``，因此固定映射公式为：

``kv_head_index = q_head_index // 2``

例如，Q 0 // 2 和 Q 1 // 2 都得到 KV 0；Q 2 // 2 和 Q 3 // 2 都得到
KV 1；依此类推，Q 14/15 使用 KV 7。这个关系来自模型训练时采用的连续 head
分组约定，不是推理时通过相似度动态选择。vLLM 的注意力层还会检查 Q head 数
能否被 KV head 数整除，以保证每个 KV head 服务相同数量的 Q head。

因此这个具体的两卡例子不会复制 KV head；两张卡各持有不同的 4 个 KV head。
更重要的是，每个 Q head 所需的 KV head 都和它位于同一张卡：GPU 0 的 Q 0~7
只需要本地 KV 0~3，GPU 1 的 Q 8~15 只需要本地 KV 4~7，所以注意力计算不必
为了取得所需的 K/V 张量而向另一张 GPU 请求数据。
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsEagle, SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, extract_layer_index, maybe_prefix

logger = init_logger(__name__)


class Qwen3Attention(nn.Module):
    """Qwen3 的自注意力子层。

    参数名中的 ``total`` 表示“整个模型所有张量并行设备合起来”的数量；没有
    ``total`` 的成员通常表示“当前 GPU/进程实际负责”的数量。

    Qwen3 使用 GQA（Grouped-Query Attention，分组查询注意力）：Q 的 head 数可
    以多于 K/V 的 head 数，多个 Q head 共享一组 K/V。Qwen3-0.6B 有 16 个 Q
    head 和 8 个 KV head，即每 2 个 Q head 共用一个 K/V head。这样可显著减小
    生成阶段的 KV cache，同时通常比只使用单个 KV head 的 MQA 保留更多能力。
    “共用”关系由 head 编号固定决定：``KV编号 = Q编号 // 2``，不是比较 Q/K/V
    的数值后临时决定；上面的模块总说明给出了完整的两卡分配例子。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        # hidden_size 是每个 token 的隐藏向量宽度。Qwen3-0.6B 的
        # hidden_size=1024，所以每个 token 进入本层时由 1024 个数表示。
        # 注意这里 hidden_size 不等于 num_heads * head_dim：本例明确配置
        # head_dim=128，而 16 * 128 = 2048。这是模型设计允许的，注意力输出
        # 最后会由 o_proj 从 2048 维重新映射到 1024 维。
        self.hidden_size = hidden_size

        # TP（Tensor Parallel，张量并行）把同一层的计算拆到多张 GPU。
        # 本例在两卡机器上设置 tp_size=2：两张卡共同计算全部 28 层，而不是
        # GPU 0 算前 14 层、GPU 1 算后 14 层；后者属于流水线并行（PP）。
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        # Q head 必须能平均分到两个 TP rank。本例 16 % 2 == 0，因此可以
        # 等分：GPU 0 负责 Q head 0~7，GPU 1 负责 Q head 8~15。
        assert self.total_num_heads % tp_size == 0
        # 本例：16 // 2 = 8，即每张 GPU 负责 8 个 Q head。
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Qwen3-0.6B 的 8 个 KV head 多于两张 GPU，因此本例进入此分支。
            # 8 % 2 == 0，可以平均切分：GPU 0 负责 KV head 0~3，GPU 1
            # 负责 KV head 4~7。每张卡各算 4 个 KV head，没有 KV 重复计算。
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # 本例不会进入这里，因为 8 >= 2。此分支是这份通用 Qwen3 实现为
            # “GPU 数多于 KV head 数”的部署保留的：head 不能切成小数，届时
            # QKVParallelLinear 会在多个 GPU 上复制相同 KV 权重，以少量重复
            # 计算换取更少的卡间通信。assert 要求每个 KV head 的副本数相同。
            assert tp_size % self.total_num_kv_heads == 0
        # 本例：max(1, 8 // 2) = 4，即每张 GPU 有 4 个本地 KV head。
        # 每卡 8 个 Q head 对应 4 个 KV head，仍保持每 2 个 Q head 共享 1 组 KV。
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        # Qwen3-0.6B 显式配置 head_dim=128，所以这里取 128，而不是采用后面的
        # 备用公式 1024 // 16 = 64。``or`` 右侧只在 head_dim 未配置时使用。
        self.head_dim = head_dim or hidden_size // self.total_num_heads

        # 当前 TP rank 上，展平所有本地 head 后 Q、K/V 各自占据的末维宽度。
        # 本例每卡有 8 个 Q head、4 个 KV head、head_dim=128，因此：
        # q_size = 8 * 128 = 1024，kv_size = 4 * 128 = 512。
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # scaled dot-product attention 使用 QK^T / sqrt(head_dim)。
        # x**-0.5 等价于 1/sqrt(x)，用于避免 head_dim 较大时点积数值过大，
        # 从而让 softmax 过早变得极端。
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        # 一个融合的线性层同时生成 Q、K、V。概念上等价于三个独立层
        # q_proj/k_proj/v_proj，但合并后能减少算子启动与内存读写。
        # 本例输入 hidden_states 是 [num_tokens, 1024]；每张卡的融合投影输出
        # 末维为 q_size + kv_size + kv_size = 1024 + 512 + 512 = 2048。
        # 投影: 做一次线性变换，矩阵乘法+偏置。 input @ weight.T + bias
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # 全部 16 个注意力 head 拼接后总宽度是 16 * 128 = 2048；每张卡持有
        # 其中 8 * 128 = 1024 维。RowParallelLinear 让两张卡分别计算输出的
        # 部分和，再通过 TP 通信相加，最终映射回 hidden_size=1024。
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE 不把位置向量直接加进 hidden states，而是在每个 head 内旋转
        # Q 和 K 的成对维度，使 Q·K 点积自然携带相对位置信息。
        # Qwen3-0.6B 模型页公布的上下文长度是 32768；目前配置文件中的
        # max_position_embeddings 是 40960，所以代码实际传入的 max_position
        # 是 40960。前者可理解为对外说明的上下文能力，后者是位置编码预留的
        # 配置上限，两者不必完全相等。rope_parameters 还可描述扩展缩放方法。
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        # decoder-only 生成模型使用带因果遮罩的 Attention：token 只能看自己和
        # 前面的 token。某些 embedding 版 Qwen3 则需要双向看完整句子，所以
        # 使用 EncoderOnlyAttention。
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        # 这里的 Attention 是 vLLM 的高性能注意力封装：它会根据运行环境选择
        # 后端，并在逐 token 生成时读写 KV cache。本类无需手工实现 softmax、
        # causal mask 或 cache 管理。
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        # Qwen3 与 Qwen2 的关键差异之一是 QK-Norm：对每一个 Q/K head 独立做
        # RMSNorm。它帮助稳定注意力分数，尤其适合长上下文训练和推理。
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """计算一次自注意力。

        ``positions`` 给出每个 token 的绝对位置，例如 ``[0, 1, 2, 3]``；在
        Qwen3-0.6B 增量解码时也可能只有新 token 的位置，如 ``[37]``。
        ``hidden_states`` 常为 ``[num_tokens, 1024]``，返回值形状与它相同。
        """
        # 融合投影一次得到 QKV。线性层还返回可选 bias，此处模型不需要，所以
        # 用下划线明确忽略第二个返回值。
        qkv, _ = self.qkv_proj(hidden_states)
        # 沿最后一维把每张卡的融合结果切回三个张量。本例末维按
        # [1024, 512, 512] 切分。此时 head 维仍被压平在最后一维中。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 添加 QK-Norm。RMSNorm 必须对“每个 head 的 head_dim”单独归一化，
        # 不能直接对所有 head 拼起来的 q_size 做一次归一化，所以先 reshape。
        # 本例 q 从 [num_tokens, 1024] 变成 [num_tokens, 8, 128]；
        # k 从 [num_tokens, 512] 变成 [num_tokens, 4, 128]。
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        # 恢复为注意力后端期望的展平格式；view 不改变元素值，只改变观察形状。
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        # 仅对 Q、K 应用位置旋转，因为注意力权重来自 Q 与 K 的点积；V 负责
        # 携带被加权汇总的内容，不参与“匹配位置”，因此无需 RoPE。
        q, k = self.rotary_emb(positions, q, k)
        # 概念公式：softmax((Q @ K^T) * scaling + mask) @ V。
        # 实际实现还负责 GQA head 映射、分页 KV cache 和高性能融合 kernel。
        attn_output = self.attn(q, k, v)
        # 把多头注意力结果映射回模型隐藏宽度。第二个返回值同样是可选 bias。
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    """一个完整的 Qwen3 Transformer block。

    数据流可简化为：

    ``输入 -> RMSNorm -> 自注意力 -> 残差相加 -> RMSNorm -> MLP -> 残差相加``

    vLLM 的 RMSNorm 接口会把“残差相加”和“归一化”融合起来以减少显存访问，
    因而代码写法与教科书中的逐行公式略有不同。
    """

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # 老配置可能没有 rope_theta。Qwen3 的默认基频是 1,000,000；只在配置
        # 缺省时补上，不覆盖模型原本明确指定的值。
        set_default_rope_theta(config, default_theta=1000000)
        # Dual Chunk Attention 是某些长上下文变体使用的配置。用 getattr 并给
        # None 默认值，可兼容不含该字段的普通 HuggingFace 配置。
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # 标准 Qwen3 是 decoder-only 模型，使用因果注意力。例如第 3 个 token
        # 可以看第 1~3 个，却不能偷看第 4 个。若 HuggingFace 配置显式设置
        # is_causal=False，则启用双向注意力，每个 token 都能看完整输入；这常见
        # 于句向量/检索 embedding 模型，如 Alibaba-NLP/gte-Qwen3-7B-instruct。
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        # MLP（前馈网络）逐 token 独立处理，不在 token 之间交换信息。
        # Qwen3 复用 Qwen2MLP，通常是带门控的结构（gate_proj、up_proj、
        # down_proj）。Qwen3-0.6B 的 intermediate_size=3072：概念上先把每个
        # token 从 1024 维扩展到 3072 维，经过 SiLU 门控，再投影回 1024 维。
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # Pre-Norm 架构：注意力和 MLP 之前各有一个 RMSNorm。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行一个 Transformer block，并返回 ``(新输出, 残差流)``。

        ``residual`` 在首层入口可能为 None。后续层会显式传递它，让 vLLM 用
        融合 kernel 完成加法与 RMSNorm。把二者分开返回不是少做了残差连接，
        而是一种减少中间张量和显存读写的性能优化。
        """
        # 第一部分：自注意力。
        if residual is None:
            # 第一层尚无单独的残差流：保存原始输入作为 residual，再归一化一份
            # 送入注意力。概念上稍后会形成 input + attention(norm(input))。
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # 融合形式同时完成：
            #   1. hidden_states = hidden_states + residual
            #   2. residual = 上述相加后的未归一化结果
            #   3. hidden_states = RMSNorm(上述结果)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # 第二部分：MLP。这里先把注意力输出加回 residual，并对结果归一化；
        # MLP 的输出与更新后的 residual 分开返回，交给下一层入口继续融合相加。
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    # Qwen2Model 的通用建层逻辑通过这个名称找到 Qwen3 block。
    # 字典形式也为未来可能加入其他层类型保留扩展空间。
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3Model(Qwen2Model):
    """Qwen3 的主干网络（词嵌入、多层 decoder block、最终归一化）。

    Qwen3 与 Qwen2 的整体堆叠、流水线并行和输入处理方式相同，所以继承
    ``Qwen2Model``；这里只把每一层的类型替换为 ``Qwen3DecoderLayer``。
    装饰器告诉 torch.compile 哪些参数维度是动态的：不同请求的 token 数会变，
    编译器应尽量复用同一份已编译计算图，而不是每种长度都重新编译。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )


class Qwen3ForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    """可用于“预测下一个 token”的完整 Qwen3 模型。

    ``Qwen3Model`` 输出隐藏向量；本类再使用 ``lm_head`` 将每个隐藏向量映射
    到词表大小的 logits。Qwen3-0.6B 的词表有 151936 项，因此每个待预测位置
    会得到 151936 个未归一化分数，采样器据此选择下一个 token。

    SupportsLoRA/PP/Eagle/Eagle3 是能力标记接口，告诉 vLLM 此模型支持 LoRA、
    Pipeline Parallel（流水线并行）以及 EAGLE 系列投机解码功能。
    """

    # HuggingFace checkpoint 通常分别保存 q_proj/k_proj/v_proj，但 vLLM 为了
    # 推理效率使用融合 qkv_proj。加载器依靠该映射把三个权重打包到一个参数。
    # gate_proj/up_proj 与融合 gate_up_proj 同理。
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA 等通用功能通过此映射识别模型的输入、输出 embedding 模块名称。
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # hf_config 描述模型架构（层数、hidden_size、head 数、词表大小等）；
        # quant_config 描述是否以及如何使用 INT8/FP8/GPTQ 等量化权重。
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # PP（Pipeline Parallel）按“层”把模型切到不同 rank。只有最后一个
        # pipeline rank 需要把最终隐藏状态转成词表 logits；其他 rank 放置一个
        # PPMissingLayer 占位，避免无意义地分配巨大 lm_head 权重。
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                # 权重绑定：输入 embedding 矩阵也作为输出投影矩阵使用。这样既
                # 减少参数量，也与训练 checkpoint 的结构保持一致。
                self.lm_head = self.model.embed_tokens
            else:
                # 未绑定时创建独立语言模型头。ParallelLMHead 会沿词表维进行
                # 张量并行，因此每张卡只保存/计算一部分词表权重。
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        # LogitsProcessor 负责调用 lm_head，并处理张量并行词表、只选所需位置等
        # vLLM 推理细节；这里并不执行 softmax，logits 仍是未归一化分数。
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # 暴露主干模型的辅助函数，供流水线并行框架创建层间传递所需的空张量。
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """把 token id 查表转换成隐藏向量。

        对 Qwen3-0.6B，``input_ids=[10, 25]`` 会取形状为
        ``[151936, 1024]`` 的 embedding 矩阵第 10、25 行，得到形状为
        ``[2, 1024]`` 的浮点张量。TP=2 时 embedding 的内部存储/查找由并行
        embedding 层负责，对上层仍表现为每个 token 得到 1024 维向量。
        """
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """运行 Qwen3 主干网络。

        通常传 ``input_ids``；若上游已经算好 embedding，则可传
        ``inputs_embeds``。在流水线中间 rank，输入/输出可能是
        ``IntermediateTensors``，其中保存从相邻 rank 传来的隐藏状态和残差。
        本方法故意不计算 logits，使主干计算与输出头解耦。
        """
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """将隐藏状态映射成词表 logits。

        对 Qwen3-0.6B，正常位于最后一个 pipeline rank 时，结果形状类似
        ``[num_selected_tokens, 151936]``。返回 None 的情况由底层并行流程决定。
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载 HuggingFace 格式的 ``(参数名, 参数张量)`` 迭代器。

        ``AutoWeightsLoader`` 会处理模块名前缀、张量并行切分，以及上面声明的
        QKV/门控权重打包。若输入输出 embedding 已绑定，checkpoint 中独立的
        ``lm_head.*`` 无需再次加载，所以通过 ``skip_prefixes`` 跳过。返回集合
        包含实际成功加载的参数名，便于框架检查是否漏载。
        """
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
