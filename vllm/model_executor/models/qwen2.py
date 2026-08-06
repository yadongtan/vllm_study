# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
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
"""只用于推理（inference-only）的 Qwen2 模型，并兼容 HuggingFace 权重。

这个文件把 HuggingFace checkpoint 中的 Qwen2 权重组织成 vLLM 推理时使用的
网络。可以从下到上把它理解为五层：

1. ``Qwen2MLP``：逐 token 运行的门控前馈网络。
2. ``Qwen2Attention``：Q/K/V 投影、RoPE、KV Cache 注意力和输出投影。
3. ``Qwen2DecoderLayer``：一个 Transformer block，即注意力加 MLP。
4. ``Qwen2Model``：Embedding、24 个 block 和最后的 RMSNorm。
5. ``Qwen2ForCausalLM``：再接 lm_head，把隐藏状态变成词表 logits。

以下注释统一使用当前目录中的 ``Qwen2-0.5B-Instruct`` 举例。它的配置是：

* 24 层，``hidden_size=896``，``intermediate_size=4864``；
* 14 个 Q head，2 个 KV head，故 ``head_dim=896 // 14=64``；
* 词表大小 151936，最大位置数 32768，输入/输出 embedding 权重绑定；
* GQA 中每 7 个 Q head 共用 1 个 K head 和 1 个 V head。

当前 Apple Silicon CPU 是 TP=1，所以一个进程持有全部 14 个 Q head 和 2 个
KV head。为了说明张量并行，涉及 TP 的注释还会补充 TP=2 的例子：GPU 0 负责
Q head 0~6 和 KV head 0；GPU 1 负责 Q head 7~13 和 KV head 1。

vLLM 常把传统的 ``[batch_size, sequence_length, hidden_size]`` 前两维压平为
``[num_tokens, hidden_size]``。例如一个请求 Prefill 4 个 token，进入模型的
``hidden_states`` 是 ``[4, 896]``；Decode 一次处理一个新输入 token 时通常是
``[1, 896]``。混合批处理中，``num_tokens`` 是本轮所有请求实际调度 token 数
之和，不等于请求数，也不一定等于某条请求的完整上下文长度。
"""

# 下面导入的名称按用途可分为：
#
# * Iterable：表示可逐个读取的对象；加载权重时无需一次把所有张量放入列表。
# * islice：只遍历层列表的一段，供流水线并行选择当前 rank 的层。
# * Any：允许可选扩展配置字典容纳不同类型的值。
# * torch/nn：PyTorch 张量操作，以及所有模型层共同的 nn.Module 基类。
# * Qwen2Config：HuggingFace 定义的 Qwen2 架构配置类型。
# * support_torch_compile：使模型支持编译；--enforce-eager 可关闭它以便调试。
# * CacheConfig/VllmConfig：分别描述 KV Cache，以及完整的 vLLM 运行配置。
# * distributed 工具：查询 PP 首末 rank 和 TP 进程总数。
# * SiluAndMul：执行 Qwen2 MLP 的 SiLU(gate) * up 门控激活。
# * Attention/EncoderOnlyAttention：分别执行因果注意力和双向注意力。
# * RMSNorm：按均方根缩放向量；它不同于把分数变概率的 Softmax。
# * 三种 ParallelLinear：实现门控、QKV 和输出投影的融合/张量并行。
# * LogitsProcessor：调用 lm_head，处理词表并行和待采样位置选择。
# * QuantizationConfig：描述 GPTQ、INT8、FP8 等可选量化方式。
# * get_rope：构造 RoPE（Rotary Position Embedding，旋转位置编码）。
# * ParallelLMHead/VocabParallelEmbedding：沿词表维并行输出头和输入查表层。
# * weight_utils：把 checkpoint 张量写入参数，并兼容量化 scale 名称。
# * IntermediateTensors：在 PP rank 之间传递 hidden_states 和 residual。
# * transformers_utils：兼容滑动窗口，并为旧配置补默认 rope_theta。
# * AttentionType：区分 decoder 因果注意力和 encoder-only 双向注意力。
# * interfaces：能力标记，告诉 vLLM 模型支持 EAGLE、LoRA 和 PP。
# * .utils：自动加载、PP 占位、创建层、解析层号和拼接参数前缀。
from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import is_interleaved, set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    EagleModelMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class Qwen2MLP(nn.Module):
    """Qwen2 的门控前馈网络（MLP，Multi-Layer Perceptron）。

    注意力负责让不同 token 交换信息；MLP 则对每个 token 独立进行非线性变换。
    对本例一个 ``[num_tokens, 896]`` 输入，概念公式是：

    ``down_proj(SiLU(gate_proj(x)) * up_proj(x))``

    ``gate_proj`` 和 ``up_proj`` 都从 896 投影到 4864，vLLM 把两次投影融合为
    ``gate_up_proj``；逐元素相乘后仍为 4864 维，再由 ``down_proj`` 降回 896。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """创建 MLP 的层和参数。

        ``hidden_size`` 是输入/输出宽度，本例为 896；``intermediate_size`` 是
        中间宽度，本例为 4864；``hidden_act`` 应为 ``"silu"``；``prefix``
        只是参数的完整路径，例如 ``model.layers.0.mlp``，不参与数学计算。
        """
        # 调用 nn.Module 的初始化逻辑，以便 PyTorch 能注册子模块和参数。
        super().__init__()
        # MergedColumnParallelLinear 把两个输出投影合并计算。
        # [intermediate_size] * 2 等价于 [4864, 4864]，表示融合输出由两个
        # 等宽分片组成，而不是把输入执行“两遍 Python 函数”。TP=1 时：
        # [num_tokens, 896] -> [num_tokens, 9728]，前 4864 维是 gate，
        # 后 4864 维是 up。TP=2 时每卡各保存/计算两个输出的一半。
        self.gate_up_proj = MergedColumnParallelLinear(
            # 输入特征数：每个 token 有 896 个隐藏值。
            hidden_size,
            # 两个逻辑输出的宽度：gate_proj=4864，up_proj=4864。
            [intermediate_size] * 2,
            # Qwen2 checkpoint 的这两个投影没有 bias（偏置）。
            bias=False,
            # 若模型量化，线性层据此选择相应的权重格式与计算实现。
            quant_config=quant_config,
            # 形成类似 model.layers.0.mlp.gate_up_proj 的内部参数名。
            prefix=f"{prefix}.gate_up_proj",
        )
        # RowParallelLinear 把门控后的 4864 维投影回 hidden_size=896。
        # TP>1 时输入宽度按卡切分，各卡计算部分结果，最后通信相加。
        self.down_proj = RowParallelLinear(
            # 输入是 SiLU(gate) * up，所以宽度为 4864。
            intermediate_size,
            # 输出回到残差流要求的模型宽度 896。
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        # 当前实现只支持 Qwen2 原生使用的 SiLU 门控。若 checkpoint 声明其他
        # 激活函数，直接报错比悄悄算错结果更安全。
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        # SiluAndMul 会把融合张量平分成 gate/up，然后计算
        # SiLU(gate) * up。SiLU(z)=z*sigmoid(z)，结果包含非线性表达能力。
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """对每个 token 执行门控 MLP，并保持首尾隐藏宽度不变。"""
        # 本例 TP=1：x [num_tokens, 896] -> gate_up [num_tokens, 9728]。
        # 并行线性层第二个返回值是可单独返回的 bias；这里层没有 bias，忽略它。
        gate_up, _ = self.gate_up_proj(x)
        # 将 9728 平分为两个 4864 维张量，计算 SiLU(gate) * up；
        # x 因而变为 [num_tokens, 4864]。
        x = self.act_fn(gate_up)
        # 把中间表示投影回 [num_tokens, 896]，以便和 residual 相加。
        x, _ = self.down_proj(x)
        # 返回 MLP 分支输出；残差相加由 Qwen2DecoderLayer 负责。
        return x


class Qwen2Attention(nn.Module):
    """Qwen2 的自注意力子层。

    本例采用 GQA（Grouped-Query Attention）：14 个 Q head 共享 2 组 K/V，
    即 Q 0~6 使用 KV 0，Q 7~13 使用 KV 1。这里的“相关”是模型结构预先规定
    的编号映射 ``kv_index = q_index // 7``，不是运行时比较内容是否相似。

    Prefill 时，当前输入的多个 token 会一起生成 Q/K/V，并在因果遮罩下计算；
    Decode 时，新输入 token 生成自己的 Q/K/V，K/V 写入缓存，其 Q 再读取所有
    可见历史 K/V。两种阶段都调用本类的同一个 ``forward``，具体 token 边界、
    因果遮罩和分页 KV Cache 由底层 Attention 后端根据调度元数据处理。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        """创建 Qwen2 注意力所需的投影、RoPE、归一化和注意力后端。"""
        super().__init__()
        # 每个 token 进入注意力层时的隐藏向量宽度；本例是 896。
        self.hidden_size = hidden_size
        # TP（Tensor Parallel）按 head/矩阵维度拆同一层。本机 CPU 默认为 1；
        # 若 TP=2，则两张卡共同计算每一层，而不是各自负责一半层数。
        tp_size = get_tensor_model_parallel_world_size()
        # total 表示所有 TP rank 合计的 Q head 数；本例是 14。
        self.total_num_heads = num_heads
        # Q head 必须可以均匀分给 TP rank。本例 TP=1：14 % 1 == 0；
        # 两卡例子：14 % 2 == 0，每卡负责 7 个 Q head。
        assert self.total_num_heads % tp_size == 0
        # 当前 rank 本地负责的 Q head 数：TP=1 是 14，TP=2 是 7。
        self.num_heads = self.total_num_heads // tp_size
        # 所有 rank 合计的 KV head 数；本例是 2。一个“KV head”表示编号相同
        # 的一组 K head 和 V head，不是说 K/V 合并成了一个张量。
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # KV head 数不少于 TP rank 数时，把它们平均切分。
            # 当前 TP=1：2 个 KV head 都在本进程；若 TP=2：每卡 1 个 KV head，
            # GPU 0 的 Q 0~6 使用本地 KV 0，GPU 1 的 Q 7~13 使用本地 KV 1，
            # 注意力不必跨卡获取另一组 KV。
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # KV head 少于 TP rank 时无法切成“小数个 head”，于是复制 KV 权重。
            # 例如本模型若强行 TP=4：只有 2 个 KV head，每个会复制到 2 张卡，
            # 保证每张卡至少有一个本地 KV head。这样会有少量重复 K/V 投影，
            # 但避免每一步注意力都跨卡索取 KV，通常更划算。
            assert tp_size % self.total_num_kv_heads == 0
        # 当前 rank 的 KV head 数，至少为 1。TP=1：2；TP=2：1；TP=4：1
        # （最后一种是复制所得，而不是 2 // 4 真能得到完整 head）。
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # 每个 head 的特征维度。本例 896 // 14 = 64，且必须整除。
        self.head_dim = hidden_size // self.total_num_heads
        # 当前 rank 全部 Q head 拼接后的宽度。TP=1：14*64=896；TP=2：448。
        self.q_size = self.num_heads * self.head_dim
        # 当前 rank 的 K 或 V 宽度。TP=1：2*64=128；TP=2：1*64=64。
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力公式使用 QK^T / sqrt(head_dim)。64**-0.5 = 1/8；缩放可避免
        # 点积随维度增大而过大，导致 Softmax 接近全 0/1、数值和梯度不稳定。
        self.scaling = self.head_dim**-0.5
        # Dual Chunk Attention 是可选的长上下文分块策略；普通 Qwen2 配置没有
        # 这个字段，因此本例为 None。这里只保存配置，交给 RoPE/Attention。
        self.dual_chunk_attention_config = dual_chunk_attention_config
        # 标准 Qwen2-0.5B 未启用 QK-Norm，所以本例为 False；这份通用实现也
        # 兼容 BAGEL 等在每个 Q/K head 上增加 RMSNorm 的变体。
        self.qk_norm = qk_norm

        # 一个融合线性投影同时生成 Q、K、V。proj 是 projection（投影）的缩写，
        # 本质是 x @ weight.T + bias。融合能减少输入读取和算子启动次数。
        # 本例 TP=1：[tokens,896] -> [tokens,896+128+128]=[tokens,1152]。
        # 两个 128 分别属于 K 和 V：K 用来与 Q 计算匹配分数；V 携带最后被
        # 加权汇总的内容，二者作用不同，所以需要各自的一份投影结果。
        self.qkv_proj = QKVParallelLinear(
            # 输入隐藏宽度 896。
            hidden_size,
            # 每个 Q/K/V head 有 64 个特征。
            self.head_dim,
            # 全模型 14 个 Q head；层内部会按 TP 切分。
            self.total_num_heads,
            # 全模型 2 个 KV head；层内部会按 TP 切分或必要时复制。
            self.total_num_kv_heads,
            # Qwen2 的 q_proj/k_proj/v_proj checkpoint 含 bias，因此设为 True。
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # o_proj（output projection）把多头注意力输出重新混合到隐藏空间。
        # 全部 14 个 head 拼接宽度为 14*64=896，投影后仍是 896，虽然本例
        # 输入输出宽度相同，它仍是一个训练得到的 896x896 矩阵，并非原样复制。
        # TP>1 时各卡先算局部 head 的贡献，再通过 all-reduce 通信求和。
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # QK-Norm 支持（BAGEL 等变体使用，标准 Qwen2-0.5B 不执行此分支）。
        if self.qk_norm:
            # 对每个 Q head 的 64 个值做独立 RMSNorm。可学习缩放权重来自训练，
            # 推理时只读取，不再更新；eps 防止均方根过小时除零。
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            # K 与 Q 分别有自己的 RMSNorm 权重，因为二者承担不同作用。
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # RoPE（Rotary Position Embedding）在每个 head 内把相邻特征两两组成
        # 二维向量，并按 token 位置旋转不同角度。位置 m 的 Q 与位置 n 的 K
        # 点积中，旋转矩阵满足 R(m)^T R(n)=R(n-m)，所以结果依赖相对位置
        # n-m。V 不参与 Q·K 匹配，因此不旋转。max_position 本例为 32768；
        # rope_parameters 含 rope_theta=1,000,000 等频率参数。
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        # attn_cls 中 cls 是 class（类）的缩写。先选择“用哪个注意力类”，再在
        # 下一段统一实例化，避免为两种类型重复写一整套构造参数。
        attn_cls = (
            # is_causal=False 的 embedding 模型需要每个 token 双向看完整输入。
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            # 标准生成模型使用因果注意力，不能看未来 token。
            else Attention
        )
        # 实例化 vLLM 注意力封装。它会按平台选择后端，读取调度器准备的元数据，
        # 隔离混合批次中的各请求，并负责 causal mask、分页 KV Cache 和 GQA。
        self.attn = attn_cls(
            # 当前 TP rank 的 Q head 数；本例 TP=1 是 14。
            self.num_heads,
            # 每个 head 的维度 64。
            self.head_dim,
            # 点积缩放 1/sqrt(64)=1/8。
            self.scaling,
            # 当前 rank 的 KV head 数；本例是 2。
            num_kv_heads=self.num_kv_heads,
            # KV Cache 的块大小、数据类型、容量等设置。
            cache_config=cache_config,
            # 注意力/KV Cache 若量化，需要从这里取得相应配置。
            quant_config=quant_config,
            # decoder 或 encoder-only，决定可见范围和后端行为。
            attn_type=attn_type,
            # 唯一模块路径，便于注册本层及匹配缓存/量化参数。
            prefix=f"{prefix}.attn",
            # 只有启用 Dual Chunk Attention 时才额外传入层号和该配置。
            **{
                # 从如 model.layers.3.self_attn 中提取层号 3。
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            # 条件表达式：有配置时展开上面的字典，否则展开空字典。
            if dual_chunk_attention_config
            else {},
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """计算一次注意力，输入输出形状均为 ``[num_tokens, 896]``。

        ``positions`` 是本轮每个 token 的逻辑位置。Prefill 4 个 token 时可为
        ``[0,1,2,3]``；已有 100 个 token 后处理新 token 时为 ``[100]``。
        注意此时输入 token 已经存在：模型是在用它预测“再下一个”token。
        """
        # 融合矩阵乘法生成 Q/K/V；第二个返回值是可选的独立 bias，这里结果
        # 已包含需要的 bias，额外返回项不使用。TP=1 输出 [tokens,1152]。
        qkv, _ = self.qkv_proj(hidden_states)
        # 沿最后一维切成 Q、K、V。本例切分宽度 [896,128,128]：
        # q=[tokens,896]，k=[tokens,128]，v=[tokens,128]。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 若变体启用 QK-Norm，在 RoPE 前按 head 归一化。本例 qk_norm=False，
        # 因而整段跳过。RMSNorm 与 Softmax 不同：前者调节向量尺度，后者把
        # 注意力分数变为总和为 1 的权重。
        if self.qk_norm:
            # vLLM 把 head 压平了，先取本轮扁平 token 数。例如形状 [4,896]
            # 的 q，其 total_tokens=4。
            total_tokens = q.shape[0]
            # view 只改变观察张量的形状，通常不复制数据：
            # q [tokens,896] -> [tokens,14,64]，使 RMSNorm 只看最后 64 维。
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            # k [tokens,128] -> [tokens,2,64]。
            k = k.view(total_tokens, self.num_kv_heads, self.head_dim)

            # 分别对每个 Q/K head 的 64 维计算 RMSNorm。
            q = self.q_norm(q)
            k = self.k_norm(k)

            # 注意力后端期望 head 再次压平，故恢复原形状；元素顺序不变。
            q = q.view(total_tokens, self.q_size)
            k = k.view(total_tokens, self.kv_size)

        # 根据 positions 旋转 Q 和 K，将位置信息写入两者；形状保持不变。
        q, k = self.rotary_emb(positions, q, k)
        # 概念公式是 softmax((Q @ K^T)/sqrt(64) + mask) @ V。
        # Prefill 会同时处理当前 chunk 的多个 Q/K/V；Decode 会把新 K/V 写入
        # KV Cache，并让新 Q 读取自己可见的历史缓存。请求之间由 slot mapping、
        # block table 和序列边界隔离，不会互相参加注意力。
        attn_output = self.attn(q, k, v)
        # 将 14 个 Q head 的结果拼接/汇合，再由 o_proj 映射回 896 维。
        output, _ = self.o_proj(attn_output)
        # 这里只返回注意力分支结果；与 residual 相加在 decoder layer 中完成。
        return output


class Qwen2DecoderLayer(nn.Module):
    """一个完整的 Qwen2 Transformer Decoder block。

    可以把数据流写成：

    ``x -> RMSNorm -> Attention -> 残差相加 -> RMSNorm -> MLP -> 残差相加``

    这里最后一次“MLP 输出 + residual”会推迟到下一层入口（或最终 norm）通过
    融合 RMSNorm 完成，所以返回 ``(hidden_states, residual)`` 两个张量。这样
    能少写回一次大张量，是性能优化，不是漏掉残差连接。
    """

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """根据模型配置创建本 block 的注意力、MLP 和两个 RMSNorm。"""
        super().__init__()
        # 本例 hidden_size=896，表示每个 token 在所有 24 层中都由 896 个数表示。
        self.hidden_size = config.hidden_size
        # 某些旧配置可能缺 rope_theta；只在缺少时补默认 1,000,000，不覆盖
        # checkpoint 已明确给出的值。本例配置本身就是 1,000,000。
        set_default_rope_theta(config, default_theta=1000000)
        # getattr(obj,name,default) 在字段不存在时返回默认值，兼容普通 Qwen2 与
        # 支持 Dual Chunk Attention 的派生模型。本例没有该字段，所以为 None。
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # 标准 Qwen2 是 decoder-only 因果语言模型：位置 i 只能看 0..i，不能
        # 偷看未来。某些 Qwen2 embedding 模型会设 is_causal=False，让所有
        # token 双向看完整句子，例如 Alibaba-NLP/gte-Qwen2-7B-instruct。
        if getattr(config, "is_causal", True):
            # 当前 Qwen2-0.5B-Instruct 没显式关闭 causal，所以进入此分支。
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        # QK-Norm 是部分兼容模型的扩展；本例配置没有 qk_norm，默认 False。
        qk_norm = getattr(config, "qk_norm", False)

        # 创建本层自注意力。prefix 会像 model.layers.0.self_attn 一样标识层号。
        self.self_attn = Qwen2Attention(
            # 每个 token 输入宽度 896。
            hidden_size=self.hidden_size,
            # 14 个 Q head。
            num_heads=config.num_attention_heads,
            # 最大 RoPE 位置 32768。
            max_position=config.max_position_embeddings,
            # 2 个 KV head，形成 14:2 的 GQA。
            num_kv_heads=config.num_key_value_heads,
            # KV Cache 的容量、块结构和 dtype。
            cache_config=cache_config,
            # 可选量化方案。
            quant_config=quant_config,
            # theta、缩放类型等 RoPE 参数。
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            # 因果或双向注意力类型。
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qk_norm=qk_norm,
            # RMSNorm 防除零常数，本例 1e-6；只在 qk_norm=True 时使用。
            rms_norm_eps=config.rms_norm_eps,
        )
        # 创建逐 token MLP：896 -> 两路 4864 -> 门控相乘 -> 896。
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # Pre-Norm 架构在注意力前归一化。RMSNorm 有 896 个训练所得的缩放
        # 权重；推理时这些权重保持不变。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 在 MLP 前，把注意力输出加回 residual 后再进行第二次 RMSNorm。
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行一个 block，并返回未相加的分支输出和当前残差流。

        ``positions`` 与当前调度 token 对齐；``hidden_states`` 和 ``residual``
        通常都是 ``[num_tokens,896]``。第一层入口的 residual 为 None。
        """
        # 第一部分：自注意力前的 residual/RMSNorm。
        if residual is None:
            # 只会发生在第一层入口：保留原始 embedding 作为残差主干。
            residual = hidden_states
            # 另取归一化结果送进注意力。概念上稍后形成
            # embedding + Attention(RMSNorm(embedding))。
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # 后续层入口时，上层返回的 hidden_states 是“上一层 MLP 分支输出”，
            # residual 是“进入上一层 MLP 前的主干”。融合 RMSNorm 同时完成：
            # 1) summed = hidden_states + residual；
            # 2) residual = summed（保留未经归一化的主干）；
            # 3) hidden_states = RMSNorm(summed)（送入本层注意力）。
            # 因而不是“把 residual 错误地改成归一化值”。第二个返回值仍是原始
            # 相加结果，只有第一个返回值经过归一化。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 对当前所有调度 token 计算注意力，输出仍为 [num_tokens,896]。
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # 第二部分：把 Attention 分支加回主干，并归一化后送入 MLP。
        # 返回的新 residual = 旧 residual + attention_output；hidden_states 则是
        # RMSNorm(new residual)。这仍是融合加法+归一化操作。
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # MLP 对每个 token 独立做 896 -> 4864 -> 896，暂不与 residual 相加。
        hidden_states = self.mlp(hidden_states)
        # 下一层入口会完成本层 MLP 输出 + residual；若这是最后一层，则模型
        # 最后的 self.norm 会完成该相加。因此残差从未丢失。
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        # input_ids 的第 0 维是动态 token 数，用符号 b 标记。
        "input_ids": {0: "b"},
        # 普通 Qwen2 的 positions 是 [num_tokens]；Qwen2-VL 使用 mRoPE 时可以
        # 是 [3,num_tokens]，所以统一把最后一维 -1 标成同一个动态符号 b。
        "positions": {-1: "b"},
        # PP 中间张量和外部 embedding 的 token 维也与 b 相同。
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    }
)
class Qwen2Model(nn.Module, EagleModelMixin):
    """Qwen2 主干：Embedding、若干 DecoderLayer、最终 RMSNorm。

    ``@support_torch_compile`` 告诉 PyTorch token 数是动态的，从而尽量复用
    编译图。学习源码和打逐行断点时应使用 ``--enforce-eager``，否则首次启动
    可能只在 Dynamo 跟踪阶段经过 Python，之后请求直接运行编译后的图。

    ``EagleModelMixin`` 增加投机解码需要的辅助隐藏状态收集能力；普通请求
    通常不会产生额外 ``aux_hidden_states``。
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer,
    ):
        """依据 VllmConfig 创建当前 PP rank 应持有的主干模块。"""
        super().__init__()

        # 多模态配置可能同时含视觉和文本子配置；get_text_config() 取得 Qwen2
        # 文本部分。本例 config 对应本地 config.json。
        config = vllm_config.model_config.hf_config.get_text_config()
        # KV Cache 配置由引擎根据可用内存、块大小等准备。
        cache_config = vllm_config.cache_config
        # 量化配置可能为 None，也可能描述 GPTQ/FP8 等格式。
        quant_config = vllm_config.quant_config

        # interleaved/sliding-window 模型可能规定一部分层使用窗口注意力、其余层
        # 使用全注意力。当前实现只接受“所有层采用同样窗口范围”的情况。
        # 本地配置 use_sliding_window=False，通常不会触发该检查。
        if is_interleaved(vllm_config.model_config.hf_text_config):
            # 若 max_window_layers 少于总层数，当前 vLLM 实现会明确报错。
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                )
            )

        # 保存 HuggingFace 配置，供权重绑定、加载器和其他组件读取。
        self.config = config
        self.quant_config = quant_config
        # 词表共 151936 个 token id，合法范围通常是 0..151935。
        self.vocab_size = config.vocab_size

        # PP（Pipeline Parallel）按层切模型。只有第一段需要把 token id 查成
        # embedding；若首尾不是同一 rank 但输入输出权重绑定，最后一段也必须
        # 持有这份 embedding，才能把它当 lm_head 使用。
        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            # 逻辑 embedding 矩阵形状 [151936,896]。给定 input_ids=[10,25]，
            # 查第 10、25 行后得到 [2,896]。VocabParallelEmbedding 在 TP>1
            # 时沿词表行切分，但会组合出正确的完整隐藏向量。
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            # 中间 PP rank 不使用 embedding；占位层避免无意义分配大矩阵，同时
            # 保持所有 rank 的模型对象具有相同属性结构。
            self.embed_tokens = PPMissingLayer()

        # make_layers 创建总计 24 个逻辑 DecoderLayer，并按 PP 决定本 rank 的
        # start_layer/end_layer。PP=1 时 start=0、end=24，当前进程持有全部层。
        self.start_layer, self.end_layer, self.layers = make_layers(
            # 总层数，本例 24。
            config.num_hidden_layers,
            # lambda 是一个临时建层函数；make_layers 每次给出具体 prefix，
            # 这里据此创建一个 Qwen2DecoderLayer（或调用者传入的兼容层类型）。
            lambda prefix: decoder_layer_type(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        # PP 框架有时要先创建接收缓冲区。这个工厂说明跨 rank 需要传两个
        # [num_tokens,896] 张量：hidden_states 和 residual。
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        # 只有最后一个 PP rank 拥有最终 RMSNorm，因为只有它收到最后一层输出。
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将整数 token id 查表为 896 维浮点隐藏向量。"""
        # 这是 embedding lookup，不是把 token id 这个整数直接当作模型数值。
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """运行当前 PP rank 所负责的 Qwen2 主干层。

        Prefill/Decode 都走这里，区别不在 Python ``if``，而在本轮输入 token 数
        和注意力元数据。Prefill 4 个 token 时 input_ids/positions 可为 [4]；
        Decode 时通常为 [1]。混合批处理会把多个请求的本轮 token 压在一起，
        Attention 后端利用请求边界和各自的 KV block table 保持逻辑隔离。
        """
        # 第一 PP rank 从 token id 或调用者已经准备好的 embedding 开始。
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                # 多模态/提示 embedding 等场景可直接提供 [tokens,896]，跳过查表。
                hidden_states = inputs_embeds
            else:
                # 常规文本路径：input_ids [tokens] -> hidden_states [tokens,896]。
                # 类型标注允许 input_ids=None 是因为另一路可传 inputs_embeds；
                # 正常调用必须保证二者至少有一个有效。
                hidden_states = self.embed_input_ids(input_ids)
            # 第一层尚未建立分离的残差流，由 DecoderLayer 保存 embedding。
            residual = None
        else:
            # 非第一 PP rank 没有 token embedding，必须接收上一 rank 的输出。
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # EAGLE 投机解码可能要求保存指定层的隐藏状态；普通模式返回空列表。
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        # islice 只遍历当前 PP rank 的 [start_layer,end_layer) 范围。
        # PP=1 时就是 0..23 共 24 层；enumerate 的 idx 从 0 开始计本地层。
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            # 每层保持 token 数和隐藏宽度不变，并显式传递融合残差流。
            hidden_states, residual = layer(positions, hidden_states, residual)
            # 若 EAGLE 配置指定此深度，则把组合后的隐藏状态加入辅助结果。
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        # 非最后 PP rank 不能做最终 norm/lm_head；把两个张量交给下一段流水线。
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        # 最后一层返回的 MLP 分支尚未加回 residual。融合 RMSNorm 先完成相加，
        # 再归一化，得到最终 [num_tokens,896]。第二返回值不再需要，故用 _。
        hidden_states, _ = self.norm(hidden_states, residual)

        # 投机解码请求可能同时需要最终状态和若干中间层状态。
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states

        # 普通生成只返回最终隐藏状态；此处还不是 logits，也没有采样 token。
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """把 HuggingFace Qwen2 checkpoint 权重加载进 vLLM 主干模型。

        HuggingFace 分开保存 ``q_proj/k_proj/v_proj`` 和
        ``gate_proj/up_proj``，而 vLLM 为高效推理把它们分别融合成
        ``qkv_proj`` 与 ``gate_up_proj``。因此不能只按同名参数直接复制，必须
        先改名，再由各并行层的 ``weight_loader`` 放到融合参数的正确分片。

        返回值是实际处理过的参数名集合，框架可据此检查漏载或重复加载。
        """
        # 每个三元组含：vLLM 融合参数名、checkpoint 原参数名、融合槽位编号。
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # q_proj 权重放进 qkv_proj 的 q 槽。
            ("qkv_proj", "q_proj", "q"),
            # k_proj 权重放进 qkv_proj 的 k 槽。
            ("qkv_proj", "k_proj", "k"),
            # v_proj 权重放进 qkv_proj 的 v 槽。
            ("qkv_proj", "v_proj", "v"),
            # gate_proj 放进 gate_up_proj 的第 0 槽。
            ("gate_up_proj", "gate_proj", 0),
            # up_proj 放进 gate_up_proj 的第 1 槽。
            ("gate_up_proj", "up_proj", 1),
        ]
        # 建立“完整参数名 -> PyTorch Parameter”字典。remove_duplicate=False
        # 很重要：权重绑定时同一 Parameter 可能通过不同模块路径被引用。
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        # 记录已加载名称。set 自动去重，最终返回给上层做完整性检查。
        loaded_params: set[str] = set()
        # weights 是惰性迭代器；逐个处理可避免同时在内存放全部 checkpoint。
        for name, loaded_weight in weights:
            # RoPE 的 inv_freq 在 vLLM 中通常由配置动态生成，不需要加载
            # HuggingFace checkpoint 里可能保存的同名缓冲区。
            if "rotary_emb.inv_freq" in name:
                continue
            # 量化 KV Cache 可能在 checkpoint 中带缩放系数。海象运算符 :=
            # 一边调用 get_cache_scale(name)，一边把结果赋给 scale_name；仅当
            # quant_config 存在且返回了有效名称时进入分支。
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # 找到 vLLM 内部对应的 KV Cache scale 参数。
                param = params_dict[scale_name]
                # 某些并行/量化 Parameter 自带定制 loader；普通参数回退到默认
                # 复制函数。getattr 的第三个参数就是属性不存在时的默认值。
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # scale 有时保存为标量 []，有时保存成含冗余首维的 [1]；后一种
                # 取第 0 项，统一成 loader 期望的标量。
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                # 把缩放值写入目标参数。
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                # scale 已处理完，不再走普通 QKV/MLP 权重路径。
                continue
            # 尝试判断当前 checkpoint 参数是否属于某个需要融合的投影。
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # 例如名称不含 q_proj，就继续尝试 k_proj、v_proj 等规则。
                if weight_name not in name:
                    continue
                # 例：model.layers.0.self_attn.q_proj.weight
                # 变为 model.layers.0.self_attn.qkv_proj.weight。
                name = name.replace(weight_name, param_name)
                # 有些 GPTQ checkpoint 保存模型实现并不使用的额外 bias；若目标
                # 字典不存在这个参数就跳过，避免 KeyError。
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # PP rank 只拥有部分层；属于其他 rank 的参数在本 rank 是占位层，
                # 不应加载，但其他 rank 会加载自己的部分。
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # 不同 checkpoint/后端对 FP8 KV scale 的命名可能不同，先
                    # 映射到本模型真正注册的参数名。
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        # 返回 None 表示本模型无需或无法对应这项 scale。
                        continue
                # 取得融合层的目标 Parameter。
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # 普通 loader 不认识 shard_id，直接加载完整张量。
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    # 融合并行层的定制 loader 根据 q/k/v 或 0/1，把权重切到正确
                    # TP 分片并写入融合矩阵相应区域。
                    weight_loader(param, loaded_weight, shard_id)
                # 当前规则已经匹配并加载，退出内层 for，避免再匹配其他规则。
                break
            else:
                # Python 的 for...else 中，else 只在循环没有 break 时运行。
                # 因而这里处理“不属于任何融合映射”的普通参数，如 RMSNorm、
                # embedding、o_proj 和 down_proj；它不是 if 的 else。
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # 普通路径也可能遇到 FP8 scale，同样先兼容参数名。
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                # 跳过属于其他 PP rank 的参数。
                if is_pp_missing_parameter(name, self):
                    continue
                # 某些兼容 checkpoint 含当前模型不使用的额外张量，安全跳过。
                if name not in params_dict:
                    continue
                # 取得同名参数及它可能携带的定制 loader。
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # 普通参数无需融合槽位编号，直接加载。
                weight_loader(param, loaded_weight)
            # 注意：若上面某个 continue 跳到外层循环，此行不会执行；成功处理
            # 的参数名称才加入集合。
            loaded_params.add(name)
        # 把加载结果交给调用方检查。
        return loaded_params


class Qwen2ForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    """完整的 Qwen2 因果语言模型（Causal Language Model）。

    ``Qwen2Model`` 只产生 896 维隐藏状态；本类增加 ``lm_head``（language
    model head，语言模型输出头）把选中的隐藏状态映射到 151936 维词表 logits。
    logits 是每个候选 token 的未归一化分数，不是概率；采样器之后才会应用
    temperature、top-k/top-p、Softmax/采样等步骤得到 token id。

    SupportsLoRA、SupportsPP、SupportsEagle/Eagle3 是能力标记接口，分别告诉
    vLLM 本模型支持低秩适配器、流水线并行和 EAGLE 投机解码。
    """

    # 声明 checkpoint 的多个独立模块可打包到哪个 vLLM 融合模块。这也会被
    # LoRA 等通用逻辑读取，不只服务于下面的 load_weights。
    packed_modules_mapping = {
        # q_proj、k_proj、v_proj 合并成一个 qkv_proj。
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        # MLP 的 gate_proj、up_proj 合并成 gate_up_proj。
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """创建主干、输出头和 logits 处理器。"""
        super().__init__()
        # 取得文本模型配置；本例含 24 层、hidden_size=896 等实际参数。
        config = vllm_config.model_config.hf_config.get_text_config()
        # 保存可选量化方式，传给线性层和加载器。
        quant_config = vllm_config.quant_config

        self.config = config

        self.quant_config = quant_config
        # 创建 Qwen2 主干。maybe_prefix("", "model") 得到 "model"；若外层
        # 已有前缀，则安全拼成如 "target_model.model"，避免多余的点号。
        self.model = Qwen2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # 只有最后一个 PP rank 得到最终隐藏状态，所以只有它需要 lm_head。
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                # 本例 tie_word_embeddings=true：复用输入 embedding 矩阵作为输出
                # 投影权重。隐藏向量 [tokens,896] 与矩阵转置相乘，逻辑上得到
                # [tokens,151936]；少存一份约 151936*896 个参数。
                self.lm_head = self.model.embed_tokens
            else:
                # 未绑定时创建独立的 language-model head。TP>1 时它沿词表维
                # 分片保存/计算，各 rank 最终协作形成所需 logits。
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            # 非末 PP rank 不计算 logits，用占位层保持统一属性。
            self.lm_head = PPMissingLayer()

        # LogitsProcessor 不只是一个普通矩阵乘法包装：它还根据本轮采样元数据
        # 选择需要 logits 的 token 位置，并处理 TP 词表分片。本例词表 151936。
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # 把主干创建 PP 中间张量的工厂暴露给外层引擎。
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """把 token id 查表成隐藏向量，直接复用主干实现。"""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """执行主干前向传播，但此方法本身不计算 logits 或采样。

        Prefill 和 Decode 都会进入这里。以 prompt 有 4 个 token、max_tokens=2
        为例：Prefill 用 4 个输入 token 产生上下文化隐藏状态，并从最后一个
        有效位置的 logits 采样第 1 个输出 token；下一轮 Decode 再把“第 1 个
        输出 token”作为新输入，利用历史 KV Cache 预测第 2 个输出 token。
        """
        # nn.Module.__call__ 最终调用 Qwen2Model.forward。使用位置参数只是这里
        # 的简写，顺序与其签名 input_ids/positions/intermediate/embeds 一致。
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        # 返回最终隐藏状态（末 PP rank）或 PP 中间张量（非末 rank）。
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """把需要预测的位置的隐藏状态映射为词表 logits。

        输入不一定包含 Prefill 的所有 token。vLLM 通常只为真正需要采样或返回
        prompt logprobs 的位置选择隐藏状态。例如普通 4-token Prefill 虽然主干
        计算 [4,896]，通常只需最后位置预测第一个输出，结果可为 [1,151936]，
        而不是必然生成 [4,151936]。Decode 一个 token 时通常也是
        [1,896] -> [1,151936]。混合批次则首维是本轮被选中的采样位置总数。
        """
        # lm_head 执行隐藏空间到词表空间的投影；LogitsProcessor 负责位置选择、
        # TP 汇合等。这里仍未 Softmax，也没有决定最终 token。
        logits = self.logits_processor(self.lm_head, hidden_states)
        # 非负责 logits 的并行 rank 可能返回 None；正常末 rank 返回浮点张量。
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """通过通用加载器加载完整 CausalLM 的 checkpoint。"""
        # AutoWeightsLoader 会递归调用 Qwen2Model.load_weights，并理解上面的
        # packed_modules_mapping、TP/PP 分片和模块前缀。
        loader = AutoWeightsLoader(
            self,
            # 本例输入 embedding 与 lm_head 绑定，同一参数只需加载一次；若
            # checkpoint 还带独立 lm_head.*，跳过它可避免重复/冲突。
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        # 执行加载并返回成功加载的参数名集合。
        return loader.load_weights(weights)
