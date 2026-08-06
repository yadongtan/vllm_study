# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

import vllm.utils.cpu_triton_utils as cpu_tl
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.tracing import instrument
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # avoid calling accelerator APIs for methods inherited from super class
        _set_torch_accelerator_to_noop()

        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        # Note: speculative decoding is now supported on CPU with C++ native impls

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tensors()
        self._postprocess_triton()

    def _postprocess_tensors(self) -> None:
        # Note: replace device tensors with cpu tensors
        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if isinstance(cpu_tensor, torch.Tensor) and isinstance(
                device_tensor, torch.Tensor
            ):
                setattr(obj, device_attr_name, cpu_tensor)

        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                v.gpu = v.cpu

        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        for block_table in self.input_batch.block_table.block_tables:
            for v in vars(block_table).values():
                if isinstance(v, CpuGpuBuffer):
                    v.gpu = v.cpu

    def _postprocess_triton(self) -> None:
        import vllm.v1.worker.block_table

        vllm.v1.worker.block_table._compute_slot_mapping_kernel = (
            cpu_tl.compute_slot_mapping_kernel
        )

        # Speculative decoding fallbacks
        import vllm.v1.sample.rejection_sampler
        import vllm.v1.spec_decode.llm_base_proposer
        import vllm.v1.spec_decode.utils

        vllm.v1.spec_decode.llm_base_proposer.eagle_prepare_inputs_padded_kernel = (
            cpu_tl.eagle_prepare_inputs_padded_kernel
        )
        vllm.v1.spec_decode.llm_base_proposer.eagle_prepare_next_token_padded_kernel = (
            cpu_tl.eagle_prepare_next_token_padded_kernel
        )
        vllm.v1.spec_decode.llm_base_proposer.copy_and_expand_eagle_inputs_kernel = (
            cpu_tl.copy_and_expand_eagle_inputs_kernel
        )
        vllm.v1.spec_decode.utils.eagle_step_slot_mapping_metadata_kernel = (
            cpu_tl.eagle_step_slot_mapping_metadata_kernel
        )
        vllm.v1.sample.rejection_sampler.rejection_greedy_sample_kernel = (
            cpu_tl.rejection_greedy_sample_kernel
        )
        vllm.v1.sample.rejection_sampler.rejection_random_sample_kernel = (
            cpu_tl.rejection_random_sample_kernel
        )
        vllm.v1.sample.rejection_sampler.expand_kernel = cpu_tl.expand_kernel
        vllm.v1.sample.rejection_sampler.sample_recovered_tokens_kernel = (
            cpu_tl.sample_recovered_tokens_kernel
        )

    @instrument(span_name="Loading (CPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        if load_dummy_weights:
            raise ValueError(
                "Loading dummy weights (needed for elastic EP scale-up) "
                "Is not supported by the CPU Model Runner."
            )
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

        if hasattr(self, "drafter"):
            logger.info_once("Loading drafter model...")
            self.drafter.load_model(self.model)

        self._setup_eagle3_aux_hidden_state_outputs()

    def get_model(self) -> nn.Module:
        return self.model

    @instrument(span_name="Warmup (CPU)")
    def warming_up_model(self) -> None:
        """用虚拟输入预热 CPU 模型，并触发需要的编译与惰性初始化。

        ``@instrument`` 是可观测性装饰器：如果启用了 tracing，它会把整个函数
        记录成名为 ``Warmup (CPU)`` 的耗时区间（span），方便分析启动速度；
        它不是 PyTorch 编译装饰器，也不会改变模型的数学结果。

        这里说的“运行一次模型”并不是生成用户输出。``profile_run`` 会创建
        dummy token、dummy positions 和虚拟调度元数据，调用与真实请求相同的
        Qwen2 前向路径，然后丢弃结果。主要目的有两个：

        1. 第一次执行 ``@support_torch_compile`` 模型时，触发 Dynamo 跟踪和
           CPU Inductor 编译，避免首个用户请求承担这段启动延迟；
        2. 让编译产物及持久缓冲区真实占用内存，随后
           ``determine_available_memory`` 才能从 Worker 预算中扣除它们，安全
           决定 KV Cache 大小。

        若启动参数包含 ``--enforce-eager``，vLLM 会关闭 torch.compile；但
        ``profile_run`` 仍会执行，用来预热算子和形成真实内存基线。因此在
        qwen2.py 打断点时，服务器尚未监听 8000 端口就可能先命中一次。
        """
        # 记录开始日志。这里的 compilation 是主要目的，但 eager 模式也会
        # 执行相同预热流程，所以日志文本不会随 --enforce-eager 改变。
        logger.info("Warming up model for the compilation...")
        # 临时设置 CPU Inductor 的全局编译选项。当前 context manager 会在
        # max_autotune=True 时临时开启 freezing，使 MKLDNN/CPPGEMM 能把不变的
        # 模型参数冻结进优化图；退出 with 后恢复原值，避免污染其他模型/任务。
        #
        # 这里只为通用的最大 token 批次形状生成/预热计算路径。真实请求的
        # num_tokens 可以较小，动态形状配置会尽量复用这份编译结果。
        with _set_global_compilation_settings(self.vllm_config):
            # CPUModelRunner 继承 GPUModelRunner.profile_run。尽管定义文件名中有
            # gpu，该方法包含跨设备的通用虚拟前向流程；CPU 的 CUDA Graph、
            # 设备同步等能力已由 CPUModelRunner 属性或 override 禁用/替换。
            self.profile_run()
        # 到这里虚拟前向、虚拟 sampler 和临时对象清理均已完成；持久编译产物
        # 则有意保留，供真实请求复用并计入后续 RSS 内存。
        logger.info("Warming up done.")

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, is_profiling)

        if self.speculative_config:
            if self.speculative_config.use_eagle():
                logger.info("EAGLE drafter KV cache initialized for CPU backend")
            elif self.speculative_config.uses_draft_model():
                logger.info("Draft model KV cache initialized for CPU backend")

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        pass

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        # CPU attention assigns -INF to logits at invalid positions,
        # so stale KV cache data never affects computation.
        pass

    # =========================================================================
    # CPU-safe overrides for speculative decoding methods
    # These methods override GPU-specific implementations that use CUDA streams
    # =========================================================================

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        """CPU-safe version: no async copy needed, tensors already on CPU."""
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return

        num_reqs = draft_token_ids.shape[0]
        if self.draft_token_ids_cpu is not None:
            if not zeros_only:
                self.draft_token_ids_cpu[:num_reqs].copy_(draft_token_ids)
            else:
                self.draft_token_ids_cpu[:num_reqs] = 0

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        """CPU-safe version: no event synchronization needed."""
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        if self.draft_token_ids_cpu is not None:
            return self.draft_token_ids_cpu[: len(req_ids)].tolist(), req_ids
        return [], []

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        """CPU-safe version: direct copy without CUDA streams."""
        if self.valid_sampled_token_count_cpu is None:
            return

        counts = valid_sampled_tokens_count
        counts_cpu = self.valid_sampled_token_count_cpu
        counts_cpu[: counts.shape[0]].copy_(counts)
        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        """CPU-safe version: no event synchronization needed."""
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        if prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        if counts_cpu is None:
            return []
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        """CPU-safe version: direct tolist() without CUDA events."""
        return sampled_token_ids.tolist()


@contextmanager
def _torch_cuda_wrapper():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    cuda_event = torch.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.Event = cuda_event
        torch.cuda.Stream = cuda_stream


@contextmanager
def _set_global_compilation_settings(config: VllmConfig):
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    # Note: The MKLDNN and CPPGEMM backend requires freezing parameters.
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value


def _set_torch_accelerator_to_noop() -> None:
    def noop(*args: Any, **kwargs: Any) -> None:
        pass

    torch.accelerator.synchronize = noop
    torch.accelerator.empty_cache = noop
