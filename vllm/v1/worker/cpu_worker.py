# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Must be imported firstly
import vllm.v1.worker.cpu.shm  # noqa # isort: skip

import math
import os
import sys
from typing import Any

import psutil
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.utils.cpu_resource_utils import (
    get_allowed_cpu_list,
    get_memory_node_info,
    get_visible_memory_node,
)
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.worker_base import CompilationTimes

logger = init_logger(__name__)


class CPUWorker(Worker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # TODO: use numactl for process setup
        # TODO: optimize for `interleaved` policy
        # Bind memory node
        allowed_memory_nodes = get_visible_memory_node()
        allowed_cpu_list = get_allowed_cpu_list()
        cpu_core = allowed_cpu_list[0]

        # TODO: some CI hosts are not correctly set, change to assertion
        # after fix
        if cpu_core.numa_node not in allowed_memory_nodes:
            logger.warning(
                "Node %s is not in available memory nodes %s.",
                cpu_core.numa_node,
                allowed_memory_nodes,
            )

        torch.ops._C.init_cpu_memory_env([cpu_core.numa_node])

        memory_status = get_memory_node_info(cpu_core.numa_node)
        memory_fraction = vllm_config.cache_config.gpu_memory_utilization
        self.requested_cpu_memory = math.ceil(
            memory_status.total_memory * memory_fraction
        )
        available_memory = memory_status.available_memory

        if (
            vllm_config.cache_config.kv_cache_memory_bytes is None
            and self.requested_cpu_memory > available_memory
        ):
            raise ValueError(
                f"Available memory on node {cpu_core.numa_node} "
                f"({format_gib(available_memory)}/"
                f"{format_gib(memory_status.total_memory)} GiB) on startup "
                f"is less than desired CPU memory utilization "
                f"({vllm_config.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_cpu_memory)} GiB). "
                "On the CPU backend, the `--gpu-memory-utilization` flag "
                "controls the fraction of CPU memory reserved (despite its "
                "name). To resolve: decrease `--gpu-memory-utilization` "
                "(e.g. `--gpu-memory-utilization 0.5`) "
                "or reduce CPU memory used by other processes."
            )

        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.parallel_config.disable_custom_all_reduce = True

        # Torch profiler. Enabled and configured through profiler_config.
        self.profiler: Any | None = None
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU"],
            )

    def init_device(self):
        self.device = torch.device("cpu")

        # Check whether critical libraries are loaded
        def check_preloaded_libs(name: str):
            ld_preload_list = os.environ.get("LD_PRELOAD", "")
            if name not in ld_preload_list:
                logger.warning(
                    "%s is not found in LD_PRELOAD. "
                    "For best performance, please follow the section "
                    "`set LD_PRELOAD` in "
                    "https://docs.vllm.ai/en/latest/getting_started/installation/cpu/ "
                    "to setup required pre-loaded libraries.",
                    name,
                )

        if sys.platform.startswith("linux"):
            check_preloaded_libs("libtcmalloc")
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                check_preloaded_libs("libiomp")

        def skip_set_num_threads(x: int):
            logger.warning(
                "CPU backend doesn't allow to use "
                "`torch.set_num_threads` after the thread binding, skip it."
            )

        torch.set_num_threads = skip_set_num_threads

        # Note: unique identifier for creating allreduce shared memory
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]
        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        if self.use_v2_model_runner:
            from vllm.v1.worker.cpu.model_runner import (
                CPUModelRunner as CPUModelRunnerV2,
            )

            self.model_runner: CPUModelRunner = CPUModelRunnerV2(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            self.model_runner = CPUModelRunner(self.vllm_config, torch.device("cpu"))

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def determine_available_memory(self) -> int:
        """计算当前 CPU Worker 最多能为 KV Cache 分配多少字节。

        函数名中的 ``available_memory`` 不是简单返回“操作系统还剩多少内存”。
        vLLM 需要把本 Worker 的内存预算分成两部分：

        1. 非 KV 内存：模型权重、PyTorch/编译产物、运行缓冲区等；
        2. KV Cache：保存每层历史 token 的 K/V，供 Decode 重用。

        自动模式使用下面的近似公式：

        ``KV Cache 字节数 = Worker 内存预算 - 预热后进程 RSS``

        CPU 后端中，命令行参数虽然叫 ``--gpu-memory-utilization``，实际控制
        CPU Worker 的预算占物理内存的比例。例如机器总内存假设为 16 GiB，
        参数为 0.2，则 ``requested_cpu_memory`` 约为 3.2 GiB；若加载模型并
        预热后 Worker RSS 为 1.5 GiB，就尝试把约 1.7 GiB 留给 KV Cache。
        这只是说明公式的示例，实际数值取决于机器和当时运行的其他程序。

        为什么计算前要“用虚拟输入运行一次模型”？很多开销是惰性产生的：
        仅构造 Python 模型对象时，torch.compile 的图、编译代码和部分运行缓冲
        还不存在。如果先把几乎所有剩余内存都分给 KV Cache，第一次真实请求
        再产生这些开销就可能 OOM。因此先执行一次 profile/warmup forward，
        让这些持久开销出现，再计算 KV Cache。虚拟运行不会处理用户请求；其
        输出会被丢弃。使用 ``--enforce-eager`` 时不会编译，但仍会执行预热，
        所以调试 ``qwen2.py`` 时可能在服务器接受 curl 之前先命中断点。

        返回值单位是 byte（字节），之后 EngineCore 会根据每个 KV block 占用
        的字节数，把它换算为可创建多少个 KV Cache block。
        """
        # 先进行一次最大通用形状的虚拟模型运行。调用路径大致为：
        # determine_available_memory -> warming_up_model -> profile_run
        # -> _dummy_run -> Qwen2ForCausalLM.forward。
        #
        # 对当前 Qwen2-0.5B 来说，这会经过 embedding、24 个 DecoderLayer、
        # 最终 RMSNorm 和虚拟采样路径。它主要用于触发 torch.compile 和持久
        # 缓冲区分配，不是在启动时替用户“提前生成一个 token”。
        self.model_runner.warming_up_model()

        # 取得当前进程被允许使用的 CPU 核心列表。容器、taskset 或 NUMA 绑定
        # 可能使它少于机器全部核心，不能直接假设从 CPU 0 开始。
        allowed_cpu_list = get_allowed_cpu_list()
        # 取第一个允许的核心作为代表，用它找到本 Worker 所绑定的 NUMA 内存
        # 节点。Apple Silicon 通常可理解为统一内存节点；多路 CPU 服务器则可能
        # 有多个 NUMA node，各节点的本地可用内存不同。
        cpu_core = allowed_cpu_list[0]

        # 查询该 NUMA node 的 total/available memory。available_memory 是此刻
        # 操作系统认为还能分配的内存，不等于 vLLM 为 Worker 设定的预算。
        memory_status = get_memory_node_info(cpu_core.numa_node)
        available_memory = memory_status.available_memory
        # 用户可通过 --kv-cache-memory-bytes（或兼容环境配置）明确指定 KV Cache
        # 大小。未指定时值为 None，进入后面的自动计算分支。
        explicit_kv_cache_size = self.cache_config.kv_cache_memory_bytes

        # 先声明结果和日志消息。两个分支都会给它们赋具体值。
        kv_cache_size = None
        msg = None
        # 分支一：用户明确控制 KV Cache 字节数，不再用内存比例自动推算。
        if explicit_kv_cache_size is not None:
            # 即使用户明确指定，也不能超过操作系统此刻实际可用内存，否则后续
            # 分配必然失败，所以在真正申请大块缓存前给出清晰错误。
            if explicit_kv_cache_size > available_memory:
                raise ValueError(
                    # 指明发生问题的 NUMA node。
                    f"Available memory on node {cpu_core.numa_node} "
                    # 同时显示当前可用量和节点总容量，便于判断内存压力。
                    f"({format_gib(available_memory)}/"
                    f"{format_gib(memory_status.total_memory)} GiB) on kv cache"
                    f" allocation is less than requested memory for kv "
                    # 显示用户要求分给 KV Cache 的大小。
                    f"({format_gib(explicit_kv_cache_size)} GiB). "
                    # 给出三种解决方向：调小显式值、调小兼容环境变量，或释放
                    # 其他进程占用的 CPU 内存。
                    "Decrease --kv-cache-memory-bytes, VLLM_CPU_KVCACHE_SPACE, "
                    "or reduce CPU memory used by other processes."
                )
            # 检查通过，直接采用用户指定值。显式模式不受
            # --gpu-memory-utilization 推导出的 Worker 预算约束。
            kv_cache_size = explicit_kv_cache_size
            # 构造一条人类可读的 GiB 日志；format_gib 只负责单位转换，内部
            # kv_cache_size 仍始终以整数 byte 保存。
            msg = (
                f"Explicitly set ({format_gib(kv_cache_size)}/"
                f"{format_gib(memory_status.total_memory)}) GiB for KV cache "
                f"on node {cpu_core.numa_node}."
            )
        else:
            # 分支二：自动计算。RSS（Resident Set Size，常驻内存集）是当前
            # Worker 真正驻留在物理内存中的总量，包含权重、Python/PyTorch、
            # 预热后留下的编译产物和持久缓冲区等非 KV 开销。
            consumed_memory = psutil.Process(os.getpid()).memory_info().rss
            # requested_cpu_memory 在 __init__ 中按下面公式得到：
            # ceil(NUMA node 总内存 * gpu_memory_utilization)。从预算扣除当前
            # Worker RSS，剩余部分就是计划留给 KV Cache 的空间。
            requested_memory_for_kv = int(self.requested_cpu_memory - consumed_memory)
            # 两类结果都不能安全执行：
            # 1. <= 0：模型/运行时本身已经用完甚至超过 Worker 内存预算；
            # 2. > available：预算算出的缓存虽为正，但系统此刻没有这么多空闲
            #    内存，可能被其他应用占用。
            if (
                requested_memory_for_kv <= 0
                or requested_memory_for_kv > available_memory
            ):
                raise ValueError(
                    f"Available memory on node {cpu_core.numa_node} "
                    f"({format_gib(available_memory)}/"
                    f"{format_gib(memory_status.total_memory)} GiB) on kv cache"
                    f" allocation is less than requested memory for kv "
                    # 同时打印自动算出的 KV 需求和整个 Worker 预算，方便判断是
                    # 内存比例过低，还是其他进程占用了太多内存。
                    f"({format_gib(requested_memory_for_kv)}/"
                    f"{format_gib(self.requested_cpu_memory)} GiB). "
                    "Reduce CPU memory used by other processes."
                )
            # 校验通过，把自动计算的正整数作为 KV Cache 字节预算。
            kv_cache_size = requested_memory_for_kv
            # 日志会列出：KV Cache/节点总内存、Worker 总预算，以及预热后已经
            # 被非 KV 用途占用的 RSS，便于核对上面的减法公式。
            msg = (
                f"Auto set ({format_gib(kv_cache_size)}/"
                f"{format_gib(memory_status.total_memory)}) GiB for KV cache "
                f"on node {cpu_core.numa_node}, with "
                f"{format_gib(self.requested_cpu_memory)} GiB requested memory"
                f" for the worker. {format_gib(consumed_memory)} GiB"
                f" memory was consumed by non-kv usages."
            )

        # 两个分支都只记录最终决策一次，避免启动日志重复。
        logger.info(msg)

        # 这里只返回“可以分多少字节”，尚未真的创建 KV Cache。调用方随后根据
        # 每层 K/V head 数、head_dim、dtype 和 block_size 计算 block 数并分配。
        return kv_cache_size

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # Note: the model has been compiled in determine_available_memory(),
        # Only compile here for models without kv cache
        if len(self.model_runner.kv_caches) == 0:
            self.model_runner.warming_up_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
