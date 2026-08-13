"""CUDA ragged GQA attention operator used by the Qwen2 study demo."""

from pathlib import Path
import os
import sys

import torch

_site_packages = Path(torch.__file__).resolve().parents[1]
_cuda_root = _site_packages / "nvidia" / "cu13"
os.environ.setdefault("CUDA_HOME", str(_cuda_root))
_venv_bin = Path(sys.prefix) / "bin"
os.environ["PATH"] = ":".join(
    (str(_venv_bin), str(_cuda_root / "bin"), os.environ.get("PATH", ""))
)

from torch.utils.cpp_extension import load

_loaded = False


def _load_extension() -> None:
    global _loaded
    if _loaded:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("study CUDA attention requires a CUDA device")
    root = Path(__file__).resolve().parent
    build_directory = root / "build"
    build_directory.mkdir(exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    load(
        name="study_cuda_attention",
        build_directory=str(build_directory),
        sources=[
            str(root / "binding.cpp"),
            str(root / "ragged_gqa_attention.cu"),
            str(root / "cuda_flash_attention_v1.cu"),
            str(root / "cuda_flash_attention_v2.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        extra_ldflags=[str(_cuda_root / "lib" / "libcudart.so.13")],
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )
    _loaded = True


def ragged_gqa_attention(query, key, value, query_start_loc, kv_start_loc,
                         past_lens, q_lens, max_kv_len, scale):
    _load_extension()
    return torch.ops.study_cuda.ragged_gqa_attention(
        query, key, value, query_start_loc, kv_start_loc, past_lens, q_lens,
        max_kv_len, scale)


def cuda_flash_attention_v1(q, k, v, mask, query_start, kv_start,
                            q_block_size, kv_block_size):
    _load_extension()
    return torch.ops.study_cuda.cuda_flash_attention_v1(
        q, k, v, mask, query_start, kv_start, q_block_size, kv_block_size)


def cuda_flash_attention_v2(q, k, v, mask, query_start, kv_start,
                            work_items, partial_start, partial_count,
                            partials_per_head, q_block_size, kv_block_size):
    _load_extension()
    return torch.ops.study_cuda.cuda_flash_attention_v2(
        q, k, v, mask, query_start, kv_start, work_items, partial_start,
        partial_count, partials_per_head, q_block_size, kv_block_size)


__all__ = [
    "cuda_flash_attention_v1",
    "cuda_flash_attention_v2",
    "ragged_gqa_attention",
]
