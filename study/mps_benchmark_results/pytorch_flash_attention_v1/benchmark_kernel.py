"""Benchmark the teaching PyTorch FlashAttention v1 implementation on MPS.

This benchmark measures the attention operator itself. It also compares every
scenario with PyTorch SDPA before reporting performance, so a fast but
incorrect result is not accepted silently.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

# Direct execution puts only this benchmark directory on sys.path. Add the
# repository root so the local ``study`` package can be imported consistently.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from study.inference_engine.pytorch_attention.pytorch_flash_attention_v1 import (
    pytorch_flash_attention_v1,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    query_len: int
    kv_len: int
    iterations: int


DEFAULT_SCENARIOS = (
    Scenario("decode-kv-1000", query_len=1, kv_len=1000, iterations=20),
    Scenario("prefill-250", query_len=250, kv_len=250, iterations=5),
    Scenario("prefill-1000", query_len=1000, kv_len=1000, iterations=3),
)


def select_device() -> tuple[torch.device, torch.dtype]:
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(
    scenario: Scenario,
    device: torch.device,
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260811)
    q = torch.randn(
        num_q_heads,
        scenario.query_len,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    k = torch.randn(
        num_kv_heads,
        scenario.kv_len,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    v = torch.randn(
        num_kv_heads,
        scenario.kv_len,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)

    if scenario.query_len == 1:
        mask = torch.ones(
            1,
            scenario.kv_len,
            dtype=torch.bool,
            device=device,
        )
    else:
        # Query i may attend to keys up to its aligned absolute position. This
        # also works when kv_len is larger than query_len because of KV cache.
        past_len = scenario.kv_len - scenario.query_len
        query_positions = torch.arange(
            scenario.query_len,
            device=device,
        ) + past_len
        key_positions = torch.arange(scenario.kv_len, device=device)
        mask = key_positions[None, :] <= query_positions[:, None]

    return q, k, v, mask


def custom_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    return pytorch_flash_attention_v1(q, k, v, mask, block_size)


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    groups = q.shape[0] // k.shape[0]
    dense_k = k.repeat_interleave(groups, dim=0)
    dense_v = v.repeat_interleave(groups, dim=0)
    return F.scaled_dot_product_attention(
        q,
        dense_k,
        dense_v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=q.shape[-1] ** -0.5,
    )


def time_calls(
    callback,
    iterations: int,
    warmups: int,
    device: torch.device,
) -> float:
    for _ in range(warmups):
        callback()
    synchronize(device)

    started = time.perf_counter()
    for _ in range(iterations):
        callback()
    synchronize(device)
    return time.perf_counter() - started


def benchmark_scenario(
    scenario: Scenario,
    device: torch.device,
    dtype: torch.dtype,
    block_size: int,
    warmups: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, object]:
    q, k, v, mask = make_inputs(
        scenario,
        device,
        dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )

    # The teaching implementation prints input shapes. Suppress that output so
    # terminal I/O does not distort the timing loop.
    with contextlib.redirect_stdout(io.StringIO()):
        actual = custom_attention(q, k, v, mask, block_size)
        expected = sdpa_attention(q, k, v, mask)
        synchronize(device)

        custom_seconds = time_calls(
            lambda: custom_attention(q, k, v, mask, block_size),
            scenario.iterations,
            warmups,
            device,
        )
        sdpa_seconds = time_calls(
            lambda: sdpa_attention(q, k, v, mask),
            scenario.iterations,
            warmups,
            device,
        )

    max_abs_error = (actual.float() - expected.float()).abs().max().item()
    tolerance = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    correct = torch.allclose(
        actual.float(),
        expected.float(),
        atol=tolerance,
        rtol=1e-3,
    )
    if not correct:
        raise AssertionError(
            f"{scenario.name} differs from SDPA: max_abs_error="
            f"{max_abs_error:.6g}"
        )

    def metrics(elapsed: float) -> dict[str, float]:
        calls_per_second = scenario.iterations / elapsed
        query_tokens_per_second = (
            scenario.iterations * scenario.query_len / elapsed
        )
        score_elements_per_second = (
            scenario.iterations
            * num_q_heads
            * scenario.query_len
            * scenario.kv_len
            / elapsed
        )
        # QK^T and P@V each cost approximately 2 FLOPs per multiply-add.
        estimated_flops = (
            scenario.iterations
            * 4
            * num_q_heads
            * scenario.query_len
            * scenario.kv_len
            * head_dim
        )
        return {
            "elapsed_seconds": elapsed,
            "mean_latency_ms": elapsed / scenario.iterations * 1000,
            "calls_per_second": calls_per_second,
            "query_tokens_per_second": query_tokens_per_second,
            "score_elements_per_second": score_elements_per_second,
            "estimated_tflops": estimated_flops / elapsed / 1e12,
        }

    custom_metrics = metrics(custom_seconds)
    sdpa_metrics = metrics(sdpa_seconds)
    return {
        **asdict(scenario),
        "block_size": block_size,
        "max_abs_error_vs_sdpa": max_abs_error,
        "correct_vs_sdpa": correct,
        "custom": custom_metrics,
        "sdpa": sdpa_metrics,
        "custom_to_sdpa_latency_ratio": (
            custom_metrics["mean_latency_ms"]
            / sdpa_metrics["mean_latency_ms"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--num-q-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--scenario",
        choices=[scenario.name for scenario in DEFAULT_SCENARIOS],
        action="append",
        help="Run only selected scenarios; may be specified more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("kernel-results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, dtype = select_device()
    selected = [
        scenario
        for scenario in DEFAULT_SCENARIOS
        if args.scenario is None or scenario.name in args.scenario
    ]

    results = []
    for scenario in selected:
        print(f"Running {scenario.name} on {device} ({dtype})...")
        result = benchmark_scenario(
            scenario,
            device,
            dtype,
            args.block_size,
            args.warmups,
            args.num_q_heads,
            args.num_kv_heads,
            args.head_dim,
        )
        results.append(result)
        print(
            f"  custom={result['custom']['query_tokens_per_second']:.2f} "
            f"query token/s, "
            f"SDPA={result['sdpa']['query_tokens_per_second']:.2f}, "
            f"max_error={result['max_abs_error_vs_sdpa']:.3g}"
        )

    payload = {
        "date": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "num_q_heads": args.num_q_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "warmups": args.warmups,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
