"""Create a compact Markdown table from vLLM bench serve JSON results."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    result_dir = Path(sys.argv[1])
    rows = []
    paths = sorted(
        result_dir.glob("concurrency-*.json"),
        key=lambda path: int(path.stem.rsplit("-", 1)[1]),
    )
    for path in paths:
        concurrency = int(path.stem.rsplit("-", 1)[1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                concurrency,
                payload["completed"],
                payload["failed"],
                payload["duration"],
                payload["request_throughput"],
                payload["output_throughput"],
                payload["total_token_throughput"],
                payload["mean_ttft_ms"],
                payload["mean_tpot_ms"],
            )
        )

    lines = [
        "# PyTorch FlashAttention v1 MPS 端到端吞吐测试",
        "",
        "测试口径：16 个请求，每请求 1000 输入 token、10 输出 token。",
        "",
        "| 并发 | 成功 | 失败 | 耗时(s) | 请求/s | 输出 token/s | "
        "总 token/s | 平均 TTFT(ms) | 平均 TPOT(ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.2f} | "
            f"{row[4]:.4f} | {row[5]:.4f} | {row[6]:.2f} | "
            f"{row[7]:.2f} | {row[8]:.2f} |"
        )
    lines.extend(
        [
            "",
            "服务端启用了 `STUDY_USE_PYTORCH_FLASH_ATTENTION_V1=1`，"
            "并禁用了自定义 CUDA Attention。",
            "",
        ]
    )
    output = result_dir / "summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
