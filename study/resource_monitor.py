#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import signal
import subprocess
import threading
import time
from pathlib import Path


STOP_EVENT = threading.Event()


def stop_monitoring(*_: object) -> None:
    STOP_EVENT.set()


def read_cpu_times() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + values[4]
    return sum(values), idle


def read_memory() -> tuple[float, float, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", maxsplit=1)
        values[key] = int(value.split()[0])
    total_mib = values["MemTotal"] / 1024
    used_mib = (values["MemTotal"] - values["MemAvailable"]) / 1024
    return used_mib, total_mib, used_mib / total_mib * 100


def read_gpu() -> tuple[float, float, float, float, float, float]:
    query = (
        "utilization.gpu,memory.used,memory.total,"
        "power.draw,temperature.gpu"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
    gpu_util, memory_used, memory_total, power, temperature = values
    return (
        gpu_util,
        memory_used,
        memory_total,
        memory_used / memory_total * 100,
        power,
        temperature,
    )


def monitor(output: Path, interval: float, duration: float | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + duration if duration is not None else None
    previous_total, previous_idle = read_cpu_times()
    fields = [
        "timestamp",
        "gpu_util_pct",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_memory_util_pct",
        "cpu_util_pct",
        "memory_used_mib",
        "memory_total_mib",
        "memory_util_pct",
        "gpu_power_w",
        "gpu_temperature_c",
    ]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        while not STOP_EVENT.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(interval)
            total, idle = read_cpu_times()
            total_delta = total - previous_total
            idle_delta = idle - previous_idle
            cpu_util = 100 * (total_delta - idle_delta) / total_delta
            previous_total, previous_idle = total, idle
            memory_used, memory_total, memory_util = read_memory()
            (
                gpu_util,
                gpu_memory_used,
                gpu_memory_total,
                gpu_memory_util,
                gpu_power,
                gpu_temperature,
            ) = read_gpu()
            writer.writerow(
                {
                    "timestamp": time.time(),
                    "gpu_util_pct": gpu_util,
                    "gpu_memory_used_mib": gpu_memory_used,
                    "gpu_memory_total_mib": gpu_memory_total,
                    "gpu_memory_util_pct": gpu_memory_util,
                    "cpu_util_pct": cpu_util,
                    "memory_used_mib": memory_used,
                    "memory_total_mib": memory_total,
                    "memory_util_pct": memory_util,
                    "gpu_power_w": gpu_power,
                    "gpu_temperature_c": gpu_temperature,
                }
            )
            file.flush()


def summarize(input_dir: Path, output: Path) -> None:
    summary: dict[str, dict[str, float | int]] = {}
    metric_names = [
        "gpu_util_pct",
        "gpu_memory_used_mib",
        "gpu_memory_util_pct",
        "cpu_util_pct",
        "memory_used_mib",
        "memory_util_pct",
        "gpu_power_w",
        "gpu_temperature_c",
    ]
    for csv_path in sorted(input_dir.glob("resources-*.csv")):
        with csv_path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue
        values = {
            metric: [float(row[metric]) for row in rows]
            for metric in metric_names
        }
        result: dict[str, float | int] = {"samples": len(rows)}
        for metric, samples in values.items():
            result[f"mean_{metric}"] = sum(samples) / len(samples)
            result[f"peak_{metric}"] = max(samples)
        summary[csv_path.stem.removeprefix("resources-")] = result
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--output", type=Path, required=True)
    monitor_parser.add_argument("--interval", type=float, default=1.0)
    monitor_parser.add_argument("--duration", type=float)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--input-dir", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "monitor":
        signal.signal(signal.SIGTERM, stop_monitoring)
        signal.signal(signal.SIGINT, stop_monitoring)
        monitor(args.output, args.interval, args.duration)
    else:
        summarize(args.input_dir, args.output)


if __name__ == "__main__":
    main()
