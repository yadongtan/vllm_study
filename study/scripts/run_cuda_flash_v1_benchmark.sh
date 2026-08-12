#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

result_stamp=${1:?usage: run_cuda_flash_v1_benchmark.sh YYYYMMDDHHMM}
result_dir="study/cuda_benchmark_results/$result_stamp"
mkdir -p "$result_dir"
concurrencies=(1 4 8)
port=8012
server_log="$result_dir/cuda-flash-v1-server.log"
server_pid=""
monitor_pid=""

cleanup() {
  if [[ -n "$monitor_pid" ]]; then
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_ready() {
  for _ in $(seq 1 240); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:$port/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -100 "$server_log"
      return 1
    fi
    sleep 1
  done
  tail -100 "$server_log"
  return 1
}

.venv/bin/python3 study/scripts/resource_monitor.py monitor \
  --output "$result_dir/resources-baseline-system-before.csv" --duration 5

STUDY_USE_CUDA_FLASH_ATTENTION_V1=1 \
  .venv/bin/python3 -m uvicorn study.openai_server.v2_openai_server:app \
  --host 127.0.0.1 --port "$port" >"$server_log" 2>&1 &
server_pid=$!
wait_ready

.venv/bin/python3 study/scripts/resource_monitor.py monitor \
  --output "$result_dir/resources-baseline-cuda-flash-v1-loaded.csv" \
  --duration 5

for concurrency in "${concurrencies[@]}"; do
  .venv/bin/python3 study/scripts/resource_monitor.py monitor \
    --output "$result_dir/resources-cuda-flash-v1-concurrency-$concurrency.csv" &
  monitor_pid=$!

  .venv/bin/vllm bench serve --backend openai \
    --base-url "http://127.0.0.1:$port" --endpoint /v1/completions \
    --model qwen2-0.5b-instruct-v2 \
    --tokenizer /opt/models/Qwen2-0.5B-Instruct \
    --dataset-name random --random-input-len 1000 --random-output-len 50 \
    --random-range-ratio 0 --num-prompts 128 \
    --max-concurrency "$concurrency" --request-rate inf --ignore-eos \
    --temperature 0 --save-result --save-detailed --result-dir "$result_dir" \
    --result-filename "cuda-flash-v1-concurrency-$concurrency.json"

  kill -TERM "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  monitor_pid=""
done

kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
server_pid=""

.venv/bin/python3 study/scripts/resource_monitor.py summarize \
  --input-dir "$result_dir" --output "$result_dir/resource-summary.json"
