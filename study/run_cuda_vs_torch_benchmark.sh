#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

result_dir=${1:?usage: run_cuda_vs_torch_benchmark.sh YYYYMMDDHHMM}
result_dir="study/cuda_benchmark_results/$result_dir"
mkdir -p "$result_dir"
concurrencies=(1 4 8)

wait_ready() {
  local port=$1 pid=$2 log=$3
  for _ in $(seq 1 240); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:$port/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -100 "$log"
      return 1
    fi
    sleep 1
  done
  tail -100 "$log"
  return 1
}

run_backend() {
  local backend=$1 port=$2
  local server_log="$result_dir/$backend-server.log"
  STUDY_USE_CUDA_ATTENTION="$([[ "$backend" == custom-cuda ]] && echo 1 || echo 0)" \
    .venv/bin/python3 -m uvicorn study.v2_openai_server:app \
    --host 127.0.0.1 --port "$port" >"$server_log" 2>&1 &
  local server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true' RETURN
  wait_ready "$port" "$server_pid" "$server_log"
  .venv/bin/python3 study/resource_monitor.py monitor \
    --output "$result_dir/resources-baseline-$backend-loaded.csv" --duration 5
  for concurrency in "${concurrencies[@]}"; do
    .venv/bin/python3 study/resource_monitor.py monitor \
      --output "$result_dir/resources-$backend-concurrency-$concurrency.csv" &
    local monitor_pid=$!
    .venv/bin/vllm bench serve --backend openai \
      --base-url "http://127.0.0.1:$port" --endpoint /v1/completions \
      --model qwen2-0.5b-instruct-v2 --tokenizer /opt/models/Qwen2-0.5B-Instruct \
      --dataset-name random --random-input-len 1000 --random-output-len 50 \
      --random-range-ratio 0 --num-prompts 128 \
      --max-concurrency "$concurrency" --request-rate inf --ignore-eos \
      --temperature 0 --save-result --save-detailed --result-dir "$result_dir" \
      --result-filename "$backend-concurrency-$concurrency.json"
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  done
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - RETURN
}

.venv/bin/python3 study/resource_monitor.py monitor \
  --output "$result_dir/resources-baseline-system-before.csv" --duration 5
run_backend custom-cuda 8012
run_backend torch-sdpa 8012
.venv/bin/python3 study/resource_monitor.py summarize \
  --input-dir "$result_dir" --output "$result_dir/resource-summary.json"
