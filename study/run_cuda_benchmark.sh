#!/usr/bin/env bash

set -euo pipefail

cd /opt/vllm_study
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

result_dir="study/cuda_benchmark_results"
concurrencies=(1 4 8 16 32)
mkdir -p "$result_dir"

cleanup_results() {
  find "$result_dir" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.log' -o -name '*.md' \) -delete
}

wait_for_server() {
  local port=$1
  local pid=$2
  local log=$3
  for _ in $(seq 1 180); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:$port/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      cat "$log"
      return 1
    fi
    sleep 1
  done
  echo "Server did not become healthy on port $port"
  cat "$log"
  return 1
}

stop_server() {
  local pid=$1
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

run_benchmarks() {
  local port=$1
  local model=$2
  local prefix=$3
  for concurrency in "${concurrencies[@]}"; do
    echo "Running $prefix benchmark with concurrency=$concurrency"
    .venv/bin/vllm bench serve \
      --backend openai \
      --base-url "http://127.0.0.1:$port" \
      --endpoint /v1/completions \
      --model "$model" \
      --tokenizer models/Qwen/Qwen2-0.5B-Instruct \
      --dataset-name random \
      --random-input-len 1000 \
      --random-output-len 50 \
      --random-range-ratio 0 \
      --num-prompts 128 \
      --max-concurrency "$concurrency" \
      --request-rate inf \
      --ignore-eos \
      --temperature 0 \
      --save-result \
      --save-detailed \
      --result-dir "$result_dir" \
      --result-filename "$prefix-concurrency-$concurrency.json"
  done
}

cleanup_results

custom_log="$result_dir/custom-server.log"
.venv/bin/python3 -m uvicorn study.v2_openai_server:app \
  --host 127.0.0.1 \
  --port 8012 \
  >"$custom_log" 2>&1 &
custom_pid=$!
trap 'stop_server "$custom_pid"' EXIT
wait_for_server 8012 "$custom_pid" "$custom_log"
run_benchmarks 8012 qwen2-0.5b-instruct-v2 custom-cuda
stop_server "$custom_pid"
trap - EXIT

vllm_log="$result_dir/vllm-server.log"
.venv/bin/vllm serve models/Qwen/Qwen2-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8013 \
  --served-model-name qwen2-0.5b-instruct-vllm \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 32 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.80 \
  --dtype bfloat16 \
  >"$vllm_log" 2>&1 &
vllm_pid=$!
trap 'stop_server "$vllm_pid"' EXIT
wait_for_server 8013 "$vllm_pid" "$vllm_log"
run_benchmarks 8013 qwen2-0.5b-instruct-vllm vllm-cuda
stop_server "$vllm_pid"
trap - EXIT
