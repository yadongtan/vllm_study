#!/usr/bin/env bash

set -euo pipefail

cd /opt/vllm_study
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

result_dir="study/cuda_benchmark_results"
mkdir -p "$result_dir"

.venv/bin/python3 -m uvicorn study.v2_openai_server:app \
  --host 127.0.0.1 \
  --port 8014 \
  >"$result_dir/default-batch-server.log" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl --noproxy '*' -fsS http://127.0.0.1:8014/health >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "$result_dir/default-batch-server.log"
    exit 1
  fi
  sleep 1
done

curl --noproxy '*' -fsS http://127.0.0.1:8014/health

for concurrency in 1 2 4; do
  echo "Running CUDA default-batch benchmark with concurrency=$concurrency"
  .venv/bin/vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8014 \
    --endpoint /v1/completions \
    --model qwen2-0.5b-instruct-v2 \
    --tokenizer models/Qwen/Qwen2-0.5B-Instruct \
    --dataset-name random \
    --random-input-len 1000 \
    --random-output-len 10 \
    --random-range-ratio 0 \
    --num-prompts 10 \
    --max-concurrency "$concurrency" \
    --request-rate inf \
    --ignore-eos \
    --temperature 0 \
    --save-result \
    --save-detailed \
    --result-dir "$result_dir" \
    --result-filename \
      "v2-0809-cuda-default-batch-concurrency-$concurrency.json"
done
