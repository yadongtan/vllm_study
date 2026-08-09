#!/usr/bin/env bash

set -euo pipefail
cd /opt/vllm_study
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

result_dir=study/cuda_benchmark_results
concurrencies=(1 4 8)
run_custom=${RUN_CUSTOM:-1}
clean_results=${CLEAN_RESULTS:-1}
mkdir -p "$result_dir"
if [[ "$clean_results" == 1 ]]; then
  rm -f "$result_dir"/custom-cuda-concurrency-*.json \
    "$result_dir"/vllm-cuda-concurrency-*.json \
    "$result_dir"/resources-*.csv "$result_dir"/resource-summary.json \
    "$result_dir"/resource-baselines.json
fi

wait_ready() {
  local port=$1 pid=$2 log=$3
  for _ in $(seq 1 180); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:$port/health" >/dev/null; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then cat "$log"; return 1; fi
    sleep 1
  done
  cat "$log"
  return 1
}

sample_baseline() {
  local name=$1
  .venv/bin/python3 study/resource_monitor.py monitor \
    --output "$result_dir/resources-baseline-$name.csv" --duration 5
}

run_group() {
  local port=$1 model=$2 prefix=$3 c=$4
  local resource="$result_dir/resources-$prefix-concurrency-$c.csv"
  .venv/bin/python3 study/resource_monitor.py monitor --output "$resource" &
  local monitor_pid=$!
  .venv/bin/vllm bench serve --backend openai \
    --base-url "http://127.0.0.1:$port" --endpoint /v1/completions \
    --model "$model" --tokenizer models/Qwen/Qwen2-0.5B-Instruct \
    --dataset-name random --random-input-len 1000 --random-output-len 50 \
    --random-range-ratio 0 --num-prompts 128 --max-concurrency "$c" \
    --request-rate inf --ignore-eos --temperature 0 --save-result \
    --save-detailed --result-dir "$result_dir" \
    --result-filename "$prefix-concurrency-$c.json"
  kill -TERM "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}

sample_baseline system_before

if [[ "$run_custom" == 1 ]]; then
  custom_log="$result_dir/custom-server.log"
  .venv/bin/python3 -m uvicorn study.v2_openai_server:app --host 127.0.0.1 \
    --port 8012 >"$custom_log" 2>&1 &
  custom_pid=$!
  trap 'kill "$custom_pid" 2>/dev/null || true' EXIT
  wait_ready 8012 "$custom_pid" "$custom_log"
  sample_baseline custom_loaded
  for c in "${concurrencies[@]}"; do
    run_group 8012 qwen2-0.5b-instruct-v2 custom-cuda "$c"
  done
  kill "$custom_pid" 2>/dev/null || true
  wait "$custom_pid" 2>/dev/null || true
  trap - EXIT
fi

vllm_log="$result_dir/vllm-server.log"
VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve models/Qwen/Qwen2-0.5B-Instruct --host 127.0.0.1 \
  --port 8013 --served-model-name qwen2-0.5b-instruct-vllm \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 32 \
  --enable-chunked-prefill --gpu-memory-utilization 0.80 --dtype bfloat16 \
  >"$vllm_log" 2>&1 &
vllm_pid=$!
trap 'kill "$vllm_pid" 2>/dev/null || true' EXIT
wait_ready 8013 "$vllm_pid" "$vllm_log"
sample_baseline vllm_loaded
for c in "${concurrencies[@]}"; do run_group 8013 qwen2-0.5b-instruct-vllm vllm-cuda "$c"; done
kill "$vllm_pid" 2>/dev/null || true
wait "$vllm_pid" 2>/dev/null || true
trap - EXIT

.venv/bin/python3 study/resource_monitor.py summarize --input-dir "$result_dir" \
  --output "$result_dir/resource-summary.json"
