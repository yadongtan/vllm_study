#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
VLLM="$REPO_ROOT/.venv/bin/vllm"
MODEL_PATH="${STUDY_MODEL_PATH:-$REPO_ROOT/models/Qwen/Qwen2-0.5B-Instruct}"
PORT="${STUDY_BENCHMARK_PORT:-8002}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="$SCRIPT_DIR/results/$STAMP"
SERVER_LOG="$RESULT_DIR/server.log"

mkdir -p "$RESULT_DIR"

if [[ ! -x "$PYTHON" || ! -x "$VLLM" ]]; then
  echo "The repository .venv is missing python or vllm." >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory not found: $MODEL_PATH" >&2
  exit 1
fi

cd "$REPO_ROOT"

STUDY_MODEL_PATH="$MODEL_PATH" \
STUDY_USE_PYTORCH_FLASH_ATTENTION_V1=1 \
STUDY_USE_CUDA_ATTENTION=0 \
"$PYTHON" -m uvicorn study.openai_server.v2_openai_server:app \
  --host 127.0.0.1 \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for the model server (pid=$SERVER_PID)..."
for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited during startup. See $SERVER_LOG" >&2
    exit 1
  fi
  if "$PYTHON" - "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/health",
    timeout=1,
) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
  then
    break
  fi
  sleep 1
done

if ! "$PYTHON" - "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/health",
    timeout=2,
) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
then
  echo "Server did not become healthy. See $SERVER_LOG" >&2
  exit 1
fi

CONCURRENCIES="${STUDY_BENCHMARK_CONCURRENCIES:-1 2 4 8}"
for concurrency in $CONCURRENCIES; do
  echo "Running end-to-end benchmark with concurrency=$concurrency..."
  "$VLLM" bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/completions \
    --model qwen2-0.5b-instruct-pytorch-flash-attention-v1 \
    --tokenizer "$MODEL_PATH" \
    --dataset-name random \
    --random-input-len 1000 \
    --random-output-len 10 \
    --random-range-ratio 0 \
    --num-prompts 16 \
    --max-concurrency "$concurrency" \
    --request-rate inf \
    --ignore-eos \
    --temperature 0 \
    --save-result \
    --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename "concurrency-$concurrency.json"
done

"$PYTHON" "$SCRIPT_DIR/summarize_results.py" "$RESULT_DIR"
echo "Results saved in $RESULT_DIR"
