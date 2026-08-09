# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Expose the educational v0 Qwen2 model through OpenAI-compatible APIs.

Run:
    .venv/bin/python -m study.openai_server.v0_openai_server

The server supports the streaming protocol consumed by ``vllm bench serve``.
It deliberately serializes model execution because v0 has no request scheduler,
continuous batching, or per-request KV cache management yet.
"""

import argparse
import asyncio
import json
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoConfig, AutoTokenizer

try:
    from study.inference_engine.qwen2_demo import (
        MODEL_PATH,
        Qwen2ForCausalLM,
        VllmStyleWeightLoader,
        get_device_and_dtype,
        set_default_dtype,
    )
except ImportError:
    from qwen2_demo import (
        MODEL_PATH,
        Qwen2ForCausalLM,
        VllmStyleWeightLoader,
        get_device_and_dtype,
        set_default_dtype,
    )


MODEL_ID = "qwen2-0.5b-instruct-v0"



class CompletionRequest(BaseModel):
    """Subset of the OpenAI Completions request used by vLLM benchmarks."""

    model_config = ConfigDict(extra="ignore")

    model: str = MODEL_ID
    prompt: str
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = 0.0
    stream: bool = False
    ignore_eos: bool = False


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI Chat Completions request."""

    model_config = ConfigDict(extra="ignore")

    model: str = MODEL_ID
    messages: list[dict[str, Any]]
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = 0.0
    stream: bool = False
    ignore_eos: bool = False

    def requested_tokens(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 16


class ModelRuntime:
    """Own the tokenizer, model, and the v0 single-request execution lock."""

    def __init__(self) -> None:
        self.device, self.dtype = get_device_and_dtype()
        self.config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )

        print(f"Building {MODEL_ID} on {self.device} with {self.dtype}...")
        with set_default_dtype(self.dtype), self.device:
            self.model = Qwen2ForCausalLM(self.config)
        VllmStyleWeightLoader(self.model, self.config).load(MODEL_PATH)
        self.model.eval()

        # v0 recomputes the full sequence for every token and cannot batch
        # independently arriving requests. The lock prevents concurrent callers
        # from racing on the shared MPS model.
        self.inference_lock = threading.Lock()

    def tokenize_prompt(self, prompt: str) -> torch.Tensor:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if input_ids.shape[1] >= self.config.max_position_embeddings:
            raise ValueError("Prompt exceeds the model context length")
        return input_ids.to(self.device)

    @torch.inference_mode()
    def iter_token_ids(
        self,
        prompt: str,
        max_tokens: int,
        ignore_eos: bool,
    ) -> Iterator[tuple[int, int]]:
        """Yield ``(token_id, prompt_tokens)`` as greedy decoding progresses."""

        input_ids = self.tokenize_prompt(prompt)
        prompt_tokens = input_ids.shape[1]
        if prompt_tokens + max_tokens > self.config.max_position_embeddings:
            raise ValueError("Prompt and output exceed the model context length")

        eos_token_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        token_ids = input_ids

        with self.inference_lock:
            for _ in range(max_tokens):
                next_token = self.model(token_ids).argmax(dim=-1)
                next_token_id = next_token.item()
                if not ignore_eos and next_token_id in eos_token_ids:
                    break
                token_ids = torch.cat((token_ids, next_token[:, None]), dim=1)
                yield next_token_id, prompt_tokens

    def generate_text(
        self,
        prompt: str,
        max_tokens: int,
        ignore_eos: bool,
    ) -> tuple[str, int, int]:
        token_ids: list[int] = []
        prompt_tokens = 0
        for token_id, prompt_tokens in self.iter_token_ids(
            prompt,
            max_tokens,
            ignore_eos,
        ):
            token_ids.append(token_id)
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text, prompt_tokens, len(token_ids)


app = FastAPI(title="Qwen2 v0 OpenAI-compatible server", version="0")


def get_runtime(request: Request) -> ModelRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return runtime


def normalize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            content = "".join(text_parts)
        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(content),
            }
        )
    return normalized


def completion_chunk(
    request_id: str,
    model: str,
    text: str,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def chat_chunk(
    request_id: str,
    model: str,
    text: str,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": text},
                "finish_reason": finish_reason,
            }
        ],
    }


async def stream_generation(
    runtime: ModelRuntime,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool,
    model: str,
    is_chat: bool,
) -> AsyncIterator[str]:
    """Bridge blocking MPS inference to an asynchronous SSE response."""

    request_id = f"cmpl-{uuid.uuid4().hex}"
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def run_generation() -> None:
        token_ids: list[int] = []
        prompt_tokens = 0
        previous_text = ""
        try:
            for token_id, prompt_tokens in runtime.iter_token_ids(
                prompt,
                max_tokens,
                ignore_eos,
            ):
                token_ids.append(token_id)
                current_text = runtime.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                )
                delta = current_text[len(previous_text) :]
                previous_text = current_text
                result_queue.put(("token", delta))
            result_queue.put(
                (
                    "done",
                    {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(token_ids),
                    },
                )
            )
        except Exception as error:
            result_queue.put(("error", str(error)))

    threading.Thread(target=run_generation, daemon=True).start()

    while True:
        event, payload = await asyncio.to_thread(result_queue.get)
        if event == "token":
            chunk = (
                chat_chunk(request_id, model, payload)
                if is_chat
                else completion_chunk(request_id, model, payload)
            )
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            continue

        if event == "error":
            error = {"error": {"message": payload, "type": "server_error"}}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        finish_chunk = (
            chat_chunk(request_id, model, "", "length")
            if is_chat
            else completion_chunk(request_id, model, "", "length")
        )
        yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
        usage = {
            "id": request_id,
            "object": "usage",
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": {
                **payload,
                "total_tokens": (
                    payload["prompt_tokens"] + payload["completion_tokens"]
                ),
            },
        }
        yield f"data: {json.dumps(usage, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    runtime = get_runtime(request)
    return {"status": "ok", "model": MODEL_ID, "device": str(runtime.device)}


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    get_runtime(request)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "study-v0",
            }
        ],
    }


@app.post("/v1/completions")
async def completions(
    request: Request,
    body: CompletionRequest,
) -> Response:
    runtime = get_runtime(request)
    if body.stream:
        return StreamingResponse(
            stream_generation(
                runtime,
                body.prompt,
                body.max_tokens,
                body.ignore_eos,
                body.model,
                False,
            ),
            media_type="text/event-stream",
        )

    try:
        text, prompt_tokens, completion_tokens = await asyncio.to_thread(
            runtime.generate_text,
            body.prompt,
            body.max_tokens,
            body.ignore_eos,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return JSONResponse(
        {
            **completion_chunk(
                f"cmpl-{uuid.uuid4().hex}",
                body.model,
                text,
                "length",
            ),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
) -> Response:
    runtime = get_runtime(request)
    messages = normalize_chat_messages(body.messages)
    prompt = runtime.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    max_tokens = body.requested_tokens()

    if body.stream:
        return StreamingResponse(
            stream_generation(
                runtime,
                prompt,
                max_tokens,
                body.ignore_eos,
                body.model,
                True,
            ),
            media_type="text/event-stream",
        )

    try:
        text, prompt_tokens, completion_tokens = await asyncio.to_thread(
            runtime.generate_text,
            prompt,
            max_tokens,
            body.ignore_eos,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app.state.runtime = ModelRuntime()
    print(f"OpenAI-compatible server: http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
