"""使用 qwen2_demo.Scheduler 的最小 OpenAI 流式服务。

这个文件只负责 HTTP 层：

* 收到请求后创建一个 ModelRequest；
* 调用 Scheduler.add_request() 将请求放入线程安全的输入队列；
* 从 Request.wait_for_new_tokens() 读取尚未消费的 token；
* 将新增文本转换成 SSE 流返回给客户端。

模型计算和 Continuous Batching 由 qwen2_demo.Scheduler 的后台线程负责。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from itertools import count
from typing import Any, AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from study.inference_engine.qwen2_demo import (
        MODEL_PATH,
        Qwen2ForCausalLM,
        Request as ModelRequest,
        Scheduler,
        VllmStyleWeightLoader,
        get_device_and_dtype,
        set_default_dtype,
    )
except ImportError:
    # 直接执行脚本时使用此分支。
    from qwen2_demo import (
        MODEL_PATH,
        Qwen2ForCausalLM,
        Request as ModelRequest,
        Scheduler,
        VllmStyleWeightLoader,
        get_device_and_dtype,
        set_default_dtype,
    )
from transformers import AutoConfig, AutoTokenizer


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen2-0.5B-Instruct"
    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int = Field(default=128, ge=1)
    ignore_eos: bool = False


class CompletionRequest(BaseModel):
    model: str = "Qwen2-0.5B-Instruct"
    prompt: str
    stream: bool = False
    max_tokens: int = Field(default=128, ge=1)
    ignore_eos: bool = False


class Runtime:
    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.tokenizer = None
        self.scheduler: Scheduler | None = None
        self.request_ids = count(10000)

    def new_request(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        ignore_eos: bool,
    ) -> ModelRequest:
        if self.scheduler is None:
            raise RuntimeError("Scheduler is not initialized")

        # Request 必须接收一个请求的一维 token 序列，而不是 [batch, seq]。
        request = ModelRequest(
            input_ids=input_ids,
            req_id=next(self.request_ids),
            max_new_tokens=max_new_tokens,
            ignore_eos=ignore_eos,
        )
        return self.scheduler.add_request(request)


runtime = Runtime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    device, dtype = get_device_and_dtype()
    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # 模型加载是启动阶段的一次性同步操作，不阻塞已经运行的请求。
    with set_default_dtype(dtype), device:
        model = Qwen2ForCausalLM(config)
    VllmStyleWeightLoader(model, config).load(MODEL_PATH)
    model.eval()

    eos_token_ids = {
        token_id
        for token_id in (
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        if token_id is not None
    }

    runtime.device = device
    runtime.tokenizer = tokenizer
    runtime.scheduler = Scheduler(eos_token_ids, model)
    yield

    # 停止 Scheduler 后台线程。
    if runtime.scheduler is not None:
        runtime.scheduler.stop_event.set()
        runtime.scheduler.wakeup_event.set()


app = FastAPI(
    title="Qwen2 v2 OpenAI-compatible server",
    lifespan=lifespan,
)


def _message_prompt(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _sse_chunk(
    request_id: str,
    model: str,
    delta: str,
    finish_reason: str | None = None,
    is_chat: bool = True,
) -> str:
    choice = {
        "index": 0,
        "finish_reason": finish_reason,
    }
    if is_chat:
        choice["delta"] = {"content": delta} if delta else {}
        object_type = "chat.completion.chunk"
    else:
        choice["text"] = delta
        choice["logprobs"] = None
        object_type = "text_completion"

    payload = {
        "id": request_id,
        "object": object_type,
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _usage_chunk(
    request_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> str:
    payload = {
        "id": request_id,
        "object": "usage",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_request(
    request: ModelRequest,
    model_name: str,
    request_id: str,
    is_chat: bool,
) -> AsyncIterator[str]:
    assert runtime.tokenizer is not None

    generated_token_ids: list[int] = []
    previous_text = ""

    while True:
        # wait_for_new_tokens() 在 Scheduler 线程没有新 token 时阻塞；
        # to_thread 避免阻塞 FastAPI 的事件循环。
        token_ids, finished = await asyncio.to_thread(
            request.wait_for_new_tokens
        )

        if token_ids:
            generated_token_ids.extend(token_ids)
            current_text = runtime.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
            )
            delta = current_text[len(previous_text):]
            previous_text = current_text

            if delta:
                yield _sse_chunk(
                    request_id,
                    model_name,
                    delta,
                    is_chat=is_chat,
                )

        if finished:
            yield _sse_chunk(
                request_id,
                model_name,
                "",
                finish_reason="stop",
                is_chat=is_chat,
            )
            yield _usage_chunk(
                request_id,
                model_name,
                request.input_ids.shape[0],
                len(generated_token_ids),
            )
            yield "data: [DONE]\n\n"
            return


async def _collect_request(request: ModelRequest) -> list[int]:
    generated_token_ids: list[int] = []
    while True:
        token_ids, finished = await asyncio.to_thread(
            request.wait_for_new_tokens
        )
        generated_token_ids.extend(token_ids)
        if finished:
            return generated_token_ids


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    if runtime.tokenizer is None or runtime.device is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    prompt = _message_prompt(body.messages, runtime.tokenizer)
    input_ids = runtime.tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids[0].to(runtime.device)
    request = runtime.new_request(
        input_ids,
        body.max_tokens,
        body.ignore_eos,
    )
    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    if body.stream:
        return StreamingResponse(
            _stream_request(request, body.model, request_id, is_chat=True),
            media_type="text/event-stream",
        )

    token_ids = await _collect_request(request)
    text = runtime.tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )
    return JSONResponse(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
    )


@app.post("/v1/completions")
async def completions(body: CompletionRequest):
    if runtime.tokenizer is None or runtime.device is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    input_ids = runtime.tokenizer(
        body.prompt,
        return_tensors="pt",
    ).input_ids[0].to(runtime.device)
    request = runtime.new_request(
        input_ids,
        body.max_tokens,
        body.ignore_eos,
    )
    request_id = f"cmpl-{uuid.uuid4().hex}"

    if body.stream:
        return StreamingResponse(
            _stream_request(request, body.model, request_id, is_chat=False),
            media_type="text/event-stream",
        )

    token_ids = await _collect_request(request)
    text = runtime.tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )
    return JSONResponse(
        {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                }
            ],
        }
    )
