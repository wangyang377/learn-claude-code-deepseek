#!/usr/bin/env python3
"""
Minimal FastAPI chat demo with a tiny agent loop.

Run:
    uv run uvicorn demo.minimal_agent_chat:app --reload
"""

from __future__ import annotations

import os
import json
import asyncio
import threading
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel


load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Minimal Agent Chat Demo")
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
async_client = AsyncOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
MODEL_NAME = os.getenv("MODEL_NAME")
CHAT_SLOT = threading.Semaphore(1)
ASYNC_CHAT_SLOT = asyncio.Semaphore(1)

SYSTEM_PROMPT = (
    "You are a minimal coding agent demo. "
    "Reply briefly and helpfully. "
    "This demo does not use tool calls yet."
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


def stream_agent_loop(messages: list[Message]):
    CHAT_SLOT.acquire()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *[message.model_dump() for message in messages],
            ],
            stream=True,
        )

        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        CHAT_SLOT.release()


async def stream_agent_loop_async(messages: list[Message]):
    async with ASYNC_CHAT_SLOT:
        response = await async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *[message.model_dump() for message in messages],
            ],
            stream=True,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "chat.html")


@app.post("/api/chat")
def chat(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_agent_loop(payload.messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/chat-async")
async def chat_async(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_agent_loop_async(payload.messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
