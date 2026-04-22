#!/usr/bin/env python3
# Harness: compare a minimal sync agent loop with an async agent loop.
"""
s01_agent_loop_compare.py - Sync vs Async Agent Loop

This file keeps the example intentionally small:

1. sync_agent_loop:
   - uses blocking OpenAI client
   - executes tool calls one by one

2. async_agent_loop:
   - uses AsyncOpenAI
   - executes tool calls in the same round concurrently

Try asking something like:
    "Run `sleep 2 && echo one` and `sleep 2 && echo two`, then summarize."

The sync version will take about 4s for the tools.
The async version will take about 2s for the same tool round.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI


load_dotenv(override=True)

WORKDIR = Path.cwd()
SYSTEM = [
    {
        "role": "system",
        "content": (
            f"You are a coding agent at {WORKDIR}. "
            "Use the exec tool when useful. "
            "If the user asks for multiple independent commands, call them in the same turn."
        ),
    }
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Execute a shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Optional working directory. Defaults to workspace root.",
                    },
                },
                "required": ["command"],
            },
        },
    }
]


class SyncLLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            stream=False,
        )


class AsyncLLMClient:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ):
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            stream=False,
        )


def run_bash(command: str, working_dir: str | None = None) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(token in command for token in dangerous):
        return "Error: Dangerous command blocked"

    cwd = working_dir or str(WORKDIR)
    try:
        import subprocess

        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as exc:
        return f"Error: {exc}"


async def run_bash_async(command: str, working_dir: str | None = None) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(token in command for token in dangerous):
        return "Error: Dangerous command blocked"

    cwd = working_dir or str(WORKDIR)
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = (stdout.decode() + stderr.decode()).strip()
        return output[:50000] if output else "(no output)"
    except TimeoutError:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as exc:
        return f"Error: {exc}"


def sync_agent_loop(messages: list[dict]) -> None:
    client = SyncLLMClient()

    while True:
        round_started = time.perf_counter()
        response = client.chat(SYSTEM + messages, TOOLS)
        choice = response.choices[0]
        message = choice.message
        messages.append(message.model_dump(exclude_none=True))

        if choice.finish_reason != "tool_calls":
            print(message.content or "")
            print(f"[sync] final round: {time.perf_counter() - round_started:.2f}s")
            return

        tool_calls = message.tool_calls or []
        print(f"[sync] tool round with {len(tool_calls)} call(s)")
        tool_started = time.perf_counter()
        for tool_call in tool_calls:
            if tool_call.function.name != "exec":
                continue

            args = json.loads(tool_call.function.arguments or "{}")
            command = args["command"]
            working_dir = args.get("working_dir")
            print(f"\033[33m[sync] $ {command}\033[0m")
            output = run_bash(command, working_dir)
            print(output[:200])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.function.name,
                    "content": output,
                }
            )
        print(f"[sync] tool round took: {time.perf_counter() - tool_started:.2f}s")


async def _run_single_tool(tool_call) -> dict[str, str]:
    if tool_call.function.name != "exec":
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.function.name,
            "content": f"Unknown tool: {tool_call.function.name}",
        }

    args = json.loads(tool_call.function.arguments or "{}")
    command = args["command"]
    working_dir = args.get("working_dir")
    print(f"\033[33m[async] $ {command}\033[0m")
    output = await run_bash_async(command, working_dir)
    print(output[:200])
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.function.name,
        "content": output,
    }


async def async_agent_loop(messages: list[dict]) -> None:
    client = AsyncLLMClient()

    while True:
        round_started = time.perf_counter()
        response = await client.chat(SYSTEM + messages, TOOLS)
        choice = response.choices[0]
        message = choice.message
        messages.append(message.model_dump(exclude_none=True))

        if choice.finish_reason != "tool_calls":
            print(message.content or "")
            print(f"[async] final round: {time.perf_counter() - round_started:.2f}s")
            return

        tool_calls = message.tool_calls or []
        print(f"[async] tool round with {len(tool_calls)} call(s)")
        tool_started = time.perf_counter()
        tool_messages = await asyncio.gather(
            *(_run_single_tool(tool_call) for tool_call in tool_calls)
        )
        messages.extend(tool_messages)
        print(f"[async] tool round took: {time.perf_counter() - tool_started:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sync", "async", "both"],
        default="both",
        help="Choose which agent loop to run.",
    )
    parser.add_argument(
        "--prompt",
        help="Optional one-shot prompt. If omitted, enter interactive mode.",
    )
    return parser.parse_args()


def run_sync_once(prompt: str) -> None:
    print("\n===== sync =====")
    history = [{"role": "user", "content": prompt}]
    started = time.perf_counter()
    sync_agent_loop(history)
    print(f"[sync] total elapsed: {time.perf_counter() - started:.2f}s")


async def run_async_once(prompt: str) -> None:
    print("\n===== async =====")
    history = [{"role": "user", "content": prompt}]
    started = time.perf_counter()
    await async_agent_loop(history)
    print(f"[async] total elapsed: {time.perf_counter() - started:.2f}s")


def interactive_loop(mode: str) -> None:
    while True:
        try:
            query = input("\033[36mcompare >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if query.strip().lower() in ("q", "exit", ""):
            return

        if mode in ("sync", "both"):
            run_sync_once(query)
        if mode in ("async", "both"):
            asyncio.run(run_async_once(query))


def main() -> None:
    args = parse_args()

    if args.prompt:
        if args.mode in ("sync", "both"):
            run_sync_once(args.prompt)
        if args.mode in ("async", "both"):
            asyncio.run(run_async_once(args.prompt))
        return

    interactive_loop(args.mode)


if __name__ == "__main__":
    main()
