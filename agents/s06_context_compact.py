#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Layer 1: micro_compact
      Replace old non-read_file tool results with placeholders.

    Layer 2: auto_compact
      Save transcript, ask LLM for a summary, replace messages with summary.

    Layer 3: compact tool
      Let the model trigger immediate summarization.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from client import LLMClient


WORKDIR = Path.cwd()
client = LLMClient("s06-agent")
SYSTEM = [
    {
        "role": "system",
        "content": f"You are a coding agent at {WORKDIR}. Use tools to solve tasks.",
    }
]

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3
PRESERVE_RESULT_TOOLS = {"read_file"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compact",
            "description": "Trigger manual conversation compression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What to preserve in the summary",
                    },
                },
            },
        },
    },
]


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, default=str)) // 4


def micro_compact(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_messages: list[dict[str, Any]] = [
        msg for msg in messages if msg.get("role") == "tool"
    ]
    if len(tool_messages) <= KEEP_RECENT:
        return messages

    to_clear = tool_messages[:-KEEP_RECENT]
    for msg in to_clear:
        content = msg.get("content")
        tool_name = msg.get("tool_name", "unknown")
        if not isinstance(content, str) or len(content) <= 100:
            continue
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue
        msg["content"] = f"[Previous: used {tool_name}]"
    return messages


def auto_compact(messages: list[dict[str, Any]], focus: str | None = None) -> list[dict[str, Any]]:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with transcript_path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    conversation_text = json.dumps(messages, ensure_ascii=False, default=str)[-80000:]
    prompt = (
        "Summarize this conversation for continuity. Include: "
        "1) What was accomplished, 2) Current state, 3) Key decisions made. "
        "Be concise but preserve critical details."
    )
    if focus:
        prompt += f" Preserve this especially: {focus}."

    response = client.chat(
        [
            {
                "role": "user",
                "content": f"{prompt}\n\n{conversation_text}",
            }
        ]
    )
    summary = response.content or "No summary generated."
    return [
        {
            "role": "user",
            "content": (
                f"[Conversation compressed. Transcript: {transcript_path}]\n\n"
                f"{summary}"
            ),
        }
    ]


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact": lambda **kw: "Manual compression requested.",
}


def agent_loop(messages: list[dict[str, Any]]):
    while True:
        micro_compact(messages)
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)

        response = client.chat(SYSTEM + messages, TOOLS)
        messages.append(response.assistant_message)

        if response.finish_reason != "tool_calls":
            if response.content:
                print(response.content)
            return

        manual_compact = False
        compact_focus = None

        for tool_call in response.tool_calls:
            args = json.loads(tool_call.arguments)
            if tool_call.name == "compact":
                manual_compact = True
                compact_focus = args.get("focus")
                output = "Compressing..."
            else:
                handler = TOOL_HANDLERS.get(tool_call.name)
                try:
                    output = handler(**args) if handler else f"Unknown tool: {tool_call.name}"
                except Exception as e:
                    output = f"Error: {e}"

            print(f"> {tool_call.name}:")
            print(str(output)[:200])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "content": str(output),
                }
            )

        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages, compact_focus)
            return


if __name__ == "__main__":
    history: list[dict[str, Any]] = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
