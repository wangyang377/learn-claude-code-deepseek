#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ----------> | while tool_calls |
    |   prompt="..."   |             |   call tools     |
    |   description="" |             |   append results |
    |                  |  summary    |                  |
    |   result = "..." | <---------- | return last text |
    +------------------+             +------------------+

Key insight: "Subagent context stays isolated. The parent only sees a summary."
"""

import json
import os
import subprocess
from pathlib import Path

from client import LLMClient


WORKDIR = Path.cwd()
parent_client = LLMClient("s04-parent")
subagent_client = LLMClient("s04-subagent")

SYSTEM = [
    {
        "role": "system",
        "content": (
            f"You are a coding agent at {WORKDIR}. "
            "Use the task tool to delegate exploration or subtasks."
        ),
    }
]

SUBAGENT_SYSTEM = [
    {
        "role": "system",
        "content": (
            f"You are a coding subagent at {WORKDIR}. "
            "Complete the given task, then summarize your findings."
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
}

CHILD_TOOLS = [
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
]

PARENT_TOOLS = CHILD_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": (
                "Spawn a subagent with fresh context. It shares the filesystem "
                "but not conversation history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "Short description of the task",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


def execute_tool(name: str, arguments: str) -> str:
    if name == "task":
        args = json.loads(arguments)
        desc = args.get("description", "subtask")
        prompt = args.get("prompt", "")
        print(f"> task ({desc}): {prompt[:80]}")
        return run_subagent(prompt)

    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    args = json.loads(arguments)
    return str(handler(**args))


def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    final_content = ""

    for _ in range(30):
        response = subagent_client.chat(SUBAGENT_SYSTEM + sub_messages, CHILD_TOOLS)
        sub_messages.append(response.assistant_message)
        final_content = response.content or final_content

        if response.finish_reason != "tool_calls":
            return final_content or "(no summary)"

        for tool_call in response.tool_calls:
            output = execute_tool(tool_call.name, tool_call.arguments)
            sub_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "content": output,
                }
            )

    return final_content or "(no summary)"


def agent_loop(messages: list):
    while True:
        response = parent_client.chat(SYSTEM + messages, PARENT_TOOLS)
        messages.append(response.assistant_message)

        if response.finish_reason != "tool_calls":
            if response.content:
                print(response.content)
            return

        for tool_call in response.tool_calls:
            output = execute_tool(tool_call.name, tool_call.arguments)
            print(f"  {str(output)[:200]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "content": output,
                }
            )


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
