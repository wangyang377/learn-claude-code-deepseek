#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess
import json

from client import LLMClient

client = LLMClient()
SYSTEM = [
    {
        "role": "system",
        "content": (
            f"You are a coding agent at {os.getcwd()}. "
            "Use bash to solve tasks. Act, don't explain."
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

"""
真实执行tool_use的地方
"""
def run_bash(command: str, working_dir: str | None = None) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        cwd = working_dir or os.getcwd()
        r = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
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


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        response = client.chat(SYSTEM + messages, TOOLS)
        messages.append(response.assistant_message)
        # If the model didn't call a tool, we're done
        if response.finish_reason != "tool_calls":
            if response.content:
                print(response.content)
            return
        # Execute each tool call, collect results
        for tool_call in response.tool_calls:
            if tool_call.name != "exec":
                continue
            args = json.loads(tool_call.arguments)
            command = args["command"]
            working_dir = args.get("working_dir")
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command, working_dir)
            print(output[:200])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                }
            )


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
