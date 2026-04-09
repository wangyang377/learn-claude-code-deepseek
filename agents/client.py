"""
封装 deepseek api，方便统一调用
"""

from dataclasses import dataclass
import os
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv
import json


load_dotenv(override=True)


"""
deepseek官方的api接口示例，非正式内容，uv run client.py执行测试
"""
def deepseek_api_test():
    client = OpenAI(
        api_key=os.getenv('API_KEY'),
        base_url=os.getenv('BASE_URL')
        )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )
    # 用json格式print(response)
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))

    # content是返回的内容，tool_calls决定了是否需要调用工具
    print(response.choices[0].message.content)
    print(response.choices[0].message.tool_calls)

def main():
    deepseek_api_test()

if __name__ == "__main__":
    main()


"""
以下是正式内容.
通过 LLMClient 封装 llm 的入参和出参
"""

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str | None
    usage: dict[str, int]
    tool_calls: list[ToolCall]
    assistant_message: dict[str, Any]
    raw: Any

class LLMClient:

    def __init__(self) -> None:
        self.api_key = os.getenv('API_KEY')
        self.base_url = os.getenv('BASE_URL')
        self.model_name = os.getenv('MODEL_NAME')

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            stream=False
        )

        choice = response.choices[0]
        usage = response.usage
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

        return ChatResult(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage_dict,
            tool_calls=[
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments or "{}",
                )
                for tc in (getattr(choice.message, "tool_calls", None) or [])
            ],
            assistant_message=choice.message.model_dump(exclude_none=True),
            raw=response,
        )
