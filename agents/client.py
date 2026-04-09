"""
封装 deepseek api，方便统一调用
"""

# Please install OpenAI SDK first: `uv add openai`
import os
from openai import OpenAI
from dotenv import load_dotenv
import json


load_dotenv(override=True)


"""
deepseek官方的api接口示例，uv run client.py执行测试
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