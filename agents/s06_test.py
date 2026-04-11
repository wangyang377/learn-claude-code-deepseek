from s06_context_compact import auto_compact, micro_compact
import json


# auto_messages = [
#     {"role": "user", "content": "请帮我修 bug"},
#     {"role": "assistant", "content": "我先看文件"},
#     {"role": "tool", "tool_name": "read_file", "content": "def buggy(): pass" * 200},
# ]

# print("=== auto_compact ===")
# new_messages = auto_compact(auto_messages)
# print(json.dumps(new_messages, indent=2, ensure_ascii=False))
# print("\n=== auto_compact content ===")
# print(new_messages[0]["content"])


micro_messages = [
    {"role": "user", "content": "第1轮"},
    {"role": "tool", "tool_name": "bash", "content": "bash-output-1 " * 20},
    {"role": "tool", "tool_name": "bash", "content": "bash-output-2 " * 20},
    {"role": "tool", "tool_name": "read_file", "content": "read-file-output " * 20},
    {"role": "tool", "tool_name": "bash", "content": "bash-output-3 " * 20},
    {"role": "tool", "tool_name": "bash", "content": "bash-output-4 " * 20},
]

print("\n=== micro_compact before ===")
print(json.dumps(micro_messages, indent=2, ensure_ascii=False))

micro_result = micro_compact(micro_messages)

print("\n=== micro_compact after ===")
print(json.dumps(micro_result, indent=2, ensure_ascii=False))
