"""一个最小但完整的 DeepSeek 命令行 Agent。"""

import ast
import json
import operator
import os
from datetime import datetime
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """你是一个可靠、简洁的中文 AI 助手。
需要精确计算或当前时间时请调用工具，不要猜测工具结果。
回答应直接、清楚；信息不足时明确说明。"""

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
MAX_TOOL_ROUNDS = 8


def calculate(expression: str) -> str:
    """安全计算只包含数字和基本算术运算的表达式。"""
    binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
            return binary_ops[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return unary_ops[type(node.op)](evaluate(node.operand))
        raise ValueError("只支持数字、括号和 + - * / // % ** 运算")

    if len(expression) > 200:
        raise ValueError("表达式过长")
    return str(evaluate(ast.parse(expression, mode="eval").body))


def current_time() -> str:
    """返回运行 Agent 的机器本地时间。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "精确计算一个基本算术表达式",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "获取 Agent 所在机器的当前本地时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

FUNCTIONS: dict[str, Callable[..., str]] = {
    "calculate": calculate,
    "current_time": current_time,
}


class Agent:
    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def ask(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return message.content or ""

            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    result = FUNCTIONS[call.function.name](**arguments)
                except Exception as error:
                    result = f"工具执行失败: {error}"
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
        raise RuntimeError("工具调用轮次过多，已停止")

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def main() -> None:
    agent = Agent()
    print(f"DeepSeek Agent ({MODEL})；输入 /reset 清空记忆，/exit 退出。")
    while True:
        try:
            text = input("\n你: ").strip()
            if text == "/exit":
                break
            if text == "/reset":
                agent.reset()
                print("Agent: 对话记忆已清空。")
            elif text:
                print("Agent:", agent.ask(text))
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        except Exception as error:
            print(f"错误: {error}")


if __name__ == "__main__":
    main()
