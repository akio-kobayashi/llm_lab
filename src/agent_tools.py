import ast
import datetime as dt
import math
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[str], str]


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def call(self, name: str, arg: str) -> str:
        if name not in self.tools:
            return f"Tool not found: {name}"
        return self.tools[name].run(arg)


def safe_calculate(expr: str) -> str:
    allowed_names = {
        "pi": math.pi,
        "e": math.e,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("Unsupported operator")
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.Name) and node.id in allowed_names:
            return allowed_names[node.id]
        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree)
        return str(result)
    except Exception as e:
        return f"計算エラー: {e}"


def today_tool(_: str) -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def make_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("calculator", "数式を安全に計算する", safe_calculate))
    reg.register(Tool("today", "今日の日付を返す", today_tool))
    return reg
