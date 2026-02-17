import ast
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
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
    # 自然文から計算に使える文字だけを抽出する。
    expr = extract_math_expression(expr)
    if not expr:
        return "計算エラー: 数式を抽出できませんでした"

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


def extract_math_expression(text: str) -> str:
    # 例: "(23 + 17) * 5 を計算して" -> "(23 + 17) * 5"
    candidates = re.findall(r"[0-9\.\+\-\*\/\(\)\s]+", text)
    candidates = [c.strip() for c in candidates if c.strip()]
    if not candidates:
        return ""
    return max(candidates, key=len)


def today_tool(_: str) -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def make_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("calculator", "数式を安全に計算する", safe_calculate))
    reg.register(Tool("today", "今日の日付を返す", today_tool))
    reg.register(Tool("anime_seed_search", "2026年2月アニメ種データを参照する", anime_seed_search))
    return reg


def anime_seed_search(query: str) -> str:
    path = Path("data/anime_202602/anime_news_202602_seed.jsonl")
    if not path.exists():
        return "参照データが見つかりません。"

    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    if not docs:
        return "参照データが空です。"

    q = query.lower()
    scored = []
    for d in docs:
        text = f"{d.get('title','')} {d.get('summary','')}".lower()
        score = 0
        for token in ["2026", "2月", "アニメ", "最新", "一覧", "ニュース"]:
            if token in q and token in text:
                score += 1
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for _, d in scored[:2]]

    lines = []
    for d in top:
        lines.append(
            f"- {d.get('date','')} | {d.get('title','')} | {d.get('source','')}\n"
            f"  {d.get('summary','')}"
        )
    return "\n".join(lines)
