from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from src.agent_tools import ToolRegistry


@dataclass
class AgentStep:
    step_no: int
    role: str
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class RoleConfig:
    name: str
    system_prompt: str
    temperature: float = 0.2
    max_new_tokens: int = 512


class RulePlannerCoderCriticAgent:
    """軽量な教育用3役エージェント（LLM不要版）。

    Planner: タスク分解
    Coder: 実行案作成（必要に応じてツール呼び出し）
    Critic: 出力形式と安全性を点検
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def _planner(self, query: str) -> Dict:
        steps = [
            "要求を1文で要約する",
            "必要ならツールを選択する",
            "回答形式を満たして出力する",
        ]
        assumptions = []
        if "最新" in query or "2026" in query:
            assumptions.append("外部最新情報は持たないため、既知情報または入力データに限定して回答する")
        return {
            "task_summary": query,
            "steps": steps,
            "assumptions": assumptions,
        }

    def _select_tool(self, query: str) -> Tuple[str, str]:
        q = query.lower()
        if any(k in q for k in ["+", "-", "*", "/", "計算", "sqrt", "("]):
            return "calculator", query
        if any(k in query for k in ["今日", "日付", "today", "date"]):
            return "today", ""
        return "none", ""

    def _coder(self, query: str, plan: Dict) -> Dict:
        tool, tool_input = self._select_tool(query)
        observation = "ツール未使用"
        if tool != "none":
            observation = self.registry.call(tool, tool_input)

        answer = f"質問: {query}"
        if tool != "none":
            answer += f"\nツール({tool})結果: {observation}"
        if plan.get("assumptions"):
            answer += "\n前提: " + " / ".join(plan["assumptions"])

        return {
            "tool": tool,
            "tool_input": tool_input,
            "observation": observation,
            "answer": answer,
        }

    def _critic(self, plan: Dict, coded: Dict) -> Dict:
        issues = []
        if not coded.get("answer"):
            issues.append("回答本文が空です")
        if str(coded.get("observation", "")).startswith("計算エラー"):
            issues.append("計算ツールが失敗しています")
        if plan.get("assumptions") and "前提:" not in coded.get("answer", ""):
            issues.append("前提の明示が不足しています")
        status = "pass" if not issues else "needs_fix"
        return {
            "status": status,
            "issues": issues,
            "fix": "前提と制約を明示して再回答" if issues else "修正不要",
        }

    def run(self, query: str, max_steps: int = 3) -> Tuple[str, List[AgentStep]]:
        steps: List[AgentStep] = []

        plan = self._planner(query)
        steps.append(
            AgentStep(
                step_no=1,
                role="Planner",
                thought="課題を分解して実行方針を作る",
                action="plan",
                action_input=query,
                observation=str(plan),
            )
        )

        coded = self._coder(query, plan)
        steps.append(
            AgentStep(
                step_no=2,
                role="Coder",
                thought="計画に沿って最小実行案を作る",
                action=coded["tool"],
                action_input=coded["tool_input"],
                observation=coded["observation"],
            )
        )

        review = self._critic(plan, coded)
        steps.append(
            AgentStep(
                step_no=3,
                role="Critic",
                thought="不足・危険・形式違反を点検する",
                action="review",
                action_input="answer",
                observation=str(review),
            )
        )

        final = coded["answer"]
        if review["status"] != "pass":
            final += "\n\n[Critic修正提案] " + review["fix"]
        return final, steps


class LLMPlannerCoderCriticAgent:
    """軽量3役マルチエージェント（LLM版）。"""

    def __init__(
        self,
        llm_chat: Callable[[str, str, int, float], str],
        role_configs: Optional[List[RoleConfig]] = None,
        max_context_chars: int = 6000,
    ):
        self.llm_chat = llm_chat
        self.max_context_chars = max_context_chars
        self.role_configs = role_configs or [
            RoleConfig("Planner", "あなたは優秀なプロジェクトプランナーです。課題を分解し箇条書きで提案してください。曖昧さは仮定を明示。", 0.2, 400),
            RoleConfig("Coder", "あなたは実装担当です。計画に沿って、動く最小構成のPythonコード案を提示してください。説明は最小限。", 0.2, 700),
            RoleConfig("Critic", "あなたは批評担当です。誤り・危険・不足を具体的に指摘し、修正方針を書いてください。", 0.1, 500),
        ]

    def _build_prompt(self, role: str, user_task: str, context: str) -> str:
        prompt = "【ユーザー課題】\n" + user_task.strip() + "\n\n"
        if context.strip():
            prompt += "【これまでの出力（共有コンテキスト）】\n" + context.strip() + "\n\n"
        prompt += "【あなたの役割】\n" + role + "\n"
        prompt += "上記を踏まえて、あなたの役割に沿って出力してください。"
        return prompt

    def run_pipeline(self, user_task: str) -> Tuple[str, str, List[AgentStep]]:
        logs: List[str] = []
        steps: List[AgentStep] = []
        context = ""

        for idx, cfg in enumerate(self.role_configs, start=1):
            prompt = self._build_prompt(cfg.name, user_task, context)
            out = self.llm_chat(
                cfg.system_prompt,
                prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )

            logs.append(f"===== {cfg.name} =====\n{out}")
            context = (context + f"\n\n{cfg.name}:\n" + out)[-self.max_context_chars :]

            steps.append(
                AgentStep(
                    step_no=idx,
                    role=cfg.name,
                    thought=f"{cfg.name} role executed",
                    action="llm_chat",
                    action_input=prompt[:300],
                    observation=out[:600],
                )
            )

        full_log = "\n\n".join(logs)
        final_answer = logs[-1] if logs else ""
        return final_answer, full_log, steps


class RagAwareAgent:
    """3役エージェントにRAG根拠を付加する。"""

    def __init__(self, base_agent, rag_search):
        self.base_agent = base_agent
        self.rag_search = rag_search

    def run(self, query: str):
        return self.base_agent.run(query)

    def run_with_rag(self, query: str, top_k: int = 3):
        docs = self.rag_search(query, top_k=top_k)
        answer, steps = self.base_agent.run(query)
        evidence = []
        for d in docs:
            title = d.get("title", "")
            date = d.get("date", "")
            source = d.get("source", "")
            summary = d.get("summary", "")
            evidence.append(f"- {date} | {title} | {source}\n  {summary}")

        answer = answer + "\n\nRAG根拠:\n" + ("\n".join(evidence) if evidence else "- 根拠なし")
        return answer, steps, docs
