import torch
from dataclasses import dataclass
from typing import List, Callable, Tuple

@dataclass
class RoleConfig:
    name: str
    system_prompt: str
    temperature: float = 0.3
    max_new_tokens: int = 512

@dataclass
class AgentStep:
    role: str
    observation: str

class LLMExecutorCriticAgent:
    """
    Executor (実行) と Critic (批評・修正) の2役をこなすエージェント。
    """
    def __init__(self, chat_fn: Callable, role_configs: List[RoleConfig] = None):
        self.chat_fn = chat_fn
        self.role_configs = role_configs or [
            RoleConfig(name="Executor", system_prompt="あなたは優秀なAIアシスタントです。論理的かつ簡潔に回答してください。"),
            RoleConfig(name="Critic", system_prompt="あなたは厳しい批評家です。回答に誤りや論理の飛躍がないか指摘してください。問題がなければ「誤りなし」とだけ答えてください。")
        ]

    def run_pipeline(self, query: str, max_iterations: int = 1) -> Tuple[str, str, List[AgentStep]]:
        steps = []
        executor = self.role_configs[0]
        critic = self.role_configs[1]

        # 1. Executor の初回生成
        current_answer = self.chat_fn(executor.system_prompt, query, executor.max_new_tokens, executor.temperature)
        steps.append(AgentStep(role=executor.name, observation=current_answer))

        # 2. Executor-Critic ループ (max_iterations 回)
        for i in range(max_iterations):
            # 2a. Critic の批評
            critic_input = (
                f"以下の回答をレビューし、システムプロンプトの指示に従ってフィードバックを出力してください。\n\n"
                f"【元の質問】\n{query}\n\n【回答】\n{current_answer}"
            )
            critique = self.chat_fn(critic.system_prompt, critic_input, critic.max_new_tokens, critic.temperature)
            steps.append(AgentStep(role=f"{critic.name} (round {i+1})", observation=critique))

            # 2b. Critic が「誤りなし」と回答した場合は早期終了
            if "誤りなし" in critique:
                break

            # 2c. Executor が批評を受けて修正
            refine_prompt = (
                f"以下の批評を参考に、回答を修正してください。修正後の回答のみを出力してください。\n\n"
                f"【元の質問】\n{query}\n\n"
                f"【現在の回答】\n{current_answer}\n\n【批評】\n{critique}"
            )
            current_answer = self.chat_fn(executor.system_prompt, refine_prompt, executor.max_new_tokens, executor.temperature)
            steps.append(AgentStep(role=f"{executor.name} (refined {i+1})", observation=current_answer))

        full_log = "\n\n".join([f"### [{s.role}]\n{s.observation}" for s in steps])

        return current_answer, full_log, steps
