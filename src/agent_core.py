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
            RoleConfig(name="Critic", system_prompt="あなたは厳しい批評家です。回答に誤りや論理の飛躍がないか指摘してください。")
        ]

    def run_pipeline(self, query: str) -> Tuple[str, str, List[AgentStep]]:
        steps = []
        
        # 1. Executor の生成
        executor = self.role_configs[0]
        initial_answer = self.chat_fn(executor.system_prompt, query, executor.max_new_tokens, executor.temperature)
        steps.append(AgentStep(role=executor.name, observation=initial_answer))
        
        # 2. Critic の批評
        critic = self.role_configs[1]
        critic_input = f"以下の回答をレビューし、修正が必要な点があれば指摘してください。\n\n【回答】\n{initial_answer}"
        critique = self.chat_fn(critic.system_prompt, critic_input, critic.max_new_tokens, critic.temperature)
        steps.append(AgentStep(role=critic.name, observation=critique))
        
        # 3. 最終回答の生成 (Criticの指摘を反映)
        final_prompt = f"以下の批評を参考に、最初の回答をより正確で分かりやすいものに修正してください。\n\n【最初の回答】\n{initial_answer}\n\n【批評】\n{critique}"
        final_answer = self.chat_fn(executor.system_prompt, final_prompt, executor.max_new_tokens, executor.temperature)
        steps.append(AgentStep(role="Refined Answer", observation=final_answer))
        
        full_log = "\n\n".join([f"### [{s.role}]\n{s.observation}" for s in steps])
        
        return final_answer, full_log, steps
