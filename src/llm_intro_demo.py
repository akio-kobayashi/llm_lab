"""生成AI入門の Colab 演習で使う処理本体。

ノートブックではフォームの値だけを受け取り、このモジュールのメソッドを呼ぶ。
"""

from __future__ import annotations

import html
import time
from datetime import datetime
from pathlib import Path


def _positive_int(value: int, name: str, upper: int) -> int:
    value = int(value)
    if not 1 <= value <= upper:
        raise ValueError(f"{name} は1～{upper}の整数にしてください。")
    return value


class LectureDemo:
    def __init__(
        self,
        model_id: str = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1",
        prefer_gpu: bool = True,
    ) -> None:
        import torch
        from src.common import load_llm

        self.torch = torch
        self.model_id = model_id
        self.device = "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model, self.tokenizer = load_llm(
            model_id=model_id,
            use_4bit=False,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else "cpu",
        )
        self.model.eval()
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.candidate_history: list[dict] = []
        self.generation_history: list[dict] = []

    def _display(self, content: str) -> None:
        from IPython.display import HTML, display

        display(HTML(content))

    def show_candidates(self, prompt: str, top_n: int = 8) -> list[dict]:
        """入力文そのものの続きに対する確率分布の上位候補を表示する。"""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("続きが気になる文章を入力してください。")
        top_n = _positive_int(top_n, "表示候補数", 30)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1]
            probabilities = self.torch.softmax(logits.float(), dim=-1)
        values, indices = self.torch.topk(probabilities, top_n)
        candidates = [
            {
                "token": self.tokenizer.decode([token_id]).replace("\n", "↵"),
                "probability": float(probability),
            }
            for probability, token_id in zip(values.tolist(), indices.tolist())
        ]
        self.candidate_history.append({"prompt": prompt, "candidates": candidates})
        self._display(self._candidate_html())
        return candidates

    def _candidate_html(self) -> str:
        sections = []
        for entry in self.candidate_history:
            rows = "".join(
                "<tr><td style='padding:5px 15px'>"
                + html.escape(repr(item["token"]))
                + "</td><td style='padding:5px 15px;text-align:right'>"
                + f"{item['probability'] * 100:.2f}%</td></tr>"
                for item in entry["candidates"]
            )
            sections.append(
                f"<h4>入力：{html.escape(entry['prompt'])}</h4>"
                "<table><tr><th>次のトークン候補</th><th>確率</th></tr>"
                f"{rows}</table>"
                "<p>上位候補だけを表示しています。表示値の合計は100%とは限りません。</p>"
            )
        return "\n".join(sections)

    def _generate(
        self, prompt: str, temperature: float, top_k: int, max_new_tokens: int
    ) -> dict:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("指示文を入力してください。")
        temperature = float(temperature)
        if not 0 < temperature <= 2:
            raise ValueError("temperature は0より大きく2以下にしてください。")
        top_k = _positive_int(top_k, "top-k", 200)
        max_new_tokens = _positive_int(max_new_tokens, "最大生成トークン数", 200)
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=1.0,
                pad_token_id=self.pad_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        eos_id = self.tokenizer.eos_token_id
        reached_limit = len(new_tokens) >= max_new_tokens and (
            eos_id is None or int(new_tokens[-1]) != eos_id
        )
        return {
            "text": self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip(),
            "reached_limit": reached_limit,
        }

    def compare_generations(
        self,
        prompt: str,
        temperature: float = 0.3,
        repetitions: int = 3,
        top_k: int = 50,
        max_new_tokens: int = 60,
    ) -> list[dict]:
        """条件ごとに複数回生成し、過去の条件も含めて比較表示する。"""
        repetitions = _positive_int(repetitions, "生成回数", 10)
        started = time.perf_counter()
        answers = [
            self._generate(prompt, temperature, top_k, max_new_tokens)
            for _ in range(repetitions)
        ]
        self.generation_history.append(
            {
                "prompt": prompt.strip(),
                "temperature": float(temperature),
                "top_k": int(top_k),
                "max_new_tokens": int(max_new_tokens),
                "seconds": round(time.perf_counter() - started, 1),
                "answers": answers,
            }
        )
        self._display(self._generation_html())
        return answers

    def _generation_html(self) -> str:
        sections = []
        reference = self.generation_history[0]
        if any(
            (batch["prompt"], batch["top_k"], batch["max_new_tokens"])
            != (reference["prompt"], reference["top_k"], reference["max_new_tokens"])
            for batch in self.generation_history[1:]
        ):
            sections.append(
                "<p style='color:#a43'><b>比較条件に注意：</b>"
                "入力文、top-k、または生成上限が途中で変わっています。"
                "temperatureだけの影響を調べるときは、これらを固定してください。</p>"
            )
        for batch in self.generation_history:
            rows = []
            for index, item in enumerate(batch["answers"], 1):
                cutoff = (
                    "<br><small>長さの上限で終了。文の途切れを内容の誤りとして数えません。</small>"
                    if item["reached_limit"]
                    else ""
                )
                rows.append(
                    "<div style='padding:9px;border:1px solid #bbb;margin:8px 0'>"
                    f"{index}回目：{html.escape(item['text'])}{cutoff}</div>"
                )
            sections.append(
                f"<h4>入力：{html.escape(batch['prompt'])}</h4>"
                f"<p>temperature={batch['temperature']}、top-k={batch['top_k']}、"
                f"上限={batch['max_new_tokens']}トークン、"
                f"生成時間={batch['seconds']}秒</p>{''.join(rows)}"
            )
        return "\n".join(sections)

    def compare_prompts(
        self,
        short_prompt: str,
        detailed_prompt: str,
        temperature: float = 0.7,
        top_k: int = 50,
        max_new_tokens: int = 100,
    ) -> list[dict]:
        results = [
            self._generate(short_prompt, temperature, top_k, max_new_tokens),
            self._generate(detailed_prompt, temperature, top_k, max_new_tokens),
        ]
        self._display(
            "".join(
                f"<h4>{label}：{html.escape(prompt)}</h4>"
                f"<p>{html.escape(item['text'])}"
                + ("<br><small>長さの上限で終了</small>" if item["reached_limit"] else "")
                + "</p>"
                for label, prompt, item in zip(
                    ("短い指示", "具体的な指示"),
                    (short_prompt, detailed_prompt),
                    results,
                )
            )
        )
        return results

    def show_answer(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_k: int = 50,
        max_new_tokens: int = 100,
    ) -> dict:
        answer = self._generate(prompt, temperature, top_k, max_new_tokens)
        cutoff = "<br><small>長さの上限で終了</small>" if answer["reached_limit"] else ""
        self._display(f"<p>{html.escape(answer['text'])}{cutoff}</p>")
        return answer

    def save_backup(self, path: str = "llm_intro_results.html") -> Path:
        """授業者用。実測結果をファイルに残す。"""
        target = Path(path).name
        if not self.candidate_history or not self.generation_history:
            raise ValueError("候補表示と文章生成を実行してから保存してください。")
        candidate_prompts = {item["prompt"] for item in self.candidate_history}
        if not {"秋といえば", "冬といえば"}.issubset(candidate_prompts):
            raise ValueError("演習1の秋と冬を実行してから保存してください。")
        if not {0.3, 0.7, 1.2}.issubset(
            {item["temperature"] for item in self.generation_history}
        ):
            raise ValueError("演習2のtemperature三条件を実行してから保存してください。")
        reference = self.generation_history[0]
        if any(
            (item["prompt"], item["top_k"], item["max_new_tokens"])
            != (reference["prompt"], reference["top_k"], reference["max_new_tokens"])
            for item in self.generation_history[1:]
        ):
            raise ValueError("演習2は入力文、top-k、生成上限を固定して実行し直してください。")
        content = (
            "<!doctype html><html lang='ja'><meta charset='utf-8'>"
            "<title>第9回 事前実行結果</title>"
            "<body style='font-family:sans-serif;max-width:1000px;margin:3em auto'>"
            f"<h1>第9回 事前実行結果</h1><p>モデル：{html.escape(self.model_id)}、"
            f"作成日時：{datetime.now().astimezone().isoformat(timespec='seconds')}</p>"
            "<h2>次のトークン候補</h2>"
            + self._candidate_html()
            + "<h2>temperature別の出力</h2>"
            + self._generation_html()
            + "</body></html>"
        )
        output = Path.cwd() / target
        output.write_text(content, encoding="utf-8")
        return output
