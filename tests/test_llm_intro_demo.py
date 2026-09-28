"""モデルを取得せずに実行できる Colab 教材の基本検査。"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.llm_intro_demo import LectureDemo, _positive_int


ROOT = Path(__file__).resolve().parents[1]


class LectureDemoTest(unittest.TestCase):
    def test_positive_int_bounds(self):
        self.assertEqual(_positive_int(8, "候補", 30), 8)
        with self.assertRaises(ValueError):
            _positive_int(0, "候補", 30)

    def test_loader_reuses_common_without_quantization(self):
        model = Mock()
        tokenizer = SimpleNamespace(pad_token_id=1, eos_token_id=2)
        load_llm = Mock(return_value=(model, tokenizer))
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            float16="fp16",
            float32="fp32",
        )
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "src.common": SimpleNamespace(load_llm=load_llm)},
        ):
            demo = LectureDemo(prefer_gpu=True)
        self.assertEqual(demo.device, "cuda")
        self.assertEqual(load_llm.call_args.kwargs["use_4bit"], False)
        self.assertEqual(load_llm.call_args.kwargs["torch_dtype"], "fp16")
        model.eval.assert_called_once()

    def test_generation_html_keeps_previous_conditions_and_escapes_text(self):
        demo = LectureDemo.__new__(LectureDemo)
        demo.generation_history = [
            {
                "prompt": "<入力>",
                "top_k": 50,
                "max_new_tokens": 60,
                "temperature": temperature,
                "seconds": 1.0,
                "answers": [{"text": "<回答>", "reached_limit": False}],
            }
            for temperature in (0.3, 0.7, 1.2)
        ]
        result = demo._generation_html()
        self.assertEqual(result.count("<h4>入力"), 3)
        self.assertIn("&lt;回答&gt;", result)
        self.assertNotIn("<回答>", result)
        self.assertNotIn("比較条件に注意", result)

    def test_all_colab_code_cells_are_forms(self):
        path = ROOT / "notebooks" / "09_llm_intro_forms.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 6)
        for cell in code_cells:
            self.assertEqual(cell["metadata"]["cellView"], "form")
            self.assertTrue(cell["source"][0].startswith("#@title"))
            self.assertIn("#@param", "".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
