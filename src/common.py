import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 固定設定 (GEMINI.md に基づく)
# 2023年世代のモデルは最新の chat_template に対応していないため、
# 高性能かつモダンな Qwen2.5-3B-Instruct を採用します。
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
EMB_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_SYSTEM_PROMPT = (
    "あなたは親切で優秀な日本語AIアシスタントです。"
    "必ず「日本語のみ」で回答してください。英語や他の言語は使用しないでください。"
    "思考過程や推論手順は出力せず、最終的な回答のみを日本語で簡潔に返してください。"
)


def _strip_reasoning_trace(text: str) -> str:
    """
    モデルが思考過程を出力した場合に、最終回答のみを返す。
    """
    cleaned = text.strip()

    # <think> ... </think> 形式
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    # "Final Answer:" がある場合はその後ろを優先
    for marker in ("Final Answer:", "最終回答:", "回答:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[-1].strip()

    # "Thinking Process:" だけが残るケースを回避
    if cleaned.lower().startswith("thinking process"):
        lines = [line for line in cleaned.splitlines() if line.strip()]
        if lines:
            cleaned = lines[-1].strip()

    return cleaned


def _is_mostly_non_japanese(text: str) -> bool:
    """
    日本語文字がほぼ含まれない回答を検出する。
    """
    if not text:
        return False
    japanese_chars = sum(
        1 for ch in text
        if ("\u3040" <= ch <= "\u30ff") or ("\u4e00" <= ch <= "\u9fff")
    )
    ascii_letters = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))

    # 日本語が十分あるなら日本語回答とみなす
    if japanese_chars >= 20:
        return False

    # 英字が優勢なら非日本語とみなす
    if ascii_letters >= 30 and ascii_letters > japanese_chars * 3:
        return True

    # 短文時の保険
    return japanese_chars < 3


def _looks_incomplete_answer(text: str) -> bool:
    """
    箇条書き断片や短すぎる中途半端回答を検出する。
    """
    s = text.strip()
    if not s:
        return True

    lower = s.lower()
    bad_prefixes = (
        "*",
        "-",
        "option ",
        "option:",
        "choice ",
        "choice:",
    )
    if lower.startswith(bad_prefixes):
        return True

    # 句点なしの極端な短文は中途半端回答のことが多い
    if len(s) < 20 and all(p not in s for p in ("。", "！", "？")):
        return True

    return False


def _looks_meta_or_leaked_reasoning(text: str) -> bool:
    """
    回答として不適切な自己言及・メタ説明・推論漏洩を検出する。
    """
    s = text.strip().lower()
    if not s:
        return True

    bad_fragments = (
        "thinking process",
        "given the context",
        "issue:",
        "i should",
        "i need to",
        "as an ai",
        "rewrite what's there",
    )
    return any(fragment in s for fragment in bad_fragments)


def _generate_once(
    model,
    tokenizer,
    messages,
    max_new_tokens,
    temperature,
    top_p=DEFAULT_TOP_P,
    top_k=DEFAULT_TOP_K,
):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Greedy search では sampling parameters を渡さない
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False
        # sampling parameters を含めないことで警告を回避

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **gen_kwargs
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return _strip_reasoning_trace(decoded)

def load_llm(model_id=MODEL_ID, use_4bit=True):
    """
    Qwen 3.5 などのLLMをロードする共通関数。
    Colab T4環境を想定し、デフォルトで4bit量子化(QLoRA対応)を使用。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if use_4bit:
        # BitsAndBytesConfig (GEMINI.md 指定)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
    top_k=DEFAULT_TOP_K,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
):
    """
    推論用共通関数。Qwen 3.5 の Chat Template を使用。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    answer = _generate_once(
        model,
        tokenizer,
        messages,
        max_new_tokens,
        temperature,
        top_p=top_p,
        top_k=top_k,
    )

    # 既定設定で英語回答が出た場合のみ、日本語化を1回フォールバックする
    if system_prompt == DEFAULT_SYSTEM_PROMPT and _is_mostly_non_japanese(answer):
        fallback_messages = [
            {
                "role": "system",
                "content": (
                    "あなたは翻訳者です。"
                    "入力文を自然な日本語に翻訳し、翻訳結果のみを出力してください。"
                    "箇条書き記号や前置きは不要です。"
                ),
            },
            {"role": "user", "content": answer},
        ]
        answer = _generate_once(
            model,
            tokenizer,
            fallback_messages,
            max_new_tokens,
            0.0,
            top_p=top_p,
            top_k=top_k,
        )

    # 断片的な回答を避けるための最終リトライ
    if _looks_incomplete_answer(answer) or _looks_meta_or_leaked_reasoning(answer):
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "あなたは日本語アシスタントです。"
                    "箇条書きや選択肢形式ではなく、自然な日本語の1〜2文で完結に回答してください。"
                    "途中で切れた表現を避け、回答本文のみを出力してください。"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        answer = _generate_once(
            model,
            tokenizer,
            retry_messages,
            max_new_tokens,
            0.3,
            top_p=top_p,
            top_k=top_k,
        )

    # それでも不適切なら、事実回答として最低限の日本語を返す
    if _looks_meta_or_leaked_reasoning(answer):
        answer = "現時点で確認できる情報では特定できません。公式情報をご確認ください。"

    return answer
