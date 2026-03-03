import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 固定設定 (GEMINI.md に基づく)
MODEL_ID = "Qwen/Qwen3.5-4B"
EMB_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SYSTEM_PROMPT = (
    "あなたは親切で優秀な日本語AIアシスタントです。"
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
    return japanese_chars < 3


def _generate_once(model, tokenizer, messages, max_new_tokens, temperature):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
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
    temperature=0.7,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
):
    """
    推論用共通関数。Qwen 3.5 の Chat Template を使用。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    answer = _generate_once(model, tokenizer, messages, max_new_tokens, temperature)

    # 既定設定で英語回答が出た場合のみ、日本語化を1回フォールバックする
    if system_prompt == DEFAULT_SYSTEM_PROMPT and _is_mostly_non_japanese(answer):
        fallback_messages = [
            {
                "role": "system",
                "content": (
                    "あなたは日本語の編集者です。"
                    "入力文を自然で簡潔な日本語に言い換え、回答本文のみを出力してください。"
                ),
            },
            {"role": "user", "content": answer},
        ]
        answer = _generate_once(model, tokenizer, fallback_messages, max_new_tokens, 0.2)

    return answer
