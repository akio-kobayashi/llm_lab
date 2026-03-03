import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 固定設定 (GEMINI.md に基づく)
MODEL_ID = "Qwen/Qwen3.5-4B"
EMB_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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
    system_prompt=(
        "あなたは親切で優秀な日本語AIアシスタントです。"
        "思考過程や推論手順は出力せず、最終的な回答のみを日本語で簡潔に返してください。"
    ),
):
    """
    推論用共通関数。Qwen 3.5 の Chat Template を使用。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
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
    
    # 入力部分をカットしてデコード
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return _strip_reasoning_trace(decoded)
