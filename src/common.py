import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, GenerationConfig

# --- 編集可能: モデル設定 ---
DEFAULT_MODEL_ID = "Rakuten/RakutenAI-2.0-mini-instruct"
# --- 編集可能ここまで ---

def load_llm(model_id: str = DEFAULT_MODEL_ID, use_4bit: bool = True):
    print(f"Loading model: {model_id}")

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, 
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 推論用にキャッシュを有効化
    model.config.use_cache = True
    model.eval()
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def generate_text(
    model,
    tokenizer,
    user_input: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
):
    """
    入力を受け取り、モデルに適した形式でテキストを生成します。
    """
    # キャッシュを確実に有効化 (学習後にFalseのままになっているのを防ぐ)
    model.config.use_cache = True
    
    # ノートブック側の手動タグを優先し、二重ラップを防ぐ
    if "### 指示:" in user_input:
        prompt = user_input
    else:
        messages = [{"role": "user", "content": user_input}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    try:
        results = text_gen(
            prompt,
            generation_config=gen_config,
            return_full_text=False,
        )
        return results[0]["generated_text"]
    except Exception as e:
        return f"Error: {e}"