import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, GenerationConfig

# --- 編集可能: モデル設定 ---
# 楽天の1.5B軽量モデル。日本語能力が高く、非常に高速です。
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
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # 楽天モデルは標準的なFastTokenizerで動作します
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    チャットテンプレートを使用して、モデルに最適な形式で生成します。
    """
    # モデル固有のフォーマット（chat_template）を適用
    # これにより「ユーザー：」などのタグを自分で書く必要がなくなります
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
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

if __name__ == '__main__':
    try:
        model, tokenizer = load_llm()

        # 質問をそのまま入力するだけ！
        user_query = "日本で一番高い山は何ですか？また、二番目についても教えてください。"
        
        print(f"\n--- 質問 ---\n{user_query}")

        generated_text = generate_text(model, tokenizer, user_query)
        
        print(f"\n--- 回答 ---\n{generated_text}")

        # メモリ解放
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Failed: {e}")