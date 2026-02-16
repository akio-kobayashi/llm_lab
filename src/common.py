import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# --- 編集可能: モデル設定 ---
# stabilityai/japanese-stablelm-3b-4e1t-instruct
# line-corporation/japanese-large-lm-3.6b-instruction-sft
DEFAULT_MODEL_ID = "stabilityai/japanese-stablelm-3b-4e1t-instruct"
# --- 編集可能ここまで ---

def load_llm(model_id: str = DEFAULT_MODEL_ID, use_4bit: bool = True):
    """
    指定されたモデルIDのLLMとTokenizerをロードする。
    4bit量子化をデフォルトで有効にする。

    Args:
        model_id (str): ロードするHugging FaceモデルのID。
        use_4bit (bool): 4bit量子化を使用するかどうか。

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_id}")
    print(f"Using 4-bit quantization: {use_4bit}")

    # T4などbf16非対応GPUではfloat16を使う
    compute_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16

    if use_4bit:
        # 4bit量子化の設定
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        bnb_config = None

    # Tokenizerのロード
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception as e:
        print(f"Error loading tokenizer for {model_id}: {e}")
        raise

    # japanese-stablelm系では <|endoftext|> をEOSとして明示し、
    # BOSがEOSと同一IDになる状態は回避する
    vocab = tokenizer.get_vocab()
    if "<|endoftext|>" in vocab:
        eos_id = vocab["<|endoftext|>"]
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.eos_token_id = eos_id
    if tokenizer.bos_token_id == tokenizer.eos_token_id or tokenizer.bos_token_id is None:
        tokenizer.bos_token = None
        tokenizer.bos_token_id = None

    # 一部モデルでpad_token_id属性が欠けるケースに備えてconfigを補完
    try:
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading config for {model_id}: {e}")
        raise

    if not hasattr(model_config, "pad_token_id") or model_config.pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            model_config.pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            model_config.pad_token_id = tokenizer.eos_token_id

    # モデルのロード
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=model_config,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",  # GPUに自動で割り当て
        )
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config") and getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = model.config.pad_token_id
        model.eval() # 評価モード
    except Exception as e:
        print(f"Error loading model for {model_id}: {e}")
        raise

    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    do_sample: bool = True,
):
    """
    ロード済みのモデルとTokenizerを使ってテキストを生成する。

    Args:
        model: ロード済みモデル。
        tokenizer: ロード済みTokenizer。
        prompt (str): 生成の元となるプロンプト。
        max_new_tokens (int): 生成する最大トークン数。
        temperature (float): 生成の多様性を制御。
        top_p (float): 上位pの確率を持つトークンからサンプリング。
        repetition_penalty (float): 同じ単語の繰り返しを抑制。
        do_sample (bool): サンプリングを使用するかどうか。

    Returns:
        str: 生成されたテキスト。
    """
    # パイプラインの作成
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # テキスト生成の実行
    try:
        generated = text_generation_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )
        # パイプラインの出力はリストなので、最初の要素の生成テキストを返す
        if generated and len(generated) > 0:
            return generated[0]["generated_text"]
        else:
            return "Error: Text generation failed."

    except Exception as e:
        print(f"Error during text generation: {e}")
        return f"Error: {e}"

if __name__ == '__main__':
    # このファイルが直接実行された場合の動作テスト
    print("--- Running common.py self-test ---")

    # 4bitでロード
    try:
        model, tokenizer = load_llm(use_4bit=True)

        # テキスト生成
        prompt = "日本で一番高い山はなんですか？"
        print(f"\nPrompt: {prompt}")

        generated_text = generate_text(model, tokenizer, prompt)
        print("\nGenerated Text:")
        print(generated_text)

        # モデルとTokenizerを解放 (メモリ節約)
        del model
        del tokenizer
        torch.cuda.empty_cache()
        print("\n--- Self-test finished ---")

    except Exception as e:
        print(f"\nSelf-test failed: {e}")
        print("Please ensure you have enough GPU memory and required libraries are installed.")
