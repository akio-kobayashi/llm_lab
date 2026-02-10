import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
import trl

# trlのバージョンに応じて、SFTConfigかTrainingArgumentsのどちらを使用するか決定
# ただし、今回はSFTTrainerではなく標準のTrainerを使用するため、この分岐は主に設定クラスの互換性のため
if hasattr(trl, "SFTConfig"):
    from trl import SFTConfig
    # SFTConfigはTrainingArgumentsのラッパーなので、TrainingArgumentsとして扱う
    ConfigClass = TrainingArguments
else:
    ConfigClass = TrainingArguments

def create_lora_model(model, lora_rank=8, lora_alpha=16, lora_dropout=0.05):
    """
    ベースモデルにLoRAアダプタを追加して、学習可能なPEFTモデルを返す。
    """
    # QLoRAではkbit_trainingのためにモデルを準備
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"], # モデル固有のQ/V層
    )
    peft_model = get_peft_model(model, lora_config)
    print("PEFT model created successfully.")
    # 学習可能パラメータの数を確認
    peft_model.print_trainable_parameters()
    return peft_model

def train_lora(
    model,
    tokenizer: AutoTokenizer,
    train_dataset_path: str,
    output_dir: str = "./lora_adapter",
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_steps: int = 100,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
):
    """
    QLoRAの学習を実行する。SFTTrainerではなく、より堅牢なtransformers.Trainerを使用。
    """
    # 1. トークナイザ設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Causal LMの学習では通常、右側パディングを使用
    tokenizer.padding_side = "right" 
    
    print(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")

    # 2. データセットの整形とトークン化
    # プロンプト形式に整形し、tokenizer.eos_tokenで終端をマーク
    def format_text(example):
        return {"text": f"### 指示:\n{example['input']}\n\n### 応答:\n{example['output']}{tokenizer.eos_token}"}

    print("Formatting prompts...")
    # 'input'と'output'カラムはもう不要なので削除
    train_dataset = train_dataset.map(format_text, remove_columns=["input", "output"])

    # トークン化関数
    def tokenize_function(examples):
        # max_lengthで切り捨て、DataCollatorでパディングを処理
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False, 
        )
        # Causal LM学習では、labelsはinput_idsと同じにする
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("Tokenizing dataset...")
    # 'text'カラムはもう不要
    tokenized_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        num_proc=os.cpu_count(), # CPUコア数に応じて並列処理
    )
    
    print(f"Dataset prepared. Columns: {tokenized_dataset.column_names}")

    # 3. TrainingArgumentsの設定
    # gradient_checkpointingはメモリを節約するが、速度が低下する。
    # "use_cache=True is incompatible with gradient checkpointing" 警告を避けるため、
    # use_cacheはFalseにするか、gradient_checkpointingを無効にする。
    # LoRAではgradient_checkpointingが推奨される。
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=True, # T4 GPUではbfloat16よりfp16が一般的
        use_cache=False,
        logging_steps=10,
        save_strategy="no", # 小規模演習のため、チェックポイントは保存しない
        report_to="none",
        remove_unused_columns=False, # DataCollatorが処理するため、未使用カラムは削除しない
        gradient_checkpointing=True, # メモリ節約のため有効化
    )

    # 4. DataCollatorの準備
    # 動的にパディングを処理し、labelsとinput_idsを適切に準備
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Trainerの初期化と学習の開始
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting LoRA training (Standard Trainer)...")
    # SFTTrainerは内部で`peft_model_prepare_for_training`を呼ぶが、
    # `transformers.Trainer`では`model`がPEFTモデルであればそのまま学習される
    trainer.train()
    print("Training finished.")

    # 学習済みモデル（アダプタ）の保存
    adapter_save_path = os.path.join(output_dir, "final_adapter")
    print(f"Saving LoRA adapter to: {adapter_save_path}")
    # base modelを渡すと完全なモデルが保存されるため、peft_model_idを指定してアダプタのみ保存
    model.save_pretrained(adapter_save_path)
    
    return trainer

def load_lora_adapter(model, adapter_path: str):
    """
    学習済みLoRAアダプタをベースモデルにロードする。
    """
    from peft import PeftModel
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
    print(f"Loading LoRA adapter from: {adapter_path}")
    # base modelとアダプタをマージしてロード
    model = PeftModel.from_pretrained(model, adapter_path, trust_remote_code=True)
    print("LoRA adapter loaded successfully.")
    return model