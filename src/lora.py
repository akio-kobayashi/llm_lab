import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
import trl
from trl import SFTTrainer

# trlのバージョンに応じて、SFTConfigかTrainingArgumentsのどちらを使用するか決定
if hasattr(trl, "SFTConfig"):
    from trl import SFTConfig
    ConfigClass = SFTConfig
else:
    ConfigClass = TrainingArguments

def create_lora_model(model, lora_rank=8, lora_alpha=16, lora_dropout=0.05):
    """
    ベースモデルにLoRAアダプタを追加して、学習可能なPEFTモデルを返す。
    """
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    peft_model = get_peft_model(model, lora_config)
    print("PEFT model created successfully.")
    peft_model.print_trainable_parameters()
    return peft_model

import inspect
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

from transformers import Trainer, DataCollatorForLanguageModeling

def train_lora(
    model,
    tokenizer,
    train_dataset_path: str,
    output_dir: str = "./lora_adapter",
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_steps: int = 100,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
):
    """
    QLoRAの学習を実行する。
    不安定なSFTTrainerを廃止し、堅牢なtransformers.Trainerを使用する最終版。
    """
    # 1. トークナイザ設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 右側パディング（標準Trainer用）
    tokenizer.padding_side = "right" 
    
    print(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")

    # ---------------------------------------------------------
    # 【変更点】SFTTrainerの自動処理に頼らず、手動で確実にデータを作る
    # ---------------------------------------------------------
    
    # 手順1: テキストを整形する
    def format_text(example):
        # input/outputを結合して一つの文章にする
        return {"text": f"### 指示:\n{example['input']}\n\n### 応答:\n{example['output']}{tokenizer.eos_token}"}

    print("Formatting prompts...")
    train_dataset = train_dataset.map(format_text, remove_columns=["input", "output"])

    # 手順2: トークナイズを行う (ID化)
    def tokenize_function(examples):
        # テキストをトークン化
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False, # DataCollatorで動的にパディングするためここではFalse
        )
        # Causal LM学習では、labels は input_ids と同じにするのが基本
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("Tokenizing dataset...")
    # 'text'カラムはもう不要なので削除し、モデルが必要とするカラムだけ残す
    tokenized_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )
    
    print(f"Dataset prepared. Columns: {tokenized_dataset.column_names}")

    # ---------------------------------------------------------
    # 2. 学習設定 (標準のTrainingArgumentsを使用)
    # ---------------------------------------------------------
    # SFTConfigではなく、親クラスのTrainingArgumentsを確実に使う
    from transformers import TrainingArguments
    
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False, # データセットのカラム削除を防ぐ
    )

    # ---------------------------------------------------------
    # 3. Trainer の初期化 (SFTTrainerではない)
    # ---------------------------------------------------------
    # DataCollator: バッチごとに長さを揃えるための機能
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting LoRA training (Standard Trainer)...")
    trainer.train()
    print("Training finished.")

    # 保存処理
    adapter_save_path = os.path.join(output_dir, "final_adapter")
    print(f"Saving LoRA adapter to: {adapter_save_path}")
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
    model = PeftModel.from_pretrained(model, adapter_path, trust_remote_code=True)
    print("LoRA adapter loaded successfully.")
    return model
