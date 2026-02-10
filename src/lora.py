import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

def create_lora_model(model, lora_rank=8, lora_alpha=16, lora_dropout=0.05):
    """
    ベースモデルにLoRAアダプタを追加。
    """
    # 学習時はキャッシュを完全に無効化
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # 安定性のためにQ/Vのみ
        target_modules=["q_proj", "v_proj"], 
    )
    peft_model = get_peft_model(model, lora_config)
    
    # PEFTモデル側でもキャッシュを無効化
    peft_model.config.use_cache = False
    
    print("PEFT model created successfully.")
    return peft_model

def train_lora(
    model,
    tokenizer,
    train_dataset_path: str,
    output_dir: str = "./lora_adapter",
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_steps: int = 40,
    learning_rate: float = 2e-5, # さらに学習率を下げて崩壊を阻止
    max_seq_length: int = 512,
):
    """
    崩壊を阻止する超安定版。
    """
    # 学習開始前に再度キャッシュを確認
    model.config.use_cache = False
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")

    def format_for_training(example):
        # ノートブック 01-07 の形式と完全に一致させる
        text = f"以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{example['input']}\n\n### 応答:\n{example['output']}{tokenizer.eos_token}"
        return {"text": text}

    train_dataset = train_dataset.map(format_for_training, remove_columns=["input", "output"])

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        lr_scheduler_type="constant",
        max_grad_norm=0.1, # 勾配を厳しく制限
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting ultra-stable LoRA training...")
    trainer.train()
    
    # 学習終了後、推論のためにキャッシュを戻す
    model.config.use_cache = True
    
    adapter_save_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_save_path)
    return trainer

def load_lora_adapter(model, adapter_path: str):
    from peft import PeftModel
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # 推論用にキャッシュとevalモードを確実に設定
    model.config.use_cache = True
    model.eval()
    
    print("LoRA adapter loaded successfully.")
    return model
