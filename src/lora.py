import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import trl

def create_lora_model(model, lora_rank=8, lora_alpha=16, lora_dropout=0.05):
    """
    ベースモデルにLoRAアダプタを追加。
    """
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # 全ての線形層を対象にすることで学習を安定化
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.config.use_cache = False
    
    print("PEFT model created successfully.")
    peft_model.print_trainable_parameters()
    return peft_model

def train_lora(
    model,
    tokenizer,
    train_dataset_path: str,
    output_dir: str = "./lora_adapter",
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_steps: int = 50,
    learning_rate: float = 5e-5, # 学習率をさらに下げる
    max_seq_length: int = 512,
):
    """
    超安定版学習ロジック。
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    print(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")

    def format_for_training(example):
        # 独自タグを廃止し、モデル標準のチャットテンプレートのみを使用
        # 指示の内容自体にフォーマット指定を含める
        user_prompt = f"{example['input']}\n回答は必ず以下のJSON形式で出力してください。\n{{ \"answer\": \"...\", \"confidence\": \"...\" }}"
        
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example['output']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    print("Formatting prompts with standard chat template...")
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
        lr_scheduler_type="constant_with_warmup", # 急激な変化を避ける
        warmup_steps=10,
        max_grad_norm=0.3, # 勾配爆発を抑制
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting LoRA training...")
    trainer.train()
    
    # 保存前にキャッシュを戻す
    model.config.use_cache = True
    
    adapter_save_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_save_path)
    return trainer

def load_lora_adapter(model, adapter_path: str):
    from peft import PeftModel
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model
