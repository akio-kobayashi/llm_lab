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
    # 勾配チェックポインティングを有効にする場合、これが必須
    model.gradient_checkpointing_enable()
    # 量子化モデル用の前処理
    model = prepare_model_for_kbit_training(model)
    # 入力に勾配を要求するように設定 (Trainer用)
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.config.use_cache = False # 学習時はFalse
    
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
    max_steps: int = 100,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
):
    """
    標準Trainerを使用した堅牢な学習。
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    print(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")

    # 推論時 (generate_text) と同じ Chat Template を使用して学習データを作成
    def format_chat_template(example):
        # ノートブックのプロンプト構造に合わせる
        user_content = f"### 指示:\n{example['input']}\n\n### 応答:\n"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example['output']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    print("Formatting prompts with Chat Template...")
    train_dataset = train_dataset.map(format_chat_template, remove_columns=["input", "output"])

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

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
        remove_unused_columns=False,
        gradient_checkpointing=True,
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
    print("Training finished.")

    adapter_save_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_save_path)
    return trainer

def load_lora_adapter(model, adapter_path: str):
    from peft import PeftModel
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path, trust_remote_code=True)
    print("LoRA adapter loaded successfully.")
    return model
