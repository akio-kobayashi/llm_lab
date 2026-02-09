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
    """
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")

    def formatting_prompts_func(example):
        # input/outputカラム名はデータセットに合わせて適宜調整してください
        # 例: {"input": "...", "output": "..."}
        return [f"### 指示:\n{i}\n\n### 応答:\n{o}" for i, o in zip(example['input'], example['output'])]

    # 基本的な学習設定
    common_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "fp16": True,
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": "none",
        # max_seq_length はここでは渡さない
    }

    # 設定オブジェクトを作成
    training_args = ConfigClass(**common_args)
    
    # SFTTrainerの初期化
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=formatting_prompts_func,
        # ここを修正: tokenizer -> processing_class
        processing_class=tokenizer,
        # ここに追加: max_seq_length (SFTConfigを使用しない場合などに必要になることがある)
        # ※ただし最新のtrl + SFTConfigの組み合わせの場合、Config側に含めるべき場合もありますが、
        #  エラー回避のためSFTTrainerの引数として渡すのが現在の安全策です。
    )

    print("Starting LoRA training...")
    trainer.train()
    print("Training finished.")

    # 学習済みアダプタの保存
    adapter_save_path = os.path.join(output_dir, "final_adapter")
    print(f"Saving LoRA adapter to: {adapter_save_path}")
    trainer.model.save_pretrained(adapter_save_path)
    
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
