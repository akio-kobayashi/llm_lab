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
    エラー回避のため、SFTTrainerに渡す前にデータセットを事前整形する版。
    """
    # 1. トークナイザの設定
    tokenizer.pad_token = tokenizer.eos_token
    # パディング側を右側に統一（SFTでは一般的）
    tokenizer.padding_side = "right" 

    print(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")

    # ---------------------------------------------------------
    # 【修正の肝】: SFTTrainerの内部関数(formatting_func)は使わない。
    # 事前に dataset.map で「text」カラムを作成し、リスト問題を回避する。
    # ---------------------------------------------------------
    def pre_process_data(batch):
        # バッチ処理されるため、batch['input'] はリストです
        texts = []
        for i, o in zip(batch['input'], batch['output']):
            # ここでプロンプトを組み立て、EOSトークンまで含める
            text = f"### 指示:\n{i}\n\n### 応答:\n{o}{tokenizer.eos_token}"
            texts.append(text)
        return {"text": texts}

    print("Formatting dataset explicitly...")
    # 元のカラム(input, output)を残すと混乱の元になるので remove_columns で消す
    train_dataset = train_dataset.map(
        pre_process_data, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    print("Dataset formatted. Example:", train_dataset[0]['text'][:50] + "...")

    # ---------------------------------------------------------
    # 2. 学習設定 (SFTConfig 対応)
    # ---------------------------------------------------------
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
        # dataset_text_fieldを使う場合は max_seq_length をここに入れないのが無難
    }

    # クラス判定による引数生成
    if ConfigClass.__name__ == "SFTConfig":
        # 最新版 trl では max_seq_length を config に含めることが推奨される場合があるが、
        # 下位互換性のため trainer 引数で渡す方式を採用する
        training_args = ConfigClass(**common_args, dataset_text_field="text")
    else:
        training_args = ConfigClass(**common_args)
    
    # ---------------------------------------------------------
    # 3. SFTTrainer の初期化
    # ---------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",  # 事前に作った「text」カラムを指定
        max_seq_length=max_seq_length,
        processing_class=tokenizer, # tokenizer引数の最新版対応
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
