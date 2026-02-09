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
    実行環境のSFTTrainerの仕様を動的に判定して引数を構築する安全版。
    """
    # 1. トークナイザ設定
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")

    # 2. フォーマット関数
    # エラー回避のため、純粋なリスト[str]を返す形にします
    def formatting_prompts_func(example):
        output_texts = []
        for i, o in zip(example['input'], example['output']):
            text = f"### 指示:\n{i}\n\n### 応答:\n{o}{tokenizer.eos_token}"
            output_texts.append(text)
        return output_texts

    # 3. 学習設定 (Config)
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
        "remove_unused_columns": False, 
    }

    # SFTConfigが使える場合は、max_seq_lengthなどはConfigに入れるのが今の流儀
    if "dataset_kwargs" in inspect.signature(SFTConfig).parameters:
         # 新しいバージョンでは dataset_kwargs を使う場合があるが、
         # ここでは安全のため max_seq_length だけチェックして入れる
         pass

    # max_seq_length を Config に入れるべきかどうかの判定
    # (ConfigClass が SFTConfig の場合のみ)
    if issubclass(ConfigClass, SFTConfig):
        # SFTConfigなら max_seq_length を受け取る可能性がある
        try:
            training_args = ConfigClass(**common_args, max_seq_length=max_seq_length)
            print("Config: SFTConfig initialized with max_seq_length.")
        except TypeError:
            # 受け取らないバージョンなら外して初期化
            training_args = ConfigClass(**common_args)
            print("Config: SFTConfig initialized without max_seq_length.")
    else:
        # TrainingArguments ならそのまま
        training_args = ConfigClass(**common_args)


    # 4. SFTTrainer の引数構築（動的検査）
    # 現在の環境の SFTTrainer.__init__ の引数を全て取得
    signature = inspect.signature(SFTTrainer.__init__)
    valid_params = signature.parameters.keys()
    
    print(f"Detected SFTTrainer arguments: {list(valid_params)}")

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    # --- 引数の動的割り当て ---

    # (A) formatting_func (ほぼ全てのバージョンで必須)
    if "formatting_func" in valid_params:
        trainer_kwargs["formatting_func"] = formatting_prompts_func

    # (B) max_seq_length (Configに入れた場合は不要だが、Trainer引数にあるなら念のため渡すか、Config優先か)
    # 通常、Trainer引数で渡すのが安全
    if "max_seq_length" in valid_params:
        trainer_kwargs["max_seq_length"] = max_seq_length

    # (C) Tokenizer / Processing Class
    if "processing_class" in valid_params:
        print("Using 'processing_class' argument.")
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in valid_params:
        print("Using 'tokenizer' argument.")
        trainer_kwargs["tokenizer"] = tokenizer
    
    # (D) dataset_text_field
    # これがあると formatting_func と競合する場合があるが、
    # formatting_func があるので、あえて指定しない方が安全なケースが多い。
    # しかし、一部バージョンで必須の場合は空文字などを入れる必要があるか確認。
    # 今回のエラーは "unexpected keyword" だったので、存在しない場合は渡さない。
    if "dataset_text_field" in valid_params:
        # formatting_funcを使う場合、dataset_text_fieldは通常不要だが、
        # 念のためNone明示か、あるいはformatting_func優先で渡さない。
        # 今回は「渡さない」を選択（formatting_funcがあれば無視されるか、競合するため）
        pass

    # 5. 初期化と実行
    print("Initializing SFTTrainer with validated arguments...")
    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except Exception as e:
        # 万が一のパッキングエラー等をキャッチ
        print(f"Initialization failed: {e}")
        print("Detailed parameters attempt:", trainer_kwargs.keys())
        raise e

    print("Starting LoRA training...")
    trainer.train()
    print("Training finished.")

    # 保存
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
