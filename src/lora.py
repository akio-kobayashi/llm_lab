import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

def create_lora_model(model, lora_rank=8, lora_alpha=16, lora_dropout=0.05):
    """
    ベースモデルにLoRAアダプタを追加して、学習可能なPEFTモデルを返す。

    Args:
        model: ベースとなるLLM。
        lora_rank (int): LoRAのランク。
        lora_alpha (int): LoRAのスケーリング係数。
        lora_dropout (float): LoRA層のドロップアウト率。

    Returns:
        peft_model: 学習可能なPEFTモデル。
    """
    # 量子化モデルを学習可能にする前処理
    model = prepare_model_for_kbit_training(model)

    # LoRAの設定
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    # PEFTモデルの作成
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

    Args:
        model: LoRAアダプタが追加されたPEFTモデル。
        tokenizer: Tokenizer。
        train_dataset_path (str): 学習データセットのパス (JSONL)。
        output_dir (str): 学習済みアダプタの保存先。
        ... (その他の学習ハイパーパラメータ)

    Returns:
        trainer: 学習済みTrainerオブジェクト。
    """
    # Tokenizerのパディング設定
    tokenizer.pad_token = tokenizer.eos_token
    
    # データセットのロード
    print(f"Loading training dataset from: {train_dataset_path}")
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")

    # プロンプト形式に整形する関数
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['input'])):
            text = f"### 指示:\n{example['input'][i]}\n\n### 応答:\n{example['output'][i]}"
            output_texts.append(text)
        return output_texts

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=True, # 量子化モデルではfp16=Trueが推奨される
        logging_steps=10,
        save_strategy="no", # 小規模演習のため保存は最後のみ
        report_to="none", # wandbなどを使わない場合はnone
    )

    # SFTTrainerの初期化
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        formatting_func=formatting_prompts_func,
        max_seq_length=max_seq_length,
        args=training_args,
    )

    # 学習の開始
    print("Starting LoRA training...")
    trainer.train()
    print("Training finished.")

    # 学習済みモデル（アダプタ）の保存
    adapter_save_path = os.path.join(output_dir, "final_adapter")
    print(f"Saving LoRA adapter to: {adapter_save_path}")
    model.save_pretrained(adapter_save_path)
    
    return trainer

def load_lora_adapter(model, adapter_path: str):
    """
    学習済みLoRAアダプタをベースモデルにロードする。
    この関数はPEFT 0.7.0以降では非推奨になる可能性があるため注意。
    from_pretrainedで直接ロードするのが一般的。

    Args:
        model: ベースモデル。
        adapter_path (str): アダプタが保存されているディレクトリのパス。

    Returns:
        model: アダプタがマージされたモデル。
    """
    from peft import PeftModel
    
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
    print(f"Loading LoRA adapter from: {adapter_path}")
    # PEFTモデルとしてアダプタをロード
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # 必要に応じてマージ
    # model = model.merge_and_unload()
    
    print("LoRA adapter loaded successfully.")
    return model

if __name__ == '__main__':
    # このファイルは直接実行してのテストは難しい
    # (GPUとベースモデルのロードが必要なため)
    # ノートブック上で各関数を呼び出してテストすることを想定
    print("--- lora.py ---")
    print("This script contains helper functions for QLoRA training.")
    print("Please use these functions within a Jupyter notebook with a loaded model.")
    print("Functions available: create_lora_model, train_lora, load_lora_adapter")
