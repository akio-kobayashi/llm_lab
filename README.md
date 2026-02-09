# 日本語LLM実践演習 (Google Colab版)

本リポジトリは、日本語LLM（大規模言語モデル）の基本的な扱い方から、RAG、LoRAチューニング、GradioによるUI作成までを段階的に学ぶための演習教材です。Google Colab (T4 GPU) での実行を想定しています。

## 演習の構成

各ノートブックは、特定のテーマに沿った演習となっています。以下の手順に従って進めてください。

### 【重要】演習の始め方

1. 下記の表にある **「Colabで開く」** ボタンをクリックします。
2. Google Colabが開いたら、まずメニューバーの **「ファイル」→「ドライブにコピーを保存」** をクリックしてください。
3. 新しいタブで「コピー」が開きます。**必ずそのコピーされたノートブックを使って**演習を進めてください。
   （元のノートブックのままでは、編集内容を保存できません）

### ノートブック一覧

| 回 | ノートブック | テーマ | Colabで開く |
| :--- | :--- | :--- | :--- |
| **00** | `00_setup_common.ipynb` | **共通セットアップ** (最初に必ず実行) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/main/notebooks/00_setup_common.ipynb) |
| **01** | `01_gpt_baseline.ipynb` | LLM単体での生成体験 (ハルシネーションの観察) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/main/notebooks/01_gpt_baseline.ipynb) |
| **02** | `02_prompting.ipynb` | プロンプトエンジニアリング基礎 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/main/notebooks/02_prompting.ipynb) |
| **03** | `03_rag_concept_demo.ipynb` | RAGの概念デモ (手動RAG) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/main/notebooks/03_rag_concept_demo.ipynb) |
| **04** | `04_rag_faiss_exercise.ipynb` | Faissを使った本格的なRAG実装 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/main/notebooks/04_rag_faiss_exercise.ipynb) |
| **05** | `05_lora_concept_demo.ipynb` | LoRAファインチューニングの概念デモ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/main/notebooks/05_lora_concept_demo.ipynb) |
| **06** | `06_lora_qlora_exercise.ipynb` | QLoRAによる学習実践 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/main/notebooks/06_lora_qlora_exercise.ipynb) |
| **07** | `07_integrate_gradio.ipynb` | 統合演習 (RAG + LoRA + Gradio UI) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/main/notebooks/07_integrate_gradio.ipynb) |

## 実行手順の詳細

1. **`00_setup_common.ipynb` の実行**:
   - 最初に必ず `00_setup_common.ipynb` を実行し、ライブラリのインストールを行ってください。
   - インストール完了後、**「ランタイムを再起動」** する必要があります（ノートブック内の指示に従ってください）。

2. **各演習の実行**:
   - `01` から順に進めてください。
   - 各ノートブックの冒頭で自動的にリポジトリのクローンが行われ、必要なプログラムが読み込まれます。

## 推奨環境

- **プラットフォーム**: Google Colab
- **GPU**: T4 GPU (無料版で利用可能)
  - ノートブックの設定で `ランタイムのタイプを変更` -> `ハードウェア アクセラレータ` を `T4 GPU` に設定してください。
- **VRAM**: 15GB程度を想定

## よくあるエラーと対処法

- **`OutOfMemoryError` (OOM)**: GPUメモリ不足のエラーです。
  - **対処法**:
    1. Colabの `ランタイム` -> `セッションを再起動` を試してください。
    2. ノートブック内の `batch_size` や `max_seq_length` などのパラメータを小さくしてみてください。
    3. 他のノートブックを開いている場合は、それらを閉じてから再度実行してください。

- **`ModuleNotFoundError`**: ライブラリが見つからないエラーです。
  - **対処法**:
    1. `00_setup_common.ipynb` のインストールセルが正常に完了しているか確認してください。
    2. `sys.path` を追加するセルが正しく実行されているか確認してください。

- **モデルのロードに失敗する**:
  - **対処法**:
    1. Hugging Face Hubがダウンしていないか確認してください。
    2. `model_id` の文字列が正しいか確認してください。
    3. インターネット接続が安定しているか確認してください。

## 動作確認チェックリスト

- [ ] `00_setup_common.ipynb` がエラーなく最後まで実行できる。
- [ ] `01_gpt_baseline.ipynb` でLLMからの応答が生成される。
- [ ] `04_rag_faiss_exercise.ipynb` で質問に関連するドキュメントが検索され、表示される。
- [ ] `06_lora_qlora_exercise.ipynb` の学習が完了し、学習済みアダプタが保存される。
- [ ] `07_integrate_gradio.ipynb` でGradio UIが起動し、RAGやLoRAのON/OFFを切り替えて応答が生成される。
