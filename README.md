# 日本語LLM実践演習 (Google Colab版)

本リポジトリは、日本語LLM（大規模言語モデル）の基本的な扱い方から、RAG、LoRAチューニング、GradioによるUI作成までを段階的に学ぶための演習教材です。Google Colab (T4 GPU) での実行を想定しています。

## 演習の構成

各ノートブックは、特定のテーマに沿った演習となっています。`notebooks/` ディレクトリにあるファイルを、番号順に実行してください。

- **00_setup_common.ipynb**:
  - 演習全体で利用するライブラリのインストールや、共通関数のセットアップを行います。**最初に必ず実行してください。**

- **01_gpt_baseline.ipynb**:
  - LLMをそのまま使った場合の基本的なテキスト生成を体験します。同じ質問でも回答が揺らぐことや、事実に基づかない回答（ハルシネーション）を観察します。

- **02_prompting.ipynb**:
  - プロンプトエンジニアリングの初歩を学びます。指示の与え方によって、LLMの振る舞いや出力形式を制御できることを確認します。

- **03_rag_concept_demo.ipynb**:
  - RAG（Retrieval-Augmented Generation）の概念を学びます。外部知識（ドキュメント）を検索し、その内容に基づいて回答を生成する仕組みを体験します。

- **04_rag_faiss_exercise.ipynb**:
  - FAISSライブラリを使って、より本格的なRAGシステムを構築します。質問を変えながら、検索されるドキュメントがどう変わるかを観察します。

- **05_lora_concept_demo.ipynb**:
  - LoRA（Low-Rank Adaptation）によるファインチューニングの効果を体験します。学習済みのアダプタを読み込み、特定の出力形式を遵守するようになったモデルの振る舞いを確認します。

- **06_lora_qlora_exercise.ipynb**:
  - QLoRA（Quantized LoRA）を使って、実際にモデルのファインチューニングを行います。少量のデータで短時間の学習を行い、学習前後でモデルの性能がどう変化するかを比較します。

- **07_integrate_gradio.ipynb**:
  - これまでの演習で作成したRAGとLoRAの機能を統合し、Gradioを使ってインタラクティブなWeb UIを作成します。

## 実行手順

1. **Google Colabでノートブックを開く**:
   - 各 `.ipynb` ファイルをGoogle Colabで開きます。GitHubリポジトリのURLをColabの「GitHub」タブに貼り付けるのが便利です。
   - 例: `https://colab.research.google.com/github/[YourUsername]/[YourRepoName]/blob/main/notebooks/00_setup_common.ipynb`

2. **リポジトリのクローン**:
   - 各ノートブックの冒頭には、本リポジトリをColab環境にクローンするセルがあります。まずそれを実行してください。
   - `!git clone https://github.com/your-account/llm_lab.git`

3. **セットアップの実行**:
   - `00_setup_common.ipynb` を開き、上から順にセルを実行します。これにより、必要なライブラリがインストールされ、以降の演習の準備が整います。

4. **各演習の実行**:
   - `01` から `07` までのノートブックを順に実行します。各ノートブック内の説明を読み、指示に従ってセルを実行・編集してください。

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
