あなたは教育用の演習教材を作るソフトウェアエンジニア兼TAです。以下の仕様に従い、GitHubリポジトリとしてそのまま置ける形で、Google Colab向けのJupyterノートブック（.ipynb）を7回分作成してください。加えて、共通セットアップノートブック、データ例、README、requirements相当のメモを作成してください。

# 0. 全体要件
- 対象：日本語LLM（3Bクラス）をGoogle Colabで扱う演習（全7回）
- 各回は「1ノートブック=1回」で完結すること
- ただし依存関係のインストールや共通関数は共通化し、初回に一度実行すれば以後は軽い状態にすること
- 目的：General-purpose TransformerとしてのLLM理解 → プロンプト → RAG → Faiss → LoRA → 統合（Gradio）までを段階的に学ぶ
- 実行環境：Google Colab（T4想定、VRAM 15GB程度）
- 量子化：推論は4bit量子化（bitsandbytes）を基本とする
- LoRA：QLoRA（4bit + peft）で最小成功体験（短時間で完走）
- RAG：Faiss（CPU）を使用し、embeddingは軽量モデルを固定で使う
- UI：最終回にGradioでデモUI
- 文章：日本語で説明、句読点は「、」「。」で統一
- 教材の方針：学生が詰まりやすい自由度は極力なくし、触る場所を限定する。重要部は「編集禁止セル」として明示する。

# 1. モデル要件（LLM）
- Hugging Faceの日本語LLM（3Bクラス）を使用すること
- 例：stabilityai/japanese-stablelm-3b-4e1t-instruct を第一候補とし、モデルIDを変数で差し替え可能にしておくこと
- tokenizer/modelロードは共通関数化すること
- 生成はtransformersのpipelineまたはgenerateで統一し、温度などのデフォルト値を固定すること

# 2. リポジトリ構成（出力してほしいファイル）
以下のディレクトリ構造でファイルを作成すること。

repo-root/
  README.md
  notebooks/
    00_setup_common.ipynb
    01_gpt_baseline.ipynb
    02_prompting.ipynb
    03_rag_concept_demo.ipynb
    04_rag_faiss_exercise.ipynb
    05_lora_concept_demo.ipynb
    06_lora_qlora_exercise.ipynb
    06_integrate_gradio.ipynb
  data/
    docs/
      anime_docs_sample.jsonl
    lora/
      lora_train_sample.jsonl
      lora_eval_questions.json
  src/
    common.py
    rag.py
    lora.py
    ui.py
  assets/
    (空でよい)
  LICENSE
  NOTICE.md

- notebooksはColabでそのまま開けること
- srcのPythonはノートブックからimportして使うこと（ただしColabなのでsys.path追加セルを用意）
- dataは最小のサンプルを同梱すること（著作権上の問題が起きないよう、創作のダミーデータでよい）
- READMEには「各回で何をするか」「実行手順」「想定GPU」「よくあるエラー対処」を書くこと
- LICENSEはMITでよい
- NOTICE.mdには、Wikipedia等の外部データを使う場合の注意（出典URLとライセンス表記が必要）を簡潔に書くこと

# 3. データ仕様（RAG用）
- data/docs/anime_docs_sample.jsonl は、1行1チャンクのJSONL
- スキーマ例（必須キー）：
  - doc_id: 作品識別子（短い英小文字）
  - title: 作品名
  - section: セクション名（概要、あらすじ、登場人物、制作など）
  - text: 本文（300〜800文字程度）
  - source_url: 出典URL（サンプルはダミーURLでよい）
  - retrieved_at: 日付文字列（YYYY-MM-DD）
- サンプルとして10作品相当のdoc_idを用意（内容は架空でよい）
- RAGの学習や評価ではなく、「検索→提示→生成」を体験する目的で設計

# 4. データ仕様（LoRA用）
- data/lora/lora_train_sample.jsonl は、1行1サンプルのJSONL
- スキーマ（必須キー）：
  - input: ユーザー入力
  - output: 望ましい出力（JSON形式で固定）
- タスクは「出力フォーマット固定」を採用すること（例：必ず以下のJSONキーを出す）
  - { "answer": "...", "evidence": "...", "confidence": "low|medium|high" }
- サンプルは50〜100件のダミーでよい（短く）
- data/lora/lora_eval_questions.json は評価用の質問を10件程度

# 5. RAG実装要件（Faissの位置付けを明確に）
- 文書チャンクをembeddingし、Faissに登録する
- 質問をembeddingし、Faiss検索でtop_k件を取得する
- 取得したチャンクをプロンプトに埋め込んでLLMに渡す
- ノートブックでは必ず「取得チャンク」を表示し、どれを読んで答えたのか見える化する
- インデックスは事前作成が基本。04回では「作成セル」もあるが、基本は配布済みを読み込む流れにする（時間短縮）
- インデックスを保存・読み込みできるようにする（faiss.write_index / read_index）
- embeddingモデルは軽量で固定（例：sentence-transformers系の小型モデル）。モデルIDは変数にする

# 6. LoRA実装要件（QLoRA）
- bitsandbytes 4bit量子化 + peftのLoRAで学習する
- 学習は短時間で完走する設定（少ステップ、少データ）
- OOM回避のため、batch sizeやgradient accumulationなどを保守的に設定
- 「学習前後の比較」を必ず実施し、フォーマット遵守が改善したことを確認する
- 学習済みアダプタの保存・読み込みを実装する（save_pretrained / from_pretrained）

# 7. 各回ノートブックの内容（必須）
## 00_setup_common.ipynb
- GPU確認、ライブラリinstall、共通関数の動作確認
- src/以下をimportできるようにパス設定
- 以降の回で共通に使うload_llm(), generate_text()を確認

## 01_gpt_baseline.ipynb
- LLM単体の生成を体験
- 同じ質問を複数回投げ、揺れや根拠のなさを観察
- 学生が触るのは質問文のみ

## 02_prompting.ipynb
- プロンプトで振る舞いを制御（制約、出力形式）
- それでも知識は増えないことを確認
- 学生はテンプレの一部のみ編集

## 03_rag_concept_demo.ipynb
- 検索拡張の概念デモ（検索→読む→生成）
- 取得チャンクを表示
- RAGなし vs RAGありを比較

## 04_rag_faiss_exercise.ipynb
- Faiss indexを使った動的検索を演習
- top_kを変えたり、質問を言い換えたりして検索結果が変わることを観察
- 失敗例（曖昧質問）も含める

## 05_lora_concept_demo.ipynb
- 学習済みLoRAアダプタを読み込み、フォーマット固定効果をデモ
- RAGとは別物であることを明確にする

## 06_lora_qlora_exercise.ipynb
- 学生が10件だけ学習データを追加できるセルを用意
- QLoRAで短時間学習し、学習前後比較
- 成功基準は「JSONが壊れない割合が増える」

## 06_integrate_gradio.ipynb
- RAG on/off、LoRA on/offを切り替えられるGradio UIを作成
- 回答と引用チャンクを表示
- ここまでの総まとめ（GPTは部品、設計がシステムを作る）

# 8. 実装上の注意（重要）
- 学生が触るセルには「ここを編集」と明示し、それ以外は「編集禁止」と明示
- 例外処理、エラー時のメッセージ、OOM時の対処（batch/seq/stepsを減らす）を各回に記載
- 依存関係はノート00に集約するが、各ノートの冒頭にも「00を実行済みか」を確認するセルを入れる
- コードは可読性優先。過度な省略（難しい内包表記など）は避け、初心者が追える形にする
- ダミーデータは創作と明示し、実データ（Wikipediaなど）を使う場合の出典表記ルールをNOTICEに書く

# 9. 生成物の品質要件
- すべてのノートブックが上から順に実行できること
- Colabでの実行を想定し、ファイルパスやpip installが動くこと
- ノートブック内の説明は日本語で簡潔に、しかし手順が省略されないこと
- READMEに「想定動作確認のチェックリスト」を含めること

以上を満たす形で、指定のファイル一式を生成してください。

# 10. 推奨の固定設定（必ず採用）
- LLM:
  - MODEL_ID = "stabilityai/japanese-stablelm-3b-4e1t-instruct"
  - 量子化: bitsandbytes 4bit (nf4), compute_dtype=bfloat16（不可ならfloat16）
  - 生成パラメータ（共通）:
    - max_new_tokens=256
    - temperature=0.7
    - top_p=0.9
    - repetition_penalty=1.05
    - do_sample=True
- Embedding:
  - EMB_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  - embeddingはCPUでよい（Colab T4でも十分）
- Faiss:
  - IndexFlatIP（内積）を使用し、embeddingはL2正規化してcos類似相当にする
  - index保存先: data/index/faiss.index
  - 対応するchunk情報: data/index/chunks.jsonl
- LoRA（QLoRA）:
  - target_modules = ["q_proj","v_proj"]（存在しない場合はモデルのnamed_modulesを表示して候補を出す）
  - r=8, lora_alpha=16, lora_dropout=0.05
  - per_device_train_batch_size=1
  - gradient_accumulation_steps=8
  - max_steps=100（3択: 50/100/200）
  - learning_rate=2e-4
  - logging_steps=10
  - save_total_limit=1
  - fp16=True（bfloat16が使える場合はbf16=True）
  - アダプタ保存先: data/lora/adapters/demo_adapter/
- Gradio:
  - RAG on/off, LoRA on/off, top_k(1-5)のUI部品を必ず入れる
