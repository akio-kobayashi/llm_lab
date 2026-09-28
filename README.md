# Qwen2.5-3B-Instruct 演習キット for Google Colab

このリポジトリは、**Qwen2.5-3B-Instruct** を Google Colab（T4 GPU）で動かし、プロンプト、RAG、そして **AIエージェント** までを段階的に学ぶための演習教材です。

## 特徴
- **採用モデルは Qwen2.5-3B-Instruct**。
- **4bit量子化 (bitsandbytes)** により、無料版Colabで安定動作。
- **AIエージェント構成**: 回答（Executor）と検証（Critic）の2役を組み合わせ、LLMの自己修正プロセスを体験。
- **RAG (Faiss)** とエージェントの統合により、信頼性の高い回答システムを構築。

## 演習内容（全8回）
各バナーから Google Colab で直接開けます（GitHub の `ai_agent` ブランチを参照します）。

1. `00_setup_common.ipynb` - 環境セットアップ  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/ai_agent/notebooks/00_setup_common.ipynb)
2. `01_gpt_baseline.ipynb` - LLM単体での生成とハルシネーションの観察  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/ai_agent/notebooks/01_gpt_baseline.ipynb)
3. `02_prompting.ipynb` - 指示による振る舞いの制御  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/ai_agent/notebooks/02_prompting.ipynb)
4. `03_rag_concept_demo.ipynb` - RAGの基本概念  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/ai_agent/notebooks/03_rag_concept_demo.ipynb)
5. `04_rag_faiss_exercise.ipynb` - ベクトル検索 (Faiss) の実習  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/ai_agent/notebooks/04_rag_faiss_exercise.ipynb)
6. `05_agent_basics.ipynb` - エージェント（自己修正ループ）の基礎  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/ai_agent/notebooks/05_agent_basics.ipynb)
7. `06_agent_gradio_ui.ipynb` - AIエージェントの処理過程の可視化  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/ai_agent/notebooks/06_agent_gradio_ui.ipynb)
8. `07_agent_rag_gradio.ipynb` - RAG統合マルチエージェント（最終課題）  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/ai_agent/notebooks/07_agent_rag_gradio.ipynb)

## 実行手順
1. Google Colab で `00_setup_common.ipynb` を開きます。
2. ランタイムのタイプを **GPU (T4)** に変更します（「ランタイム」メニュー → 「ランタイムのタイプを変更」）。
3. ノートブック内の指示に従い、順番にセルを実行してください。

## 生成AI入門：次の言葉と文章生成

`demo` ブランチには、文系初学者向けの別教材として [生成AIの答えはなぜ変わるのか](notebooks/09_llm_intro_forms.ipynb) を置きます。このノートでは軽量な Sarashina2.2-0.5B-Instruct を使い、次のトークン候補、temperatureによる出力の違い、指示文の具体化、事実確認を順に扱います。学習者が操作する値はColabのフォームにまとめ、処理本体は [src/llm_intro_demo.py](src/llm_intro_demo.py) に分けました。

この教材は `ai_agent` ブランチを基点にし、`src/common.py` の `load_llm` をモデルの読込に再利用します。一方、`generate_text` は使いません。同関数には回答の再試行・整形があり、temperatureだけを変えた結果を観察する演習では条件が混ざるためです。生成と表示には授業用クラスを追加しました。既存教材で使う共通関数の既定の動作は変えていません。

Colabで開く場合は、次のURLを使用します（`demo` ブランチをGitHubへ公開した後に有効）。

<https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/demo/notebooks/09_llm_intro_forms.ipynb>

フォーム表示はコードの表示を折りたたむ機能であり、ソースの閲覧・編集を禁止するものではありません。授業前に実機でモデルの取得時間と生成時間を測り、結果のHTMLをMacへ保存してください。

## ライセンス
- 本リポジトリのソースコードは MIT License の下で公開しています。
- ただし、使用するモデル、重み、データ、外部ライブラリには、それぞれ別のライセンスが適用されます。
- 本教材は教育目的での利用を想定しています。
- 商用利用を行う場合は、利用者自身の責任で各モデル・データ・ライブラリの利用条件を確認してください。
