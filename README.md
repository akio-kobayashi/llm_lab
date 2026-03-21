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
7. `06_agent_gradio_ui.ipynb` - エージェント思考プロセスの可視化UI  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/ai_agent/notebooks/06_agent_gradio_ui.ipynb)
8. `07_agent_rag_gradio.ipynb` - RAG統合マルチエージェント（最終課題）  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akio-kobayashi/llm_lab/blob/ai_agent/notebooks/07_agent_rag_gradio.ipynb)

## 実行手順
1. Google Colab で `00_setup_common.ipynb` を開きます。
2. ランタイムのタイプを **GPU (T4)** に変更します（「ランタイム」メニュー → 「ランタイムのタイプを変更」）。
3. ノートブック内の指示に従い、順番にセルを実行してください。

## ノートブック更新日の自動反映
各 notebook の先頭に表示する `最終更新` は、ファイルごとの最終 commit 日から自動生成できます。

```bash
./scripts/install_git_hooks.sh
```

これを一度実行すると、以後 `git commit` の前に `scripts/update_notebook_dates.py` が自動実行され、更新後の notebook も自動で stage されます。

## ライセンス
- 本リポジトリのソースコードは MIT License の下で公開しています。
- ただし、使用するモデル、重み、データ、外部ライブラリには、それぞれ別のライセンスが適用されます。
- 本教材は教育目的での利用を想定しています。
- 商用利用を行う場合は、利用者自身の責任で各モデル・データ・ライブラリの利用条件を確認してください。
