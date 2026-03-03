# 日本語LLM (Qwen 3.5-4B) 演習キット for Google Colab

このリポジトリは、最新の日本語LLMをGoogle Colab（T4 GPU）で動かし、プロンプト、RAG、そして **AIエージェント** までを段階的に学ぶための演習教材です。

## 特徴
- **最新モデル Qwen 3.5-4B-Instruct** を採用。
- **4bit量子化 (bitsandbytes)** により、無料版Colabで安定動作。
- **AIエージェント構成**: 回答（Executor）と検証（Critic）の2役を組み合わせ、LLMの自己修正プロセスを体験。
- **RAG (Faiss)** とエージェントの統合により、信頼性の高い回答システムを構築。

## 演習内容（全7回）
1. **00_setup_common**: 環境セットアップ
2. **01_gpt_baseline**: LLM単体での生成とハルシネーションの観察
3. **02_prompting**: 指示による振る舞いの制御
4. **03_rag_concept_demo**: RAGの基本概念
5. **04_rag_faiss_exercise**: ベクトル検索 (Faiss) の実習
6. **05_agent_basics**: エージェント（自己修正ループ）の基礎
7. **06_agent_gradio_ui**: エージェント思考プロセスの可視化UI
8. **07_agent_rag_gradio**: RAG統合マルチエージェント（最終課題）

## 実行手順
1. Google Colab で `notebooks/00_setup_common.ipynb` を開きます。
2. ランタイムのタイプを **GPU (T4)** に変更します。
3. 順番に実行してください。

## ライセンス
- MIT License
