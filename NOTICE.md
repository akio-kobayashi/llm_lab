# 注意事項 (NOTICE)

本リポジトリおよび演習教材では、以下のデータやモデルを使用・参照しています。

## 1. 言語モデル (LLM)
- **Qwen 3.5-4B-Instruct**: Alibaba Group が提供するモデル。使用にあたっては [Qwen License](https://github.com/QwenLM/Qwen) および利用規約を遵守してください。

## 2. 埋め込みモデル (Embedding)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**: Hugging Face Hub で公開されている多言語対応の埋め込みモデルです。

## 3. サンプルデータ
- **data/docs/anime_docs_sample.jsonl**: このファイルに含まれるアニメ作品の情報はすべて「架空のもの（創作）」であり、実在の人物・団体・作品とは関係ありません。演習目的で作成されたダミーデータです。
- 実際に Wikipedia 等の外部データを利用する場合は、各データのライセンス（CC BY-SA 3.0 等）に従い、適切に出典を明記してください。

## 4. 外部ライブラリ
- bitsandbytes, transformers, peft, faiss, gradio 等、多くのオープンソースソフトウェアを利用しています。各ライブラリのライセンスについてはそれぞれの配布元を確認してください。
