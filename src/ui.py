import gradio as gr

def create_gradio_ui(
    generate_func_plain,
    generate_func_rag,
    generate_func_lora,
    generate_func_rag_lora
):
    """
    RAGとLoRAのON/OFFを切り替えられるGradioデモUIを作成する。

    Args:
        generate_func_plain: RAGなし, LoRAなしの生成関数。
        generate_func_rag: RAGあり, LoRAなしの生成関数。
        generate_func_lora: RAGなし, LoRAありの生成関数。
        generate_func_rag_lora: RAGあり, LoRAありの生成関数。
        
    Returns:
        gradio.Blocks: 生成されたGradio UI。
    """
    
    def combined_generate(query, use_rag, use_lora):
        """
        チェックボックスの状態に応じて適切な生成関数を呼び出す。
        """
        print(f"Query: '{query}', RAG: {use_rag}, LoRA: {use_lora}")
        
        if use_rag and use_lora:
            # RAG + LoRA
            output, context = generate_func_rag_lora(query)
            return output, context
        elif use_rag:
            # RAGのみ
            output, context = generate_func_rag(query)
            return output, context
        elif use_lora:
            # LoRAのみ
            output = generate_func_lora(query)
            return output, "LoRAモードでは引用はありません。"
        else:
            # Plain (素のLLM)
            output = generate_func_plain(query)
            return output, "RAGを使用していないため、引用はありません。"

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 統合デモ: RAGとLoRAの連携
            
            このUIでは、これまでの演習で学んだ技術を組み合わせて、LLMの応答をどのように制御できるかを試すことができます。
            - **RAG (Retrieval-Augmented Generation)**: ONにすると、外部知識（ドキュメント）を検索し、その内容を参考にして回答を生成します。
            - **LoRA (Low-Rank Adaptation)**: ONにすると、特定のタスク（この場合はJSON形式での出力）にファインチューニングされたモデル（アダプタ）を使用して回答を生成します。
            
            それぞれのスイッチを切り替えて、LLMの振る舞いがどう変わるか観察してみましょう。
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="質問入力",
                    placeholder="ここに質問を入力してください...",
                    lines=3
                )
                
                with gr.Row():
                    use_rag_checkbox = gr.Checkbox(label="RAGを有効にする", value=True)
                    use_lora_checkbox = gr.Checkbox(label="LoRAを有効にする", value=False)
                
                submit_button = gr.Button("生成", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 生成結果")
                output_text = gr.Markdown(label="回答")
                
                gr.Markdown("### 引用/コンテキスト")
                context_text = gr.Textbox(
                    label="RAGが参照した情報",
                    lines=8,
                    interactive=False
                )

        submit_button.click(
            fn=combined_generate,
            inputs=[query_input, use_rag_checkbox, use_lora_checkbox],
            outputs=[output_text, context_text]
        )
        
        gr.Examples(
            examples=[
                ["「星屑のメモリー」の主人公は誰ですか？", True, False],
                ["「東京サイバーパンク2042」について教えてください。", True, True],
                ["日本の首都はどこですか？", False, True],
                ["世界で一番高い山は何ですか？", False, False],
            ],
            inputs=[query_input, use_rag_checkbox, use_lora_checkbox],
            outputs=[output_text, context_text],
            fn=combined_generate,
            cache_examples=False, # デモ用にキャッシュ無効
        )

    return demo

if __name__ == '__main__':
    # このファイルが直接実行された場合のダミー動作テスト
    print("--- Running ui.py self-test ---")

    # ダミーの生成関数を定義
    def dummy_plain(query):
        return f"【Plain】'{query}' に答えます。"

    def dummy_rag(query):
        context = f"'{query}' に関連するドキュメントを見つけました。"
        return f"【RAG】'{query}' に答えます。", context

    def dummy_lora(query):
        return f"【LoRA】'{query}' に答えます。 (JSON形式風)"

    def dummy_rag_lora(query):
        context = f"'{query}' に関連するドキュメントを見つけました。"
        return f"【RAG+LoRA】'{query}' に答えます。 (JSON形式風)", context

    # UIの作成と起動
    demo_ui = create_gradio_ui(
        generate_func_plain=dummy_plain,
        generate_func_rag=dummy_rag,
        generate_func_lora=dummy_lora,
        generate_func_rag_lora=dummy_rag_lora,
    )
    
    print("Gradio UI created. Launching demo...")
    # demo_ui.launch() # このまま実行するとプロセスがブロックされるためコメントアウト
    print("Self-test finished. To run the demo, uncomment 'demo_ui.launch()' and run as a script.")
