import gradio as gr

def create_gradio_ui(
    generate_func_plain,
    generate_func_rag,
    examples=None
):
    """
    RAGのON/OFFを切り替えられるGradioデモUIを作成する。

    Args:
        generate_func_plain: RAGなしの生成関数。
        generate_func_rag: RAGありの生成関数。
        examples: UIに表示するサンプル入力のリスト。
        
    Returns:
        gradio.Blocks: 生成されたGradio UI。
    """
    
    # デフォルトのサンプル入力
    if examples is None:
        examples = [
            ["「星屑のメモリー」の主人公は誰ですか？", True],
            ["「古都の探偵録」のあらすじを教えてください。", True],
            ["富士山の高さは？", False],
            ["最近の流行りのアニメについて教えて。", False],
        ]
    
    def combined_generate(query, use_rag):
        print(f"Query: '{query}', RAG: {use_rag}")
        if use_rag:
            output, context = generate_func_rag(query)
            return output, context
        else:
            output = generate_func_plain(query)
            return output, "RAGを使用していないため引用はありません。素のLLMの知識で回答しています。"

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 統合デモ: RAG連携
            
            このUIでは、これまでの演習で学んだ技術を組み合わせて、LLMの応答をどのように制御できるかを試すことができます。
            - **RAG (検索拡張生成)**: **ON**にすると、外部知識を検索し、その事実に基づいて回答します。
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
                
                submit_button = gr.Button("生成実行", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 1. 生成された回答")
                output_text = gr.Markdown(label="回答内容")
                
                gr.Markdown("### 2. RAGが参照した根拠 (Evidence)")
                context_text = gr.Textbox(
                    label="検索されたドキュメント（JSON形式）",
                    lines=8,
                    interactive=False
                )

        submit_button.click(
            fn=combined_generate,
            inputs=[query_input, use_rag_checkbox],
            outputs=[output_text, context_text]
        )
        
        gr.Examples(
            examples=examples,
            inputs=[query_input, use_rag_checkbox],
            outputs=[output_text, context_text],
            fn=combined_generate,
            cache_examples=False,
            label="試してみる質問例 (クリックすると入力されます)"
        )

    return demo
