import gradio as gr

def create_agent_ui(run_agent_fn):
    """
    エージェントの思考プロセスを可視化するUI
    """
    with gr.Blocks(title="AIエージェントの処理過程の可視化") as demo:
        gr.Markdown("# AIエージェントの処理過程の可視化")
        gr.Markdown("Executor (実行) と Critic (批評) による自己修正ループを確認できます。")
        
        with gr.Row():
            with gr.Column(scale=2):
                query = gr.Textbox(label="ユーザーの質問", placeholder="難しい数学の問題やコードの作成など", lines=3)
                submit_btn = gr.Button("エージェントに依頼", variant="primary")
            
            with gr.Column(scale=3):
                final_answer = gr.Textbox(label="最終回答", lines=10)

        with gr.Accordion("思考プロセスの詳細ログ", open=True):
            log_display = gr.Markdown(label="ログ")

        submit_btn.click(
            fn=run_agent_fn,
            inputs=[query],
            outputs=[final_answer, log_display]
        )
        
    return demo
