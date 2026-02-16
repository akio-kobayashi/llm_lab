import gradio as gr


def create_agent_ui(run_plain, run_rag=None):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Agent Demo")
        gr.Markdown("OpenClaw風の step-by-step ログを表示します。")

        with gr.Row():
            query = gr.Textbox(label="質問", lines=3)
            use_rag = gr.Checkbox(label="RAGを使う", value=False, visible=run_rag is not None)

        run_btn = gr.Button("実行", variant="primary")
        answer = gr.Markdown(label="回答")
        trace = gr.JSON(label="実行ログ")

        def _run(q, rag_flag=False):
            if run_rag is not None and rag_flag:
                a, steps, docs = run_rag(q)
                return a, [s.__dict__ for s in steps] + [{"retrieved_docs": docs}]
            a, steps = run_plain(q)
            return a, [s.__dict__ for s in steps]

        if run_rag is not None:
            run_btn.click(_run, inputs=[query, use_rag], outputs=[answer, trace])
        else:
            run_btn.click(lambda q: _run(q, False), inputs=[query], outputs=[answer, trace])

    return demo


def create_multiagent_ui(run_pipeline, run_pipeline_rag=None):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Multi-Agent Demo")
        gr.Markdown("Planner / Coder / Critic の出力を順番に可視化します。")

        with gr.Row():
            query = gr.Textbox(label="質問", lines=4)
            use_rag = gr.Checkbox(label="RAGを使う", value=False, visible=run_pipeline_rag is not None)

        run_btn = gr.Button("パイプライン実行", variant="primary")
        final_answer = gr.Markdown(label="最終出力")
        role_log = gr.Textbox(label="ロール実行ログ", lines=16)
        trace = gr.JSON(label="ステップ情報")

        def _run(q, rag_flag=False):
            if run_pipeline_rag is not None and rag_flag:
                final, full_log, steps, docs = run_pipeline_rag(q)
                return final, full_log, [s.__dict__ for s in steps] + [{"retrieved_docs": docs}]
            final, full_log, steps = run_pipeline(q)
            return final, full_log, [s.__dict__ for s in steps]

        if run_pipeline_rag is not None:
            run_btn.click(_run, inputs=[query, use_rag], outputs=[final_answer, role_log, trace])
        else:
            run_btn.click(lambda q: _run(q, False), inputs=[query], outputs=[final_answer, role_log, trace])

    return demo
