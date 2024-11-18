import gradio as gr
from button_interface import Backend
from generator import APIHelper
from utils.header import ConfigReader

DEBUG_MODE = False

def generate_page(backend):
    with gr.Blocks(title="Scientific Paper Idea Proposer") as demo:
        ## Background, keywords parts
        gr.Markdown(
            """
    # Scientific Paper Idea Proposer
    """
        )
        # with gr.Blocks(theme="earneleh/paris") as d:
        with gr.Blocks() as d:
            with gr.Tab("Keywords"):
                key_words = gr.Textbox(placeholder="Interested key words", label="Keywords (Provide at least 1 keyword)")
            with gr.Tab("Background"):
                background = gr.Textbox(placeholder="Background", label="Background")
            if DEBUG_MODE:
                with gr.Tab("Json"):
                    json_file = gr.File()
                    json_background = gr.Textbox(placeholder="Background", label="Background")
                    json_strs = gr.Textbox(visible=False)
                    json_file.upload(backend.upload_json_callback, inputs=[json_file], outputs=[json_background])
            else:
                json_strs = None

        ## brainstorm ideas parts
        # background2brainstorm = gr.Button("Continue (background2brainstorm)")
        with gr.Row(equal_height=True):
            gr.ClearButton(value="🆑 Clear", components=[background], scale=1)
            background2brainstorm = gr.Button("😈 Brainstorm", scale=1)
        # @gr.render(inputs=None, triggers=[background2brainstorm.click])
        # def show_brainstorm():
        # with gr.Accordion("Braining Ideas", open=False) as a1:
        with gr.Row(equal_height=True):
            brainstorm_txt = gr.Textbox(placeholder="Generated brainstorm ideas", label="Brainstorm ideas", info="Feel free to improve them before next step", max_lines=500)
            brainstorm_md = gr.Markdown(label="Brainstorm ideas")

        ## Expanded key words parts
        # brainstorm2entities = gr.Button("Continue (brainstorm2entities)")
        with gr.Row(equal_height=True):
            gr.ClearButton(value="🆑 Clear", components=[brainstorm_txt], scale=1)
            brainstorm2entities = gr.Button("Extract Entities", scale=1)
        entities = gr.CheckboxGroup([], label="Expanded key words", visible=True)
        entities2literature = gr.Button("📖 Retrieve Literature")
        literature_intact = gr.State()
        # entities2literature = gr.Button("Continue (retrieve literature)")

        ## Retrieved literature parts
        retrieved_literature = gr.Textbox(placeholder="Retrieved literature", label="Retrieved related works", info="", max_lines=500)
        # literature2initial_ideas = gr.Button("Continue (generate initial ideas)")
        with gr.Row(equal_height=True):
            gr.ClearButton(value="🆑 Clear", components=[retrieved_literature], scale=1)
            literature2initial_ideas = gr.Button("🤖 Generate Initial ideas", scale=1)


        ## Initial ideas parts
        with gr.Row():
            initial_ideas_txt = gr.Textbox(placeholder="Initial ideas", label="Initial ideas", info="Feel free to improve them before next step", max_lines=500)
            initial_ideas_md = gr.Markdown(label="Initial ideas")
        # initial2final = gr.Button("Continue (generate final ideas)")
        with gr.Row(equal_height=True):
            gr.ClearButton(value="🆑 Clear", components=[initial_ideas_txt], scale=1)
            initial2final = gr.Button("🔥 Refine Ideas")

        ## Final ideas parts
        with gr.Row():
            final_ideas_txt = gr.Textbox(placeholder="Final ideas", label="Final ideas", info="", max_lines=500)
            final_ideas_md = gr.Markdown(label="Final ideas")

        # register callback
        background2brainstorm.click(backend.background2brainstorm_callback, inputs=[background], outputs=[brainstorm_txt])
        brainstorm2entities.click(backend.brainstorm2entities_callback, inputs=[background, brainstorm_txt], outputs=[entities])
        brainstorm_txt.change(lambda input: input, inputs=brainstorm_txt, outputs=brainstorm_md)
        initial_ideas_txt.change(lambda input: input, inputs=initial_ideas_txt, outputs=initial_ideas_md)
        final_ideas_txt.change(lambda input: input, inputs=final_ideas_txt, outputs=final_ideas_md)
        entities2literature.click(backend.entities2literature_callback, inputs=[background, entities], outputs=[retrieved_literature, literature_intact])
        literature2initial_ideas.click(backend.literature2initial_ideas_callback, inputs=[background, literature_intact], outputs=[initial_ideas_txt, final_ideas_txt])
        initial2final.click(backend.initial2final_callback, inputs=[initial_ideas_txt], outputs=[final_ideas_txt])
    return demo

if __name__ == "__main__":
    backend = Backend()
    demo = generate_page(backend)
    demo.launch(server_name="0.0.0.0", share=True)