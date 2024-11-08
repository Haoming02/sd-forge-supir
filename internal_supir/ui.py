import gradio as gr

POS_PROMPT = "cinematic, high contrast, detailed, canon camera, photorealistic, maximum detail, 4k, color grading, ultra hd, sharpness, perfect"
NEG_PROMPT = "painting, illustration, drawing, art, sketch, anime, cartoon, CG Style, 3D render, blur, aliasing, unsharp, weird textures, ugly, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, lowres"


def supir_ui() -> dict[str, gr.components.Component]:
    args = {}

    with gr.Row():
        args["prompt"] = gr.Textbox(
            label="Image Description / Caption",
            info="Describe as much as possible, especially the details not present in the original image",
            value="",
            placeholder="A 33 years old man, walking on the street on a summer morning, Santiago",
            lines=10,
            max_lines=10,
        )

        with gr.Column():
            args["p_prompt"] = gr.Textbox(
                label="Positive Prompt",
                info="Quality description that gets appended after the main caption",
                value=POS_PROMPT,
                lines=3,
                max_lines=3,
            )
            args["n_prompt"] = gr.Textbox(
                label="Negative Prompt",
                info="Description for what the image should not contain",
                value=NEG_PROMPT,
                lines=3,
                max_lines=3,
            )

    with gr.Accordion(label="Advanced Settings", open=False):
        with gr.Row():
            args["edm_steps"] = gr.Slider(
                label="Steps",
                minimum=1,
                maximum=128,
                value=48,
                step=1,
            )
            args["seed"] = gr.Slider(
                label="Seed",
                value=-1,
                minimum=-1,
                maximum=4294967295,
                step=1,
            )
            args["s_cfg"] = gr.Slider(
                label="Text Guidance Scale",
                # info="Guided by Image | Guided by Prompt",
                minimum=1.0,
                maximum=15.0,
                value=7.5,
                step=0.5,
            )

        with gr.Row():
            args["s_stage2"] = gr.Slider(
                label="Restoring Guidance Strength",
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.05,
            )
            args["s_stage1"] = gr.Slider(
                label="Pre-denoising Guidance Strength",
                minimum=-1.0,
                maximum=6.0,
                value=-1.0,
                step=1.0,
            )

        with gr.Row():
            args["s_churn"] = gr.Slider(
                label="S-Churn", minimum=0, maximum=40, value=5, step=1
            )
            args["s_noise"] = gr.Slider(
                label="S-Noise", minimum=1.0, maximum=1.1, value=1.003, step=0.001
            )

        with gr.Row():
            args["linear_CFG"] = gr.Checkbox(True, label="Linear CFG")
            args["linear_s_stage2"] = gr.Checkbox(
                False, label="Linear Restoring Guidance"
            )

        with gr.Row():
            args["spt_linear_CFG"] = gr.Slider(
                label="CFG Start",
                minimum=1.0,
                maximum=9.0,
                value=4,
                step=0.5,
            )
            args["spt_linear_s_stage2"] = gr.Slider(
                label="Guidance Start",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.05,
            )

        args["color_fix_type"] = gr.Radio(
            label="Color-Fix Mode",
            choices=("None", "AdaIn", "Wavelet"),
            info="AdaIn for photo ; Wavelet for JPEG artifacts",
            value="AdaIn",
        )

    return args
