from __future__ import annotations

import gradio as gr

APP_CSS = """
:root {
    --studio-radius: 18px;
}

.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
}

.studio-header {
    padding: 22px 26px;
    border-radius: 22px;
    background:
        linear-gradient(
            135deg,
            rgba(90, 80, 220, 0.18),
            rgba(40, 160, 210, 0.10)
        );
    margin-bottom: 16px;
}

.studio-card {
    border-radius: var(--studio-radius) !important;
    overflow: hidden;
}

.compact-gallery {
    min-height: 150px !important;
}

.compact-gallery .grid-wrap {
    min-height: 140px !important;
}

.face-gallery {
    min-height: 230px !important;
}

.primary-action {
    min-height: 48px !important;
    font-weight: 700 !important;
}

.selection-status {
    padding: 12px 16px;
    border-radius: 14px;
    background: rgba(120, 120, 140, 0.10);
}
"""


def create_processing_controls() -> dict[str, gr.Component]:
    swap_model = gr.Dropdown(
        choices=["InSwapper 128"],
        value="InSwapper 128",
        label="Модель замены",
        interactive=True,
    )

    enhance_faces = gr.Checkbox(
        value=False,
        label="Улучшить лица",
    )

    enhancement_weight = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.25,
        step=0.05,
        label="Сила улучшения лица",
    )

    upscale_image = gr.Checkbox(
        value=False,
        label="Улучшить всё изображение",
    )

    upscale_factor = gr.Dropdown(
        choices=[1.5, 2.0, 3.0, 4.0],
        value=2.0,
        label="Масштаб изображения",
    )

    tile_size = gr.Dropdown(
        choices=[128, 192, 256, 384, 512],
        value=256,
        label="Размер блока обработки",
    )

    return {
        "swap_model": swap_model,
        "enhance_faces": enhance_faces,
        "enhancement_weight": enhancement_weight,
        "upscale_image": upscale_image,
        "upscale_factor": upscale_factor,
        "tile_size": tile_size,
    }
