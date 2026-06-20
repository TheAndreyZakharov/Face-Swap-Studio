from __future__ import annotations

from html import escape

import gradio as gr

from src.face_swap_studio.models.manifest import (
    is_model_ready,
    selectable_swap_models,
)


def model_choices() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []

    for definition in selectable_swap_models():
        status = (
            "Available"
            if is_model_ready(definition)
            else "Unavailable"
        )

        choices.append(
            (
                f"{definition.name} — {status}",
                definition.id,
            )
        )

    return choices


def default_model_id() -> str | None:
    definitions = selectable_swap_models()

    preferred_ids = (
        "inswapper_128",
        "simswap_512",
        "ghost_unet_3blocks",
        "ghost2_head",
    )

    for preferred_id in preferred_ids:
        for definition in definitions:
            if (
                definition.id == preferred_id
                and is_model_ready(definition)
            ):
                return definition.id

    for definition in definitions:
        if is_model_ready(definition):
            return definition.id

    if definitions:
        return definitions[0].id

    return None


def model_cards_html() -> str:
    cards: list[str] = []

    for definition in selectable_swap_models():
        ready = is_model_ready(
            definition
        )

        status_class = (
            "model-status-ready"
            if ready
            else "model-status-unavailable"
        )

        status_text = (
            "Available"
            if ready
            else "Unavailable"
        )

        cards.append(
            f"""
<div class="model-card">
    <div class="model-card-header">
        <span class="model-card-name">
            {escape(definition.name)}
        </span>
        <span class="model-status {status_class}">
            {status_text}
        </span>
    </div>
    <div class="model-card-description">
        {escape(definition.description)}
    </div>
</div>
"""
        )

    if not cards:
        return """
<div class="empty-message">
    No face-swap models are registered.
</div>
"""

    return (
        '<div class="model-card-list">'
        + "".join(cards)
        + "</div>"
    )


def create_processing_controls() -> dict[str, gr.Component]:
    swap_model = gr.Dropdown(
        choices=model_choices(),
        value=default_model_id(),
        label="Face-swap model",
        info="Select the model used for all assigned face replacements.",
        interactive=True,
        elem_id="swap-model",
    )

    with gr.Accordion(
        "Post-processing",
        open=False,
        elem_classes=["settings-accordion"],
    ):
        enhance_faces = gr.Checkbox(
            value=False,
            label="Restore generated faces with GFPGAN",
            info=(
                "Can improve skin detail and facial structure, "
                "but high values may alter identity."
            ),
        )

        enhancement_weight = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.35,
            step=0.05,
            label="GFPGAN restoration strength",
            interactive=True,
        )

        upscale_image = gr.Checkbox(
            value=False,
            label="Upscale the complete image with Real-ESRGAN",
            info=(
                "Increases resolution after face replacement. "
                "This can require significantly more processing time."
            ),
        )

        upscale_factor = gr.Dropdown(
            choices=[
                ("1.5×", 1.5),
                ("2×", 2.0),
                ("3×", 3.0),
                ("4×", 4.0),
            ],
            value=2.0,
            label="Upscale factor",
            interactive=True,
        )

        tile_size = gr.Dropdown(
            choices=[
                ("128 px", 128),
                ("192 px", 192),
                ("256 px", 256),
                ("384 px", 384),
                ("512 px", 512),
            ],
            value=256,
            label="Processing tile size",
            info=(
                "Use a smaller tile if upscaling consumes too much memory."
            ),
            interactive=True,
        )

    return {
        "swap_model": swap_model,
        "enhance_faces": enhance_faces,
        "enhancement_weight": enhancement_weight,
        "upscale_image": upscale_image,
        "upscale_factor": upscale_factor,
        "tile_size": tile_size,
    }