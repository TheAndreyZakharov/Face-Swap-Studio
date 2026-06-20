from __future__ import annotations

from typing import Any

import numpy as np

from src.face_swap_studio.models.model_manager import get_face_swapper

SUPPORTED_SWAPPERS = {
    "InSwapper 128": "inswapper_128.onnx",
}


def swap_face(
    image_bgr: np.ndarray,
    source_face: Any,
    target_face: Any,
    model_label: str = "InSwapper 128",
) -> np.ndarray:
    if model_label not in SUPPORTED_SWAPPERS:
        raise ValueError(f"Неизвестная swap-модель: {model_label}")

    model_name = SUPPORTED_SWAPPERS[model_label]
    swapper = get_face_swapper(model_name)

    return swapper.get(
        image_bgr,
        target_face,
        source_face,
        paste_back=True,
    )
