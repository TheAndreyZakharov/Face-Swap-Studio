from __future__ import annotations

from typing import Any

import numpy as np

from src.face_swap_studio.models.model_manager import (
    get_face_swapper,
)

INSWAPPER_MODEL_FILE = "inswapper_128.onnx"


def swap_face(
    image_bgr: np.ndarray,
    source_face: Any,
    target_face: Any,
) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Передано пустое изображение.")

    if source_face is None:
        raise ValueError("Не передано исходное лицо.")

    if target_face is None:
        raise ValueError("Не передано целевое лицо.")

    swapper = get_face_swapper(INSWAPPER_MODEL_FILE)

    result = swapper.get(
        image_bgr,
        target_face,
        source_face,
        paste_back=True,
    )

    if result is None:
        raise RuntimeError("InSwapper не вернул результат.")

    return result
