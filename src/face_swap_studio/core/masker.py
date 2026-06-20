from __future__ import annotations

import cv2
import numpy as np


def feather_mask(
    width: int,
    height: int,
    feather_ratio: float = 0.12,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Размер маски должен быть положительным.")

    normalized_ratio = float(np.clip(feather_ratio, 0.0, 0.49))

    mask = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    margin_x = max(
        1,
        int(width * normalized_ratio),
    )
    margin_y = max(
        1,
        int(height * normalized_ratio),
    )

    cv2.ellipse(
        mask,
        center=(width // 2, height // 2),
        axes=(
            max(1, width // 2 - margin_x),
            max(1, height // 2 - margin_y),
        ),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=1.0,
        thickness=-1,
    )

    blur_size = max(
        3,
        int(min(width, height) * normalized_ratio),
    )

    if blur_size % 2 == 0:
        blur_size += 1

    return cv2.GaussianBlur(
        mask,
        (blur_size, blur_size),
        0,
    )


def alpha_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if background.shape != foreground.shape:
        raise ValueError("Фон и foreground должны иметь одинаковый размер.")

    if background.ndim != 3:
        raise ValueError("Ожидается цветное изображение формата H×W×C.")

    if mask.shape[:2] != background.shape[:2]:
        raise ValueError("Размер маски не совпадает с размером изображений.")

    normalized_mask = mask.astype(np.float32)

    if normalized_mask.ndim == 2:
        normalized_mask = normalized_mask[..., None]

    normalized_mask = np.clip(
        normalized_mask,
        0.0,
        1.0,
    )

    background_float = background.astype(np.float32)
    foreground_float = foreground.astype(np.float32)

    result = foreground_float * normalized_mask + background_float * (1.0 - normalized_mask)

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)
