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

    mask = np.zeros((height, width), dtype=np.float32)

    margin_x = max(1, int(width * feather_ratio))
    margin_y = max(1, int(height * feather_ratio))

    cv2.ellipse(
        mask,
        center=(width // 2, height // 2),
        axes=(max(1, width // 2 - margin_x), max(1, height // 2 - margin_y)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=1.0,
        thickness=-1,
    )

    blur_size = max(3, int(min(width, height) * feather_ratio))
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

    return cv2.GaussianBlur(mask, (blur_size, blur_size), 0)


def alpha_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if background.shape != foreground.shape:
        raise ValueError("Фон и foreground должны иметь одинаковый размер.")

    if mask.ndim == 2:
        mask = mask[..., None]

    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)

    result = foreground.astype(np.float32) * mask + background.astype(np.float32) * (1.0 - mask)

    return np.clip(result, 0, 255).astype(np.uint8)
