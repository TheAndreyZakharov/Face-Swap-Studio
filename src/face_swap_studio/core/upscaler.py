from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from src.face_swap_studio.models.model_manager import (
    realesrgan_model_path,
)
from src.face_swap_studio.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=8)
def get_upscaler(
    tile_size: int = 256,
) -> RealESRGANer:
    normalized_tile_size = max(
        0,
        int(tile_size),
    )

    logger.info(
        "Загрузка Real-ESRGAN x4plus, tile=%s",
        normalized_tile_size,
    )

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    return RealESRGANer(
        scale=4,
        model_path=str(realesrgan_model_path()),
        model=model,
        tile=normalized_tile_size,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )


def upscale_image(
    image_bgr: np.ndarray,
    output_scale: float = 2.0,
    tile_size: int = 256,
) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Нельзя увеличить пустое изображение.")

    normalized_scale = float(output_scale)

    if normalized_scale <= 1.0:
        return image_bgr.copy()

    upscaler = get_upscaler(int(tile_size))

    result, _ = upscaler.enhance(
        image_bgr,
        outscale=normalized_scale,
    )

    if result is None:
        raise RuntimeError("Real-ESRGAN не вернул результат.")

    return result


def resize_if_too_large(
    image_bgr: np.ndarray,
    maximum_side: int = 8000,
) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Передано пустое изображение.")

    if maximum_side <= 0:
        raise ValueError("Максимальная сторона должна быть положительной.")

    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)

    if longest_side <= maximum_side:
        return image_bgr

    ratio = maximum_side / longest_side

    resized_width = max(
        1,
        int(round(width * ratio)),
    )
    resized_height = max(
        1,
        int(round(height * ratio)),
    )

    return cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
