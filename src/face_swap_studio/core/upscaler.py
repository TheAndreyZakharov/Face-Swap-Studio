from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from src.face_swap_studio.models.model_manager import realesrgan_model_path
from src.face_swap_studio.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_upscaler(tile_size: int = 256) -> RealESRGANer:
    logger.info("Загрузка Real-ESRGAN x4plus, tile=%s", tile_size)

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
        tile=max(0, int(tile_size)),
        tile_pad=10,
        pre_pad=0,
        half=False,
    )


def upscale_image(
    image_bgr: np.ndarray,
    output_scale: float = 2.0,
    tile_size: int = 256,
) -> np.ndarray:
    if output_scale <= 1.0:
        return image_bgr

    upscaler = get_upscaler(tile_size)
    result, _ = upscaler.enhance(
        image_bgr,
        outscale=float(output_scale),
    )

    return result


def resize_if_too_large(
    image_bgr: np.ndarray,
    maximum_side: int = 8000,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    longest = max(height, width)

    if longest <= maximum_side:
        return image_bgr

    ratio = maximum_side / longest

    return cv2.resize(
        image_bgr,
        (
            max(1, int(width * ratio)),
            max(1, int(height * ratio)),
        ),
        interpolation=cv2.INTER_AREA,
    )
