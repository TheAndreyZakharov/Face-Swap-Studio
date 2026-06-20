from __future__ import annotations

from functools import lru_cache

import numpy as np
from gfpgan import GFPGANer

from src.face_swap_studio.models.model_manager import gfpgan_model_path
from src.face_swap_studio.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_face_enhancer() -> GFPGANer:
    logger.info("Загрузка GFPGAN v1.4")

    return GFPGANer(
        model_path=str(gfpgan_model_path()),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def enhance_faces(
    image_bgr: np.ndarray,
    weight: float = 0.35,
) -> np.ndarray:
    weight = float(np.clip(weight, 0.0, 1.0))

    enhancer = get_face_enhancer()
    _, _, restored = enhancer.enhance(
        image_bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=weight,
    )

    if restored is None:
        return image_bgr

    return restored
