from __future__ import annotations

import numpy as np

from src.face_swap_studio.core.detector import detection_preview


def test_detection_preview_without_faces() -> None:
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = detection_preview(
        image,
        [],
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result, image)
    assert result is not image
