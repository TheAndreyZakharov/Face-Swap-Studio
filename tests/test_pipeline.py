from __future__ import annotations

from pathlib import Path

import numpy as np

from src.face_swap_studio.core.pipeline import write_image


def test_write_image(tmp_path: Path) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    destination = tmp_path / "result.png"

    returned_path = write_image(destination, image)

    assert returned_path == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0
