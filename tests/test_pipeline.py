from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.face_swap_studio.core.pipeline import (
    read_image,
    write_image,
)


def test_write_and_read_image(
    tmp_path: Path,
) -> None:
    image = np.zeros(
        (64, 64, 3),
        dtype=np.uint8,
    )
    image[10:30, 15:40] = (
        20,
        120,
        240,
    )

    destination = tmp_path / "result.png"

    returned_path = write_image(
        destination,
        image,
    )

    assert returned_path == destination.resolve()
    assert returned_path.is_file()
    assert returned_path.stat().st_size > 0

    restored = read_image(returned_path)

    assert restored.shape == image.shape
    assert restored.dtype == np.uint8
    assert np.array_equal(restored, image)


def test_write_image_adds_png_extension(
    tmp_path: Path,
) -> None:
    image = np.zeros(
        (16, 16, 3),
        dtype=np.uint8,
    )

    destination = tmp_path / "without-extension"

    returned_path = write_image(
        destination,
        image,
    )

    assert returned_path.suffix == ".png"
    assert returned_path.is_file()


def test_write_image_rejects_empty_image(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "empty.png"
    image = np.array(
        [],
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="пустое изображение",
    ):
        write_image(
            destination,
            image,
        )


def test_read_image_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.png"

    with pytest.raises(
        FileNotFoundError,
        match="Не найдено изображение",
    ):
        read_image(missing)


def test_written_file_is_valid_opencv_image(
    tmp_path: Path,
) -> None:
    image = np.full(
        (32, 48, 3),
        127,
        dtype=np.uint8,
    )

    destination = tmp_path / "opencv.png"

    returned_path = write_image(
        destination,
        image,
    )

    decoded = cv2.imread(
        str(returned_path),
        cv2.IMREAD_COLOR,
    )

    assert decoded is not None
    assert decoded.shape == image.shape
