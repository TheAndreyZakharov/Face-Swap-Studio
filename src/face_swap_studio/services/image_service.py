from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.face_swap_studio.core.detector import (
    detect_faces,
    detection_preview,
    face_crop,
)
from src.face_swap_studio.core.pipeline import read_image


def normalize_uploaded_paths(files: Any) -> list[str]:
    if files is None:
        return []

    if not isinstance(files, list):
        files = [files]

    paths: list[str] = []

    for item in files:
        if item is None:
            continue

        if isinstance(item, (str, Path)):
            paths.append(str(item))
            continue

        name = getattr(item, "name", None)

        if name:
            paths.append(str(name))

    return paths


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )


def build_source_gallery(
    paths: list[str],
    confidence_threshold: float,
) -> list[tuple[np.ndarray, str]]:
    gallery: list[tuple[np.ndarray, str]] = []

    for path_string in paths:
        path = Path(path_string)
        image = read_image(path)
        faces = detect_faces(
            image,
            confidence_threshold,
        )

        for face in faces:
            crop = face_crop(
                image,
                face,
            )
            caption = f"{path.name} · лицо {face.index + 1}"
            gallery.append(
                (
                    bgr_to_rgb(crop),
                    caption,
                )
            )

    return gallery


def build_target_gallery(
    paths: list[str],
    confidence_threshold: float,
) -> list[tuple[np.ndarray, str]]:
    gallery: list[tuple[np.ndarray, str]] = []

    for path_string in paths:
        path = Path(path_string)
        image = read_image(path)
        faces = detect_faces(
            image,
            confidence_threshold,
        )
        preview = detection_preview(
            image,
            faces,
        )

        gallery.append(
            (
                bgr_to_rgb(preview),
                f"{path.name} · найдено лиц: {len(faces)}",
            )
        )

    return gallery
