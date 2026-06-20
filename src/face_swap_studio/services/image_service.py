from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.face_swap_studio.core.detector import (
    create_detection_preview,
    face_crop,
)
from src.face_swap_studio.core.pipeline import (
    SourceIdentity,
    TargetAnalysis,
)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def normalize_uploaded_paths(files: Any) -> list[str]:
    if files is None:
        return []

    if not isinstance(files, list):
        files = [files]

    paths: list[str] = []

    for item in files:
        if item is None:
            continue

        if isinstance(item, str):
            paths.append(item)
            continue

        if isinstance(item, Path):
            paths.append(str(item))
            continue

        name = getattr(item, "name", None)

        if name:
            paths.append(str(name))

    return paths


def uploaded_image_gallery(
    paths: list[str],
) -> list[tuple[str, str]]:
    return [
        (
            path,
            Path(path).name,
        )
        for path in paths
    ]


def source_gallery(
    sources: list[SourceIdentity],
) -> list[tuple[np.ndarray, str]]:
    gallery: list[tuple[np.ndarray, str]] = []

    for source in sources:
        crop = face_crop(
            source.image,
            source.detected_face,
            padding_ratio=0.18,
        )

        caption = f"Источник {source.index + 1} · {source.path.name}"
        gallery.append((bgr_to_rgb(crop), caption))

    return gallery


def target_preview_gallery(
    targets: list[TargetAnalysis],
) -> list[tuple[np.ndarray, str]]:
    gallery: list[tuple[np.ndarray, str]] = []

    for target in targets:
        preview = create_detection_preview(
            target.image,
            target.faces,
        )

        caption = f"{target.path.name} · найдено лиц: {len(target.faces)}"

        gallery.append((bgr_to_rgb(preview), caption))

    return gallery


def target_face_gallery(
    targets: list[TargetAnalysis],
    assignments: dict[tuple[int, int], int | None],
) -> tuple[
    list[tuple[np.ndarray, str]],
    list[tuple[int, int]],
]:
    gallery: list[tuple[np.ndarray, str]] = []
    flattened_faces: list[tuple[int, int]] = []

    for target in targets:
        for face in target.faces:
            target_key = (target.index, face.index)
            flattened_faces.append(target_key)

            crop = face_crop(
                target.image,
                face,
                padding_ratio=0.22,
            )

            if target_key not in assignments:
                assignment_text = "не назначено"
            elif assignments[target_key] is None:
                assignment_text = "не заменять"
            else:
                source_index = assignments[target_key]
                assignment_text = f"источник {source_index + 1}"

            caption = f"{target.path.name} · лицо {face.index + 1} · {assignment_text}"

            gallery.append((bgr_to_rgb(crop), caption))

    return gallery, flattened_faces


def image_dimensions(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
