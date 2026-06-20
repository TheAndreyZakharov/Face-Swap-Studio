from __future__ import annotations

import cv2
import numpy as np

from src.face_swap_studio.domain.entities import DetectedFace
from src.face_swap_studio.models.model_manager import get_face_analyser
from src.face_swap_studio.utils.paths import load_settings


def _clamp_bbox(
    bbox: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return x1, y1, x2, y2


def detect_faces(
    image_bgr: np.ndarray,
    confidence_threshold: float | None = None,
) -> list[DetectedFace]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Передано пустое изображение.")

    detection_settings = load_settings().get("detection", {})

    threshold = (
        float(confidence_threshold)
        if confidence_threshold is not None
        else float(
            detection_settings.get(
                "confidence_threshold",
                0.5,
            )
        )
    )

    maximum_faces = int(
        detection_settings.get(
            "maximum_faces",
            100,
        )
    )

    analyser = get_face_analyser()
    raw_faces = analyser.get(image_bgr)

    height, width = image_bgr.shape[:2]
    detected: list[DetectedFace] = []

    for raw_face in raw_faces:
        confidence = float(
            getattr(
                raw_face,
                "det_score",
                0.0,
            )
        )

        if confidence < threshold:
            continue

        bbox = _clamp_bbox(
            raw_face.bbox,
            width,
            height,
        )
        x1, y1, x2, y2 = bbox

        detected.append(
            DetectedFace(
                index=0,
                raw_face=raw_face,
                bbox=bbox,
                confidence=confidence,
                area=(x2 - x1) * (y2 - y1),
            )
        )

    detected.sort(
        key=lambda item: (
            item.bbox[0],
            item.bbox[1],
        )
    )

    detected = detected[:maximum_faces]

    for index, face in enumerate(detected):
        face.index = index

    return detected


def face_crop(
    image_bgr: np.ndarray,
    face: DetectedFace,
    padding_ratio: float = 0.28,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox

    face_width = x2 - x1
    face_height = y2 - y1

    padding_x = int(face_width * padding_ratio)
    padding_y = int(face_height * padding_ratio)

    crop_x1 = max(0, x1 - padding_x)
    crop_y1 = max(0, y1 - padding_y)
    crop_x2 = min(width, x2 + padding_x)
    crop_y2 = min(height, y2 + padding_y)

    return image_bgr[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()


def detection_preview(
    image_bgr: np.ndarray,
    faces: list[DetectedFace],
) -> np.ndarray:
    result = image_bgr.copy()

    for face in faces:
        x1, y1, x2, y2 = face.bbox

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (70, 220, 120),
            3,
        )
        cv2.putText(
            result,
            str(face.index + 1),
            (x1 + 5, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (70, 220, 120),
            2,
            cv2.LINE_AA,
        )

    return result
