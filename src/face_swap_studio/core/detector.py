from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.face_swap_studio.models.model_manager import get_face_analyser


@dataclass(slots=True)
class DetectedFace:
    index: int
    face: Any
    bbox: tuple[int, int, int, int]
    confidence: float
    area: int


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
    confidence_threshold: float = 0.5,
) -> list[DetectedFace]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Передано пустое изображение.")

    analyser = get_face_analyser()
    raw_faces = analyser.get(image_bgr)

    height, width = image_bgr.shape[:2]
    detected: list[DetectedFace] = []

    for raw_face in raw_faces:
        confidence = float(getattr(raw_face, "det_score", 0.0))

        if confidence < confidence_threshold:
            continue

        bbox = _clamp_bbox(raw_face.bbox, width, height)
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        detected.append(
            DetectedFace(
                index=0,
                face=raw_face,
                bbox=bbox,
                confidence=confidence,
                area=area,
            )
        )

    detected.sort(key=lambda item: (item.bbox[0], item.bbox[1]))

    for index, item in enumerate(detected):
        item.index = index

    return detected


def largest_face(faces: list[DetectedFace]) -> DetectedFace:
    if not faces:
        raise ValueError("На изображении не найдено ни одного лица.")

    return max(faces, key=lambda item: item.area)


def face_crop(
    image_bgr: np.ndarray,
    detected_face: DetectedFace,
    padding_ratio: float = 0.25,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = detected_face.bbox

    face_width = x2 - x1
    face_height = y2 - y1

    padding_x = int(face_width * padding_ratio)
    padding_y = int(face_height * padding_ratio)

    crop_x1 = max(0, x1 - padding_x)
    crop_y1 = max(0, y1 - padding_y)
    crop_x2 = min(width, x2 + padding_x)
    crop_y2 = min(height, y2 + padding_y)

    return image_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()


def create_detection_preview(
    image_bgr: np.ndarray,
    faces: list[DetectedFace],
) -> np.ndarray:
    preview = image_bgr.copy()

    for item in faces:
        x1, y1, x2, y2 = item.bbox

        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            preview,
            f"Face {item.index}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    return preview
