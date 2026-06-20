from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.face_swap_studio.core.detector import (
    DetectedFace,
    detect_faces,
    largest_face,
)
from src.face_swap_studio.core.enhancer import enhance_faces
from src.face_swap_studio.core.swapper import swap_face
from src.face_swap_studio.core.upscaler import resize_if_too_large, upscale_image
from src.face_swap_studio.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SourceIdentity:
    index: int
    path: Path
    image: np.ndarray
    detected_face: DetectedFace


@dataclass(slots=True)
class TargetAnalysis:
    index: int
    path: Path
    image: np.ndarray
    faces: list[DetectedFace]


@dataclass(slots=True)
class FaceAssignment:
    target_index: int
    face_index: int
    source_index: int | None
    enabled: bool = True


def read_image(path: str | Path) -> np.ndarray:
    resolved = Path(path)

    if not resolved.is_file():
        raise FileNotFoundError(f"Не найдено изображение: {resolved}")

    data = np.fromfile(resolved, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {resolved}")

    return image


def write_image(path: str | Path, image_bgr: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    extension = destination.suffix.lower() or ".png"
    success, encoded = cv2.imencode(extension, image_bgr)

    if not success:
        raise RuntimeError(f"Не удалось закодировать изображение: {destination}")

    encoded.tofile(destination)
    return destination


def analyse_sources(
    source_paths: list[str | Path],
    confidence_threshold: float = 0.5,
) -> list[SourceIdentity]:
    sources: list[SourceIdentity] = []

    for source_path in source_paths:
        image = read_image(source_path)
        faces = detect_faces(image, confidence_threshold)

        if not faces:
            logger.warning("В source-файле не найдено лицо: %s", source_path)
            continue

        selected = largest_face(faces)

        sources.append(
            SourceIdentity(
                index=len(sources),
                path=Path(source_path),
                image=image,
                detected_face=selected,
            )
        )

    if not sources:
        raise ValueError("Ни в одном исходном изображении не найдено лицо.")

    return sources


def analyse_targets(
    target_paths: list[str | Path],
    confidence_threshold: float = 0.5,
) -> list[TargetAnalysis]:
    targets: list[TargetAnalysis] = []

    for index, target_path in enumerate(target_paths):
        image = read_image(target_path)
        faces = detect_faces(image, confidence_threshold)

        targets.append(
            TargetAnalysis(
                index=index,
                path=Path(target_path),
                image=image,
                faces=faces,
            )
        )

    return targets


def process_target(
    target: TargetAnalysis,
    sources: list[SourceIdentity],
    assignments: list[FaceAssignment],
    swap_model: str = "InSwapper 128",
    enhance_face_regions: bool = False,
    face_enhancement_weight: float = 0.35,
    upscale_full_image: bool = False,
    upscale_factor: float = 2.0,
    tile_size: int = 256,
) -> np.ndarray:
    result = target.image.copy()

    target_assignments = {
        assignment.face_index: assignment
        for assignment in assignments
        if assignment.target_index == target.index and assignment.enabled
    }

    for target_face in target.faces:
        assignment = target_assignments.get(target_face.index)

        if assignment is None or assignment.source_index is None:
            continue

        if assignment.source_index < 0 or assignment.source_index >= len(sources):
            logger.warning(
                "Некорректный source index %s для target %s face %s",
                assignment.source_index,
                target.index,
                target_face.index,
            )
            continue

        source = sources[assignment.source_index]

        result = swap_face(
            image_bgr=result,
            source_face=source.detected_face.face,
            target_face=target_face.face,
            model_label=swap_model,
        )

    if enhance_face_regions:
        result = enhance_faces(
            result,
            weight=face_enhancement_weight,
        )

    if upscale_full_image:
        result = upscale_image(
            result,
            output_scale=upscale_factor,
            tile_size=tile_size,
        )
        result = resize_if_too_large(result)

    return result
