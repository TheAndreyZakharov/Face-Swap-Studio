from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class DetectedFace:
    index: int
    raw_face: Any
    bbox: tuple[int, int, int, int]
    confidence: float
    area: int


@dataclass(slots=True)
class SourceFace:
    id: str
    file_path: Path
    image_bgr: np.ndarray
    face: DetectedFace
    label: str


@dataclass(slots=True)
class TargetImage:
    id: str
    file_path: Path
    image_bgr: np.ndarray
    faces: list[DetectedFace]


@dataclass(slots=True)
class FaceAssignment:
    target_id: str
    target_face_index: int
    source_id: str | None
    enabled: bool = True


@dataclass(slots=True)
class ProcessingOptions:
    model_id: str = "inswapper_128"
    enhance_faces: bool = False
    face_enhancement_weight: float = 0.35
    upscale_image: bool = False
    upscale_factor: float = 2.0
    tile_size: int = 256


@dataclass(slots=True)
class StudioSession:
    sources: list[SourceFace] = field(default_factory=list)
    targets: list[TargetImage] = field(default_factory=list)
    assignments: list[FaceAssignment] = field(default_factory=list)

    selected_source_id: str | None = None
    selected_target_id: str | None = None
    selected_target_face_index: int | None = None

    def source_by_id(self, source_id: str) -> SourceFace:
        for source in self.sources:
            if source.id == source_id:
                return source

        raise KeyError(f"Не найдено source-лицо: {source_id}")

    def target_by_id(self, target_id: str) -> TargetImage:
        for target in self.targets:
            if target.id == target_id:
                return target

        raise KeyError(f"Не найдено target-изображение: {target_id}")

    def assignment_for(
        self,
        target_id: str,
        target_face_index: int,
    ) -> FaceAssignment | None:
        for assignment in self.assignments:
            if (
                assignment.target_id == target_id
                and assignment.target_face_index == target_face_index
            ):
                return assignment

        return None
