from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileCollectionState:
    paths: list[str] = field(default_factory=list)
    selected_index: int | None = None

    def add(self, new_paths: list[str]) -> None:
        existing = set(self.paths)

        for path in new_paths:
            resolved = str(Path(path))

            if resolved not in existing:
                self.paths.append(resolved)
                existing.add(resolved)

    def remove_selected(self) -> None:
        if self.selected_index is None:
            return

        if 0 <= self.selected_index < len(self.paths):
            self.paths.pop(self.selected_index)

        self.selected_index = None


@dataclass
class VisualAssignmentState:
    assignments: dict[tuple[int, int], int | None] = field(default_factory=dict)
    selected_source_index: int | None = None
    selected_target_face_index: int | None = None

    def assign(
        self,
        flattened_target_faces: list[tuple[int, int]],
    ) -> None:
        if self.selected_target_face_index is None:
            raise ValueError("Сначала выберите целевое лицо.")

        if self.selected_source_index is None:
            raise ValueError("Сначала выберите исходное лицо.")

        if not 0 <= self.selected_target_face_index < len(flattened_target_faces):
            raise ValueError("Выбрано некорректное целевое лицо.")

        target_key = flattened_target_faces[self.selected_target_face_index]
        self.assignments[target_key] = self.selected_source_index

    def skip(
        self,
        flattened_target_faces: list[tuple[int, int]],
    ) -> None:
        if self.selected_target_face_index is None:
            raise ValueError("Сначала выберите целевое лицо.")

        if not 0 <= self.selected_target_face_index < len(flattened_target_faces):
            raise ValueError("Выбрано некорректное целевое лицо.")

        target_key = flattened_target_faces[self.selected_target_face_index]
        self.assignments[target_key] = None


@dataclass
class StudioSessionState:
    analysis: Any | None = None
    flattened_target_faces: list[tuple[int, int]] = field(default_factory=list)
    visual_assignments: VisualAssignmentState = field(default_factory=VisualAssignmentState)
