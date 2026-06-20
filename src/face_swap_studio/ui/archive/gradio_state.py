from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UIState:
    source_paths: list[str] = field(
        default_factory=list
    )
    target_paths: list[str] = field(
        default_factory=list
    )

    source_faces: list[Any] = field(
        default_factory=list
    )
    target_faces: list[Any] = field(
        default_factory=list
    )

    selected_source_face_index: int | None = None
    selected_target_face_index: int | None = None
    selected_target_image_index: int = 0

    face_mappings: dict[int, int | None] = field(
        default_factory=dict
    )

    analysis_completed: bool = False

    def reset_analysis(self) -> None:
        self.source_faces = []
        self.target_faces = []

        self.selected_source_face_index = None
        self.selected_target_face_index = None
        self.selected_target_image_index = 0

        self.face_mappings = {}
        self.analysis_completed = False

    def reset_all(self) -> None:
        self.source_paths = []
        self.target_paths = []
        self.reset_analysis()