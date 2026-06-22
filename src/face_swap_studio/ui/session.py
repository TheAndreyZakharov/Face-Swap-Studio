from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

SESSION_ROOT = (
    Path(
        tempfile.gettempdir()
    )
    / "face-swap-studio-sessions"
)


@dataclass(slots=True)
class UploadedImage:
    id: str
    name: str
    path: Path


@dataclass(slots=True)
class DetectedFace:
    id: str
    index: int
    path: Path
    label: str

    bbox: tuple[int, int, int, int] | None = None
    crop_box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class TargetAnalysis:
    target_image_id: str
    target_faces: list[DetectedFace] = field(
        default_factory=list
    )
    face_mappings: dict[int, int | None] = field(
        default_factory=dict
    )
    analysis_completed: bool = False


@dataclass(slots=True)
class StudioSession:
    id: str
    directory: Path

    source_images: list[UploadedImage] = field(
        default_factory=list
    )
    target_images: list[UploadedImage] = field(
        default_factory=list
    )

    source_faces: list[DetectedFace] = field(
        default_factory=list
    )

    target_analyses: dict[str, TargetAnalysis] = field(
        default_factory=dict
    )
    selected_target_image_ids: set[str] = field(
        default_factory=set
    )

    active_target_image_id: str | None = None

    result_paths: dict[str, Path] = field(
        default_factory=dict
    )

    @property
    def uploads_directory(self) -> Path:
        directory = (
            self.directory
            / "uploads"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @property
    def source_uploads_directory(self) -> Path:
        directory = (
            self.uploads_directory
            / "sources"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @property
    def target_uploads_directory(self) -> Path:
        directory = (
            self.uploads_directory
            / "targets"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @property
    def faces_directory(self) -> Path:
        directory = (
            self.directory
            / "faces"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @property
    def source_faces_directory(self) -> Path:
        directory = (
            self.faces_directory
            / "sources"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @property
    def target_faces_root_directory(self) -> Path:
        directory = (
            self.faces_directory
            / "targets"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @property
    def results_directory(self) -> Path:
        directory = (
            self.directory
            / "results"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @property
    def target_faces(self) -> list[DetectedFace]:
        analysis = self.active_target_analysis()

        if analysis is None:
            return []

        return analysis.target_faces

    @property
    def face_mappings(self) -> dict[int, int | None]:
        analysis = self.active_target_analysis()

        if analysis is None:
            return {}

        return analysis.face_mappings

    @property
    def analysis_completed(self) -> bool:
        analysis = self.active_target_analysis()

        return bool(
            analysis
            and analysis.analysis_completed
        )

    @property
    def result_path(self) -> Path | None:
        if self.active_target_image_id is None:
            return None

        return self.result_paths.get(
            self.active_target_image_id
        )

    def target_faces_directory_for(
        self,
        target_image_id: str,
    ) -> Path:
        safe_target_id = "".join(
            character
            if character.isalnum()
            else "_"
            for character in target_image_id
        )

        directory = (
            self.target_faces_root_directory
            / safe_target_id
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    def result_path_for(
        self,
        target_image_id: str,
    ) -> Path:
        safe_target_id = "".join(
            character
            if character.isalnum()
            else "_"
            for character in target_image_id
        )

        return (
            self.results_directory
            / f"generated-result-{safe_target_id[:12]}.png"
        )

    def active_target_image(
        self,
    ) -> UploadedImage | None:
        if self.active_target_image_id is None:
            return None

        return self.target_image_by_id(
            self.active_target_image_id
        )

    def target_image_by_id(
        self,
        target_image_id: str,
    ) -> UploadedImage | None:
        for image in self.target_images:
            if image.id == target_image_id:
                return image

        return None

    def source_image_by_id(
        self,
        source_image_id: str,
    ) -> UploadedImage | None:
        for image in self.source_images:
            if image.id == source_image_id:
                return image

        return None

    def active_target_analysis(
        self,
    ) -> TargetAnalysis | None:
        if self.active_target_image_id is None:
            return None

        return self.target_analyses.get(
            self.active_target_image_id
        )

    def ensure_target_analysis(
        self,
        target_image_id: str,
    ) -> TargetAnalysis:
        analysis = self.target_analyses.get(
            target_image_id
        )

        if analysis is None:
            analysis = TargetAnalysis(
                target_image_id=target_image_id,
            )

            self.target_analyses[
                target_image_id
            ] = analysis

        return analysis

    def reset_analysis(
        self,
    ) -> None:
        self.source_faces = []
        self.target_analyses.clear()
        self.result_paths.clear()

        shutil.rmtree(
            self.faces_directory,
            ignore_errors=True,
        )

        self.faces_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.rmtree(
            self.results_directory,
            ignore_errors=True,
        )

        self.results_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def reset_target_analysis(
        self,
        target_image_id: str,
    ) -> None:
        self.target_analyses.pop(
            target_image_id,
            None,
        )

        self.result_paths.pop(
            target_image_id,
            None,
        )

        shutil.rmtree(
            self.target_faces_directory_for(
                target_image_id
            ),
            ignore_errors=True,
        )

    def remove_source_image(
        self,
        source_image_id: str,
    ) -> bool:
        image = self.source_image_by_id(
            source_image_id
        )

        if image is None:
            return False

        self.source_images = [
            item
            for item in self.source_images
            if item.id != source_image_id
        ]

        image.path.unlink(
            missing_ok=True
        )

        self.reset_analysis()

        return True

    def remove_target_image(
        self,
        target_image_id: str,
    ) -> bool:
        image = self.target_image_by_id(
            target_image_id
        )

        if image is None:
            return False

        self.target_images = [
            item
            for item in self.target_images
            if item.id != target_image_id
        ]

        image.path.unlink(
            missing_ok=True
        )

        self.reset_target_analysis(
            target_image_id
        )

        self.selected_target_image_ids.discard(
            target_image_id
        )

        if self.active_target_image_id == target_image_id:
            self.active_target_image_id = (
                self.target_images[0].id
                if self.target_images
                else None
            )

        return True


class SessionStore:
    def __init__(
        self,
    ) -> None:
        self._sessions: dict[str, StudioSession] = {}
        self._lock = threading.RLock()

        self.clear_temporary_storage()

    def clear_temporary_storage(
        self,
    ) -> None:
        shutil.rmtree(
            SESSION_ROOT,
            ignore_errors=True,
        )

        SESSION_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create(
        self,
    ) -> StudioSession:
        session_id = uuid.uuid4().hex

        directory = (
            SESSION_ROOT
            / session_id
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        session = StudioSession(
            id=session_id,
            directory=directory,
        )

        with self._lock:
            self._sessions[
                session_id
            ] = session

        return session

    def get(
        self,
        session_id: str,
    ) -> StudioSession | None:
        with self._lock:
            return self._sessions.get(
                session_id
            )

    def require(
        self,
        session_id: str,
    ) -> StudioSession:
        session = self.get(
            session_id
        )

        if session is None:
            raise KeyError(
                f"Session not found: {session_id}"
            )

        return session

    def delete(
        self,
        session_id: str,
    ) -> None:
        with self._lock:
            session = self._sessions.pop(
                session_id,
                None,
            )

        if session is not None:
            shutil.rmtree(
                session.directory,
                ignore_errors=True,
            )

    def clear(
        self,
    ) -> None:
        with self._lock:
            self._sessions.clear()

        self.clear_temporary_storage()

session_store = SessionStore()