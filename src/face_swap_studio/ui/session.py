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
    target_faces: list[DetectedFace] = field(
        default_factory=list
    )

    face_mappings: dict[int, int | None] = field(
        default_factory=dict
    )

    active_target_image_id: str | None = None
    analysis_completed: bool = False

    result_path: Path | None = None

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
    def target_faces_directory(self) -> Path:
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

    def active_target_image(
        self,
    ) -> UploadedImage | None:
        if self.active_target_image_id is None:
            return None

        for image in self.target_images:
            if image.id == self.active_target_image_id:
                return image

        return None

    def reset_analysis(
        self,
    ) -> None:
        self.source_faces = []
        self.target_faces = []
        self.face_mappings = {}
        self.analysis_completed = False
        self.result_path = None

        shutil.rmtree(
            self.faces_directory,
            ignore_errors=True,
        )

        self.faces_directory.mkdir(
            parents=True,
            exist_ok=True,
        )


class SessionStore:
    def __init__(
        self,
    ) -> None:
        self._sessions: dict[str, StudioSession] = {}
        self._lock = threading.RLock()

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
            sessions = list(
                self._sessions.values()
            )
            self._sessions.clear()

        for session in sessions:
            shutil.rmtree(
                session.directory,
                ignore_errors=True,
            )

        shutil.rmtree(
            SESSION_ROOT,
            ignore_errors=True,
        )

        SESSION_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )


session_store = SessionStore()