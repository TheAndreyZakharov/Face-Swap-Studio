from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class AdapterRequest:
    """Универсальный запрос к адаптеру замены лица или головы."""

    def __init__(
        self,
        source_path: str | Path,
        target_path: str | Path,
        output_path: str | Path,
        model_id: str | None = None,
        target_face_index: int | None = None,
        **extra: Any,
    ) -> None:
        self.source_path = Path(source_path).expanduser().resolve()
        self.target_path = Path(target_path).expanduser().resolve()
        self.output_path = Path(output_path).expanduser().resolve()

        self.model_id = model_id
        self.target_face_index = target_face_index

        for name, value in extra.items():
            setattr(self, name, value)

    @property
    def source(self) -> Path:
        return self.source_path

    @property
    def target(self) -> Path:
        return self.target_path

    @property
    def output(self) -> Path:
        return self.output_path

    def validate(self) -> None:
        if not self.source_path.is_file():
            raise FileNotFoundError(f"Source-файл не найден: {self.source_path}")

        if not self.target_path.is_file():
            raise FileNotFoundError(f"Target-файл не найден: {self.target_path}")

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


class SwapAdapter(ABC):
    """Основной интерфейс адаптера модели."""

    model_ids: tuple[str, ...] = ()

    def supports_model(
        self,
        model_id: str,
    ) -> bool:
        return model_id in self.model_ids

    @abstractmethod
    def process(
        self,
        request: AdapterRequest,
    ) -> Path:
        """Запускает модель и возвращает путь к результату."""

        raise NotImplementedError


class BaseAdapter(SwapAdapter):
    """
    Совместимое базовое имя для существующих адаптеров.

    Старые адаптеры могут наследоваться от BaseAdapter, а новые —
    непосредственно от SwapAdapter.
    """

    pass


def ensure_output_created(
    output_path: str | Path,
    model_name: str,
) -> Path:
    """Проверяет наличие непустого файла результата."""

    resolved = Path(output_path).expanduser().resolve()

    if not resolved.is_file():
        raise RuntimeError(f"{model_name} завершился без создания результата: {resolved}")

    if resolved.stat().st_size == 0:
        raise RuntimeError(f"{model_name} создал пустой файл результата: {resolved}")

    return resolved


__all__ = [
    "AdapterRequest",
    "BaseAdapter",
    "SwapAdapter",
    "ensure_output_created",
]
