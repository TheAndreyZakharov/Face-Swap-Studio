from __future__ import annotations

import atexit
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from src.face_swap_studio.core.enhancer import enhance_faces
from src.face_swap_studio.core.upscaler import (
    resize_if_too_large,
    upscale_image,
)
from src.face_swap_studio.domain.entities import ProcessingOptions
from src.face_swap_studio.utils.paths import (
    load_settings,
    output_directory,
)

_OUTPUT_CLEANUP_LOCK = threading.RLock()
_OUTPUT_CLEANUP_REGISTERED = False


def cleanup_output_directory() -> None:
    """
    Полностью очищает data/output, сохраняя только .gitkeep.

    Вызывается:
    1. При запуске приложения — удаляет остатки после аварийного завершения.
    2. При нормальном завершении Python-процесса.
    3. При Ctrl+C, поскольку Python выполняет atexit-обработчики.
    """
    with _OUTPUT_CLEANUP_LOCK:
        directory = output_directory().expanduser().resolve()

        if not directory.exists():
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            return

        if not directory.is_dir():
            raise RuntimeError(
                f"Путь output существует, но не является директорией: "
                f"{directory}"
            )

        for child in directory.iterdir():
            if child.name == ".gitkeep":
                continue

            try:
                if child.is_symlink() or child.is_file():
                    child.unlink(
                        missing_ok=True,
                    )
                elif child.is_dir():
                    shutil.rmtree(
                        child,
                        ignore_errors=False,
                    )
            except FileNotFoundError:
                continue
            except OSError as error:
                print(
                    f"[cleanup] Не удалось удалить {child}: {error}",
                    flush=True,
                )


def install_output_cleanup() -> None:
    """
    Один раз устанавливает автоматическую очистку output.
    """
    global _OUTPUT_CLEANUP_REGISTERED

    with _OUTPUT_CLEANUP_LOCK:
        if _OUTPUT_CLEANUP_REGISTERED:
            return

        _OUTPUT_CLEANUP_REGISTERED = True

        # Сразу удаляем результаты, оставшиеся после предыдущего запуска.
        cleanup_output_directory()

        # Очищаем результаты при нормальном закрытии приложения и Ctrl+C.
        atexit.register(
            cleanup_output_directory
        )


install_output_cleanup()


def read_image(
    path: str | Path,
) -> np.ndarray:
    resolved = Path(
        path
    ).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Не найдено изображение: {resolved}"
        )

    encoded_data = np.fromfile(
        resolved,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        encoded_data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Не удалось прочитать изображение: {resolved}"
        )

    return image


def write_image(
    path: str | Path,
    image_bgr: np.ndarray,
) -> Path:
    if (
        image_bgr is None
        or image_bgr.size == 0
    ):
        raise ValueError(
            "Нельзя сохранить пустое изображение."
        )

    destination = Path(
        path
    ).expanduser()

    if not destination.suffix:
        destination = destination.with_suffix(
            ".png"
        )

    destination = destination.resolve()
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extension = destination.suffix.lower()

    supported_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    if extension not in supported_extensions:
        raise ValueError(
            f"Неподдерживаемый формат изображения: {extension}"
        )

    encode_parameters: list[int] = []

    if extension in {
        ".jpg",
        ".jpeg",
    }:
        settings = load_settings()

        quality = int(
            settings.get(
                "processing",
                {},
            ).get(
                "jpeg_quality",
                95,
            )
        )

        quality = max(
            1,
            min(
                quality,
                100,
            ),
        )

        encode_parameters = [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
        ]

    success, encoded_image = cv2.imencode(
        extension,
        image_bgr,
        encode_parameters,
    )

    if not success:
        raise RuntimeError(
            f"Не удалось закодировать изображение: {destination}"
        )

    encoded_image.tofile(
        destination
    )

    return destination


def create_output_path(
    target_path: Path,
    model_id: str,
) -> Path:
    """
    Создаёт временный результат внутри data/output.

    Файл существует только во время работы приложения.
    При закрытии приложения весь output автоматически очищается.
    """
    timestamp = time.strftime(
        "%Y%m%d-%H%M%S"
    )
    unique_suffix = uuid.uuid4().hex[
        :8
    ]

    run_directory = (
        output_directory()
        / f"run-{timestamp}-{unique_suffix}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_model_id = "".join(
        character
        if (
            character.isalnum()
            or character in {
                "-",
                "_",
            }
        )
        else "_"
        for character in model_id
    )

    filename = (
        f"{target_path.stem}_"
        f"{safe_model_id}.png"
    )

    return run_directory / filename


def apply_postprocessing(
    result_path: Path,
    options: ProcessingOptions,
) -> Path:
    if (
        not options.enhance_faces
        and not options.upscale_image
    ):
        return result_path

    result_image = read_image(
        result_path
    )

    if options.enhance_faces:
        result_image = enhance_faces(
            result_image,
            weight=options.face_enhancement_weight,
        )

    if options.upscale_image:
        result_image = upscale_image(
            result_image,
            output_scale=options.upscale_factor,
            tile_size=options.tile_size,
        )

        settings = load_settings()

        maximum_side = int(
            settings.get(
                "image_enhancement",
                {},
            ).get(
                "maximum_output_side",
                8000,
            )
        )

        result_image = resize_if_too_large(
            result_image,
            maximum_side=maximum_side,
        )

    return write_image(
        result_path,
        result_image,
    )


def process_single_pair(
    source_path: str | Path,
    target_path: str | Path,
    options: ProcessingOptions,
    target_face_index: int | None = None,
) -> Path:
    source = Path(
        source_path
    ).expanduser().resolve()

    target = Path(
        target_path
    ).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"Не найден source-файл: {source}"
        )

    if not target.is_file():
        raise FileNotFoundError(
            f"Не найден target-файл: {target}"
        )

    output_path = create_output_path(
        target_path=target,
        model_id=options.model_id,
    )

    # Эти импорты намеренно находятся внутри функции.
    # Перенос наверх создаст циклический импорт:
    # pipeline -> adapters -> inswapper -> pipeline.
    from src.face_swap_studio.adapters.base import AdapterRequest
    from src.face_swap_studio.adapters.registry import get_adapter

    adapter = get_adapter(
        options.model_id
    )

    if not adapter.is_available():
        raise RuntimeError(
            f"Модель не готова к работе: {options.model_id}"
        )

    request = AdapterRequest(
        source_path=source,
        target_path=target,
        output_path=output_path,
        target_face_index=target_face_index,
    )

    result_path = Path(
        adapter.process(
            request
        )
    ).expanduser().resolve()

    if not result_path.is_file():
        raise RuntimeError(
            "Адаптер завершил работу, но не создал "
            "результирующий файл."
        )

    return apply_postprocessing(
        result_path=result_path,
        options=options,
    )