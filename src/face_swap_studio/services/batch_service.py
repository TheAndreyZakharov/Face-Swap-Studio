from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path

from src.face_swap_studio.core.pipeline import process_single_pair
from src.face_swap_studio.domain.entities import ProcessingOptions
from src.face_swap_studio.utils.paths import (
    output_directory,
    temp_directory,
)


def process_batch(
    source_path: str | Path,
    target_paths: list[str | Path],
    options: ProcessingOptions,
    target_face_index: int | None = None,
) -> tuple[list[Path], Path]:
    if not target_paths:
        raise ValueError("Список target-файлов пуст.")

    results: list[Path] = []

    for target_path in target_paths:
        result = process_single_pair(
            source_path=source_path,
            target_path=target_path,
            options=options,
            target_face_index=target_face_index,
        )
        results.append(result)

    archive_path = create_results_archive(results)

    return results, archive_path


def create_results_archive(
    result_paths: list[Path],
) -> Path:
    if not result_paths:
        raise ValueError("Нет результатов для создания архива.")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    archive_path = temp_directory() / f"face-swap-results-{timestamp}.zip"
    archive_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        used_names: set[str] = set()

        for index, result_path in enumerate(
            result_paths,
            start=1,
        ):
            resolved = Path(result_path)

            if not resolved.is_file():
                raise FileNotFoundError(f"Не найден результат: {resolved}")

            archive_name = resolved.name

            if archive_name in used_names:
                archive_name = f"{index:03d}_{archive_name}"

            used_names.add(archive_name)

            archive.write(
                resolved,
                arcname=archive_name,
            )

    return archive_path


def clear_temporary_files() -> str:
    directory = temp_directory()
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    protected_names = {
        ".gitkeep",
        "face-swap-studio.pid",
        "face-swap-studio.log",
    }

    removed_count = 0

    for item in directory.iterdir():
        if item.name in protected_names:
            continue

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)

        removed_count += 1

    return f"Временные файлы очищены. Удалено объектов: {removed_count}."


def clear_output_files() -> str:
    directory = output_directory()
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    removed_count = 0

    for item in directory.iterdir():
        if item.name == ".gitkeep":
            continue

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)

        removed_count += 1

    return f"Результаты очищены. Удалено объектов: {removed_count}."
