from __future__ import annotations

import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.face_swap_studio.core.pipeline import (
    FaceAssignment,
    SourceIdentity,
    TargetAnalysis,
    analyse_sources,
    analyse_targets,
    process_target,
    write_image,
)
from src.face_swap_studio.services.image_service import (
    source_gallery,
    target_face_gallery,
    target_preview_gallery,
)
from src.face_swap_studio.utils.paths import (
    output_directory,
    temp_directory,
)


@dataclass(slots=True)
class AnalysisSession:
    sources: list[SourceIdentity]
    targets: list[TargetAnalysis]


def create_analysis(
    source_paths: list[str],
    target_paths: list[str],
    confidence_threshold: float,
) -> tuple[
    AnalysisSession,
    list[Any],
    list[Any],
    list[Any],
    list[tuple[int, int]],
    str,
]:
    if not source_paths:
        raise ValueError("Добавьте хотя бы одну исходную фотографию лица.")

    if not target_paths:
        raise ValueError("Добавьте хотя бы одну целевую фотографию.")

    sources = analyse_sources(
        source_paths,
        confidence_threshold,
    )
    targets = analyse_targets(
        target_paths,
        confidence_threshold,
    )

    session = AnalysisSession(
        sources=sources,
        targets=targets,
    )

    initial_assignments: dict[tuple[int, int], int | None] = {}

    faces_gallery, flattened_faces = target_face_gallery(
        targets,
        initial_assignments,
    )

    status = (
        f"Исходных лиц: {len(sources)}. "
        f"Целевых фотографий: {len(targets)}. "
        f"Найдено целевых лиц: "
        f"{sum(len(target.faces) for target in targets)}."
    )

    return (
        session,
        source_gallery(sources),
        target_preview_gallery(targets),
        faces_gallery,
        flattened_faces,
        status,
    )


def visual_assignments_to_pipeline(
    assignments: dict[tuple[int, int], int | None],
) -> list[FaceAssignment]:
    return [
        FaceAssignment(
            target_index=target_index,
            face_index=face_index,
            source_index=source_index,
            enabled=source_index is not None,
        )
        for (
            target_index,
            face_index,
        ), source_index in assignments.items()
    ]


def process_analysis(
    session: AnalysisSession | None,
    visual_assignments: dict[tuple[int, int], int | None],
    swap_model: str,
    enhance_face_regions: bool,
    face_enhancement_weight: float,
    upscale_full_image: bool,
    upscale_factor: float,
    tile_size: int,
) -> tuple[list[tuple[str, str]], str, str]:
    if session is None:
        raise ValueError("Сначала нажмите «Найти лица».")

    assignments = visual_assignments_to_pipeline(visual_assignments)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_directory = output_directory() / f"run-{timestamp}"
    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    gallery: list[tuple[str, str]] = []
    output_files: list[Path] = []

    for target in session.targets:
        result = process_target(
            target=target,
            sources=session.sources,
            assignments=assignments,
            swap_model=swap_model,
            enhance_face_regions=enhance_face_regions,
            face_enhancement_weight=face_enhancement_weight,
            upscale_full_image=upscale_full_image,
            upscale_factor=upscale_factor,
            tile_size=tile_size,
        )

        destination = run_directory / f"{target.path.stem}_swapped.png"

        write_image(destination, result)

        output_files.append(destination)
        gallery.append(
            (
                str(destination),
                destination.name,
            )
        )

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
        for output_file in output_files:
            archive.write(
                output_file,
                arcname=output_file.name,
            )

    status = f"Обработка завершена. Создано файлов: {len(output_files)}. Папка: {run_directory}"

    return (
        gallery,
        str(archive_path),
        status,
    )


def clear_temporary_files() -> str:
    directory = temp_directory()

    for item in directory.iterdir():
        if item.name in {
            ".gitkeep",
            "face-swap-studio.pid",
            "face-swap-studio.log",
        }:
            continue

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)

    return "Временные файлы очищены."
