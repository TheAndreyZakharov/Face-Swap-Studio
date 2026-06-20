from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.face_swap_studio.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str


def run_external_process(
    command: Sequence[str | Path],
    *,
    cwd: str | Path,
    environment: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> ProcessResult:
    normalized_command = tuple(str(part) for part in command)
    working_directory = Path(cwd).expanduser().resolve()

    if not normalized_command:
        raise ValueError("Передана пустая команда.")

    if not working_directory.is_dir():
        raise FileNotFoundError(f"Рабочая директория не найдена: {working_directory}")

    executable = Path(normalized_command[0]).expanduser()

    if executable.is_absolute() and not executable.is_file():
        raise FileNotFoundError(f"Интерпретатор не найден: {executable}")

    process_environment = os.environ.copy()
    process_environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    if environment:
        process_environment.update(environment)

    logger.info(
        "Запуск внешнего процесса: %s",
        list(normalized_command),
    )
    logger.info(
        "Рабочая директория: %s",
        working_directory,
    )

    try:
        completed = subprocess.run(
            normalized_command,
            cwd=working_directory,
            env=process_environment,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Внешняя модель превысила допустимое время выполнения.") from error
    except OSError as error:
        raise RuntimeError(f"Не удалось запустить внешний процесс: {error}") from error

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if stdout:
        logger.info(
            "STDOUT внешней модели:\n%s",
            stdout,
        )

    if stderr:
        logger.warning(
            "STDERR внешней модели:\n%s",
            stderr,
        )

    result = ProcessResult(
        command=normalized_command,
        return_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
    )

    if completed.returncode != 0:
        details = stderr or stdout or "Текст ошибки отсутствует."

        raise RuntimeError(
            f"Внешняя модель завершилась с ошибкой (код {completed.returncode}).\n\n{details}"
        )

    return result


def require_output_file(
    path: str | Path,
    *,
    model_name: str,
    process_result: ProcessResult,
) -> Path:
    output_path = Path(path).expanduser().resolve()

    if output_path.is_file() and output_path.stat().st_size > 0:
        return output_path

    parts = [
        f"{model_name} не создал результирующий файл.",
        f"Ожидаемый путь: {output_path}",
        f"Код процесса: {process_result.return_code}",
    ]

    if process_result.stdout:
        parts.extend(
            [
                "",
                "STDOUT:",
                process_result.stdout,
            ]
        )

    if process_result.stderr:
        parts.extend(
            [
                "",
                "STDERR:",
                process_result.stderr,
            ]
        )

    raise RuntimeError("\n".join(parts))
