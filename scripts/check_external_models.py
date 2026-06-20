from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.face_swap_studio.models.manifest import (  # noqa: E402
    is_model_ready,
    model_definitions,
)


def missing_paths(
    required_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    return tuple(
        path.expanduser().resolve()
        for path in required_paths
        if not path.expanduser().resolve().exists()
    )


def runtime_missing_paths(
    environment_python: Path | None,
    runner_path: Path | None,
) -> tuple[Path, ...]:
    missing: list[Path] = []

    for path in (
        environment_python,
        runner_path,
    ):
        if path is None:
            continue

        resolved = path.expanduser().resolve()

        if not resolved.is_file():
            missing.append(resolved)

    return tuple(missing)


def main() -> int:
    failed = False

    for model in model_definitions():
        files_missing = missing_paths(model.required_paths)
        runtime_missing = runtime_missing_paths(
            model.environment_python,
            model.runner_path,
        )
        all_missing = files_missing + runtime_missing

        ready = is_model_ready(model) and not all_missing

        print()
        print(f"{model.name} [{model.id}]")
        print(f"  Тип: {model.kind.value}")
        print(f"  Backend: {model.backend.value}")
        print(f"  Несколько target-лиц: {'да' if model.supports_multiple_faces else 'нет'}")
        print(f"  Индивидуальные назначения: {'да' if model.supports_face_assignments else 'нет'}")

        if ready:
            print("  Статус: READY")
            continue

        failed = True
        print("  Статус: INCOMPLETE")

        if all_missing:
            print("  Отсутствуют:")

            for path in all_missing:
                print(f"    - {path}")
        else:
            print("  Файлы найдены, но проверка готовности модели завершилась неуспешно.")

    print()

    if failed:
        print("Некоторые модели не готовы к запуску.")
        return 1

    print("Все зарегистрированные модели готовы к запуску.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
