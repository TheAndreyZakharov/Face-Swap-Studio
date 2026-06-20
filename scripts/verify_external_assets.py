from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.face_swap_studio.models.manifest import (  # noqa: E402
    model_definitions,
)

TORCH_SUFFIXES = {
    ".pth",
    ".pt",
    ".ckpt",
}

EXTERNALLY_VALIDATED_FILES = {
    "arcface_checkpoint.tar": (
        "Проверяется внутри окружения SimSwap командой `python scripts/smoke_test_simswap.py`."
    ),
    "generic_model.pkl": (
        "Проверяется внутри окружения GHOST 2.0 командой `python scripts/smoke_test_ghost2.py`."
    ),
}


def describe_checkpoint(
    value: Any,
) -> str:
    if isinstance(value, dict):
        keys = list(value.keys())

        return f"dict, keys={len(keys)}, sample={keys[:5]}"

    return type(value).__name__


def verify_torch_file(
    path: Path,
) -> tuple[bool, str]:
    try:
        import torch

        value = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        return (
            False,
            str(error),
        )

    return (
        True,
        describe_checkpoint(value),
    )


def verify_onnx_file(
    path: Path,
) -> tuple[bool, str]:
    try:
        import onnx

        model = onnx.load(
            path,
            load_external_data=False,
        )
        onnx.checker.check_model(
            model,
            full_check=False,
        )
    except Exception as error:
        return (
            False,
            str(error),
        )

    return (
        True,
        (f"inputs={len(model.graph.input)}, outputs={len(model.graph.output)}"),
    )


def model_runtime_paths(
    environment_python: Path | None,
    runner_path: Path | None,
) -> tuple[Path, ...]:
    paths: list[Path] = []

    for path in (
        environment_python,
        runner_path,
    ):
        if path is not None:
            paths.append(path.expanduser().resolve())

    return tuple(paths)


def verify_path(
    path: Path,
    inspected: set[Path],
) -> bool:
    resolved = path.expanduser().resolve()

    if not resolved.exists():
        print(f"[MISSING] {resolved}")
        return False

    if resolved.is_dir():
        print(f"[OK DIR] {resolved}")
        return True

    size_mb = resolved.stat().st_size / 1024 / 1024

    print(f"[OK FILE] {resolved} ({size_mb:.1f} MB)")

    if resolved in inspected:
        print("  [ALREADY CHECKED]")
        return True

    inspected.add(resolved)

    external_message = EXTERNALLY_VALIDATED_FILES.get(resolved.name)

    if external_message is not None:
        print(f"  [EXTERNAL TEST] {external_message}")
        return True

    suffix = resolved.suffix.lower()

    if suffix == ".onnx":
        success, details = verify_onnx_file(resolved)
    elif suffix in TORCH_SUFFIXES:
        success, details = verify_torch_file(resolved)
    else:
        print("  [PRESENT] Дополнительная проверка не требуется.")
        return True

    state = "VALID" if success else "INVALID"

    print(f"  [{state}] {details}")

    return success


def main() -> int:
    failed = False
    inspected: set[Path] = set()

    for model in model_definitions():
        print()
        print("=" * 72)
        print(f"{model.name} [{model.id}]")
        print("=" * 72)

        paths = tuple(model.required_paths) + model_runtime_paths(
            model.environment_python,
            model.runner_path,
        )

        for path in paths:
            if not verify_path(
                path,
                inspected,
            ):
                failed = True

    print()

    if failed:
        print("Обнаружены отсутствующие или повреждённые файлы.")
        return 1

    print("Все основные файлы моделей присутствуют и структурно исправны.")
    print("Специфические модели проверяются отдельными smoke-тестами.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
