from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_swap_studio.models.manifest import available_models  # noqa: E402

TORCH_SUFFIXES = {".pth", ".pt", ".ckpt"}

EXTERNALLY_VALIDATED_FILES = {
    "arcface_checkpoint.tar": (
        "Проверяется внутри окружения SimSwap командой `python scripts/smoke_test_simswap.py`."
    ),
    "generic_model.pkl": (
        "Проверяется внутри окружения GHOST 2.0 командой `python scripts/smoke_test_ghost2.py`."
    ),
}


def describe_checkpoint(value: Any) -> str:
    if isinstance(value, dict):
        keys = list(value.keys())
        return f"dict, keys={len(keys)}, sample={keys[:5]}"

    return type(value).__name__


def verify_torch_file(path: Path) -> tuple[bool, str]:
    try:
        import torch

        value = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        return False, str(error)

    return True, describe_checkpoint(value)


def verify_onnx_file(path: Path) -> tuple[bool, str]:
    try:
        import onnx

        model = onnx.load(path)
        onnx.checker.check_model(model)
    except Exception as error:
        return False, str(error)

    return (
        True,
        f"inputs={len(model.graph.input)}, outputs={len(model.graph.output)}",
    )


def main() -> int:
    failed = False
    inspected: set[Path] = set()

    for model in available_models():
        print()
        print("=" * 72)
        print(f"{model.label} [{model.key}]")
        print("=" * 72)

        for path in model.required_paths:
            if not path.exists():
                print(f"[MISSING] {path}")
                failed = True
                continue

            if path.is_dir():
                print(f"[OK DIR] {path}")
                continue

            size_mb = path.stat().st_size / 1024 / 1024
            print(f"[OK FILE] {path} ({size_mb:.1f} MB)")

            resolved = path.resolve()

            if resolved in inspected:
                continue

            inspected.add(resolved)

            external_message = EXTERNALLY_VALIDATED_FILES.get(path.name)

            if external_message is not None:
                print(f"  [EXTERNAL TEST] {external_message}")
                continue

            suffix = path.suffix.lower()

            if suffix == ".onnx":
                success, details = verify_onnx_file(path)
            elif suffix in TORCH_SUFFIXES:
                success, details = verify_torch_file(path)
            else:
                print("  [PRESENT] Дополнительная проверка не требуется.")
                continue

            state = "VALID" if success else "INVALID"
            print(f"  [{state}] {details}")

            if not success:
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
