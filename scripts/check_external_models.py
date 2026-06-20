from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_swap_studio.models.manifest import available_models  # noqa: E402


def main() -> int:
    failed = False

    for model in available_models():
        print()
        print(f"{model.label} [{model.key}]")
        print(f"  Тип: {model.kind}")
        print(f"  Backend: {model.backend}")

        missing = model.missing_paths()

        if not missing:
            print("  Статус: READY")
            continue

        failed = True
        print("  Статус: INCOMPLETE")
        print("  Отсутствуют:")

        for path in missing:
            print(f"    - {path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())