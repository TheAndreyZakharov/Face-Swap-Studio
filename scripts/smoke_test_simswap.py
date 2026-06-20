from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMSWAP_ROOT = PROJECT_ROOT / "vendor" / "simswap"

sys.path.insert(0, str(SIMSWAP_ROOT))


def import_attribute(module_name: str, attribute_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def main() -> int:
    import torch

    create_model = import_attribute(
        "models.models",
        "create_model",
    )
    test_options_class = import_attribute(
        "options.test_options",
        "TestOptions",
    )
    bisenet_class = import_attribute(
        "parsing_model.model",
        "BiSeNet",
    )

    arcface_path = SIMSWAP_ROOT / "arcface_model" / "arcface_checkpoint.tar"
    generator_path = SIMSWAP_ROOT / "checkpoints" / "512" / "550000_net_G.pth"
    parsing_path = SIMSWAP_ROOT / "parsing_model" / "checkpoint" / "79999_iter.pth"

    required_paths = (
        arcface_path,
        generator_path,
        parsing_path,
    )

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Не найден файл SimSwap: {path}")

    arcface = torch.load(
        arcface_path,
        map_location="cpu",
        weights_only=False,
    )
    print(f"[OK] ArcFace object: {type(arcface)}")

    generator_state = torch.load(
        generator_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(generator_state, dict):
        raise TypeError("Checkpoint SimSwap 512 не является словарём параметров.")

    print(f"[OK] SimSwap 512 state keys: {len(generator_state)}")

    parsing_model = bisenet_class(n_classes=19)
    parsing_state = torch.load(
        parsing_path,
        map_location="cpu",
        weights_only=False,
    )
    parsing_model.load_state_dict(parsing_state)
    parsing_model.eval()

    print("[OK] BiSeNet checkpoint loaded")
    print(f"[OK] create_model: {create_model}")
    print(f"[OK] TestOptions: {test_options_class}")
    print(f"[INFO] MPS available: {torch.backends.mps.is_available()}")
    print("SimSwap smoke test passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
