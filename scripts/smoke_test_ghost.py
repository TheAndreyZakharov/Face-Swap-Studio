from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GHOST_ROOT = PROJECT_ROOT / "vendor" / "ghost"

sys.path.insert(0, str(GHOST_ROOT))


def import_attribute(module_name: str, attribute_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def generator_checkpoint(blocks: int) -> Path:
    suffix = "block" if blocks == 1 else "blocks"

    return GHOST_ROOT / "weights" / f"G_unet_{blocks}{suffix}.pth"


def load_generator(blocks: int) -> None:
    import torch

    generator_class = import_attribute(
        "network.AEI_Net",
        "AEI_Net",
    )

    checkpoint_path = generator_checkpoint(blocks)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Не найден checkpoint GHOST: {checkpoint_path}")

    model = generator_class(
        "unet",
        num_blocks=blocks,
        c_id=512,
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(state)
    model.eval()

    parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"[OK] GHOST {blocks} block(s): {parameters:,} parameters")


def main() -> int:
    import torch

    arcface_factory = import_attribute(
        "arcface_model.iresnet",
        "iresnet100",
    )

    for blocks in (1, 2, 3):
        load_generator(blocks)

    arcface_path = GHOST_ROOT / "arcface_model" / "backbone.pth"

    if not arcface_path.is_file():
        raise FileNotFoundError(f"Не найден ArcFace GHOST: {arcface_path}")

    arcface = arcface_factory(fp16=False)
    arcface_state = torch.load(
        arcface_path,
        map_location="cpu",
        weights_only=False,
    )

    arcface.load_state_dict(arcface_state)
    arcface.eval()

    print("[OK] GHOST ArcFace loaded")
    print(f"[INFO] MPS available: {torch.backends.mps.is_available()}")
    print("GHOST smoke test passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
