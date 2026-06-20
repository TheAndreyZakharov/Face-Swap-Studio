from __future__ import annotations

import importlib
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GHOST2_ROOT = PROJECT_ROOT / "vendor" / "ghost2"

sys.path.insert(0, str(GHOST2_ROOT))


def import_attribute(module_name: str, attribute_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл GHOST 2.0: {path}")

    return path


def main() -> int:
    import onnxruntime as ort
    import torch

    stylematte_class = import_attribute(
        "repos.stylematte.stylematte.models",
        "StyleMatte",
    )
    aligner_generator_class = import_attribute(
        "src.aligner.generator",
        "Generator",
    )
    blender_generator_class = import_attribute(
        "src.blender.generator",
        "BlenderGenerator",
    )

    aligner_path = require_file(
        GHOST2_ROOT / "aligner_checkpoints" / "aligner_1020_gaze_final.ckpt"
    )
    blender_path = require_file(GHOST2_ROOT / "blender_checkpoints" / "blender_lama.ckpt")
    stylematte_path = require_file(
        GHOST2_ROOT / "repos" / "stylematte" / "stylematte" / "checkpoints" / "stylematte_synth.pth"
    )
    flame_path = require_file(GHOST2_ROOT / "repos" / "deca" / "data" / "generic_model.pkl")
    segformer_path = require_file(GHOST2_ROOT / "weights" / "segformer_B5_ce.onnx")

    aligner_checkpoint = torch.load(
        aligner_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(aligner_checkpoint, dict):
        raise TypeError("Aligner checkpoint имеет неожиданный формат.")

    print(f"[OK] Aligner checkpoint keys: {len(aligner_checkpoint)}")

    blender_checkpoint = torch.load(
        blender_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(blender_checkpoint, dict):
        raise TypeError("Blender checkpoint имеет неожиданный формат.")

    if "state_dict" not in blender_checkpoint:
        raise KeyError("В Blender checkpoint отсутствует state_dict.")

    print(f"[OK] Blender checkpoint keys: {len(blender_checkpoint)}")
    print(f"[OK] Blender state keys: {len(blender_checkpoint['state_dict'])}")

    stylematte = stylematte_class()
    stylematte_state = torch.load(
        stylematte_path,
        map_location="cpu",
        weights_only=False,
    )

    stylematte.load_state_dict(stylematte_state)
    stylematte.eval()

    print("[OK] StyleMatte loaded")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"scipy\.sparse.*",
        )

        with flame_path.open("rb") as file:
            flame = pickle.load(
                file,
                encoding="latin1",
            )

    if not isinstance(flame, dict):
        raise TypeError("FLAME model имеет неожиданный формат.")

    expected_flame_keys = {
        "f",
        "J_regressor",
        "kintree_table",
        "weights",
        "v_template",
        "shapedirs",
    }

    missing_flame_keys = expected_flame_keys.difference(flame)

    if missing_flame_keys:
        raise KeyError(f"В FLAME отсутствуют ключи: {sorted(missing_flame_keys)}")

    print(f"[OK] FLAME keys: {list(flame)[:10]}")

    session = ort.InferenceSession(
        str(segformer_path),
        providers=["CPUExecutionProvider"],
    )

    print(f"[OK] SegFormer inputs: {[item.name for item in session.get_inputs()]}")
    print(f"[OK] SegFormer outputs: {[item.name for item in session.get_outputs()]}")

    print(f"[OK] Aligner generator class: {aligner_generator_class}")
    print(f"[OK] Blender generator class: {blender_generator_class}")
    print(f"[INFO] MPS available: {torch.backends.mps.is_available()}")
    print("GHOST 2.0 asset smoke test passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
