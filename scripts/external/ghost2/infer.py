from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
GHOST2_ROOT = PROJECT_ROOT / "vendor" / "ghost2"

sys.path.insert(
    0,
    str(EXTERNAL_SCRIPTS_ROOT),
)

from runtime_compat import (  # noqa: E402
    configure_apple_silicon_environment,
    install_numpy_compatibility,
    install_onnxruntime_compatibility,
    install_torch_compatibility,
    install_warning_filters,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GHOST 2.0 CPU wrapper for Apple Silicon."
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--use-kandinsky",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser.parse_args()


def require_file(
    path: Path,
    label: str,
) -> Path:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{label} не найден: {resolved}"
        )

    return resolved


def require_directory(
    path: Path,
    label: str,
) -> Path:
    resolved = path.expanduser().resolve()

    if not resolved.is_dir():
        raise FileNotFoundError(
            f"{label} не найден: {resolved}"
        )

    return resolved


def find_generated_result(
    temporary_directory: Path,
    expected_path: Path,
) -> Path | None:
    if (
        expected_path.is_file()
        and expected_path.stat().st_size > 0
    ):
        return expected_path

    candidates = sorted(
        (
            *temporary_directory.rglob(
                "*.png"
            ),
            *temporary_directory.rglob(
                "*.jpg"
            ),
            *temporary_directory.rglob(
                "*.jpeg"
            ),
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for candidate in candidates:
        if (
            candidate.is_file()
            and candidate.stat().st_size > 0
        ):
            return candidate

    return None


def main() -> int:
    arguments = parse_arguments()

    configure_apple_silicon_environment(
        force_cpu=True
    )
    install_warning_filters()

    import numpy as np

    install_numpy_compatibility(
        np
    )

    source = require_file(
        arguments.source,
        "Source-файл",
    )
    target = require_file(
        arguments.target,
        "Target-файл",
    )

    output = (
        arguments.output
        .expanduser()
        .resolve()
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    require_directory(
        GHOST2_ROOT,
        "Репозиторий GHOST 2.0",
    )

    inference_script = require_file(
        GHOST2_ROOT
        / "inference.py",
        "GHOST 2.0 inference.py",
    )
    aligner_config = require_file(
        GHOST2_ROOT
        / "configs"
        / "aligner.yaml",
        "GHOST 2.0 aligner config",
    )
    blender_config = require_file(
        GHOST2_ROOT
        / "configs"
        / "blender.yaml",
        "GHOST 2.0 blender config",
    )
    aligner_checkpoint = require_file(
        GHOST2_ROOT
        / "aligner_checkpoints"
        / "aligner_1020_gaze_final.ckpt",
        "GHOST 2.0 aligner checkpoint",
    )
    blender_checkpoint = require_file(
        GHOST2_ROOT
        / "blender_checkpoints"
        / "blender_lama.ckpt",
        "GHOST 2.0 blender checkpoint",
    )

    require_file(
        GHOST2_ROOT
        / "weights"
        / "backbone50_1.pth",
        "GHOST 2.0 backbone",
    )
    require_file(
        GHOST2_ROOT
        / "weights"
        / "vgg19-d01eb7cb.pth",
        "GHOST 2.0 VGG19 model",
    )
    require_file(
        GHOST2_ROOT
        / "weights"
        / "segformer_B5_ce.onnx",
        "GHOST 2.0 SegFormer model",
    )
    require_file(
        GHOST2_ROOT
        / "repos"
        / "stylematte"
        / "stylematte"
        / "checkpoints"
        / "stylematte_synth.pth",
        "GHOST 2.0 StyleMatte model",
    )
    require_file(
        GHOST2_ROOT
        / "repos"
        / "deca"
        / "data"
        / "generic_model.pkl",
        "GHOST 2.0 FLAME model",
    )
    require_file(
        GHOST2_ROOT
        / "repos"
        / "BlazeFace_PyTorch"
        / "blazeface.pth",
        "GHOST 2.0 BlazeFace model",
    )

    sys.path.insert(
        0,
        str(
            GHOST2_ROOT
        ),
    )

    import onnxruntime as ort
    import torch

    selected_device = install_torch_compatibility(
        torch
    )
    selected_providers = install_onnxruntime_compatibility(
        ort
    )

    print(
        f"[GHOST2] device={selected_device}",
        flush=True,
    )
    print(
        f"[GHOST2] providers={selected_providers}",
        flush=True,
    )
    print(
        "[GHOST2] safe_cpu_mode=true",
        flush=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="face-swap-ghost2-"
    ) as temporary_name:
        temporary_directory = Path(
            temporary_name
        )
        temporary_result = (
            temporary_directory
            / "ghost2-result.png"
        )

        script_arguments = [
            str(
                inference_script
            ),
            "--config_a",
            str(
                aligner_config
            ),
            "--config_b",
            str(
                blender_config
            ),
            "--source",
            str(
                source
            ),
            "--target",
            str(
                target
            ),
            "--ckpt_a",
            str(
                aligner_checkpoint
            ),
            "--ckpt_b",
            str(
                blender_checkpoint
            ),
            "--save_path",
            str(
                temporary_result
            ),
        ]

        if arguments.use_kandinsky:
            script_arguments.append(
                "--use_kandi"
            )

        previous_arguments = sys.argv[:]
        previous_directory = Path.cwd()

        try:
            os.chdir(
                GHOST2_ROOT
            )
            sys.argv = script_arguments

            runpy.run_path(
                str(
                    inference_script
                ),
                run_name="__main__",
            )
        finally:
            sys.argv = previous_arguments
            os.chdir(
                previous_directory
            )

        generated_result = find_generated_result(
            temporary_directory,
            temporary_result,
        )

        if generated_result is None:
            created_files = [
                str(
                    path.relative_to(
                        temporary_directory
                    )
                )
                for path in temporary_directory.rglob(
                    "*"
                )
                if path.is_file()
            ]

            raise RuntimeError(
                "GHOST 2.0 завершился без создания изображения. "
                f"Созданные файлы: {created_files}"
            )

        shutil.copy2(
            generated_result,
            output,
        )

    if (
        not output.is_file()
        or output.stat().st_size == 0
    ):
        output.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"GHOST 2.0 не создал корректный результат: {output}"
        )

    print(
        f"[GHOST2] result={output}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )