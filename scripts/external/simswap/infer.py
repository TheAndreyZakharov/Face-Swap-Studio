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
SIMSWAP_ROOT = PROJECT_ROOT / "vendor" / "simswap"

sys.path.insert(
    0,
    str(EXTERNAL_SCRIPTS_ROOT),
)

from runtime_compat import (  # noqa: E402
    configure_apple_silicon_environment,
    install_numpy_compatibility,
    install_onnxruntime_compatibility,
    install_torch_compatibility,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face Swap Studio SimSwap 512 wrapper.")

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
        "--use-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def require_file(
    path: Path,
    label: str,
) -> Path:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"{label} не найден: {resolved}")

    return resolved


def find_result(
    output_directory: Path,
) -> Path | None:
    preferred_names = (
        "result_whole_swapsingle.jpg",
        "result_whole_swapsingle.png",
        "result_whole_swapmulti.jpg",
        "result_whole_swapmulti.png",
    )

    for name in preferred_names:
        candidate = output_directory / name

        if candidate.is_file():
            return candidate

    candidates = sorted(
        (
            *output_directory.glob("result_whole_swap*.jpg"),
            *output_directory.glob("result_whole_swap*.jpeg"),
            *output_directory.glob("result_whole_swap*.png"),
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return candidates[0] if candidates else None


def main() -> int:
    arguments = parse_arguments()

    source = require_file(
        arguments.source,
        "Source-файл",
    )
    target = require_file(
        arguments.target,
        "Target-файл",
    )

    output = arguments.output.expanduser().resolve()

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SIMSWAP_ROOT.is_dir():
        raise FileNotFoundError(f"Репозиторий SimSwap не найден: {SIMSWAP_ROOT}")

    generator_checkpoint = require_file(
        SIMSWAP_ROOT / "checkpoints" / "512" / "550000_net_G.pth",
        "SimSwap generator checkpoint",
    )

    arcface_checkpoint = require_file(
        SIMSWAP_ROOT / "arcface_model" / "arcface_checkpoint.tar",
        "SimSwap ArcFace checkpoint",
    )

    require_file(
        SIMSWAP_ROOT / "parsing_model" / "checkpoint" / "79999_iter.pth",
        "SimSwap parsing checkpoint",
    )

    configure_apple_silicon_environment()

    sys.path.insert(
        0,
        str(SIMSWAP_ROOT),
    )

    import numpy as np
    import onnxruntime as ort
    import torch

    install_numpy_compatibility(np)

    selected_device = install_torch_compatibility(torch)

    selected_providers = install_onnxruntime_compatibility(ort)

    print(
        f"[SimSwap] device={selected_device}",
        flush=True,
    )
    print(
        f"[SimSwap] providers={selected_providers}",
        flush=True,
    )
    print(
        f"[SimSwap] checkpoint={generator_checkpoint}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="face-swap-simswap-") as temporary_name:
        temporary_output = Path(temporary_name)

        script_path = SIMSWAP_ROOT / "test_wholeimage_swapsingle.py"

        script_arguments = [
            str(script_path),
            "--crop_size",
            "512",
            "--name",
            "512",
            "--which_epoch",
            "550000",
            "--checkpoints_dir",
            str(SIMSWAP_ROOT / "checkpoints"),
            "--Arc_path",
            str(arcface_checkpoint),
            "--pic_a_path",
            str(source),
            "--pic_b_path",
            str(target),
            "--output_path",
            str(temporary_output),
            "--no_simswaplogo",
            "--gpu_ids",
            "-1",
        ]

        if arguments.use_mask:
            script_arguments.append("--use_mask")

        previous_argv = sys.argv[:]
        previous_directory = Path.cwd()

        try:
            os.chdir(SIMSWAP_ROOT)

            sys.argv = script_arguments

            runpy.run_path(
                str(script_path),
                run_name="__main__",
            )
        finally:
            sys.argv = previous_argv
            os.chdir(previous_directory)

        generated_result = find_result(temporary_output)

        if generated_result is None:
            created_files = [
                str(path.relative_to(temporary_output))
                for path in temporary_output.rglob("*")
                if path.is_file()
            ]

            raise RuntimeError(
                f"SimSwap завершился без создания результата. Созданные файлы: {created_files}"
            )

        shutil.copy2(
            generated_result,
            output,
        )

    if not output.is_file():
        raise RuntimeError(f"Не удалось сохранить результат SimSwap: {output}")

    if output.stat().st_size == 0:
        output.unlink(missing_ok=True)

        raise RuntimeError("SimSwap создал пустой файл результата.")

    print(
        f"[SimSwap] result={output}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
