from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )

from src.face_swap_studio.adapters.modern_onnx import (  # noqa: E402
    MODEL_CONFIGURATIONS,
)
from src.face_swap_studio.models.manifest import (  # noqa: E402
    is_model_ready,
    model_by_id,
)
from src.face_swap_studio.models.model_manager import (  # noqa: E402
    get_modern_swapper_session,
)
from src.face_swap_studio.utils.paths import (  # noqa: E402
    swapper_directory,
)


def inspect_model(
    model_id: str,
) -> bool:
    configuration = MODEL_CONFIGURATIONS[
        model_id
    ]

    definition = model_by_id(
        model_id
    )

    model_path = (
        swapper_directory()
        / configuration.model_filename
    ).expanduser().resolve()

    print()
    print(
        "="
        * 72
    )
    print(
        f"{model_id}"
    )
    print(
        "="
        * 72
    )
    print(
        f"Path: {model_path}"
    )
    print(
        f"Ready: {is_model_ready(definition)}"
    )

    if not model_path.is_file():
        print(
            "Result: MISSING"
        )
        return False

    print(
        "Size: "
        f"{model_path.stat().st_size / 1024 / 1024:.1f} MB"
    )

    try:
        session = get_modern_swapper_session(
            configuration.model_filename,
            force_cpu=(
                configuration.model_type
                == "uniface"
            ),
        )
    except Exception as error:
        print(
            f"Result: FAILED TO LOAD: {error}"
        )
        return False

    print(
        "Providers: "
        f"{session.get_providers()}"
    )

    print(
        "Inputs:"
    )

    for input_node in session.get_inputs():
        print(
            "  - "
            f"name={input_node.name}, "
            f"shape={input_node.shape}, "
            f"type={input_node.type}"
        )

    print(
        "Outputs:"
    )

    for output_node in session.get_outputs():
        print(
            "  - "
            f"name={output_node.name}, "
            f"shape={output_node.shape}, "
            f"type={output_node.type}"
        )

    print(
        "Result: OK"
    )

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Face Swap Studio modern ONNX swappers."
        )
    )

    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help=(
            "Models to inspect. When omitted, "
            "all modern ONNX swappers are inspected."
        ),
    )

    arguments = parser.parse_args()

    unknown_models = [
        model_id
        for model_id in arguments.models
        if model_id not in MODEL_CONFIGURATIONS
    ]

    if unknown_models:
        parser.error(
            "unknown model(s): "
            + ", ".join(
                unknown_models
            )
            + ". Available models: "
            + ", ".join(
                MODEL_CONFIGURATIONS
            )
        )

    selected_models = (
        arguments.models
        or list(
            MODEL_CONFIGURATIONS
        )
    )

    success = True

    for model_id in selected_models:
        if not inspect_model(
            model_id
        ):
            success = False

    return (
        0
        if success
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )