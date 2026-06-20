from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from pathlib import Path

import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PACKAGES = {
    "numpy": "numpy",
    "Pillow": "PIL",
    "OpenCV": "cv2",
    "ONNX": "onnx",
    "ONNX Runtime": "onnxruntime",
    "Gradio": "gradio",
    "PyYAML": "yaml",
    "PyTorch": "torch",
    "TorchVision": "torchvision",
    "SciPy": "scipy",
    "scikit-image": "skimage",
}

DIRECTORIES = [
    "data/input",
    "data/output",
    "data/temp",
    "models/detectors",
    "models/swappers",
    "models/enhancers",
    "models/upscalers",
]

SYSTEM_TOOLS = (
    "git",
    "ffmpeg",
    "cmake",
)


def format_status(condition: bool) -> str:
    return "OK" if condition else "MISSING"


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Architecture: {platform.machine()}")
    print(f"Platform: {platform.platform()}")
    print(f"Project root: {PROJECT_ROOT}")

    failed = False

    print("\nPython packages:")
    for display_name, module_name in PACKAGES.items():
        installed = importlib.util.find_spec(module_name) is not None
        status = format_status(installed)
        print(f"  [{status}] {display_name}")
        failed = failed or not installed

    print("\nSystem tools:")
    for command in SYSTEM_TOOLS:
        executable_path = shutil.which(command)
        installed = executable_path is not None
        status = format_status(installed)
        displayed_path = executable_path or "not found"
        print(f"  [{status}] {command}: {displayed_path}")
        failed = failed or not installed

    print("\nProject directories:")
    for relative_directory in DIRECTORIES:
        directory_path = PROJECT_ROOT / relative_directory
        exists = directory_path.is_dir()
        status = format_status(exists)
        print(f"  [{status}] {relative_directory}")
        failed = failed or not exists

    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()

    print("\nPyTorch:")
    print(f"  Version: {torch.__version__}")
    print(f"  MPS built: {mps_built}")
    print(f"  MPS available: {mps_available}")

    failed = failed or not mps_built
    failed = failed or not mps_available

    providers = ort.get_available_providers()
    cpu_provider_available = "CPUExecutionProvider" in providers
    coreml_provider_available = "CoreMLExecutionProvider" in providers

    print("\nONNX Runtime:")
    print(f"  Version: {ort.__version__}")
    print(f"  Providers: {providers}")
    print(f"  CoreML available: {coreml_provider_available}")
    print(f"  CPU fallback available: {cpu_provider_available}")

    failed = failed or not cpu_provider_available

    if failed:
        print("\nEnvironment has missing components.")
        return 1

    print("\nEnvironment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())