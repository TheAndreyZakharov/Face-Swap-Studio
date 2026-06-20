from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import insightface
import onnxruntime as ort
from insightface.app import FaceAnalysis

from src.face_swap_studio.utils.logging import get_logger
from src.face_swap_studio.utils.paths import (
    enhancer_directory,
    swapper_directory,
    upscaler_directory,
)

logger = get_logger(__name__)


def available_onnx_providers() -> list[str]:
    return ort.get_available_providers()


def preferred_onnx_providers() -> list[str]:
    available = set(available_onnx_providers())
    providers: list[str] = []

    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")

    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    if not providers:
        raise RuntimeError("ONNX Runtime не обнаружил подходящих execution providers.")

    return providers


@lru_cache(maxsize=1)
def get_face_analyser() -> FaceAnalysis:
    providers = preferred_onnx_providers()
    logger.info("Загрузка FaceAnalysis с providers: %s", providers)

    analyser = FaceAnalysis(
        name="buffalo_l",
        root=str(Path.home() / ".insightface"),
        providers=providers,
    )
    analyser.prepare(ctx_id=0, det_size=(640, 640))

    return analyser


@lru_cache(maxsize=4)
def get_face_swapper(model_name: str = "inswapper_128.onnx") -> Any:
    model_path = swapper_directory() / model_name

    if not model_path.is_file():
        raise FileNotFoundError(f"Не найдена модель замены лица: {model_path}")

    providers = preferred_onnx_providers()
    logger.info("Загрузка swap-модели %s с providers: %s", model_name, providers)

    return insightface.model_zoo.get_model(
        str(model_path),
        providers=providers,
    )


def gfpgan_model_path() -> Path:
    path = enhancer_directory() / "GFPGANv1.4.pth"

    if not path.is_file():
        raise FileNotFoundError(f"Не найдена модель GFPGAN: {path}")

    return path


def realesrgan_model_path() -> Path:
    path = upscaler_directory() / "RealESRGAN_x4plus.pth"

    if not path.is_file():
        raise FileNotFoundError(f"Не найдена модель Real-ESRGAN: {path}")

    return path


def model_status() -> dict[str, bool]:
    buffalo = Path.home() / ".insightface" / "models" / "buffalo_l"

    return {
        "buffalo_l_detector": (buffalo / "det_10g.onnx").is_file(),
        "buffalo_l_recognition": (buffalo / "w600k_r50.onnx").is_file(),
        "inswapper_128": (swapper_directory() / "inswapper_128.onnx").is_file(),
        "gfpgan_v1_4": gfpgan_model_path().is_file(),
        "realesrgan_x4plus": realesrgan_model_path().is_file(),
    }