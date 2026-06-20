from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import insightface
import onnxruntime as ort
from insightface.app import FaceAnalysis

from src.face_swap_studio.models.manifest import (
    is_model_ready,
    model_definitions,
)
from src.face_swap_studio.utils.logging import get_logger
from src.face_swap_studio.utils.paths import (
    enhancer_directory,
    load_settings,
    swapper_directory,
    upscaler_directory,
)

logger = get_logger(__name__)


def available_onnx_providers() -> list[str]:
    return ort.get_available_providers()


def preferred_onnx_providers() -> list[str]:
    runtime = load_settings().get("runtime", {})
    configured = runtime.get(
        "preferred_onnx_providers",
        [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    available = set(available_onnx_providers())
    providers = [
        provider
        for provider in configured
        if provider in available
    ]

    if "CPUExecutionProvider" in available:
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")

    if not providers:
        raise RuntimeError(
            "ONNX Runtime не обнаружил подходящих execution providers."
        )

    return providers


@lru_cache(maxsize=1)
def get_face_analyser() -> FaceAnalysis:
    settings = load_settings()
    detection = settings.get("detection", {})

    model_name = str(detection.get("model", "buffalo_l"))
    input_size = int(detection.get("input_size", 640))

    providers = preferred_onnx_providers()

    logger.info(
        "Загрузка FaceAnalysis %s с providers=%s",
        model_name,
        providers,
    )

    analyser = FaceAnalysis(
        name=model_name,
        root=str(Path.home() / ".insightface"),
        providers=providers,
    )
    analyser.prepare(
        ctx_id=0,
        det_size=(input_size, input_size),
    )

    return analyser


@lru_cache(maxsize=4)
def get_face_swapper(
    model_name: str = "inswapper_128.onnx",
) -> Any:
    model_path = swapper_directory() / model_name

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Не найдена модель замены лица: {model_path}"
        )

    providers = preferred_onnx_providers()

    logger.info(
        "Загрузка swap-модели %s с providers=%s",
        model_path.name,
        providers,
    )

    return insightface.model_zoo.get_model(
        str(model_path),
        providers=providers,
    )


def gfpgan_model_path() -> Path:
    path = enhancer_directory() / "GFPGANv1.4.pth"

    if not path.is_file():
        raise FileNotFoundError(
            f"Не найдена модель GFPGAN: {path}"
        )

    return path


def realesrgan_model_path() -> Path:
    path = upscaler_directory() / "RealESRGAN_x4plus.pth"

    if not path.is_file():
        raise FileNotFoundError(
            f"Не найдена модель Real-ESRGAN: {path}"
        )

    return path


def model_status() -> dict[str, bool]:
    return {
        definition.id: is_model_ready(definition)
        for definition in model_definitions()
    }