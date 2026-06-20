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

logger = get_logger(
    __name__
)


def available_onnx_providers() -> list[str]:
    return list(
        ort.get_available_providers()
    )


def preferred_onnx_providers() -> list[str]:
    runtime = load_settings().get(
        "runtime",
        {},
    )

    configured = runtime.get(
        "preferred_onnx_providers",
        [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    available = set(
        available_onnx_providers()
    )

    providers = [
        str(
            provider
        )
        for provider in configured
        if provider in available
    ]

    if (
        "CPUExecutionProvider"
        in available
        and "CPUExecutionProvider"
        not in providers
    ):
        providers.append(
            "CPUExecutionProvider"
        )

    if not providers:
        raise RuntimeError(
            "ONNX Runtime did not expose a usable "
            "execution provider."
        )

    return providers


def cpu_onnx_providers() -> list[str]:
    available = set(
        available_onnx_providers()
    )

    if "CPUExecutionProvider" not in available:
        raise RuntimeError(
            "CPUExecutionProvider is unavailable."
        )

    return [
        "CPUExecutionProvider",
    ]


@lru_cache(maxsize=1)
def get_face_analyser() -> FaceAnalysis:
    settings = load_settings()

    detection = settings.get(
        "detection",
        {},
    )

    model_name = str(
        detection.get(
            "model",
            "buffalo_l",
        )
    )

    input_size = int(
        detection.get(
            "input_size",
            640,
        )
    )

    providers = preferred_onnx_providers()

    logger.info(
        "Загрузка FaceAnalysis %s с providers=%s",
        model_name,
        providers,
    )

    analyser = FaceAnalysis(
        name=model_name,
        root=str(
            Path.home()
            / ".insightface"
        ),
        providers=providers,
    )

    analyser.prepare(
        ctx_id=0,
        det_size=(
            input_size,
            input_size,
        ),
    )

    return analyser


@lru_cache(maxsize=4)
def get_face_swapper(
    model_name: str = "inswapper_128.onnx",
) -> Any:
    model_path = (
        swapper_directory()
        / model_name
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            "Не найдена модель замены лица: "
            f"{model_path}"
        )

    providers = preferred_onnx_providers()

    logger.info(
        "Загрузка swap-модели %s с providers=%s",
        model_path.name,
        providers,
    )

    return insightface.model_zoo.get_model(
        str(
            model_path
        ),
        providers=providers,
    )


@lru_cache(maxsize=8)
def get_modern_swapper_session(
    model_name: str,
    force_cpu: bool = False,
) -> ort.InferenceSession:
    model_path = (
        swapper_directory()
        / model_name
    ).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            "Modern ONNX swapper not found: "
            f"{model_path}"
        )

    if model_path.stat().st_size == 0:
        raise RuntimeError(
            "Modern ONNX swapper is empty: "
            f"{model_path}"
        )

    providers = (
        cpu_onnx_providers()
        if force_cpu
        else preferred_onnx_providers()
    )

    logger.info(
        "Загрузка ONNX swap-модели %s с providers=%s",
        model_path.name,
        providers,
    )

    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session_options.log_severity_level = 3

    try:
        session_options.intra_op_num_threads = 2
        session_options.inter_op_num_threads = 1
    except AttributeError:
        pass

    session = ort.InferenceSession(
        str(
            model_path
        ),
        sess_options=session_options,
        providers=providers,
    )

    input_description = [
        (
            input_node.name,
            input_node.shape,
            input_node.type,
        )
        for input_node in session.get_inputs()
    ]

    output_description = [
        (
            output_node.name,
            output_node.shape,
            output_node.type,
        )
        for output_node in session.get_outputs()
    ]

    logger.info(
        "ONNX-модель %s inputs=%s outputs=%s",
        model_path.name,
        input_description,
        output_description,
    )

    return session


def clear_model_caches() -> None:
    get_face_analyser.cache_clear()
    get_face_swapper.cache_clear()
    get_modern_swapper_session.cache_clear()


def gfpgan_model_path() -> Path:
    path = (
        enhancer_directory()
        / "GFPGANv1.4.pth"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Не найдена модель GFPGAN: "
            f"{path}"
        )

    return path


def realesrgan_model_path() -> Path:
    path = (
        upscaler_directory()
        / "RealESRGAN_x4plus.pth"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Не найдена модель Real-ESRGAN: "
            f"{path}"
        )

    return path


def model_status() -> dict[str, bool]:
    return {
        definition.id: is_model_ready(
            definition
        )
        for definition in model_definitions()
    }


__all__ = [
    "available_onnx_providers",
    "clear_model_caches",
    "cpu_onnx_providers",
    "get_face_analyser",
    "get_face_swapper",
    "get_modern_swapper_session",
    "gfpgan_model_path",
    "model_status",
    "preferred_onnx_providers",
    "realesrgan_model_path",
]