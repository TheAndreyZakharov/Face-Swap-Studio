from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.face_swap_studio.utils.paths import (
    PROJECT_ROOT,
    enhancer_directory,
    environments_directory,
    swapper_directory,
    upscaler_directory,
    vendor_directory,
)


class ModelKind(StrEnum):
    FACE_SWAP = "face_swap"
    HEAD_SWAP = "head_swap"
    FACE_ENHANCEMENT = "face_enhancement"
    IMAGE_UPSCALING = "image_upscaling"


class BackendKind(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    id: str
    name: str
    description: str
    kind: ModelKind
    backend: BackendKind
    required_paths: tuple[Path, ...]
    environment_python: Path | None = None
    runner_path: Path | None = None
    supports_multiple_faces: bool = False
    supports_face_assignments: bool = False


def model_definitions() -> tuple[ModelDefinition, ...]:
    vendor = vendor_directory()
    environments = environments_directory()
    scripts = PROJECT_ROOT / "scripts" / "external"

    return (
        ModelDefinition(
            id="inswapper_128",
            name="InSwapper 128",
            description=(
                "Быстрая локальная ONNX-модель. Поддерживает выбор "
                "конкретного target-лица."
            ),
            kind=ModelKind.FACE_SWAP,
            backend=BackendKind.INTERNAL,
            required_paths=(
                swapper_directory() / "inswapper_128.onnx",
            ),
            supports_multiple_faces=True,
            supports_face_assignments=True,
        ),
        ModelDefinition(
            id="hyperswap_1a_256",
            name="HyperSwap 1A 256",
            description=(
                "ONNX face-swap model with a 256×256 working crop. "
                "Focused on identity preservation and facial detail."
            ),
            kind=ModelKind.FACE_SWAP,
            backend=BackendKind.INTERNAL,
            required_paths=(
                swapper_directory()
                / "hyperswap_1a_256.onnx",
            ),
            supports_multiple_faces=True,
            supports_face_assignments=True,
        ),
        ModelDefinition(
            id="hyperswap_1b_256",
            name="HyperSwap 1B 256",
            description=(
                "Alternative HyperSwap 256 variant for comparing "
                "identity similarity, shape and texture."
            ),
            kind=ModelKind.FACE_SWAP,
            backend=BackendKind.INTERNAL,
            required_paths=(
                swapper_directory()
                / "hyperswap_1b_256.onnx",
            ),
            supports_multiple_faces=True,
            supports_face_assignments=True,
        ),
        ModelDefinition(
            id="uniface_256",
            name="UniFace 256",
            description=(
                "ONNX face-swap model using aligned source and target "
                "face images at 256×256 resolution."
            ),
            kind=ModelKind.FACE_SWAP,
            backend=BackendKind.INTERNAL,
            required_paths=(
                swapper_directory()
                / "uniface_256.onnx",
            ),
            supports_multiple_faces=True,
            supports_face_assignments=True,
        ),
        ModelDefinition(
            id="simswap_512",
            name="SimSwap 512 Beta",
            description=(
                "Высокодетализированная модель с рабочим разрешением "
                "лица 512×512."
            ),
            kind=ModelKind.FACE_SWAP,
            backend=BackendKind.EXTERNAL,
            required_paths=(
                vendor
                / "simswap"
                / "checkpoints"
                / "512"
                / "550000_net_G.pth",
                vendor
                / "simswap"
                / "arcface_model"
                / "arcface_checkpoint.tar",
                vendor
                / "simswap"
                / "parsing_model"
                / "checkpoint"
                / "79999_iter.pth",
                vendor
                / "simswap"
                / "insightface_func"
                / "models"
                / "antelope"
                / "scrfd_10g_bnkps.onnx",
                vendor
                / "simswap"
                / "insightface_func"
                / "models"
                / "antelope"
                / "glintr100.onnx",
            ),
            environment_python=(
                environments / "simswap" / "bin" / "python"
            ),
            runner_path=scripts / "simswap" / "infer.py",
            supports_multiple_faces=True,
            supports_face_assignments=False,
        ),
        ModelDefinition(
            id="ghost_unet_1block",
            name="GHOST U-Net 1 Block",
            description=(
                "Самый лёгкий вариант GHOST. Требует меньше памяти."
            ),
            kind=ModelKind.FACE_SWAP,
            backend=BackendKind.EXTERNAL,
            required_paths=(
                vendor / "ghost" / "weights" / "G_unet_1block.pth",
                vendor / "ghost" / "arcface_model" / "backbone.pth",
                vendor
                / "ghost"
                / "insightface_func"
                / "models"
                / "antelope"
                / "scrfd_10g_bnkps.onnx",
                vendor
                / "ghost"
                / "insightface_func"
                / "models"
                / "antelope"
                / "glintr100.onnx",
            ),
            environment_python=(
                environments / "ghost" / "bin" / "python"
            ),
            runner_path=scripts / "ghost" / "infer.py",
            supports_multiple_faces=True,
            supports_face_assignments=False,
        ),
        ModelDefinition(
            id="ghost_unet_2blocks",
            name="GHOST U-Net 2 Blocks",
            description=(
                "Сбалансированный вариант GHOST по качеству и памяти."
            ),
            kind=ModelKind.FACE_SWAP,
            backend=BackendKind.EXTERNAL,
            required_paths=(
                vendor / "ghost" / "weights" / "G_unet_2blocks.pth",
                vendor / "ghost" / "arcface_model" / "backbone.pth",
                vendor
                / "ghost"
                / "insightface_func"
                / "models"
                / "antelope"
                / "scrfd_10g_bnkps.onnx",
                vendor
                / "ghost"
                / "insightface_func"
                / "models"
                / "antelope"
                / "glintr100.onnx",
            ),
            environment_python=(
                environments / "ghost" / "bin" / "python"
            ),
            runner_path=scripts / "ghost" / "infer.py",
            supports_multiple_faces=True,
            supports_face_assignments=False,
        ),
        ModelDefinition(
            id="ghost_unet_3blocks",
            name="GHOST U-Net 3 Blocks",
            description=(
                "Самый тяжёлый из установленных вариантов GHOST."
            ),
            kind=ModelKind.FACE_SWAP,
            backend=BackendKind.EXTERNAL,
            required_paths=(
                vendor / "ghost" / "weights" / "G_unet_3blocks.pth",
                vendor / "ghost" / "arcface_model" / "backbone.pth",
                vendor
                / "ghost"
                / "insightface_func"
                / "models"
                / "antelope"
                / "scrfd_10g_bnkps.onnx",
                vendor
                / "ghost"
                / "insightface_func"
                / "models"
                / "antelope"
                / "glintr100.onnx",
            ),
            environment_python=(
                environments / "ghost" / "bin" / "python"
            ),
            runner_path=scripts / "ghost" / "infer.py",
            supports_multiple_faces=True,
            supports_face_assignments=False,
        ),
        ModelDefinition(
            id="ghost2_head",
            name="GHOST 2.0 Head Swap",
            description=(
                "Полная замена головы. Самая тяжёлая модель в проекте."
            ),
            kind=ModelKind.HEAD_SWAP,
            backend=BackendKind.EXTERNAL,
            required_paths=(
                vendor
                / "ghost2"
                / "aligner_checkpoints"
                / "aligner_1020_gaze_final.ckpt",
                vendor
                / "ghost2"
                / "blender_checkpoints"
                / "blender_lama.ckpt",
                vendor
                / "ghost2"
                / "weights"
                / "backbone50_1.pth",
                vendor
                / "ghost2"
                / "weights"
                / "vgg19-d01eb7cb.pth",
                vendor
                / "ghost2"
                / "weights"
                / "segformer_B5_ce.onnx",
                vendor
                / "ghost2"
                / "repos"
                / "stylematte"
                / "stylematte"
                / "checkpoints"
                / "stylematte_synth.pth",
                vendor
                / "ghost2"
                / "repos"
                / "deca"
                / "data"
                / "generic_model.pkl",
            ),
            environment_python=(
                environments / "ghost2" / "bin" / "python"
            ),
            runner_path=scripts / "ghost2" / "infer.py",
            supports_multiple_faces=False,
            supports_face_assignments=False,
        ),
        ModelDefinition(
            id="gfpgan_v1_4",
            name="GFPGAN v1.4",
            description="Восстановление и улучшение областей лица.",
            kind=ModelKind.FACE_ENHANCEMENT,
            backend=BackendKind.INTERNAL,
            required_paths=(
                enhancer_directory() / "GFPGANv1.4.pth",
            ),
        ),
        ModelDefinition(
            id="realesrgan_x4plus",
            name="Real-ESRGAN x4plus",
            description="Улучшение и увеличение всего изображения.",
            kind=ModelKind.IMAGE_UPSCALING,
            backend=BackendKind.INTERNAL,
            required_paths=(
                upscaler_directory() / "RealESRGAN_x4plus.pth",
            ),
        ),
    )


MODEL_MAP: dict[str, ModelDefinition] = {
    definition.id: definition
    for definition in model_definitions()
}


def model_by_id(
    model_id: str,
) -> ModelDefinition:
    try:
        return MODEL_MAP[model_id]
    except KeyError as error:
        raise ValueError(
            f"Неизвестная модель: {model_id}"
        ) from error


def selectable_swap_models() -> tuple[ModelDefinition, ...]:
    return tuple(
        definition
        for definition in model_definitions()
        if definition.kind
        in {
            ModelKind.FACE_SWAP,
            ModelKind.HEAD_SWAP,
        }
    )


def enhancement_models() -> tuple[ModelDefinition, ...]:
    return tuple(
        definition
        for definition in model_definitions()
        if definition.kind
        in {
            ModelKind.FACE_ENHANCEMENT,
            ModelKind.IMAGE_UPSCALING,
        }
    )


def _valid_file(
    path: Path,
) -> bool:
    resolved = path.expanduser().resolve()

    return (
        resolved.is_file()
        and resolved.stat().st_size > 0
    )


def _valid_directory(
    path: Path,
) -> bool:
    return path.expanduser().resolve().is_dir()


def is_model_ready(
    definition: ModelDefinition,
) -> bool:
    for required_path in definition.required_paths:
        resolved = required_path.expanduser().resolve()

        if resolved.is_dir():
            if not _valid_directory(resolved):
                return False
        elif not _valid_file(resolved):
            return False

    if definition.backend == BackendKind.EXTERNAL:
        if definition.environment_python is None:
            return False

        if not _valid_file(definition.environment_python):
            return False

        if definition.runner_path is None:
            return False

        if not _valid_file(definition.runner_path):
            return False

    return True


def missing_model_paths(
    definition: ModelDefinition,
) -> tuple[Path, ...]:
    missing: list[Path] = []

    for path in definition.required_paths:
        resolved = path.expanduser().resolve()

        if not resolved.exists():
            missing.append(resolved)
        elif resolved.is_file() and resolved.stat().st_size == 0:
            missing.append(resolved)

    if definition.backend == BackendKind.EXTERNAL:
        if definition.environment_python is None:
            pass
        elif not _valid_file(definition.environment_python):
            missing.append(
                definition.environment_python.expanduser().resolve()
            )

        if definition.runner_path is None:
            pass
        elif not _valid_file(definition.runner_path):
            missing.append(
                definition.runner_path.expanduser().resolve()
            )

    return tuple(missing)