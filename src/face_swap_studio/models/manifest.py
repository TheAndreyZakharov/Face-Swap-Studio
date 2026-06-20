from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from src.face_swap_studio.utils.paths import PROJECT_ROOT


class ModelKind(StrEnum):
    FACE_SWAP = "face_swap"
    HEAD_SWAP = "head_swap"


class BackendKind(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    key: str
    label: str
    description: str
    kind: ModelKind
    backend: BackendKind
    environment: Path | None
    entrypoint: Path | None
    required_paths: tuple[Path, ...] = field(default_factory=tuple)

    def missing_paths(self) -> tuple[Path, ...]:
        return tuple(path for path in self.required_paths if not path.exists())

    @property
    def installed(self) -> bool:
        return not self.missing_paths()


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


SIMSWAP_ROOT = project_path("vendor", "simswap")
GHOST_ROOT = project_path("vendor", "ghost")
GHOST2_ROOT = project_path("vendor", "ghost2")


MODELS: dict[str, ModelDefinition] = {
    "inswapper_128": ModelDefinition(
        key="inswapper_128",
        label="InSwapper 128",
        description="Быстрая ONNX-модель замены лица.",
        kind=ModelKind.FACE_SWAP,
        backend=BackendKind.INTERNAL,
        environment=None,
        entrypoint=None,
        required_paths=(
            project_path(
                "models",
                "swappers",
                "inswapper_128.onnx",
            ),
        ),
    ),
    "simswap_512": ModelDefinition(
        key="simswap_512",
        label="SimSwap 512 Beta",
        description=("Высокодетализированная замена лица с маской BiSeNet."),
        kind=ModelKind.FACE_SWAP,
        backend=BackendKind.EXTERNAL,
        environment=project_path(
            ".environments",
            "simswap",
        ),
        entrypoint=project_path(
            "scripts",
            "external",
            "simswap",
            "infer.py",
        ),
        required_paths=(
            project_path(
                ".environments",
                "simswap",
                "bin",
                "python",
            ),
            SIMSWAP_ROOT / "checkpoints" / "512" / "550000_net_G.pth",
            SIMSWAP_ROOT / "arcface_model" / "arcface_checkpoint.tar",
            SIMSWAP_ROOT / "parsing_model" / "checkpoint" / "79999_iter.pth",
            SIMSWAP_ROOT / "insightface_func" / "models" / "antelope" / "scrfd_10g_bnkps.onnx",
            SIMSWAP_ROOT / "insightface_func" / "models" / "antelope" / "glintr100.onnx",
        ),
    ),
    "ghost_unet_1block": ModelDefinition(
        key="ghost_unet_1block",
        label="GHOST U-Net 1 Block",
        description=("Более лёгкая конфигурация GHOST."),
        kind=ModelKind.FACE_SWAP,
        backend=BackendKind.EXTERNAL,
        environment=project_path(
            ".environments",
            "ghost",
        ),
        entrypoint=project_path(
            "scripts",
            "external",
            "ghost",
            "infer.py",
        ),
        required_paths=(
            project_path(
                ".environments",
                "ghost",
                "bin",
                "python",
            ),
            GHOST_ROOT / "weights" / "G_unet_1block.pth",
            GHOST_ROOT / "arcface_model" / "backbone.pth",
            GHOST_ROOT / "coordinate_reg" / "model" / "2d106det",
            GHOST_ROOT / "insightface_func" / "models" / "antelope" / "scrfd_10g_bnkps.onnx",
            GHOST_ROOT / "insightface_func" / "models" / "antelope" / "glintr100.onnx",
        ),
    ),
    "ghost_unet_2blocks": ModelDefinition(
        key="ghost_unet_2blocks",
        label="GHOST U-Net 2 Blocks",
        description=("Основная рекомендованная конфигурация GHOST."),
        kind=ModelKind.FACE_SWAP,
        backend=BackendKind.EXTERNAL,
        environment=project_path(
            ".environments",
            "ghost",
        ),
        entrypoint=project_path(
            "scripts",
            "external",
            "ghost",
            "infer.py",
        ),
        required_paths=(
            project_path(
                ".environments",
                "ghost",
                "bin",
                "python",
            ),
            GHOST_ROOT / "weights" / "G_unet_2blocks.pth",
            GHOST_ROOT / "arcface_model" / "backbone.pth",
            GHOST_ROOT / "coordinate_reg" / "model" / "2d106det",
            GHOST_ROOT / "insightface_func" / "models" / "antelope" / "scrfd_10g_bnkps.onnx",
            GHOST_ROOT / "insightface_func" / "models" / "antelope" / "glintr100.onnx",
        ),
    ),
    "ghost_unet_3blocks": ModelDefinition(
        key="ghost_unet_3blocks",
        label="GHOST U-Net 3 Blocks",
        description=("Самая тяжёлая доступная конфигурация GHOST."),
        kind=ModelKind.FACE_SWAP,
        backend=BackendKind.EXTERNAL,
        environment=project_path(
            ".environments",
            "ghost",
        ),
        entrypoint=project_path(
            "scripts",
            "external",
            "ghost",
            "infer.py",
        ),
        required_paths=(
            project_path(
                ".environments",
                "ghost",
                "bin",
                "python",
            ),
            GHOST_ROOT / "weights" / "G_unet_3blocks.pth",
            GHOST_ROOT / "arcface_model" / "backbone.pth",
            GHOST_ROOT / "coordinate_reg" / "model" / "2d106det",
            GHOST_ROOT / "insightface_func" / "models" / "antelope" / "scrfd_10g_bnkps.onnx",
            GHOST_ROOT / "insightface_func" / "models" / "antelope" / "glintr100.onnx",
        ),
    ),
    "ghost2_head": ModelDefinition(
        key="ghost2_head",
        label="GHOST 2.0 Head Swap",
        description=("Полная замена головы через Aligner и Blender."),
        kind=ModelKind.HEAD_SWAP,
        backend=BackendKind.EXTERNAL,
        environment=project_path(
            ".environments",
            "ghost2",
        ),
        entrypoint=project_path(
            "scripts",
            "external",
            "ghost2",
            "infer.py",
        ),
        required_paths=(
            project_path(
                ".environments",
                "ghost2",
                "bin",
                "python",
            ),
            GHOST2_ROOT / "aligner_checkpoints" / "aligner_1020_gaze_final.ckpt",
            GHOST2_ROOT / "blender_checkpoints" / "blender_lama.ckpt",
            GHOST2_ROOT / "weights" / "backbone50_1.pth",
            GHOST2_ROOT / "weights" / "vgg19-d01eb7cb.pth",
            GHOST2_ROOT / "weights" / "segformer_B5_ce.onnx",
            GHOST2_ROOT / "src" / "losses" / "gaze_models" / "vgg_16_2_forward_sum.pt",
            GHOST2_ROOT / "src" / "losses" / "gaze_models" / "resnet_18_2_forward_sum.pt",
            GHOST2_ROOT
            / "repos"
            / "stylematte"
            / "stylematte"
            / "checkpoints"
            / "stylematte_synth.pth",
            GHOST2_ROOT / "repos" / "deca" / "data" / "generic_model.pkl",
            GHOST2_ROOT / "repos" / "BlazeFace_PyTorch" / "blazeface.pth",
        ),
    ),
}


def available_models() -> list[ModelDefinition]:
    return list(MODELS.values())


def installed_models() -> list[ModelDefinition]:
    return [model for model in MODELS.values() if model.installed]


def get_model(model_key: str) -> ModelDefinition:
    try:
        return MODELS[model_key]
    except KeyError as error:
        raise ValueError(f"Неизвестная модель: {model_key}") from error
