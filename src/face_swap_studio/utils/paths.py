from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Не найден конфигурационный файл: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file) or {}

    if not isinstance(settings, dict):
        raise ValueError("Корневой элемент settings.yaml должен быть объектом.")

    return settings


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()

    if resolved.is_absolute():
        return resolved

    return PROJECT_ROOT / resolved


def configured_path(key: str) -> Path:
    paths = load_settings().get("paths", {})

    if key not in paths:
        raise KeyError(f"В settings.yaml отсутствует paths.{key}")

    return project_path(paths[key])


def input_directory() -> Path:
    return configured_path("input")


def output_directory() -> Path:
    return configured_path("output")


def temp_directory() -> Path:
    return configured_path("temp")


def detector_directory() -> Path:
    return configured_path("detectors")


def swapper_directory() -> Path:
    return configured_path("swappers")


def enhancer_directory() -> Path:
    return configured_path("enhancers")


def upscaler_directory() -> Path:
    return configured_path("upscalers")


def vendor_directory() -> Path:
    return PROJECT_ROOT / "vendor"


def environments_directory() -> Path:
    return PROJECT_ROOT / ".environments"


def ensure_directories() -> None:
    for directory in (
        input_directory(),
        output_directory(),
        temp_directory(),
        detector_directory(),
        swapper_directory(),
        enhancer_directory(),
        upscaler_directory(),
    ):
        directory.mkdir(parents=True, exist_ok=True)
