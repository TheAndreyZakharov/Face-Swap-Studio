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

    return settings


def project_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def ensure_directories() -> None:
    settings = load_settings()

    for configured_path in settings.get("paths", {}).values():
        project_path(configured_path).mkdir(parents=True, exist_ok=True)


def input_directory() -> Path:
    return project_path(load_settings()["paths"]["input"])


def output_directory() -> Path:
    return project_path(load_settings()["paths"]["output"])


def temp_directory() -> Path:
    return project_path(load_settings()["paths"]["temp"])


def detector_directory() -> Path:
    return project_path(load_settings()["paths"]["detectors"])


def swapper_directory() -> Path:
    return project_path(load_settings()["paths"]["swappers"])


def enhancer_directory() -> Path:
    return project_path(load_settings()["paths"]["enhancers"])


def upscaler_directory() -> Path:
    return project_path(load_settings()["paths"]["upscalers"])
