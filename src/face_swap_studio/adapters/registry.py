from __future__ import annotations

from collections.abc import Callable

from src.face_swap_studio.adapters.base import (
    SwapAdapter,
)
from src.face_swap_studio.adapters.ghost import (
    GhostAdapter,
)
from src.face_swap_studio.adapters.ghost2 import (
    Ghost2Adapter,
)
from src.face_swap_studio.adapters.inswapper import (
    InSwapperAdapter,
)
from src.face_swap_studio.adapters.modern_onnx import (
    ModernOnnxSwapperAdapter,
)
from src.face_swap_studio.adapters.simswap import (
    SimSwapAdapter,
)

AdapterFactory = Callable[
    [],
    SwapAdapter,
]


def _create_inswapper() -> SwapAdapter:
    return InSwapperAdapter()


def _create_hyperswap_1a() -> SwapAdapter:
    return ModernOnnxSwapperAdapter(
        model_id="hyperswap_1a_256",
    )


def _create_hyperswap_1b() -> SwapAdapter:
    return ModernOnnxSwapperAdapter(
        model_id="hyperswap_1b_256",
    )


def _create_uniface() -> SwapAdapter:
    return ModernOnnxSwapperAdapter(
        model_id="uniface_256",
    )


def _create_simswap() -> SwapAdapter:
    return SimSwapAdapter()


def _create_ghost_1block() -> SwapAdapter:
    return GhostAdapter(
        model_id="ghost_unet_1block",
        blocks=1,
    )


def _create_ghost_2blocks() -> SwapAdapter:
    return GhostAdapter(
        model_id="ghost_unet_2blocks",
        blocks=2,
    )


def _create_ghost_3blocks() -> SwapAdapter:
    return GhostAdapter(
        model_id="ghost_unet_3blocks",
        blocks=3,
    )


def _create_ghost2() -> SwapAdapter:
    return Ghost2Adapter()


_ADAPTER_FACTORIES: dict[
    str,
    AdapterFactory,
] = {
    "inswapper_128": _create_inswapper,
    "hyperswap_1a_256": _create_hyperswap_1a,
    "hyperswap_1b_256": _create_hyperswap_1b,
    "uniface_256": _create_uniface,
    "simswap_512": _create_simswap,
    "ghost_unet_1block": _create_ghost_1block,
    "ghost_unet_2blocks": _create_ghost_2blocks,
    "ghost_unet_3blocks": _create_ghost_3blocks,
    "ghost2_head": _create_ghost2,
}


def supported_model_ids() -> tuple[str, ...]:
    return tuple(
        _ADAPTER_FACTORIES
    )


def get_adapter(
    model_id: str,
) -> SwapAdapter:
    normalized_model_id = str(
        model_id
    ).strip()

    try:
        factory = _ADAPTER_FACTORIES[
            normalized_model_id
        ]
    except KeyError as error:
        supported = ", ".join(
            supported_model_ids()
        )

        raise ValueError(
            f"Для модели {normalized_model_id!r} "
            "не зарегистрирован адаптер. "
            f"Поддерживаются: {supported}"
        ) from error

    adapter = factory()

    if not isinstance(
        adapter,
        SwapAdapter,
    ):
        raise TypeError(
            f"Фабрика модели "
            f"{normalized_model_id!r} вернула "
            "некорректный объект: "
            f"{type(adapter).__name__}."
        )

    return adapter


__all__ = [
    "get_adapter",
    "supported_model_ids",
]