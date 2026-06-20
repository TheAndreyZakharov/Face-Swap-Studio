from __future__ import annotations

import pytest

from src.face_swap_studio.adapters.base import (
    BaseAdapter,
    SwapAdapter,
)
from src.face_swap_studio.adapters.ghost import GhostAdapter
from src.face_swap_studio.adapters.ghost2 import Ghost2Adapter
from src.face_swap_studio.adapters.inswapper import InSwapperAdapter
from src.face_swap_studio.adapters.registry import (
    get_adapter,
    supported_model_ids,
)
from src.face_swap_studio.adapters.simswap import SimSwapAdapter


def test_base_adapter_is_swap_adapter() -> None:
    assert issubclass(
        BaseAdapter,
        SwapAdapter,
    )


def test_all_adapter_classes_import() -> None:
    adapter_classes = (
        InSwapperAdapter,
        SimSwapAdapter,
        GhostAdapter,
        Ghost2Adapter,
    )

    assert len(adapter_classes) == 4


@pytest.mark.parametrize(
    "model_id",
    supported_model_ids(),
)
def test_registry_creates_adapter(
    model_id: str,
) -> None:
    adapter = get_adapter(model_id)

    assert isinstance(
        adapter,
        SwapAdapter,
    )
