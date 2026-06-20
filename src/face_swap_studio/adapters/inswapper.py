from __future__ import annotations

from pathlib import Path

from src.face_swap_studio.adapters.base import (
    AdapterRequest,
    SwapAdapter,
)
from src.face_swap_studio.core.detector import detect_faces
from src.face_swap_studio.core.pipeline import (
    read_image,
    write_image,
)
from src.face_swap_studio.models.manifest import (
    is_model_ready,
    model_by_id,
)
from src.face_swap_studio.models.model_manager import (
    get_face_swapper,
)


class InSwapperAdapter(SwapAdapter):
    model_id = "inswapper_128"

    def is_available(self) -> bool:
        return is_model_ready(model_by_id(self.model_id))

    def process(self, request: AdapterRequest) -> Path:
        source_image = read_image(request.source_path)
        target_image = read_image(request.target_path)

        source_faces = detect_faces(source_image)
        target_faces = detect_faces(target_image)

        if not source_faces:
            raise ValueError("На исходном изображении не найдено лицо.")

        if not target_faces:
            raise ValueError("На целевом изображении не найдено лицо.")

        source_face = max(
            source_faces,
            key=lambda item: item.area,
        )

        if request.target_face_index is None:
            selected_targets = target_faces
        else:
            selected_targets = [
                face for face in target_faces if face.index == request.target_face_index
            ]

        if not selected_targets:
            raise ValueError("Не найдено выбранное target-лицо.")

        swapper = get_face_swapper()
        result = target_image.copy()

        for target_face in selected_targets:
            result = swapper.get(
                result,
                target_face.raw_face,
                source_face.raw_face,
                paste_back=True,
            )

        return write_image(
            request.output_path,
            result,
        )
