from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from src.face_swap_studio.adapters.base import (
    AdapterRequest,
    SwapAdapter,
    ensure_output_created,
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
    get_modern_swapper_session,
)

ARC_FACE_128_TEMPLATE = np.array(
    [
        [0.36167656, 0.40387734],
        [0.63696719, 0.40235469],
        [0.50019687, 0.56044219],
        [0.38710391, 0.72160547],
        [0.61507734, 0.72034453],
    ],
    dtype=np.float32,
)

FFHQ_512_TEMPLATE = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class ModernSwapperConfiguration:
    model_id: str
    model_filename: str
    model_type: str
    template: np.ndarray
    crop_size: int = 256


MODEL_CONFIGURATIONS: dict[
    str,
    ModernSwapperConfiguration,
] = {
    "hyperswap_1a_256": ModernSwapperConfiguration(
        model_id="hyperswap_1a_256",
        model_filename="hyperswap_1a_256.onnx",
        model_type="hyperswap",
        template=ARC_FACE_128_TEMPLATE,
    ),
    "hyperswap_1b_256": ModernSwapperConfiguration(
        model_id="hyperswap_1b_256",
        model_filename="hyperswap_1b_256.onnx",
        model_type="hyperswap",
        template=ARC_FACE_128_TEMPLATE,
    ),
    "uniface_256": ModernSwapperConfiguration(
        model_id="uniface_256",
        model_filename="uniface_256.onnx",
        model_type="uniface",
        template=FFHQ_512_TEMPLATE,
    ),
}


def raw_face_keypoints(
    raw_face: Any,
) -> np.ndarray:
    keypoints = getattr(
        raw_face,
        "kps",
        None,
    )

    if keypoints is None:
        raise ValueError(
            "InsightFace did not return five-point landmarks."
        )

    array = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if array.shape != (
        5,
        2,
    ):
        raise ValueError(
            "Unexpected landmark shape: "
            f"{array.shape}"
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "Face landmarks contain NaN or Inf."
        )

    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


def raw_face_embedding(
    raw_face: Any,
) -> np.ndarray:
    embedding = getattr(
        raw_face,
        "normed_embedding",
        None,
    )

    if embedding is None:
        embedding = getattr(
            raw_face,
            "embedding_norm",
            None,
        )

    if embedding is None:
        original_embedding = getattr(
            raw_face,
            "embedding",
            None,
        )

        if original_embedding is None:
            raise ValueError(
                "InsightFace did not return a face embedding."
            )

        embedding_array = np.asarray(
            original_embedding,
            dtype=np.float32,
        ).reshape(
            -1
        )

        norm = float(
            np.linalg.norm(
                embedding_array
            )
        )

        if norm <= 0.0:
            raise ValueError(
                "Source face embedding has zero length."
            )

        embedding = (
            embedding_array
            / norm
        )

    normalized = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(
        1,
        -1,
    )

    if normalized.shape[1] != 512:
        raise ValueError(
            "Expected a 512-dimensional source embedding, "
            f"received {normalized.shape}."
        )

    if not np.all(
        np.isfinite(
            normalized
        )
    ):
        raise ValueError(
            "Source embedding contains NaN or Inf."
        )

    norm = np.linalg.norm(
        normalized,
        axis=1,
        keepdims=True,
    )

    norm = np.maximum(
        norm,
        1e-8,
    )

    return np.ascontiguousarray(
        normalized / norm,
        dtype=np.float32,
    )


def estimate_alignment_matrix(
    keypoints: np.ndarray,
    template: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    destination = (
        template
        * float(
            crop_size
        )
    ).astype(
        np.float32
    )

    matrix, inliers = cv2.estimateAffinePartial2D(
        keypoints.astype(
            np.float32
        ),
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=100.0,
    )

    del inliers

    if matrix is None:
        matrix, inliers = cv2.estimateAffinePartial2D(
            keypoints.astype(
                np.float32
            ),
            destination,
            method=cv2.LMEDS,
        )

        del inliers

    if matrix is None:
        raise RuntimeError(
            "Could not calculate the face alignment matrix."
        )

    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if matrix.shape != (
        2,
        3,
    ):
        raise RuntimeError(
            "Unexpected alignment matrix shape: "
            f"{matrix.shape}"
        )

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        raise RuntimeError(
            "Alignment matrix contains NaN or Inf."
        )

    return np.ascontiguousarray(
        matrix,
        dtype=np.float32,
    )


def align_face(
    image_bgr: np.ndarray,
    raw_face: Any,
    template: np.ndarray,
    crop_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    keypoints = raw_face_keypoints(
        raw_face
    )

    matrix = estimate_alignment_matrix(
        keypoints,
        template,
        crop_size,
    )

    crop = cv2.warpAffine(
        image_bgr,
        matrix,
        (
            crop_size,
            crop_size,
        ),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if crop.shape != (
        crop_size,
        crop_size,
        3,
    ):
        raise RuntimeError(
            "Unexpected aligned face shape: "
            f"{crop.shape}"
        )

    return (
        np.ascontiguousarray(
            crop,
            dtype=np.uint8,
        ),
        matrix,
    )


def prepare_target_tensor(
    crop_bgr: np.ndarray,
) -> np.ndarray:
    rgb = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2RGB,
    )

    tensor = rgb.astype(
        np.float32
    )

    tensor = (
        tensor
        / 255.0
        - 0.5
    ) / 0.5

    tensor = tensor.transpose(
        2,
        0,
        1,
    )

    tensor = np.expand_dims(
        tensor,
        axis=0,
    )

    return np.ascontiguousarray(
        tensor,
        dtype=np.float32,
    )


def prepare_uniface_source_tensor(
    source_image_bgr: np.ndarray,
    source_raw_face: Any,
    configuration: ModernSwapperConfiguration,
) -> np.ndarray:
    source_crop, _ = align_face(
        source_image_bgr,
        source_raw_face,
        configuration.template,
        configuration.crop_size,
    )

    source_rgb = cv2.cvtColor(
        source_crop,
        cv2.COLOR_BGR2RGB,
    )

    source_tensor = (
        source_rgb.astype(
            np.float32
        )
        / 255.0
    )

    source_tensor = source_tensor.transpose(
        2,
        0,
        1,
    )

    source_tensor = np.expand_dims(
        source_tensor,
        axis=0,
    )

    return np.ascontiguousarray(
        source_tensor,
        dtype=np.float32,
    )


def normalize_output_tensor(
    output: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    array = np.asarray(
        output,
        dtype=np.float32,
    )

    while (
        array.ndim > 4
        and array.shape[0] == 1
    ):
        array = array[0]

    if array.ndim == 4:
        if array.shape[0] < 1:
            raise RuntimeError(
                "The face swapper returned an empty batch."
            )

        array = array[0]

    if (
        array.ndim != 3
        or array.shape[0] != 3
    ):
        raise RuntimeError(
            "Unexpected face-swap output shape: "
            f"{array.shape}"
        )

    array = array.transpose(
        1,
        2,
        0,
    )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise RuntimeError(
            "The face swapper returned NaN or Inf."
        )

    minimum = float(
        np.min(
            array
        )
    )
    float(
        np.max(
            array
        )
    )

    if minimum < -0.05:
        array = (
            array
            * 0.5
            + 0.5
        )

    array = np.clip(
        array,
        0.0,
        1.0,
    )

    rgb = (
        array
        * 255.0
    ).round().astype(
        np.uint8
    )

    if rgb.shape[
        :2
    ] != (
        crop_size,
        crop_size,
    ):
        rgb = cv2.resize(
            rgb,
            (
                crop_size,
                crop_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    return np.ascontiguousarray(
        cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        ),
        dtype=np.uint8,
    )


def create_face_mask(
    crop_size: int,
) -> np.ndarray:
    mask = np.zeros(
        (
            crop_size,
            crop_size,
        ),
        dtype=np.float32,
    )

    cv2.ellipse(
        mask,
        (
            crop_size // 2,
            int(
                crop_size
                * 0.50
            ),
        ),
        (
            int(
                crop_size
                * 0.43
            ),
            int(
                crop_size
                * 0.48
            ),
        ),
        0,
        0,
        360,
        1.0,
        -1,
    )

    blur_size = max(
        15,
        int(
            crop_size
            * 0.12
        ),
    )

    if blur_size % 2 == 0:
        blur_size += 1

    mask = cv2.GaussianBlur(
        mask,
        (
            blur_size,
            blur_size,
        ),
        0,
    )

    return np.clip(
        mask,
        0.0,
        1.0,
    )


def paste_face_back(
    original_bgr: np.ndarray,
    swapped_crop_bgr: np.ndarray,
    alignment_matrix: np.ndarray,
) -> np.ndarray:
    inverse_matrix = cv2.invertAffineTransform(
        alignment_matrix
    )

    image_height, image_width = original_bgr.shape[
        :2
    ]

    output_size = (
        image_width,
        image_height,
    )

    warped_face = cv2.warpAffine(
        swapped_crop_bgr,
        inverse_matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    crop_mask = create_face_mask(
        swapped_crop_bgr.shape[0]
    )

    warped_mask = cv2.warpAffine(
        crop_mask,
        inverse_matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_mask = np.clip(
        warped_mask,
        0.0,
        1.0,
    )[
        ...,
        None,
    ]

    result = (
        warped_face.astype(
            np.float32
        )
        * warped_mask
        + original_bgr.astype(
            np.float32
        )
        * (
            1.0
            - warped_mask
        )
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )


def build_session_inputs(
    session: ort.InferenceSession,
    *,
    source_value: np.ndarray,
    target_value: np.ndarray,
) -> dict[
    str,
    np.ndarray,
]:
    inputs: dict[
        str,
        np.ndarray,
    ] = {}

    session_inputs = session.get_inputs()

    if len(
        session_inputs
    ) != 2:
        input_description = [
            (
                input_node.name,
                input_node.shape,
                input_node.type,
            )
            for input_node in session_inputs
        ]

        raise RuntimeError(
            "Expected exactly two model inputs, received: "
            f"{input_description}"
        )

    for input_node in session_inputs:
        normalized_name = input_node.name.lower()

        if (
            "source" in normalized_name
            or "identity" in normalized_name
            or "embedding" in normalized_name
            or "latent" in normalized_name
            or normalized_name in {
                "id",
                "z",
            }
        ):
            inputs[
                input_node.name
            ] = source_value
            continue

        if (
            "target" in normalized_name
            or "image" in normalized_name
            or "input" in normalized_name
            or "crop" in normalized_name
        ):
            inputs[
                input_node.name
            ] = target_value
            continue

        shape = input_node.shape

        has_image_shape = (
            isinstance(
                shape,
                list,
            )
            and len(
                shape
            ) == 4
        )

        if has_image_shape:
            inputs[
                input_node.name
            ] = target_value
        else:
            inputs[
                input_node.name
            ] = source_value

    if len(
        inputs
    ) != 2:
        raise RuntimeError(
            "Could not map the ONNX model inputs."
        )

    return inputs


class ModernOnnxSwapperAdapter(
    SwapAdapter
):
    model_ids = tuple(
        MODEL_CONFIGURATIONS
    )

    def __init__(
        self,
        model_id: str,
    ) -> None:
        try:
            self.configuration = (
                MODEL_CONFIGURATIONS[
                    model_id
                ]
            )
        except KeyError as error:
            raise ValueError(
                "Unsupported modern ONNX swapper: "
                f"{model_id}"
            ) from error

        self.model_id = model_id

    def is_available(
        self,
    ) -> bool:
        return is_model_ready(
            model_by_id(
                self.model_id
            )
        )

    def process(
        self,
        request: AdapterRequest,
    ) -> Path:
        request.validate()

        configuration = self.configuration

        source_image = read_image(
            request.source_path
        )
        target_image = read_image(
            request.target_path
        )

        source_faces = detect_faces(
            source_image
        )
        target_faces = detect_faces(
            target_image
        )

        if not source_faces:
            raise ValueError(
                "No face was detected in the source image."
            )

        if not target_faces:
            raise ValueError(
                "No face was detected in the target image."
            )

        source_face = max(
            source_faces,
            key=lambda face: face.area,
        )

        if request.target_face_index is None:
            selected_targets = target_faces
        else:
            selected_targets = [
                face
                for face in target_faces
                if face.index
                == request.target_face_index
            ]

        if not selected_targets:
            raise ValueError(
                "The selected target face was not found."
            )

        session = get_modern_swapper_session(
            configuration.model_filename,
            force_cpu=(
                configuration.model_type
                == "uniface"
            ),
        )

        if configuration.model_type == "hyperswap":
            source_value = raw_face_embedding(
                source_face.raw_face
            )
        elif configuration.model_type == "uniface":
            source_value = (
                prepare_uniface_source_tensor(
                    source_image,
                    source_face.raw_face,
                    configuration,
                )
            )
        else:
            raise RuntimeError(
                "Unknown modern swapper type: "
                f"{configuration.model_type}"
            )

        result = target_image.copy()

        for target_face in selected_targets:
            target_crop, alignment_matrix = align_face(
                result,
                target_face.raw_face,
                configuration.template,
                configuration.crop_size,
            )

            target_value = prepare_target_tensor(
                target_crop
            )

            inference_inputs = build_session_inputs(
                session,
                source_value=source_value,
                target_value=target_value,
            )

            outputs = session.run(
                None,
                inference_inputs,
            )

            if not outputs:
                raise RuntimeError(
                    "The ONNX face swapper returned no outputs."
                )

            swapped_crop = normalize_output_tensor(
                outputs[0],
                configuration.crop_size,
            )

            result = paste_face_back(
                result,
                swapped_crop,
                alignment_matrix,
            )

        output_path = write_image(
            request.output_path,
            result,
        )

        return ensure_output_created(
            output_path,
            configuration.model_id,
        )


__all__ = [
    "MODEL_CONFIGURATIONS",
    "ModernOnnxSwapperAdapter",
]