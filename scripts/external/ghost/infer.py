from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = Path(__file__).resolve().parents[1]
GHOST_ROOT = PROJECT_ROOT / "vendor" / "ghost"

for search_path in (
    EXTERNAL_ROOT,
    GHOST_ROOT,
):
    search_path_string = str(
        search_path
    )

    if search_path_string not in sys.path:
        sys.path.insert(
            0,
            search_path_string,
        )

from runtime_compat import (  # noqa: E402
    configure_apple_silicon_environment,
    install_numpy_compatibility,
    install_onnxruntime_compatibility,
    install_torch_compatibility,
    install_warning_filters,
)

ARCFACE_REFERENCE_112 = np.array(
    [
        [
            38.2946,
            51.6963,
        ],
        [
            73.5318,
            51.5014,
        ],
        [
            56.0252,
            71.7366,
        ],
        [
            41.5493,
            92.3655,
        ],
        [
            70.7299,
            92.2041,
        ],
    ],
    dtype=np.float32,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GHOST inference wrapper for Apple Silicon."
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--blocks",
        required=True,
        choices=(
            1,
            2,
            3,
        ),
        type=int,
    )
    parser.add_argument(
        "--similarity-threshold",
        default=0.15,
        type=float,
    )
    parser.add_argument(
        "--crop-size",
        default=256,
        type=int,
    )

    arguments = parser.parse_args()

    if arguments.crop_size != 256:
        raise ValueError(
            "GHOST AEI-Net поддерживает только crop-size=256. "
            f"Получено: {arguments.crop_size}"
        )

    return arguments

def require_file(
    path: Path,
    label: str,
) -> Path:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{label} не найден: {resolved}"
        )

    return resolved


def read_image(
    path: Path,
) -> np.ndarray:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Не удалось прочитать изображение: {path}"
        )

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError(
            f"Некорректная форма изображения {path}: "
            f"{image.shape}"
        )

    return np.ascontiguousarray(
        image,
        dtype=np.uint8,
    )


def checkpoint_path(
    blocks: int,
) -> Path:
    suffix = (
        "block"
        if blocks == 1
        else "blocks"
    )

    return (
        GHOST_ROOT
        / "weights"
        / f"G_unet_{blocks}{suffix}.pth"
    )


def load_vendor_attribute(
    module_name: str,
    attribute_name: str,
) -> Any:
    try:
        module = importlib.import_module(
            module_name
        )
    except ImportError as error:
        raise ImportError(
            f"Не удалось импортировать GHOST-модуль "
            f"{module_name!r}: {error}"
        ) from error

    try:
        return getattr(
            module,
            attribute_name,
        )
    except AttributeError as error:
        raise ImportError(
            f"В GHOST-модуле {module_name!r} отсутствует "
            f"объект {attribute_name!r}."
        ) from error


def run_face_detection(
    detector: Any,
    image: np.ndarray,
    *,
    max_num: int,
) -> tuple[np.ndarray, np.ndarray]:
    detection_result = detector.det_model.detect(
        image,
        max_num=max_num,
        metric="default",
    )

    if (
        not isinstance(
            detection_result,
            (
                tuple,
                list,
            ),
        )
        or len(
            detection_result
        ) < 2
    ):
        raise ValueError(
            "SCRFD вернул результат в неподдерживаемом формате."
        )

    raw_bboxes = detection_result[0]
    raw_keypoints = detection_result[1]

    if raw_bboxes is None:
        bboxes = np.empty(
            (
                0,
                5,
            ),
            dtype=np.float32,
        )
    else:
        bboxes = np.asarray(
            raw_bboxes,
            dtype=np.float32,
        )

    if raw_keypoints is None:
        keypoints = np.empty(
            (
                0,
                5,
                2,
            ),
            dtype=np.float32,
        )
    else:
        keypoints = np.asarray(
            raw_keypoints,
            dtype=np.float32,
        )

    if bboxes.size == 0:
        bboxes = np.empty(
            (
                0,
                5,
            ),
            dtype=np.float32,
        )

    if keypoints.size == 0:
        keypoints = np.empty(
            (
                0,
                5,
                2,
            ),
            dtype=np.float32,
        )

    if (
        bboxes.ndim != 2
        or bboxes.shape[1] < 5
    ):
        raise ValueError(
            f"SCRFD вернул неправильную форму bbox: "
            f"{bboxes.shape}"
        )

    if (
        keypoints.ndim != 3
        or keypoints.shape[1:] != (
            5,
            2,
        )
    ):
        raise ValueError(
            f"SCRFD вернул неправильную форму keypoints: "
            f"{keypoints.shape}"
        )

    if len(
        bboxes
    ) != len(
        keypoints
    ):
        raise ValueError(
            "Количество bbox и keypoints не совпадает: "
            f"{len(bboxes)} != {len(keypoints)}"
        )

    return (
        bboxes,
        keypoints,
    )


def select_best_face_index(
    bboxes: np.ndarray,
    label: str,
) -> int:
    if len(
        bboxes
    ) == 0:
        raise ValueError(
            f"На {label}-изображении не найдено лицо."
        )

    scores = bboxes[
        :,
        4,
    ]

    if not np.all(
        np.isfinite(
            scores
        )
    ):
        raise ValueError(
            f"Детектор вернул некорректные confidence scores "
            f"для {label}."
        )

    return int(
        np.argmax(
            scores
        )
    )


def extract_transform_matrix(
    estimate_result: Any,
) -> np.ndarray | None:
    """
    Поддерживает разные версии InsightFace.

    Возможные результаты estimate_norm():

    1. ndarray формы (2, 3)
    2. ndarray формы (3, 3)
    3. tuple: (ndarray формы (2, 3), index)
    4. list с матрицей внутри
    """
    candidates: list[Any] = []

    if isinstance(
        estimate_result,
        np.ndarray,
    ):
        candidates.append(
            estimate_result
        )
    elif isinstance(
        estimate_result,
        (
            tuple,
            list,
        ),
    ):
        candidates.extend(
            estimate_result
        )
    else:
        candidates.append(
            estimate_result
        )

    for candidate in candidates:
        try:
            matrix = np.asarray(
                candidate,
                dtype=np.float32,
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if matrix.shape == (
            2,
            3,
        ):
            if np.all(
                np.isfinite(
                    matrix
                )
            ):
                return np.ascontiguousarray(
                    matrix,
                    dtype=np.float32,
                )

        if matrix.shape == (
            3,
            3,
        ):
            matrix = matrix[
                :2,
                :
            ]

            if np.all(
                np.isfinite(
                    matrix
                )
            ):
                return np.ascontiguousarray(
                    matrix,
                    dtype=np.float32,
                )

    return None


def estimate_transform_with_opencv(
    keypoints: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    scale = float(
        crop_size
    ) / 112.0

    destination = (
        ARCFACE_REFERENCE_112
        * scale
    )

    transform, inliers = cv2.estimateAffinePartial2D(
        keypoints.astype(
            np.float32
        ),
        destination.astype(
            np.float32
        ),
        method=cv2.LMEDS,
    )

    del inliers

    if transform is None:
        transform, inliers = cv2.estimateAffinePartial2D(
            keypoints.astype(
                np.float32
            ),
            destination.astype(
                np.float32
            ),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )

        del inliers

    if transform is None:
        raise ValueError(
            "OpenCV не смог вычислить матрицу выравнивания лица."
        )

    transform_array = np.asarray(
        transform,
        dtype=np.float32,
    )

    if transform_array.shape != (
        2,
        3,
    ):
        raise ValueError(
            "OpenCV вернул неправильную форму матрицы: "
            f"{transform_array.shape}"
        )

    if not np.all(
        np.isfinite(
            transform_array
        )
    ):
        raise ValueError(
            "OpenCV вернул матрицу с NaN или Inf."
        )

    return np.ascontiguousarray(
        transform_array,
        dtype=np.float32,
    )


def estimate_face_transform(
    keypoints: np.ndarray,
    crop_size: int,
    face_align_module: Any,
    label: str,
) -> np.ndarray:
    selected_keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if selected_keypoints.shape != (
        5,
        2,
    ):
        raise ValueError(
            f"Некорректная форма keypoints для {label}: "
            f"{selected_keypoints.shape}"
        )

    if not np.all(
        np.isfinite(
            selected_keypoints
        )
    ):
        raise ValueError(
            f"Keypoints для {label} содержат NaN или Inf."
        )

    estimate_result: Any = None

    try:
        estimate_result = face_align_module.estimate_norm(
            selected_keypoints,
            crop_size,
            mode="None",
        )
    except (
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
    ):
        estimate_result = None

    transform = extract_transform_matrix(
        estimate_result
    )

    if transform is None:
        transform = estimate_transform_with_opencv(
            selected_keypoints,
            crop_size,
        )

    if transform.shape != (
        2,
        3,
    ):
        raise ValueError(
            f"Некорректная матрица преобразования для {label}: "
            f"{transform.shape}"
        )

    return transform


def align_face(
    image: np.ndarray,
    transform: np.ndarray,
    crop_size: int,
    label: str,
) -> np.ndarray:
    crop = cv2.warpAffine(
        image,
        transform,
        (
            crop_size,
            crop_size,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            0,
            0,
            0,
        ),
    )

    if crop.shape != (
        crop_size,
        crop_size,
        3,
    ):
        raise ValueError(
            f"Некорректная форма aligned crop для {label}: "
            f"{crop.shape}"
        )

    return np.ascontiguousarray(
        crop,
        dtype=np.uint8,
    )


def detect_and_align_face(
    image: np.ndarray,
    detector: Any,
    face_align_module: Any,
    crop_size: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    bboxes, keypoints = run_face_detection(
        detector,
        image,
        max_num=1,
    )

    best_index = select_best_face_index(
        bboxes,
        label,
    )

    selected_keypoints = keypoints[
        best_index
    ]

    transform = estimate_face_transform(
        selected_keypoints,
        crop_size,
        face_align_module,
        label,
    )

    crop = align_face(
        image,
        transform,
        crop_size,
        label,
    )

    return (
        crop,
        transform,
    )


def normalize_crop(
    image: np.ndarray,
    torch_module: Any,
    device: Any,
) -> Any:
    if image.shape != (
        image.shape[0],
        image.shape[1],
        3,
    ):
        raise ValueError(
            f"Ожидался трёхканальный crop, получено: "
            f"{image.shape}"
        )

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    tensor = torch_module.from_numpy(
        np.ascontiguousarray(
            rgb_image
        )
    ).to(
        dtype=torch_module.float32
    )

    tensor = tensor.permute(
        2,
        0,
        1,
    ).unsqueeze(
        0
    )

    tensor = (
        tensor
        / 127.5
        - 1.0
    )

    return tensor.to(
        device
    )


def create_soft_mask(
    size: int,
) -> np.ndarray:
    mask = np.zeros(
        (
            size,
            size,
        ),
        dtype=np.float32,
    )

    cv2.ellipse(
        mask,
        (
            size // 2,
            int(
                size
                * 0.52
            ),
        ),
        (
            int(
                size
                * 0.38
            ),
            int(
                size
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
            size
            * 0.12
        ),
    )

    if blur_size % 2 == 0:
        blur_size += 1

    blurred = cv2.GaussianBlur(
        mask,
        (
            blur_size,
            blur_size,
        ),
        0,
    )

    return np.clip(
        blurred,
        0.0,
        1.0,
    )


def paste_back(
    swapped_crop: np.ndarray,
    target_image: np.ndarray,
    target_transform: np.ndarray,
) -> np.ndarray:
    inverse_transform = cv2.invertAffineTransform(
        target_transform
    )

    output_size = (
        target_image.shape[1],
        target_image.shape[0],
    )

    warped_face = cv2.warpAffine(
        swapped_crop,
        inverse_transform,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    crop_mask = create_soft_mask(
        swapped_crop.shape[0]
    )

    warped_mask = cv2.warpAffine(
        crop_mask,
        inverse_transform,
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
        + target_image.astype(
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


def unwrap_generator_output(
    generated: Any,
    torch_module: Any,
) -> Any:
    current = generated

    while isinstance(
        current,
        (
            tuple,
            list,
        ),
    ):
        if len(
            current
        ) == 0:
            raise ValueError(
                "GHOST generator вернул пустой tuple/list."
            )

        current = current[0]

    if not isinstance(
        current,
        torch_module.Tensor,
    ):
        raise TypeError(
            "GHOST generator вернул неожиданный тип: "
            f"{type(current).__name__}"
        )

    if current.ndim != 4:
        raise ValueError(
            "GHOST generator вернул tensor неправильной формы: "
            f"{tuple(current.shape)}"
        )

    if current.shape[0] < 1:
        raise ValueError(
            "GHOST generator вернул пустой batch."
        )

    if current.shape[1] != 3:
        raise ValueError(
            "GHOST generator вернул неправильное число каналов: "
            f"{tuple(current.shape)}"
        )

    return current


def tensor_to_bgr_image(
    tensor: Any,
    torch_module: Any,
    expected_size: int,
) -> np.ndarray:
    array = (
        tensor[0]
        .detach()
        .to(
            device="cpu",
            dtype=torch_module.float32,
        )
        .permute(
            1,
            2,
            0,
        )
        .contiguous()
        .numpy()
    )

    if (
        array.ndim != 3
        or array.shape[2] != 3
    ):
        raise ValueError(
            f"Некорректная форма результата GHOST: "
            f"{array.shape}"
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "Результат GHOST содержит NaN или Inf."
        )

    array = np.clip(
        (
            array
            + 1.0
        )
        * 127.5,
        0,
        255,
    ).astype(
        np.uint8
    )

    if array.shape[:2] != (
        expected_size,
        expected_size,
    ):
        array = cv2.resize(
            array,
            (
                expected_size,
                expected_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    return np.ascontiguousarray(
        cv2.cvtColor(
            array,
            cv2.COLOR_RGB2BGR,
        )
    )


def extract_state_dict(
    loaded_value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(
        loaded_value,
        dict,
    ):
        raise TypeError(
            f"{label} checkpoint имеет неожиданный тип: "
            f"{type(loaded_value).__name__}"
        )

    for key in (
        "state_dict",
        "model",
        "net",
        "generator",
        "backbone",
    ):
        nested = loaded_value.get(
            key
        )

        if isinstance(
            nested,
            dict,
        ):
            return nested

    return loaded_value


def remove_module_prefix(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    if not state_dict:
        return state_dict

    if not all(
        isinstance(
            key,
            str,
        )
        for key in state_dict
    ):
        return state_dict

    if not all(
        key.startswith(
            "module."
        )
        for key in state_dict
    ):
        return state_dict

    return {
        key[
            len(
                "module."
            ):
        ]: value
        for key, value in state_dict.items()
    }


def main() -> int:
    arguments = parse_arguments()

    configure_apple_silicon_environment(
        force_cpu=True
    )
    install_warning_filters()
    install_numpy_compatibility(
        np
    )

    source_path = require_file(
        arguments.source,
        "Source-файл",
    )

    target_path = require_file(
        arguments.target,
        "Target-файл",
    )

    output_path = (
        arguments.output
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    generator_checkpoint = require_file(
        checkpoint_path(
            arguments.blocks
        ),
        "GHOST checkpoint",
    )

    arcface_checkpoint = require_file(
        GHOST_ROOT
        / "arcface_model"
        / "backbone.pth",
        "GHOST ArcFace checkpoint",
    )

    require_file(
        GHOST_ROOT
        / "insightface_func"
        / "models"
        / "antelope"
        / "scrfd_10g_bnkps.onnx",
        "GHOST SCRFD checkpoint",
    )

    require_file(
        GHOST_ROOT
        / "insightface_func"
        / "models"
        / "antelope"
        / "glintr100.onnx",
        "GHOST recognition checkpoint",
    )

    import onnxruntime as ort
    import torch
    from insightface.utils import face_align

    selected_device = install_torch_compatibility(
        torch
    )

    selected_providers = install_onnxruntime_compatibility(
        ort
    )

    device = torch.device(
        "cpu"
    )

    print(
        f"[GHOST] device={selected_device}",
        flush=True,
    )
    print(
        f"[GHOST] blocks={arguments.blocks}",
        flush=True,
    )
    print(
        f"[GHOST] providers={selected_providers}",
        flush=True,
    )

    iresnet100 = load_vendor_attribute(
        "arcface_model.iresnet",
        "iresnet100",
    )

    face_detect_crop_class = load_vendor_attribute(
        "insightface_func.face_detect_crop_multi",
        "Face_detect_crop",
    )

    aei_net_class = load_vendor_attribute(
        "network.AEI_Net",
        "AEI_Net",
    )

    detector = face_detect_crop_class(
        name="antelope",
        root=str(
            GHOST_ROOT
            / "insightface_func"
            / "models"
        ),
    )

    detector.prepare(
        ctx_id=-1,
        det_thresh=0.5,
        det_size=(
            640,
            640,
        ),
    )

    generator = aei_net_class(
        "unet",
        num_blocks=arguments.blocks,
        c_id=512,
    )

    loaded_generator_state = torch.load(
        generator_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    generator_state = remove_module_prefix(
        extract_state_dict(
            loaded_generator_state,
            "GHOST generator",
        )
    )

    generator.load_state_dict(
        generator_state,
        strict=True,
    )
    generator.eval()
    generator.to(
        device
    )

    arcface = iresnet100(
        fp16=False
    )

    loaded_arcface_state = torch.load(
        arcface_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    arcface_state = remove_module_prefix(
        extract_state_dict(
            loaded_arcface_state,
            "GHOST ArcFace",
        )
    )

    arcface.load_state_dict(
        arcface_state,
        strict=True,
    )
    arcface.eval()
    arcface.to(
        device
    )

    source_image = read_image(
        source_path
    )

    target_image = read_image(
        target_path
    )

    source_crop, _ = detect_and_align_face(
        source_image,
        detector,
        face_align,
        arguments.crop_size,
        "source",
    )

    target_crop, target_transform = detect_and_align_face(
        target_image,
        detector,
        face_align,
        arguments.crop_size,
        "target",
    )

    print(
        f"[GHOST] source_crop={source_crop.shape}",
        flush=True,
    )
    print(
        f"[GHOST] target_crop={target_crop.shape}",
        flush=True,
    )

    source_tensor = normalize_crop(
        source_crop,
        torch,
        device,
    )

    target_tensor = normalize_crop(
        target_crop,
        torch,
        device,
    )

    arcface_input = torch.nn.functional.interpolate(
        source_tensor,
        size=(
            112,
            112,
        ),
        mode="bilinear",
        align_corners=False,
    )

    with torch.inference_mode():
        identity = arcface(
            arcface_input
        )

        identity = torch.nn.functional.normalize(
            identity,
            p=2,
            dim=1,
        )

        generated = generator(
            target_tensor,
            identity,
        )

    generated_tensor = unwrap_generator_output(
        generated,
        torch,
    )

    swapped_crop = tensor_to_bgr_image(
        generated_tensor,
        torch,
        arguments.crop_size,
    )

    result = paste_back(
        swapped_crop,
        target_image,
        target_transform,
    )

    if not cv2.imwrite(
        str(
            output_path
        ),
        result,
    ):
        raise RuntimeError(
            f"Не удалось записать результат: {output_path}"
        )

    if (
        not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        output_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"GHOST не создал корректный результат: "
            f"{output_path}"
        )

    print(
        f"[GHOST] result={output_path}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )