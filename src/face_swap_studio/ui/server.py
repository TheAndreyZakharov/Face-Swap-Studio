from __future__ import annotations

import asyncio
import mimetypes
import shutil
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.face_swap_studio.core.detector import (
    detect_faces as detect_faces_in_image,
)
from src.face_swap_studio.core.pipeline import (
    apply_postprocessing,
    process_single_pair,
    read_image,
    write_image,
)
from src.face_swap_studio.domain.entities import ProcessingOptions
from src.face_swap_studio.models.manifest import (
    is_model_ready,
    selectable_swap_models,
)
from src.face_swap_studio.services.image_service import (
    build_source_gallery,
)
from src.face_swap_studio.ui.api_models import (
    DetectionRequest,
    FaceMappingRequest,
    FaceResponse,
    GeneratedResultResponse,
    GenerationRequest,
    ModelResponse,
    SessionCreateResponse,
    SessionResponse,
    TargetAnalysisResponse,
    TargetBatchSelectionRequest,
    TargetSelectionRequest,
    UploadedImageResponse,
)
from src.face_swap_studio.ui.session import (
    DetectedFace,
    StudioSession,
    TargetAnalysis,
    UploadedImage,
    session_store,
)
from src.face_swap_studio.utils.logging import get_logger
from src.face_swap_studio.utils.paths import ensure_directories

logger = get_logger(
    __name__
)

STATIC_DIRECTORY = (
    Path(
        __file__
    ).resolve().parent
    / "static"
)

INDEX_FILE = (
    STATIC_DIRECTORY
    / "index.html"
)

ALLOWED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


def require_session(
    session_id: str,
) -> StudioSession:
    try:
        return session_store.require(
            session_id
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        ) from error


def safe_filename(
    original_name: str | None,
) -> str:
    original = Path(
        original_name
        or "image.png"
    )

    suffix = original.suffix.lower()

    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        suffix = ".png"

    safe_stem = "".join(
        character
        if (
            character.isalnum()
            or character in {
                "-",
                "_",
            }
        )
        else "_"
        for character in original.stem
    ).strip(
        "_"
    )

    if not safe_stem:
        safe_stem = "image"

    return (
        f"{safe_stem}-"
        f"{uuid.uuid4().hex[:10]}"
        f"{suffix}"
    )


async def save_upload(
    upload: UploadFile,
    destination_directory: Path,
) -> UploadedImage:
    filename = safe_filename(
        upload.filename
    )

    destination = (
        destination_directory
        / filename
    )

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with destination.open(
            "wb"
        ) as output_file:
            while True:
                chunk = await upload.read(
                    1024
                    * 1024
                )

                if not chunk:
                    break

                output_file.write(
                    chunk
                )
    finally:
        await upload.close()

    if (
        not destination.is_file()
        or destination.stat().st_size == 0
    ):
        destination.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded image is empty "
                "or could not be saved."
            ),
        )

    image = cv2.imread(
        str(
            destination
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        destination.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported or damaged image: "
                f"{upload.filename}"
            ),
        )

    return UploadedImage(
        id=uuid.uuid4().hex,
        name=(
            upload.filename
            or destination.name
        ),
        path=destination,
    )


def normalize_gallery_item(
    item: Any,
) -> Any:
    if isinstance(
        item,
        (
            tuple,
            list,
        ),
    ):
        if not item:
            return None

        return item[0]

    return item


def write_gallery_item(
    item: Any,
    destination: Path,
) -> Path:
    value = normalize_gallery_item(
        item
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if isinstance(
        value,
        Path,
    ):
        source = value.expanduser().resolve()

        if not source.is_file():
            raise ValueError(
                f"Gallery file not found: {source}"
            )

        shutil.copy2(
            source,
            destination,
        )

        return destination

    if isinstance(
        value,
        str,
    ):
        source = Path(
            value
        ).expanduser().resolve()

        if not source.is_file():
            raise ValueError(
                f"Gallery file not found: {source}"
            )

        shutil.copy2(
            source,
            destination,
        )

        return destination

    if not isinstance(
        value,
        np.ndarray,
    ):
        raise TypeError(
            "Unsupported gallery item type: "
            f"{type(value).__name__}"
        )

    image = value

    if image.ndim == 2:
        encoded_image = image
    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):
        encoded_image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )
    elif (
        image.ndim == 3
        and image.shape[2] == 4
    ):
        encoded_image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2BGRA,
        )
    else:
        raise ValueError(
            "Unsupported gallery image shape: "
            f"{image.shape}"
        )

    if not cv2.imwrite(
        str(
            destination
        ),
        encoded_image,
    ):
        raise RuntimeError(
            f"Could not save thumbnail: {destination}"
        )

    return destination


def save_detected_faces(
    gallery: list[Any],
    destination_directory: Path,
    *,
    prefix: str,
) -> list[DetectedFace]:
    detected_faces: list[DetectedFace] = []

    shutil.rmtree(
        destination_directory,
        ignore_errors=True,
    )

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, item in enumerate(
        gallery
    ):
        face_id = uuid.uuid4().hex

        destination = (
            destination_directory
            / (
                f"{prefix}-{index + 1}-"
                f"{face_id[:8]}.png"
            )
        )

        write_gallery_item(
            item,
            destination,
        )

        detected_faces.append(
            DetectedFace(
                id=face_id,
                index=index,
                path=destination,
                label=(
                    f"{prefix.title()} face "
                    f"{index + 1}"
                ),
            )
        )

    return detected_faces


def expanded_face_crop_box(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    *,
    padding_ratio: float = 0.45,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    face_width = max(
        1,
        x2 - x1,
    )
    face_height = max(
        1,
        y2 - y1,
    )

    padding_x = int(
        round(
            face_width
            * padding_ratio
        )
    )
    padding_y = int(
        round(
            face_height
            * padding_ratio
        )
    )

    crop_x1 = max(
        0,
        x1 - padding_x,
    )
    crop_y1 = max(
        0,
        y1 - padding_y,
    )
    crop_x2 = min(
        image_width,
        x2 + padding_x,
    )
    crop_y2 = min(
        image_height,
        y2 + padding_y,
    )

    if (
        crop_x2 <= crop_x1
        or crop_y2 <= crop_y1
    ):
        raise ValueError(
            "Could not create a valid face crop."
        )

    return (
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2,
    )


def save_target_detected_faces(
    image_path: Path,
    destination_directory: Path,
    confidence_threshold: float,
) -> list[DetectedFace]:
    image = read_image(
        image_path
    )

    detected = detect_faces_in_image(
        image,
        confidence_threshold=confidence_threshold,
    )

    if not detected:
        return []

    shutil.rmtree(
        destination_directory,
        ignore_errors=True,
    )

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_height, image_width = image.shape[
        :2
    ]

    saved_faces: list[DetectedFace] = []

    for output_index, face in enumerate(
        detected
    ):
        bbox = (
            int(
                face.bbox[0]
            ),
            int(
                face.bbox[1]
            ),
            int(
                face.bbox[2]
            ),
            int(
                face.bbox[3]
            ),
        )

        crop_box = expanded_face_crop_box(
            bbox,
            image_width,
            image_height,
        )

        crop_x1, crop_y1, crop_x2, crop_y2 = (
            crop_box
        )

        crop = image[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ].copy()

        if crop.size == 0:
            continue

        face_id = uuid.uuid4().hex

        destination = (
            destination_directory
            / (
                f"target-{output_index + 1}-"
                f"{face_id[:8]}.png"
            )
        )

        write_image(
            destination,
            crop,
        )

        saved_faces.append(
            DetectedFace(
                id=face_id,
                index=output_index,
                path=destination,
                label=(
                    f"Target face "
                    f"{output_index + 1}"
                ),
                bbox=bbox,
                crop_box=crop_box,
            )
        )

    return saved_faces


def create_target_working_copy(
    session: StudioSession,
    target_image: UploadedImage,
) -> Path:
    destination = (
        session.results_directory
        / f"working-target-{target_image.id[:12]}.png"
    )

    image = read_image(
        target_image.path
    )

    return write_image(
        destination,
        image,
    )


def clamp_crop_box(
    crop_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    crop_x1, crop_y1, crop_x2, crop_y2 = (
        crop_box
    )

    crop_x1 = max(
        0,
        min(
            crop_x1,
            image_width - 1,
        ),
    )
    crop_y1 = max(
        0,
        min(
            crop_y1,
            image_height - 1,
        ),
    )
    crop_x2 = max(
        crop_x1 + 1,
        min(
            crop_x2,
            image_width,
        ),
    )
    crop_y2 = max(
        crop_y1 + 1,
        min(
            crop_y2,
            image_height,
        ),
    )

    return (
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2,
    )


def extract_target_face_crop(
    target_image_path: Path,
    target_face: DetectedFace,
    destination: Path,
) -> Path:
    if target_face.crop_box is None:
        raise ValueError(
            "Target face does not contain crop coordinates."
        )

    image = read_image(
        target_image_path
    )

    image_height, image_width = image.shape[
        :2
    ]

    crop_x1, crop_y1, crop_x2, crop_y2 = (
        clamp_crop_box(
            target_face.crop_box,
            image_width,
            image_height,
        )
    )

    crop = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()

    if crop.size == 0:
        raise ValueError(
            f"Target face {target_face.index + 1} "
            "produced an empty crop."
        )

    return write_image(
        destination,
        crop,
    )


def create_crop_blend_mask(
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    centre = (
        width // 2,
        height // 2,
    )

    axes = (
        max(
            1,
            int(
                width
                * 0.46
            ),
        ),
        max(
            1,
            int(
                height
                * 0.46
            ),
        ),
    )

    cv2.ellipse(
        mask,
        centre,
        axes,
        0,
        0,
        360,
        1.0,
        -1,
    )

    minimum_side = min(
        width,
        height,
    )

    blur_size = max(
        3,
        int(
            minimum_side
            * 0.15
        ),
    )

    if blur_size % 2 == 0:
        blur_size += 1

    maximum_blur_size = minimum_side

    if maximum_blur_size % 2 == 0:
        maximum_blur_size -= 1

    maximum_blur_size = max(
        1,
        maximum_blur_size,
    )

    blur_size = min(
        blur_size,
        maximum_blur_size,
    )

    if blur_size >= 3:
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
    )[
        ...,
        None,
    ]


def paste_processed_face_crop(
    target_image_path: Path,
    processed_crop_path: Path,
    target_face: DetectedFace,
) -> Path:
    if target_face.crop_box is None:
        raise ValueError(
            "Target face does not contain crop coordinates."
        )

    target_image = read_image(
        target_image_path
    )

    processed_crop = read_image(
        processed_crop_path
    )

    image_height, image_width = target_image.shape[
        :2
    ]

    crop_x1, crop_y1, crop_x2, crop_y2 = (
        clamp_crop_box(
            target_face.crop_box,
            image_width,
            image_height,
        )
    )

    expected_width = (
        crop_x2
        - crop_x1
    )
    expected_height = (
        crop_y2
        - crop_y1
    )

    if processed_crop.shape[
        :2
    ] != (
        expected_height,
        expected_width,
    ):
        processed_crop = cv2.resize(
            processed_crop,
            (
                expected_width,
                expected_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    original_crop = target_image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()

    if original_crop.shape != processed_crop.shape:
        raise ValueError(
            "Processed crop and target region "
            "have incompatible dimensions."
        )

    mask = create_crop_blend_mask(
        expected_width,
        expected_height,
    )

    blended_crop = (
        processed_crop.astype(
            np.float32
        )
        * mask
        + original_crop.astype(
            np.float32
        )
        * (
            1.0
            - mask
        )
    )

    target_image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ] = np.clip(
        blended_crop,
        0,
        255,
    ).astype(
        np.uint8
    )

    return write_image(
        target_image_path,
        target_image,
    )


def session_file_url(
    session: StudioSession,
    path: Path,
) -> str:
    relative_path = path.resolve().relative_to(
        session.directory.resolve()
    )

    return (
        f"/api/sessions/{session.id}/files/"
        f"{relative_path.as_posix()}"
    )


def uploaded_image_response(
    session: StudioSession,
    image: UploadedImage,
    *,
    selected: bool,
    included: bool,
) -> UploadedImageResponse:
    return UploadedImageResponse(
        id=image.id,
        name=image.name,
        url=session_file_url(
            session,
            image.path,
        ),
        selected=selected,
        included=included,
    )


def detected_face_response(
    session: StudioSession,
    face: DetectedFace,
) -> FaceResponse:
    return FaceResponse(
        id=face.id,
        index=face.index,
        image_url=session_file_url(
            session,
            face.path,
        ),
        label=face.label,
    )


def generated_result_response(
    session: StudioSession,
    target_image: UploadedImage,
    result_path: Path,
) -> GeneratedResultResponse:
    return GeneratedResultResponse(
        target_image_id=target_image.id,
        target_image_name=target_image.name,
        url=session_file_url(
            session,
            result_path,
        ),
        download_url=(
            f"/api/sessions/{session.id}/download/"
            f"{target_image.id}"
        ),
    )

def target_analysis_response(
    session: StudioSession,
    target_image: UploadedImage,
) -> TargetAnalysisResponse:
    analysis = session.target_analyses.get(
        target_image.id
    )

    if analysis is None:
        return TargetAnalysisResponse(
            target_image_id=target_image.id,
            target_image_name=target_image.name,
            analysis_completed=False,
            target_faces=[],
            mappings={},
        )

    return TargetAnalysisResponse(
        target_image_id=target_image.id,
        target_image_name=target_image.name,
        analysis_completed=analysis.analysis_completed,
        target_faces=[
            detected_face_response(
                session,
                face,
            )
            for face in analysis.target_faces
        ],
        mappings=dict(
            analysis.face_mappings
        ),
    )

def build_session_response(
    session: StudioSession,
) -> SessionResponse:
    active_analysis = session.active_target_analysis()

    target_faces = (
        active_analysis.target_faces
        if active_analysis is not None
        else []
    )

    mappings = (
        active_analysis.face_mappings
        if active_analysis is not None
        else {}
    )

    generated_results: list[GeneratedResultResponse] = []

    for target_image in session.target_images:
        result_path = session.result_paths.get(
            target_image.id
        )

        if (
            result_path is not None
            and result_path.is_file()
        ):
            generated_results.append(
                generated_result_response(
                    session,
                    target_image,
                    result_path,
                )
            )

    result_url: str | None = None
    download_url: str | None = None

    active_result = (
        session.result_paths.get(
            session.active_target_image_id
        )
        if session.active_target_image_id is not None
        else None
    )

    if (
        active_result is not None
        and active_result.is_file()
    ):
        result_url = session_file_url(
            session,
            active_result,
        )

        download_url = (
            f"/api/sessions/{session.id}/download"
        )
    elif generated_results:
        result_url = generated_results[0].url
        download_url = generated_results[0].download_url

    return SessionResponse(
        session_id=session.id,
        source_images=[
            uploaded_image_response(
                session,
                image,
                selected=False,
                included=False,
            )
            for image in session.source_images
        ],
        target_images=[
            uploaded_image_response(
                session,
                image,
                selected=(
                    image.id
                    == session.active_target_image_id
                ),
                included=(
                    image.id
                    in session.selected_target_image_ids
                ),
            )
            for image in session.target_images
        ],
        source_faces=[
            detected_face_response(
                session,
                face,
            )
            for face in session.source_faces
        ],
        target_faces=[
            detected_face_response(
                session,
                face,
            )
            for face in target_faces
        ],
        mappings=dict(
            mappings
        ),
        active_target_image_id=(
            session.active_target_image_id
        ),
        selected_target_image_ids=[
            image.id
            for image in session.target_images
            if image.id in session.selected_target_image_ids
        ],
        target_analyses=[
            target_analysis_response(
                session,
                image,
            )
            for image in session.target_images
        ],
        analysis_completed=(
            bool(
                active_analysis
                and active_analysis.analysis_completed
            )
        ),
        result_url=result_url,
        download_url=download_url,
        generated_results=generated_results,
    )


def resolve_model(
    model_id: str,
) -> None:
    for definition in selectable_swap_models():
        if definition.id != model_id:
            continue

        if not is_model_ready(
            definition
        ):
            raise ValueError(
                f"Model is unavailable: {definition.name}"
            )

        return

    raise ValueError(
        f"Unknown model: {model_id}"
    )

def safe_download_stem(
    name: str,
) -> str:
    stem = Path(
        name
    ).stem

    safe_stem = "".join(
        character
        if (
            character.isalnum()
            or character in {
                "-",
                "_",
            }
        )
        else "_"
        for character in stem
    ).strip(
        "_"
    )

    if not safe_stem:
        safe_stem = "face-swap-result"

    return safe_stem

def copy_result_to_session(
    session: StudioSession,
    target_image: UploadedImage,
    source_path: Path,
) -> Path:
    destination = session.result_path_for(
        target_image.id
    )

    source = source_path.expanduser().resolve()
    destination = destination.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"Generated result not found: {source}"
        )

    if source != destination:
        shutil.copy2(
            source,
            destination,
        )

    if (
        not destination.is_file()
        or destination.stat().st_size == 0
    ):
        raise RuntimeError(
            "Could not create the session result file."
        )

    session.result_paths[
        target_image.id
    ] = destination

    return destination


def target_images_for_generation(
    session: StudioSession,
) -> list[UploadedImage]:
    return [
        image
        for image in session.target_images
        if image.id in session.selected_target_image_ids
    ]


def run_generation_for_target(
    session: StudioSession,
    target_image: UploadedImage,
    analysis: TargetAnalysis,
    request: GenerationRequest,
) -> Path:
    if not analysis.analysis_completed:
        raise ValueError(
            f"Detect faces for target image first: {target_image.name}"
        )

    active_mappings = [
        (
            target_index,
            source_index,
        )
        for target_index, source_index
        in sorted(
            analysis.face_mappings.items()
        )
        if source_index is not None
    ]

    if not active_mappings:
        raise ValueError(
            f"Assign at least one replacement for: {target_image.name}"
        )

    working_target_path = create_target_working_copy(
        session,
        target_image,
    )

    processing_options = ProcessingOptions(
        model_id=request.model_id,
        enhance_faces=False,
        face_enhancement_weight=(
            request.enhancement_weight
        ),
        upscale_image=False,
        upscale_factor=request.upscale_factor,
        tile_size=request.tile_size,
    )

    crop_directory = (
        session.results_directory
        / "face-crops"
        / target_image.id
    )

    shutil.rmtree(
        crop_directory,
        ignore_errors=True,
    )

    crop_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        target_face_index,
        source_face_index,
    ) in active_mappings:
        if not (
            0
            <= source_face_index
            < len(
                session.source_faces
            )
        ):
            raise ValueError(
                "A source-face mapping is invalid."
            )

        matching_target_face = next(
            (
                face
                for face in analysis.target_faces
                if face.index
                == target_face_index
            ),
            None,
        )

        if matching_target_face is None:
            raise ValueError(
                f"Target face {target_face_index + 1} "
                f"is no longer available for: {target_image.name}"
            )

        if matching_target_face.crop_box is None:
            raise ValueError(
                f"Target face {target_face_index + 1} "
                f"does not have crop coordinates for: {target_image.name}"
            )

        source_face = session.source_faces[
            source_face_index
        ]

        target_crop_path = (
            crop_directory
            / (
                f"target-face-"
                f"{target_face_index + 1}.png"
            )
        )

        extract_target_face_crop(
            working_target_path,
            matching_target_face,
            target_crop_path,
        )

        processed_crop_path = process_single_pair(
            source_path=source_face.path,
            target_path=target_crop_path,
            options=processing_options,
            target_face_index=None,
        )

        paste_processed_face_crop(
            working_target_path,
            Path(
                processed_crop_path
            ),
            matching_target_face,
        )

    if (
        request.enhance_faces
        or request.upscale_image
    ):
        postprocessing_options = ProcessingOptions(
            model_id=request.model_id,
            enhance_faces=request.enhance_faces,
            face_enhancement_weight=(
                request.enhancement_weight
            ),
            upscale_image=request.upscale_image,
            upscale_factor=request.upscale_factor,
            tile_size=request.tile_size,
        )

        working_target_path = apply_postprocessing(
            result_path=working_target_path,
            options=postprocessing_options,
        )

    return copy_result_to_session(
        session,
        target_image,
        Path(
            working_target_path
        ),
    )

def generated_result_items(
    session: StudioSession,
) -> list[tuple[UploadedImage, Path]]:
    items: list[tuple[UploadedImage, Path]] = []

    for target_image in session.target_images:
        result_path = session.result_paths.get(
            target_image.id
        )

        if (
            result_path is not None
            and result_path.is_file()
        ):
            items.append(
                (
                    target_image,
                    result_path,
                )
            )

    return items


def build_results_archive(
    session: StudioSession,
    items: list[tuple[UploadedImage, Path]],
) -> Path:
    archive_path = (
        session.results_directory
        / "face-swap-results.zip"
    )

    archive_path.unlink(
        missing_ok=True
    )

    used_names: set[str] = set()

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for index, (
            target_image,
            result_path,
        ) in enumerate(
            items,
            start=1,
        ):
            base_name = safe_download_stem(
                target_image.name
            )

            archive_name = (
                f"{index:02d}-{base_name}-face-swap.png"
            )

            if archive_name in used_names:
                archive_name = (
                    f"{index:02d}-{base_name}-"
                    f"{target_image.id[:8]}-face-swap.png"
                )

            used_names.add(
                archive_name
            )

            archive.write(
                result_path,
                arcname=archive_name,
            )

    if (
        not archive_path.is_file()
        or archive_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "Could not create results archive."
        )

    return archive_path


def run_generation(
    session: StudioSession,
    request: GenerationRequest,
) -> dict[str, Path]:
    resolve_model(
        request.model_id
    )

    if not session.source_faces:
        raise ValueError(
            "Detect source faces before generating."
        )

    targets = target_images_for_generation(
        session
    )

    if not targets:
        raise ValueError(
            "Select at least one target image with the green border."
        )

    generated: dict[str, Path] = {}

    for target_image in targets:
        analysis = session.target_analyses.get(
            target_image.id
        )

        if analysis is None:
            raise ValueError(
                f"Detect faces for target image first: {target_image.name}"
            )

        generated[
            target_image.id
        ] = run_generation_for_target(
            session,
            target_image,
            analysis,
            request,
        )

    return generated


@asynccontextmanager
async def application_lifespan(
    application: FastAPI,
):
    del application

    ensure_directories()

    yield

    session_store.clear()


def create_application() -> FastAPI:
    application = FastAPI(
        title="Face Swap Studio",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=application_lifespan,
    )

    @application.get(
        "/",
        include_in_schema=False,
    )
    async def index() -> FileResponse:
        return FileResponse(
            INDEX_FILE,
            media_type="text/html",
        )

    @application.get(
        "/api/health",
    )
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
        }

    @application.get(
        "/api/models",
        response_model=list[ModelResponse],
    )
    async def models() -> list[ModelResponse]:
        return [
            ModelResponse(
                id=definition.id,
                name=definition.name,
                description=definition.description,
                available=is_model_ready(
                    definition
                ),
            )
            for definition
            in selectable_swap_models()
        ]

    @application.post(
        "/api/sessions",
        response_model=SessionCreateResponse,
    )
    async def create_session() -> SessionCreateResponse:
        session = session_store.create()

        return SessionCreateResponse(
            session_id=session.id,
        )

    @application.get(
        "/api/sessions/{session_id}",
        response_model=SessionResponse,
    )
    async def get_session(
        session_id: str,
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        return build_session_response(
            session
        )

    @application.post(
        "/api/sessions/{session_id}/sources",
        response_model=SessionResponse,
    )
    async def upload_sources(
        session_id: str,
        files: Annotated[
            list[UploadFile],
            File(),
        ],
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        if not files:
            raise HTTPException(
                status_code=400,
                detail="No source images were provided.",
            )

        for upload in files:
            image = await save_upload(
                upload,
                session.source_uploads_directory,
            )

            session.source_images.append(
                image
            )

        session.reset_analysis()

        return build_session_response(
            session
        )

    @application.delete(
        "/api/sessions/{session_id}/sources/{source_image_id}",
        response_model=SessionResponse,
    )
    async def delete_source_image(
        session_id: str,
        source_image_id: str,
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        if not session.remove_source_image(
            source_image_id
        ):
            raise HTTPException(
                status_code=404,
                detail="Source image not found.",
            )

        return build_session_response(
            session
        )

    @application.post(
        "/api/sessions/{session_id}/targets",
        response_model=SessionResponse,
    )
    async def upload_targets(
        session_id: str,
        files: Annotated[
            list[UploadFile],
            File(),
        ],
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        if not files:
            raise HTTPException(
                status_code=400,
                detail="No target images were provided.",
            )

        for upload in files:
            image = await save_upload(
                upload,
                session.target_uploads_directory,
            )

            session.target_images.append(
                image
            )

            if session.active_target_image_id is None:
                session.active_target_image_id = image.id

        return build_session_response(
            session
        )

    @application.delete(
        "/api/sessions/{session_id}/targets/{target_image_id}",
        response_model=SessionResponse,
    )
    async def delete_target_image(
        session_id: str,
        target_image_id: str,
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        if not session.remove_target_image(
            target_image_id
        ):
            raise HTTPException(
                status_code=404,
                detail="Target image not found.",
            )

        return build_session_response(
            session
        )

    @application.post(
        "/api/sessions/{session_id}/target-selection",
        response_model=SessionResponse,
    )
    async def select_target_image(
        session_id: str,
        request: TargetSelectionRequest,
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        matching_image = session.target_image_by_id(
            request.target_image_id
        )

        if matching_image is None:
            raise HTTPException(
                status_code=404,
                detail="Target image not found.",
            )

        session.active_target_image_id = (
            matching_image.id
        )

        return build_session_response(
            session
        )

    @application.put(
        "/api/sessions/{session_id}/target-batch-selection",
        response_model=SessionResponse,
    )
    async def update_target_batch_selection(
        session_id: str,
        request: TargetBatchSelectionRequest,
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        valid_ids = {
            image.id
            for image in session.target_images
        }

        requested_ids = set(
            request.target_image_ids
        )

        invalid_ids = requested_ids - valid_ids

        if invalid_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid target image id(s): "
                    + ", ".join(
                        sorted(
                            invalid_ids
                        )
                    )
                ),
            )

        session.selected_target_image_ids = requested_ids

        return build_session_response(
            session
        )

    @application.post(
        "/api/sessions/{session_id}/detect",
        response_model=SessionResponse,
    )
    async def detect_faces(
        session_id: str,
        request: DetectionRequest,
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        if not session.source_images:
            raise HTTPException(
                status_code=400,
                detail="Add at least one source image.",
            )

        selected_target_ids = [
            image.id
            for image in session.target_images
            if image.id in session.selected_target_image_ids
        ]

        if selected_target_ids:
            target_images = [
                image
                for image in session.target_images
                if image.id in selected_target_ids
            ]
        else:
            active_target = session.active_target_image()

            if active_target is None:
                raise HTTPException(
                    status_code=400,
                    detail="Add and select a target image.",
                )

            target_images = [
                active_target,
            ]

        try:
            source_gallery = await asyncio.to_thread(
                build_source_gallery,
                [
                    str(
                        image.path
                    )
                    for image in session.source_images
                ],
                request.confidence_threshold,
            )

            if not source_gallery:
                raise ValueError(
                    "No source faces were detected."
                )

            session.source_faces = save_detected_faces(
                list(
                    source_gallery
                ),
                session.source_faces_directory,
                prefix="source",
            )

            detected_target_count = 0
            failed_targets: list[str] = []

            for target_image in target_images:
                analysis = session.ensure_target_analysis(
                    target_image.id
                )

                target_faces = await asyncio.to_thread(
                    save_target_detected_faces,
                    target_image.path,
                    session.target_faces_directory_for(
                        target_image.id
                    ),
                    request.confidence_threshold,
                )

                if not target_faces:
                    analysis.target_faces = []
                    analysis.face_mappings = {}
                    analysis.analysis_completed = False

                    session.result_paths.pop(
                        target_image.id,
                        None,
                    )

                    failed_targets.append(
                        target_image.name
                    )

                    continue

                previous_mappings = dict(
                    analysis.face_mappings
                )

                analysis.target_faces = target_faces

                analysis.face_mappings = {
                    face.index: previous_mappings.get(
                        face.index
                    )
                    for face in analysis.target_faces
                }

                analysis.analysis_completed = True

                session.result_paths.pop(
                    target_image.id,
                    None,
                )

                detected_target_count += 1

            if detected_target_count == 0:
                if failed_targets:
                    raise ValueError(
                        "No target faces were detected in selected target images."
                    )

                raise ValueError(
                    "No target faces were detected."
                )

            if session.active_target_image_id is None:
                session.active_target_image_id = (
                    target_images[0].id
                )

            if (
                session.active_target_image_id
                not in {
                    image.id
                    for image in target_images
                }
            ):
                session.active_target_image_id = (
                    target_images[0].id
                )

            for target_image in target_images:
                session.selected_target_image_ids.add(
                    target_image.id
                )

            return build_session_response(
                session
            )
        except HTTPException:
            raise
        except Exception as error:
            logger.exception(
                "Face detection failed"
            )

            raise HTTPException(
                status_code=500,
                detail=str(
                    error
                ),
            ) from error

    @application.put(
        "/api/sessions/{session_id}/mappings",
        response_model=SessionResponse,
    )
    async def update_mappings(
        session_id: str,
        request: FaceMappingRequest,
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        target_image_id = (
            request.target_image_id
            or session.active_target_image_id
        )

        if target_image_id is None:
            raise HTTPException(
                status_code=400,
                detail="Select a target image first.",
            )

        active_target = session.target_image_by_id(
            target_image_id
        )

        if active_target is None:
            raise HTTPException(
                status_code=404,
                detail="Target image not found.",
            )

        analysis = session.target_analyses.get(
            target_image_id
        )

        if (
            analysis is None
            or not analysis.analysis_completed
        ):
            raise HTTPException(
                status_code=400,
                detail="Detect faces first.",
            )

        updated_mappings: dict[
            int,
            int | None,
        ] = {
            face.index: None
            for face in analysis.target_faces
        }

        for mapping in request.mappings:
            if (
                mapping.target_face_index
                not in updated_mappings
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid target-face index: "
                        f"{mapping.target_face_index}"
                    ),
                )

            if (
                mapping.source_face_index
                is not None
                and not (
                    0
                    <= mapping.source_face_index
                    < len(
                        session.source_faces
                    )
                )
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid source-face index: "
                        f"{mapping.source_face_index}"
                    ),
                )

            updated_mappings[
                mapping.target_face_index
            ] = mapping.source_face_index

        analysis.face_mappings = updated_mappings

        session.result_paths.pop(
            active_target.id,
            None,
        )

        return build_session_response(
            session
        )

    @application.post(
        "/api/sessions/{session_id}/generate",
        response_model=SessionResponse,
    )
    async def generate(
        session_id: str,
        request: GenerationRequest,
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        try:
            await asyncio.to_thread(
                run_generation,
                session,
                request,
            )

            return build_session_response(
                session
            )
        except Exception as error:
            logger.exception(
                "Generation failed"
            )

            raise HTTPException(
                status_code=500,
                detail=str(
                    error
                ),
            ) from error

    @application.get(
        "/api/sessions/{session_id}/files/{relative_path:path}",
        include_in_schema=False,
    )
    async def session_file(
        session_id: str,
        relative_path: str,
    ) -> FileResponse:
        session = require_session(
            session_id
        )

        session_root = session.directory.resolve()

        requested_path = (
            session_root
            / relative_path
        ).resolve()

        try:
            requested_path.relative_to(
                session_root
            )
        except ValueError as error:
            raise HTTPException(
                status_code=403,
                detail="File access denied.",
            ) from error

        if not requested_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="File not found.",
            )

        media_type = (
            mimetypes.guess_type(
                requested_path.name
            )[0]
            or "application/octet-stream"
        )

        return FileResponse(
            requested_path,
            media_type=media_type,
        )

    @application.delete(
        "/api/sessions/{session_id}/results/{target_image_id}",
        response_model=SessionResponse,
    )
    async def delete_target_result(
        session_id: str,
        target_image_id: str,
    ) -> SessionResponse:
        session = require_session(
            session_id
        )

        target_image = session.target_image_by_id(
            target_image_id
        )

        if target_image is None:
            raise HTTPException(
                status_code=404,
                detail="Target image not found.",
            )

        result_path = session.result_paths.pop(
            target_image_id,
            None,
        )

        if result_path is not None:
            result_path.unlink(
                missing_ok=True
            )

        if (
            session.active_target_image_id
            == target_image_id
        ):
            remaining_result_ids = [
                image.id
                for image in session.target_images
                if image.id in session.result_paths
            ]

            if remaining_result_ids:
                session.active_target_image_id = (
                    remaining_result_ids[0]
                )

        return build_session_response(
            session
        )
    
    @application.get(
        "/api/sessions/{session_id}/download",
        include_in_schema=False,
    )
    async def download_result(
        session_id: str,
    ) -> FileResponse:
        session = require_session(
            session_id
        )

        items = generated_result_items(
            session
        )

        if not items:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No generated result is available."
                ),
            )

        if len(
            items
        ) == 1:
            target_image, result_path = items[0]

            return FileResponse(
                result_path,
                media_type="image/png",
                filename=(
                    f"{safe_download_stem(target_image.name)}"
                    "-face-swap.png"
                ),
            )

        archive_path = build_results_archive(
            session,
            items,
        )

        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename="face-swap-results.zip",
        )

    @application.get(
        "/api/sessions/{session_id}/download/{target_image_id}",
        include_in_schema=False,
    )
    async def download_target_result(
        session_id: str,
        target_image_id: str,
    ) -> FileResponse:
        session = require_session(
            session_id
        )

        target_image = session.target_image_by_id(
            target_image_id
        )

        if target_image is None:
            raise HTTPException(
                status_code=404,
                detail="Target image not found.",
            )

        result_path = session.result_paths.get(
            target_image_id
        )

        if (
            result_path is None
            or not result_path.is_file()
        ):
            raise HTTPException(
                status_code=404,
                detail=(
                    "No generated result is available."
                ),
            )

        return FileResponse(
            result_path,
            media_type="image/png",
            filename=(
                f"{Path(target_image.name).stem}-face-swap.png"
            ),
        )

    @application.delete(
        "/api/sessions/{session_id}",
    )
    async def delete_session(
        session_id: str,
    ) -> dict[str, bool]:
        require_session(
            session_id
        )

        session_store.delete(
            session_id
        )

        return {
            "deleted": True,
        }

    application.mount(
        "/static",
        StaticFiles(
            directory=STATIC_DIRECTORY,
        ),
        name="static",
    )

    return application