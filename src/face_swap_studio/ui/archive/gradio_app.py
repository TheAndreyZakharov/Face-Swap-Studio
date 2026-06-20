from __future__ import annotations

import base64
import mimetypes
import tempfile
from html import escape
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from src.face_swap_studio.ui.components import (
    create_processing_controls,
    model_cards_html,
)
from src.face_swap_studio.ui.state import UIState

from src.face_swap_studio.core.pipeline import process_single_pair
from src.face_swap_studio.domain.entities import ProcessingOptions
from src.face_swap_studio.services.image_service import (
    build_source_gallery,
    build_target_gallery,
    normalize_uploaded_paths,
)
from src.face_swap_studio.utils.logging import get_logger
from src.face_swap_studio.utils.paths import ensure_directories

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOGO_PATH = (
    PROJECT_ROOT
    / "assets"
    / "icons"
    / "logo.png"
)


CSS = """
:root {
    --studio-background: #0c0d10;
    --studio-surface: #15171b;
    --studio-surface-raised: #1b1e23;
    --studio-surface-hover: #22262c;
    --studio-border: rgba(255, 255, 255, 0.10);
    --studio-border-strong: rgba(255, 255, 255, 0.18);
    --studio-text: #f2f3f5;
    --studio-text-muted: #9ba1aa;
    --studio-text-subtle: #6f7680;
    --studio-success: #76b787;
    --studio-danger: #cf7676;
    --studio-radius-large: 24px;
    --studio-radius-medium: 16px;
}

html,
body {
    background: var(--studio-background) !important;
}

.gradio-container {
    max-width: 1540px !important;
    margin: 0 auto !important;
    padding: 22px 24px 80px !important;
    color: var(--studio-text) !important;
    background: var(--studio-background) !important;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif !important;
}

.gradio-container .prose {
    color: var(--studio-text);
}

.app-shell {
    width: 100%;
}

.app-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 22px 24px;
    margin-bottom: 18px;
    border: 1px solid var(--studio-border);
    border-radius: var(--studio-radius-large);
    background:
        linear-gradient(
            135deg,
            rgba(255, 255, 255, 0.055),
            rgba(255, 255, 255, 0.018)
        );
}

.app-logo {
    width: 58px;
    height: 58px;
    flex: 0 0 58px;
    object-fit: contain;
    border-radius: 14px;
}

.app-logo-placeholder {
    width: 58px;
    height: 58px;
    flex: 0 0 58px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--studio-border-strong);
    border-radius: 14px;
    color: var(--studio-text-muted);
    font-size: 21px;
    font-weight: 700;
}

.app-header-copy {
    min-width: 0;
}

.app-title {
    margin: 0;
    color: var(--studio-text);
    font-size: 30px;
    font-weight: 720;
    letter-spacing: -0.035em;
    line-height: 1.1;
}

.app-subtitle {
    max-width: 760px;
    margin: 7px 0 0;
    color: var(--studio-text-muted);
    font-size: 14px;
    line-height: 1.55;
}

.workflow-panel {
    padding: 20px !important;
    border: 1px solid var(--studio-border) !important;
    border-radius: var(--studio-radius-large) !important;
    background: var(--studio-surface) !important;
    box-shadow: none !important;
}

.section-header {
    margin-bottom: 15px;
}

.section-number {
    display: inline-flex;
    min-width: 27px;
    height: 27px;
    align-items: center;
    justify-content: center;
    margin-right: 9px;
    border: 1px solid var(--studio-border-strong);
    border-radius: 999px;
    color: var(--studio-text-muted);
    font-size: 12px;
    font-weight: 700;
    vertical-align: 2px;
}

.section-title {
    color: var(--studio-text);
    font-size: 19px;
    font-weight: 680;
    letter-spacing: -0.02em;
}

.section-description {
    margin: 6px 0 0 38px;
    color: var(--studio-text-muted);
    font-size: 13px;
    line-height: 1.5;
}

.upload-column {
    min-width: 0;
}

.upload-heading {
    margin-bottom: 8px;
    color: var(--studio-text);
    font-size: 14px;
    font-weight: 650;
}

.upload-description {
    min-height: 38px;
    margin-bottom: 10px;
    color: var(--studio-text-muted);
    font-size: 12px;
    line-height: 1.45;
}

.upload-box {
    min-height: 148px;
}

.upload-box > div {
    border-radius: var(--studio-radius-medium) !important;
}

.detect-button,
.generate-button {
    min-height: 50px !important;
    border-radius: 13px !important;
    font-weight: 720 !important;
}

.secondary-button {
    min-height: 44px !important;
    border-radius: 12px !important;
}

.face-gallery {
    min-height: 270px;
    border-radius: var(--studio-radius-medium);
}

.face-gallery button {
    overflow: hidden;
    border-radius: 12px !important;
}

.face-gallery img {
    transition:
        transform 150ms ease,
        filter 150ms ease;
}

.face-gallery button:hover img {
    transform: scale(1.025);
}

.selection-note {
    min-height: 46px;
    margin-top: 10px;
    padding: 11px 13px;
    border: 1px solid var(--studio-border);
    border-radius: 12px;
    background: var(--studio-surface-raised);
    color: var(--studio-text-muted);
    font-size: 12px;
    line-height: 1.5;
}

.mapping-toolbar {
    margin-top: 12px;
}

.mapping-panel {
    min-height: 180px;
    margin-top: 14px;
    padding: 14px;
    overflow-x: auto;
    border: 1px solid var(--studio-border);
    border-radius: var(--studio-radius-medium);
    background: var(--studio-surface-raised);
}

.mapping-empty {
    display: flex;
    min-height: 140px;
    align-items: center;
    justify-content: center;
    padding: 20px;
    color: var(--studio-text-subtle);
    text-align: center;
    font-size: 13px;
    line-height: 1.55;
}

.mapping-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.mapping-row {
    display: grid;
    grid-template-columns:
        74px minmax(130px, 1fr)
        44px
        74px minmax(130px, 1fr);
    gap: 12px;
    align-items: center;
    padding: 10px;
    border: 1px solid var(--studio-border);
    border-radius: 14px;
    background: var(--studio-surface);
}

.mapping-thumbnail {
    width: 74px;
    height: 74px;
    object-fit: cover;
    border: 1px solid var(--studio-border);
    border-radius: 12px;
    background: #090a0c;
}

.mapping-thumbnail-empty {
    display: flex;
    width: 74px;
    height: 74px;
    align-items: center;
    justify-content: center;
    border: 1px dashed var(--studio-border-strong);
    border-radius: 12px;
    color: var(--studio-text-subtle);
    font-size: 22px;
}

.mapping-label {
    color: var(--studio-text);
    font-size: 13px;
    font-weight: 650;
}

.mapping-caption {
    margin-top: 4px;
    color: var(--studio-text-muted);
    font-size: 11px;
    line-height: 1.4;
}

.mapping-arrow {
    color: var(--studio-text-subtle);
    font-size: 23px;
    text-align: center;
}

.model-card-list {
    display: grid;
    grid-template-columns: repeat(
        auto-fit,
        minmax(230px, 1fr)
    );
    gap: 10px;
    margin-top: 12px;
}

.model-card {
    padding: 13px;
    border: 1px solid var(--studio-border);
    border-radius: 13px;
    background: var(--studio-surface-raised);
}

.model-card-header {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    justify-content: space-between;
}

.model-card-name {
    color: var(--studio-text);
    font-size: 13px;
    font-weight: 680;
}

.model-card-description {
    margin-top: 8px;
    color: var(--studio-text-muted);
    font-size: 11px;
    line-height: 1.5;
}

.model-status {
    flex: 0 0 auto;
    padding: 3px 7px;
    border-radius: 999px;
    font-size: 9px;
    font-weight: 720;
    letter-spacing: 0.045em;
    text-transform: uppercase;
}

.model-status-ready {
    background: rgba(118, 183, 135, 0.14);
    color: var(--studio-success);
}

.model-status-unavailable {
    background: rgba(207, 118, 118, 0.13);
    color: var(--studio-danger);
}

.result-frame {
    min-height: 560px;
    overflow: hidden;
    border: 1px solid var(--studio-border);
    border-radius: var(--studio-radius-medium);
    background:
        linear-gradient(
            45deg,
            rgba(255, 255, 255, 0.017) 25%,
            transparent 25%
        ),
        linear-gradient(
            -45deg,
            rgba(255, 255, 255, 0.017) 25%,
            transparent 25%
        ),
        linear-gradient(
            45deg,
            transparent 75%,
            rgba(255, 255, 255, 0.017) 75%
        ),
        linear-gradient(
            -45deg,
            transparent 75%,
            rgba(255, 255, 255, 0.017) 75%
        );
    background-position:
        0 0,
        0 8px,
        8px -8px,
        -8px 0;
    background-size: 16px 16px;
}

.result-frame > div {
    min-height: 560px;
}

.status-box textarea {
    font-family:
        SFMono-Regular,
        Menlo,
        Monaco,
        Consolas,
        monospace !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}

footer {
    display: none !important;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 12px 12px 60px !important;
    }

    .app-header {
        padding: 17px;
    }

    .app-logo,
    .app-logo-placeholder {
        width: 48px;
        height: 48px;
        flex-basis: 48px;
    }

    .app-title {
        font-size: 25px;
    }

    .workflow-panel {
        padding: 14px !important;
    }

    .mapping-row {
        grid-template-columns:
            58px minmax(90px, 1fr)
            28px
            58px minmax(90px, 1fr);
        gap: 8px;
    }

    .mapping-thumbnail,
    .mapping-thumbnail-empty {
        width: 58px;
        height: 58px;
    }

    .result-frame,
    .result-frame > div {
        min-height: 420px;
    }
}
"""


def file_to_data_uri(
    path: Path,
) -> str:
    if not path.is_file():
        return ""

    mime_type = (
        mimetypes.guess_type(
            path.name
        )[0]
        or "image/png"
    )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode(
        "ascii"
    )

    return f"data:{mime_type};base64,{encoded}"


def logo_data_uri() -> str:
    return file_to_data_uri(
        LOGO_PATH
    )


def page_head_html() -> str:
    logo_uri = logo_data_uri()

    if not logo_uri:
        return """
<meta
    name="theme-color"
    content="#0c0d10"
>
"""

    return f"""
<link
    rel="icon"
    type="image/png"
    href="{logo_uri}"
>
<link
    rel="apple-touch-icon"
    href="{logo_uri}"
>
<meta
    name="theme-color"
    content="#0c0d10"
>
"""


def header_html() -> str:
    logo_uri = logo_data_uri()

    if logo_uri:
        logo_element = (
            f'<img class="app-logo" '
            f'src="{logo_uri}" '
            f'alt="Face Swap Studio">'
        )
    else:
        logo_element = (
            '<div class="app-logo-placeholder">FS</div>'
        )

    return f"""
<div class="app-shell">
    <header class="app-header">
        {logo_element}
        <div class="app-header-copy">
            <h1 class="app-title">
                Face Swap Studio
            </h1>
            <p class="app-subtitle">
                Detect faces, define precise replacement mappings,
                select a model and generate the final image locally.
            </p>
        </div>
    </header>
</div>
"""


def section_header(
    number: str,
    title: str,
    description: str,
) -> str:
    return f"""
<div class="section-header">
    <div>
        <span class="section-number">
            {escape(number)}
        </span>
        <span class="section-title">
            {escape(title)}
        </span>
    </div>
    <p class="section-description">
        {escape(description)}
    </p>
</div>
"""


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


def gallery_image_to_path(
    item: Any,
    *,
    prefix: str,
) -> str | None:
    value = normalize_gallery_item(
        item
    )

    if value is None:
        return None

    if isinstance(
        value,
        Path,
    ):
        path = value.expanduser().resolve()

        if path.is_file():
            return str(path)

        return None

    if isinstance(
        value,
        str,
    ):
        path = Path(
            value
        ).expanduser().resolve()

        if path.is_file():
            return str(path)

        return None

    if isinstance(
        value,
        np.ndarray,
    ):
        array = value

        if (
            array.ndim != 3
            or array.shape[2] not in {
                3,
                4,
            }
        ):
            return None

        temporary_directory = (
            Path(
                tempfile.gettempdir()
            )
            / "face-swap-studio-ui"
        )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            temporary_directory
            / f"{prefix}-{id(value)}.png"
        )

        if array.shape[2] == 4:
            converted = cv2.cvtColor(
                array,
                cv2.COLOR_RGBA2BGRA,
            )
        else:
            converted = cv2.cvtColor(
                array,
                cv2.COLOR_RGB2BGR,
            )

        if cv2.imwrite(
            str(destination),
            converted,
        ):
            return str(destination)

    return None


def gallery_item_data_uri(
    item: Any,
) -> str:
    path_value = gallery_image_to_path(
        item,
        prefix="preview",
    )

    if path_value is None:
        return ""

    return file_to_data_uri(
        Path(path_value)
    )


def merge_paths(
    existing_paths: list[str],
    uploaded_value: Any,
) -> list[str]:
    uploaded_paths = normalize_uploaded_paths(
        uploaded_value
    )

    merged: list[str] = []
    seen: set[str] = set()

    for value in [
        *existing_paths,
        *uploaded_paths,
    ]:
        normalized = str(
            Path(value).expanduser().resolve()
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )
        merged.append(
            normalized
        )

    return merged


def mapping_html(
    state: UIState | None,
) -> str:
    if (
        state is None
        or not state.analysis_completed
        or not state.target_faces
    ):
        return """
<div class="mapping-panel">
    <div class="mapping-empty">
        Detect faces to create replacement mappings.
    </div>
</div>
"""

    rows: list[str] = []

    for target_index, target_item in enumerate(
        state.target_faces
    ):
        source_index = state.face_mappings.get(
            target_index
        )

        target_uri = gallery_item_data_uri(
            target_item
        )

        if target_uri:
            target_image = (
                f'<img class="mapping-thumbnail" '
                f'src="{target_uri}" '
                f'alt="Target face {target_index + 1}">'
            )
        else:
            target_image = (
                '<div class="mapping-thumbnail-empty">'
                '—'
                '</div>'
            )

        if (
            source_index is not None
            and 0 <= source_index < len(
                state.source_faces
            )
        ):
            source_uri = gallery_item_data_uri(
                state.source_faces[
                    source_index
                ]
            )

            if source_uri:
                source_image = (
                    f'<img class="mapping-thumbnail" '
                    f'src="{source_uri}" '
                    f'alt="Source face {source_index + 1}">'
                )
            else:
                source_image = (
                    '<div class="mapping-thumbnail-empty">'
                    '—'
                    '</div>'
                )

            source_label = (
                f"Source face {source_index + 1}"
            )
            source_caption = (
                "This identity will replace the target face."
            )
        else:
            source_image = (
                '<div class="mapping-thumbnail-empty">'
                '—'
                '</div>'
            )
            source_label = "Leave unchanged"
            source_caption = (
                "No replacement will be applied to this face."
            )

        rows.append(
            f"""
<div class="mapping-row">
    {target_image}
    <div>
        <div class="mapping-label">
            Target face {target_index + 1}
        </div>
        <div class="mapping-caption">
            Detected target identity
        </div>
    </div>

    <div class="mapping-arrow">
        →
    </div>

    {source_image}
    <div>
        <div class="mapping-label">
            {escape(source_label)}
        </div>
        <div class="mapping-caption">
            {escape(source_caption)}
        </div>
    </div>
</div>
"""
        )

    return f"""
<div class="mapping-panel">
    <div class="mapping-list">
        {''.join(rows)}
    </div>
</div>
"""


def upload_summary(
    state: UIState | None,
) -> str:
    if state is None:
        return "No images have been added."

    return (
        f"Source images: {len(state.source_paths)}. "
        f"Target images: {len(state.target_paths)}."
    )


def add_source_images(
    source_files: Any,
    state: UIState | None,
) -> tuple[
    UIState,
    None,
    str,
]:
    try:
        updated_state = state or UIState()

        updated_state.source_paths = merge_paths(
            updated_state.source_paths,
            source_files,
        )

        updated_state.reset_analysis()

        return (
            updated_state,
            None,
            upload_summary(
                updated_state
            ),
        )
    except Exception as error:
        logger.exception(
            "Failed to add source images"
        )

        raise gr.Error(
            str(error)
        ) from error


def add_target_images(
    target_files: Any,
    state: UIState | None,
) -> tuple[
    UIState,
    None,
    str,
]:
    try:
        updated_state = state or UIState()

        updated_state.target_paths = merge_paths(
            updated_state.target_paths,
            target_files,
        )

        updated_state.reset_analysis()

        return (
            updated_state,
            None,
            upload_summary(
                updated_state
            ),
        )
    except Exception as error:
        logger.exception(
            "Failed to add target images"
        )

        raise gr.Error(
            str(error)
        ) from error


def detect_faces(
    confidence_threshold: float,
    state: UIState | None,
) -> tuple[
    UIState,
    list[Any],
    list[Any],
    str,
    str,
]:
    try:
        updated_state = state or UIState()

        if not updated_state.source_paths:
            raise ValueError(
                "Add at least one source image."
            )

        if not updated_state.target_paths:
            raise ValueError(
                "Add at least one target image."
            )

        source_gallery = build_source_gallery(
            updated_state.source_paths,
            float(
                confidence_threshold
            ),
        )

        target_gallery = build_target_gallery(
            updated_state.target_paths,
            float(
                confidence_threshold
            ),
        )

        if not source_gallery:
            raise ValueError(
                "No source faces were detected."
            )

        if not target_gallery:
            raise ValueError(
                "No target faces were detected."
            )

        updated_state.source_faces = list(
            source_gallery
        )
        updated_state.target_faces = list(
            target_gallery
        )

        updated_state.selected_source_face_index = 0
        updated_state.selected_target_face_index = 0
        updated_state.selected_target_image_index = 0

        updated_state.face_mappings = {
            target_index: None
            for target_index in range(
                len(
                    updated_state.target_faces
                )
            )
        }

        updated_state.analysis_completed = True

        status = (
            "Detection complete. "
            f"Found {len(updated_state.source_faces)} "
            "source face(s) and "
            f"{len(updated_state.target_faces)} "
            "target face(s). "
            "Select a target face and a source face, "
            "then assign the replacement."
        )

        return (
            updated_state,
            source_gallery,
            target_gallery,
            mapping_html(
                updated_state
            ),
            status,
        )
    except Exception as error:
        logger.exception(
            "Face detection failed"
        )

        raise gr.Error(
            str(error)
        ) from error


def event_index(
    event: gr.SelectData,
) -> int:
    index = event.index

    if isinstance(
        index,
        tuple,
    ):
        return int(
            index[0]
        )

    return int(
        index
    )


def select_source_face(
    state: UIState | None,
    event: gr.SelectData,
) -> tuple[
    UIState,
    str,
]:
    updated_state = state or UIState()

    selected_index = event_index(
        event
    )

    if not (
        0
        <= selected_index
        < len(
            updated_state.source_faces
        )
    ):
        raise gr.Error(
            "The selected source face is no longer available."
        )

    updated_state.selected_source_face_index = (
        selected_index
    )

    return (
        updated_state,
        (
            f"Selected source face {selected_index + 1}. "
            "Select a target face and use Assign replacement."
        ),
    )


def select_target_face(
    state: UIState | None,
    event: gr.SelectData,
) -> tuple[
    UIState,
    str,
]:
    updated_state = state or UIState()

    selected_index = event_index(
        event
    )

    if not (
        0
        <= selected_index
        < len(
            updated_state.target_faces
        )
    ):
        raise gr.Error(
            "The selected target face is no longer available."
        )

    updated_state.selected_target_face_index = (
        selected_index
    )

    return (
        updated_state,
        (
            f"Selected target face {selected_index + 1}. "
            "Select a source face or choose Leave unchanged."
        ),
    )


def source_selection_html(
    state: UIState | None,
) -> str:
    if (
        state is not None
        and state.selected_source_face_index is not None
    ):
        return (
            '<div class="selection-note">'
            "Selected source face "
            f"{state.selected_source_face_index + 1}."
            "</div>"
        )

    return (
        '<div class="selection-note">'
        "No source face selected."
        "</div>"
    )


def target_selection_html(
    state: UIState | None,
) -> str:
    if (
        state is not None
        and state.selected_target_face_index is not None
    ):
        return (
            '<div class="selection-note">'
            "Selected target face "
            f"{state.selected_target_face_index + 1}."
            "</div>"
        )

    return (
        '<div class="selection-note">'
        "No target face selected."
        "</div>"
    )


def assign_selected_mapping(
    state: UIState | None,
) -> tuple[
    UIState,
    str,
    str,
]:
    try:
        updated_state = state or UIState()

        target_index = (
            updated_state.selected_target_face_index
        )
        source_index = (
            updated_state.selected_source_face_index
        )

        if target_index is None:
            raise ValueError(
                "Select a target face first."
            )

        if source_index is None:
            raise ValueError(
                "Select a source face first."
            )

        updated_state.face_mappings[
            target_index
        ] = source_index

        return (
            updated_state,
            mapping_html(
                updated_state
            ),
            (
                f"Target face {target_index + 1} "
                "will be replaced with source face "
                f"{source_index + 1}."
            ),
        )
    except Exception as error:
        raise gr.Error(
            str(error)
        ) from error


def leave_selected_unchanged(
    state: UIState | None,
) -> tuple[
    UIState,
    str,
    str,
]:
    try:
        updated_state = state or UIState()

        target_index = (
            updated_state.selected_target_face_index
        )

        if target_index is None:
            raise ValueError(
                "Select a target face first."
            )

        updated_state.face_mappings[
            target_index
        ] = None

        return (
            updated_state,
            mapping_html(
                updated_state
            ),
            (
                f"Target face {target_index + 1} "
                "will remain unchanged."
            ),
        )
    except Exception as error:
        raise gr.Error(
            str(error)
        ) from error


def clear_all_mappings(
    state: UIState | None,
) -> tuple[
    UIState,
    str,
    str,
]:
    updated_state = state or UIState()

    updated_state.face_mappings = {
        target_index: None
        for target_index in range(
            len(
                updated_state.target_faces
            )
        )
    }

    return (
        updated_state,
        mapping_html(
            updated_state
        ),
        "All target faces are set to remain unchanged.",
    )


def resolve_source_path(
    state: UIState,
    source_face_index: int,
) -> str:
    if (
        0
        <= source_face_index
        < len(
            state.source_faces
        )
    ):
        face_path = gallery_image_to_path(
            state.source_faces[
                source_face_index
            ],
            prefix=(
                f"source-face-"
                f"{source_face_index}"
            ),
        )

        if face_path is not None:
            return face_path

    if not state.source_paths:
        raise ValueError(
            "No source image is available."
        )

    fallback_index = min(
        source_face_index,
        len(
            state.source_paths
        )
        - 1,
    )

    return state.source_paths[
        fallback_index
    ]


def process_images(
    state: UIState | None,
    model_id: str,
    enhance_faces: bool,
    enhancement_weight: float,
    upscale_image: bool,
    upscale_factor: float,
    tile_size: int,
    progress: gr.Progress = gr.Progress(  # noqa: B008
        track_tqdm=False
    ),
) -> tuple[
    str,
    str,
    str,
]:
    try:
        if state is None:
            raise ValueError(
                "Add images and detect faces first."
            )

        if not state.analysis_completed:
            raise ValueError(
                "Detect faces before generating."
            )

        if not state.target_paths:
            raise ValueError(
                "No target image is available."
            )

        active_mappings = [
            (
                target_index,
                source_index,
            )
            for target_index, source_index
            in sorted(
                state.face_mappings.items()
            )
            if source_index is not None
        ]

        if not active_mappings:
            raise ValueError(
                "Assign at least one target face "
                "to a source face."
            )

        options = ProcessingOptions(
            model_id=str(
                model_id
            ),
            enhance_faces=False,
            face_enhancement_weight=float(
                enhancement_weight
            ),
            upscale_image=False,
            upscale_factor=float(
                upscale_factor
            ),
            tile_size=int(
                tile_size
            ),
        )

        target_image_index = min(
            max(
                0,
                state.selected_target_image_index,
            ),
            len(
                state.target_paths
            )
            - 1,
        )

        current_target_path = (
            state.target_paths[
                target_image_index
            ]
        )

        total_steps = len(
            active_mappings
        )

        for step_index, (
            target_face_index,
            source_face_index,
        ) in enumerate(
            active_mappings,
            start=1,
        ):
            progress(
                (
                    step_index
                    - 1
                )
                / max(
                    total_steps,
                    1,
                ),
                desc=(
                    "Replacing target face "
                    f"{target_face_index + 1}"
                ),
            )

            source_path = resolve_source_path(
                state,
                source_face_index,
            )

            current_target_path = str(
                process_single_pair(
                    source_path=source_path,
                    target_path=current_target_path,
                    options=options,
                    target_face_index=target_face_index,
                )
            )

        if (
            enhance_faces
            or upscale_image
        ):
            progress(
                0.9,
                desc="Applying post-processing",
            )

            postprocessing_options = ProcessingOptions(
                model_id=str(
                    model_id
                ),
                enhance_faces=bool(
                    enhance_faces
                ),
                face_enhancement_weight=float(
                    enhancement_weight
                ),
                upscale_image=bool(
                    upscale_image
                ),
                upscale_factor=float(
                    upscale_factor
                ),
                tile_size=int(
                    tile_size
                ),
            )

            from src.face_swap_studio.core.pipeline import (
                apply_postprocessing,
            )

            current_target_path = str(
                apply_postprocessing(
                    result_path=Path(
                        current_target_path
                    ),
                    options=postprocessing_options,
                )
            )

        progress(
            1.0,
            desc="Generation complete",
        )

        status = (
            "Generation complete. "
            f"Model: {model_id}. "
            "Applied replacements: "
            f"{len(active_mappings)}."
        )

        return (
            current_target_path,
            current_target_path,
            status,
        )
    except Exception as error:
        logger.exception(
            "Image generation failed"
        )

        raise gr.Error(
            str(error)
        ) from error


def reset_session() -> tuple[
    UIState,
    None,
    None,
    list[Any],
    list[Any],
    str,
    None,
    None,
    str,
    str,
    str,
    str,
]:
    state = UIState()

    return (
        state,
        None,
        None,
        [],
        [],
        mapping_html(
            state
        ),
        None,
        None,
        "Session cleared.",
        "No images have been added.",
        source_selection_html(
            state
        ),
        target_selection_html(
            state
        ),
    )


def create_app() -> gr.Blocks:
    ensure_directories()

    with gr.Blocks(
        title="Face Swap Studio",
    ) as demo:
        state = gr.State(
            UIState()
        )

        gr.HTML(
            header_html()
        )

        with gr.Column(
            elem_classes=[
                "workflow-panel",
            ],
        ):
            gr.HTML(
                section_header(
                    "1",
                    "Add images",
                    (
                        "Add source identities and target images "
                        "independently. Additional images can be "
                        "added at any time before face detection."
                    ),
                )
            )

            with gr.Row(
                equal_height=True
            ):
                with gr.Column(
                    elem_classes=[
                        "upload-column",
                    ],
                ):
                    gr.HTML(
                        """
<div class="upload-heading">
    Source images
</div>
<div class="upload-description">
    Images containing the identities that can be used
    as replacements.
</div>
"""
                    )

                    source_files = gr.File(
                        label="Add source images",
                        file_count="multiple",
                        file_types=[
                            "image",
                        ],
                        type="filepath",
                        elem_classes=[
                            "upload-box",
                        ],
                    )

                    add_source_button = gr.Button(
                        "Add source images",
                        variant="secondary",
                        elem_classes=[
                            "secondary-button",
                        ],
                    )

                with gr.Column(
                    elem_classes=[
                        "upload-column",
                    ],
                ):
                    gr.HTML(
                        """
<div class="upload-heading">
    Target images
</div>
<div class="upload-description">
    Images containing the faces that may be replaced.
</div>
"""
                    )

                    target_files = gr.File(
                        label="Add target images",
                        file_count="multiple",
                        file_types=[
                            "image",
                        ],
                        type="filepath",
                        elem_classes=[
                            "upload-box",
                        ],
                    )

                    add_target_button = gr.Button(
                        "Add target images",
                        variant="secondary",
                        elem_classes=[
                            "secondary-button",
                        ],
                    )

            upload_status = gr.Textbox(
                label="Uploaded images",
                value="No images have been added.",
                interactive=False,
                lines=1,
            )

            with gr.Row():
                confidence_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=0.5,
                    step=0.05,
                    label="Face detection confidence",
                    info=(
                        "Lower values detect more faces but may "
                        "include false detections."
                    ),
                    scale=3,
                )

                detect_button = gr.Button(
                    "Detect faces",
                    variant="primary",
                    scale=1,
                    elem_classes=[
                        "detect-button",
                    ],
                )

        with gr.Column(
            elem_classes=[
                "workflow-panel",
            ],
        ):
            gr.HTML(
                section_header(
                    "2",
                    "Select detected faces",
                    (
                        "Select one target face and one source face. "
                        "All thumbnails can be opened in a larger "
                        "preview."
                    ),
                )
            )

            with gr.Row(
                equal_height=True
            ):
                with gr.Column():
                    gr.Markdown(
                        "### Source identities"
                    )

                    source_gallery = gr.Gallery(
                        label=None,
                        columns=5,
                        rows=2,
                        height=300,
                        object_fit="cover",
                        allow_preview=True,
                        preview=True,
                        elem_classes=[
                            "face-gallery",
                        ],
                    )

                    source_selection_status = gr.HTML(
                        source_selection_html(
                            UIState()
                        )
                    )

                with gr.Column():
                    gr.Markdown(
                        "### Target faces"
                    )

                    target_gallery = gr.Gallery(
                        label=None,
                        columns=5,
                        rows=2,
                        height=300,
                        object_fit="cover",
                        allow_preview=True,
                        preview=True,
                        elem_classes=[
                            "face-gallery",
                        ],
                    )

                    target_selection_status = gr.HTML(
                        target_selection_html(
                            UIState()
                        )
                    )

        with gr.Column(
            elem_classes=[
                "workflow-panel",
            ],
        ):
            gr.HTML(
                section_header(
                    "3",
                    "Define replacements",
                    (
                        "Assign the selected source identity to the "
                        "selected target face, or explicitly leave "
                        "the target face unchanged."
                    ),
                )
            )

            with gr.Row(
                elem_classes=[
                    "mapping-toolbar",
                ],
            ):
                assign_button = gr.Button(
                    "Assign replacement",
                    variant="primary",
                    elem_classes=[
                        "secondary-button",
                    ],
                )

                unchanged_button = gr.Button(
                    "Leave selected face unchanged",
                    variant="secondary",
                    elem_classes=[
                        "secondary-button",
                    ],
                )

                clear_mappings_button = gr.Button(
                    "Clear all mappings",
                    variant="secondary",
                    elem_classes=[
                        "secondary-button",
                    ],
                )

            mappings_display = gr.HTML(
                mapping_html(
                    UIState()
                )
            )

        with gr.Row(
            equal_height=False
        ):
            with gr.Column(
                scale=2,
                elem_classes=[
                    "workflow-panel",
                ],
            ):
                gr.HTML(
                    section_header(
                        "4",
                        "Choose a model",
                        (
                            "Select the face-swap model and optional "
                            "post-processing settings."
                        ),
                    )
                )

                controls = create_processing_controls()

                gr.HTML(
                    model_cards_html()
                )

            with gr.Column(
                scale=3,
                elem_classes=[
                    "workflow-panel",
                ],
            ):
                gr.HTML(
                    section_header(
                        "5",
                        "Generate",
                        (
                            "The result remains available during the "
                            "current application session."
                        ),
                    )
                )

                generate_button = gr.Button(
                    "Generate image",
                    variant="primary",
                    size="lg",
                    elem_classes=[
                        "generate-button",
                    ],
                )

                result_image = gr.Image(
                    label="Generated result",
                    type="filepath",
                    height=560,
                    interactive=False,
                    elem_classes=[
                        "result-frame",
                    ],
                )

                result_file = gr.File(
                    label="Save generated image",
                    interactive=False,
                )

        with gr.Column(
            elem_classes=[
                "workflow-panel",
            ],
        ):
            gr.HTML(
                section_header(
                    "6",
                    "Session",
                    (
                        "Review the current operation status or clear "
                        "all uploaded images and mappings."
                    ),
                )
            )

            status = gr.Textbox(
                label="Status",
                value="Ready.",
                interactive=False,
                lines=3,
                elem_classes=[
                    "status-box",
                ],
            )

            reset_button = gr.Button(
                "Clear session",
                variant="secondary",
                elem_classes=[
                    "secondary-button",
                ],
            )

        add_source_button.click(
            fn=add_source_images,
            inputs=[
                source_files,
                state,
            ],
            outputs=[
                state,
                source_files,
                upload_status,
            ],
            show_progress="minimal",
        )

        add_target_button.click(
            fn=add_target_images,
            inputs=[
                target_files,
                state,
            ],
            outputs=[
                state,
                target_files,
                upload_status,
            ],
            show_progress="minimal",
        )

        detect_button.click(
            fn=detect_faces,
            inputs=[
                confidence_threshold,
                state,
            ],
            outputs=[
                state,
                source_gallery,
                target_gallery,
                mappings_display,
                status,
            ],
            show_progress="full",
        ).then(
            fn=source_selection_html,
            inputs=[
                state,
            ],
            outputs=[
                source_selection_status,
            ],
            show_progress="hidden",
        ).then(
            fn=target_selection_html,
            inputs=[
                state,
            ],
            outputs=[
                target_selection_status,
            ],
            show_progress="hidden",
        )

        source_gallery.select(
            fn=select_source_face,
            inputs=[
                state,
            ],
            outputs=[
                state,
                status,
            ],
            show_progress="hidden",
        ).then(
            fn=source_selection_html,
            inputs=[
                state,
            ],
            outputs=[
                source_selection_status,
            ],
            show_progress="hidden",
        )

        target_gallery.select(
            fn=select_target_face,
            inputs=[
                state,
            ],
            outputs=[
                state,
                status,
            ],
            show_progress="hidden",
        ).then(
            fn=target_selection_html,
            inputs=[
                state,
            ],
            outputs=[
                target_selection_status,
            ],
            show_progress="hidden",
        )

        assign_button.click(
            fn=assign_selected_mapping,
            inputs=[
                state,
            ],
            outputs=[
                state,
                mappings_display,
                status,
            ],
            show_progress="minimal",
        )

        unchanged_button.click(
            fn=leave_selected_unchanged,
            inputs=[
                state,
            ],
            outputs=[
                state,
                mappings_display,
                status,
            ],
            show_progress="minimal",
        )

        clear_mappings_button.click(
            fn=clear_all_mappings,
            inputs=[
                state,
            ],
            outputs=[
                state,
                mappings_display,
                status,
            ],
            show_progress="minimal",
        )

        generate_button.click(
            fn=process_images,
            inputs=[
                state,
                controls[
                    "swap_model"
                ],
                controls[
                    "enhance_faces"
                ],
                controls[
                    "enhancement_weight"
                ],
                controls[
                    "upscale_image"
                ],
                controls[
                    "upscale_factor"
                ],
                controls[
                    "tile_size"
                ],
            ],
            outputs=[
                result_image,
                result_file,
                status,
            ],
            show_progress="full",
        )

        reset_button.click(
            fn=reset_session,
            inputs=[],
            outputs=[
                state,
                source_files,
                target_files,
                source_gallery,
                target_gallery,
                mappings_display,
                result_image,
                result_file,
                status,
                upload_status,
                source_selection_status,
                target_selection_status,
            ],
            show_progress="minimal",
        )

    return demo