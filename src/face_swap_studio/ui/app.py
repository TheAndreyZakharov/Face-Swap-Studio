from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.face_swap_studio.models.model_manager import (
    model_status,
)
from src.face_swap_studio.services.batch_service import (
    clear_temporary_files,
    create_analysis,
    process_analysis,
)
from src.face_swap_studio.services.image_service import (
    normalize_uploaded_paths,
    target_face_gallery,
    uploaded_image_gallery,
)
from src.face_swap_studio.ui.components import (
    APP_CSS,
    create_processing_controls,
)
from src.face_swap_studio.ui.state import (
    FileCollectionState,
    StudioSessionState,
)
from src.face_swap_studio.utils.logging import get_logger
from src.face_swap_studio.utils.paths import ensure_directories

logger = get_logger(__name__)


def model_status_html() -> str:
    statuses = model_status()

    items = []

    for name, available in statuses.items():
        icon = "●"
        state = "готово" if available else "не найдено"
        css_class = "ready" if available else "missing"

        items.append(f'<span class="{css_class}">{icon} {name}: {state}</span>')

    return "<div class='model-status'>" + " &nbsp; ".join(items) + "</div>"


def add_uploaded_files(
    files,
    collection: FileCollectionState | None,
):
    state = collection or FileCollectionState()
    state.add(normalize_uploaded_paths(files))

    return (
        state,
        uploaded_image_gallery(state.paths),
        None,
        f"В коллекции файлов: {len(state.paths)}",
    )


def select_uploaded_file(
    collection: FileCollectionState,
    event: gr.SelectData,
):
    collection.selected_index = int(event.index)

    path = collection.paths[collection.selected_index]

    return (
        collection,
        f"Выбран файл: {Path(path).name}",
    )


def remove_uploaded_file(
    collection: FileCollectionState | None,
):
    if collection is None:
        collection = FileCollectionState()

    collection.remove_selected()

    return (
        collection,
        uploaded_image_gallery(collection.paths),
        f"В коллекции файлов: {len(collection.paths)}",
    )


def analyse_ui(
    source_collection: FileCollectionState | None,
    target_collection: FileCollectionState | None,
    studio_state: StudioSessionState | None,
    confidence_threshold: float,
):
    try:
        if source_collection is None:
            source_collection = FileCollectionState()

        if target_collection is None:
            target_collection = FileCollectionState()

        state = studio_state or StudioSessionState()

        (
            analysis,
            sources_gallery,
            target_preview,
            target_faces,
            flattened_faces,
            status,
        ) = create_analysis(
            source_paths=source_collection.paths,
            target_paths=target_collection.paths,
            confidence_threshold=float(confidence_threshold),
        )

        state.analysis = analysis
        state.flattened_target_faces = flattened_faces
        state.visual_assignments.assignments.clear()
        state.visual_assignments.selected_source_index = None
        state.visual_assignments.selected_target_face_index = None

        return (
            state,
            sources_gallery,
            target_preview,
            target_faces,
            "Исходное лицо не выбрано",
            "Целевое лицо не выбрано",
            status,
        )
    except Exception as error:
        logger.exception("Ошибка анализа изображений")
        raise gr.Error(str(error)) from error


def select_source_face(
    studio_state: StudioSessionState,
    event: gr.SelectData,
):
    if studio_state.analysis is None:
        raise gr.Error("Сначала выполните поиск лиц.")

    source_index = int(event.index)

    studio_state.visual_assignments.selected_source_index = source_index

    source = studio_state.analysis.sources[source_index]

    return (
        studio_state,
        f"Выбран источник: {source.path.name}",
    )


def select_target_face(
    studio_state: StudioSessionState,
    event: gr.SelectData,
):
    if studio_state.analysis is None:
        raise gr.Error("Сначала выполните поиск лиц.")

    flattened_index = int(event.index)

    studio_state.visual_assignments.selected_target_face_index = flattened_index

    target_index, face_index = studio_state.flattened_target_faces[flattened_index]

    target = studio_state.analysis.targets[target_index]

    return (
        studio_state,
        (f"Выбрано: {target.path.name}, лицо {face_index + 1}"),
    )


def assign_selected_face(
    studio_state: StudioSessionState,
):
    try:
        if studio_state.analysis is None:
            raise ValueError("Сначала выполните поиск лиц.")

        studio_state.visual_assignments.assign(studio_state.flattened_target_faces)

        gallery, flattened = target_face_gallery(
            studio_state.analysis.targets,
            studio_state.visual_assignments.assignments,
        )

        studio_state.flattened_target_faces = flattened

        return (
            studio_state,
            gallery,
            "Назначение сохранено.",
        )
    except Exception as error:
        raise gr.Error(str(error)) from error


def skip_selected_face(
    studio_state: StudioSessionState,
):
    try:
        if studio_state.analysis is None:
            raise ValueError("Сначала выполните поиск лиц.")

        studio_state.visual_assignments.skip(studio_state.flattened_target_faces)

        gallery, flattened = target_face_gallery(
            studio_state.analysis.targets,
            studio_state.visual_assignments.assignments,
        )

        studio_state.flattened_target_faces = flattened

        return (
            studio_state,
            gallery,
            "Лицо отмечено как «не заменять».",
        )
    except Exception as error:
        raise gr.Error(str(error)) from error


def assign_source_to_all(
    studio_state: StudioSessionState,
):
    try:
        if studio_state.analysis is None:
            raise ValueError("Сначала выполните поиск лиц.")

        source_index = studio_state.visual_assignments.selected_source_index

        if source_index is None:
            raise ValueError("Сначала выберите исходное лицо.")

        for target_key in studio_state.flattened_target_faces:
            studio_state.visual_assignments.assignments[target_key] = source_index

        gallery, flattened = target_face_gallery(
            studio_state.analysis.targets,
            studio_state.visual_assignments.assignments,
        )

        studio_state.flattened_target_faces = flattened

        return (
            studio_state,
            gallery,
            "Источник назначен всем найденным лицам.",
        )
    except Exception as error:
        raise gr.Error(str(error)) from error


def process_ui(
    studio_state: StudioSessionState,
    swap_model,
    enhance_faces,
    enhancement_weight,
    upscale_image,
    upscale_factor,
    tile_size,
):
    try:
        if studio_state is None:
            raise ValueError("Сначала добавьте фотографии и найдите лица.")

        return process_analysis(
            session=studio_state.analysis,
            visual_assignments=(studio_state.visual_assignments.assignments),
            swap_model=str(swap_model),
            enhance_face_regions=bool(enhance_faces),
            face_enhancement_weight=float(enhancement_weight),
            upscale_full_image=bool(upscale_image),
            upscale_factor=float(upscale_factor),
            tile_size=int(tile_size),
        )
    except Exception as error:
        logger.exception("Ошибка обработки изображений")
        raise gr.Error(str(error)) from error


def create_app() -> gr.Blocks:
    ensure_directories()

    with gr.Blocks(
        title="Face Swap Studio",
        css=APP_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="sky",
            neutral_hue="slate",
        ),
    ) as demo:
        source_collection = gr.State(FileCollectionState())
        target_collection = gr.State(FileCollectionState())
        studio_state = gr.State(StudioSessionState())

        gr.HTML(
            """
            <div class="studio-header">
                <h1>Face Swap Studio</h1>
                <p>
                    Визуальная локальная замена лиц
                    для фотографий
                </p>
            </div>
            """
        )

        gr.HTML(model_status_html())

        with gr.Tabs():
            with gr.Tab("1 · Фотографии"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Исходные лица")

                        source_upload = gr.UploadButton(
                            "Добавить фотографии лиц",
                            file_count="multiple",
                            file_types=["image"],
                            variant="primary",
                        )

                        source_upload_status = gr.Markdown("В коллекции файлов: 0")

                        source_files_gallery = gr.Gallery(
                            label=None,
                            columns=4,
                            rows=2,
                            height=220,
                            object_fit="cover",
                            preview=True,
                            elem_classes=["compact-gallery"],
                        )

                        remove_source_button = gr.Button("Удалить выбранную фотографию")

                    with gr.Column():
                        gr.Markdown("## Целевые фотографии")

                        target_upload = gr.UploadButton(
                            "Добавить целевые фотографии",
                            file_count="multiple",
                            file_types=["image"],
                            variant="primary",
                        )

                        target_upload_status = gr.Markdown("В коллекции файлов: 0")

                        target_files_gallery = gr.Gallery(
                            label=None,
                            columns=4,
                            rows=2,
                            height=220,
                            object_fit="cover",
                            preview=True,
                            elem_classes=["compact-gallery"],
                        )

                        remove_target_button = gr.Button("Удалить выбранную фотографию")

                confidence_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=0.5,
                    step=0.05,
                    label="Чувствительность поиска лиц",
                )

                analyse_button = gr.Button(
                    "Найти лица на фотографиях",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-action"],
                )

            with gr.Tab("2 · Назначение лиц"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Исходные лица")

                        source_faces_gallery = gr.Gallery(
                            label=None,
                            columns=3,
                            height=300,
                            object_fit="cover",
                            preview=True,
                            elem_classes=["face-gallery"],
                        )

                        selected_source_text = gr.Markdown(
                            "Исходное лицо не выбрано",
                            elem_classes=["selection-status"],
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("## Лица на целевых фотографиях")

                        target_faces_gallery = gr.Gallery(
                            label=None,
                            columns=5,
                            height=420,
                            object_fit="cover",
                            preview=True,
                            elem_classes=["face-gallery"],
                        )

                        selected_target_text = gr.Markdown(
                            "Целевое лицо не выбрано",
                            elem_classes=["selection-status"],
                        )

                with gr.Row():
                    assign_button = gr.Button(
                        "Назначить выбранное лицо",
                        variant="primary",
                    )

                    assign_all_button = gr.Button("Назначить всем")

                    skip_button = gr.Button("Не заменять выбранное")

                with gr.Accordion(
                    "Просмотр исходных фотографий",
                    open=False,
                ):
                    target_preview_gallery_component = gr.Gallery(
                        label=None,
                        columns=3,
                        height=380,
                        object_fit="contain",
                        preview=True,
                    )

            with gr.Tab("3 · Обработка"):
                with gr.Accordion(
                    "Настройки качества",
                    open=True,
                ):
                    controls = create_processing_controls()

                process_button = gr.Button(
                    "Запустить замену лиц",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-action"],
                )

                status_output = gr.Textbox(
                    label="Состояние",
                    interactive=False,
                    lines=3,
                )

                result_gallery = gr.Gallery(
                    label="Результаты",
                    columns=2,
                    height=620,
                    object_fit="contain",
                    preview=True,
                )

                download_archive = gr.File(
                    label="Скачать ZIP-архив",
                    interactive=False,
                )

                clear_button = gr.Button("Очистить временные файлы")

        source_upload.upload(
            fn=add_uploaded_files,
            inputs=[
                source_upload,
                source_collection,
            ],
            outputs=[
                source_collection,
                source_files_gallery,
                source_upload,
                source_upload_status,
            ],
        )

        target_upload.upload(
            fn=add_uploaded_files,
            inputs=[
                target_upload,
                target_collection,
            ],
            outputs=[
                target_collection,
                target_files_gallery,
                target_upload,
                target_upload_status,
            ],
        )

        source_files_gallery.select(
            fn=select_uploaded_file,
            inputs=[source_collection],
            outputs=[
                source_collection,
                source_upload_status,
            ],
        )

        target_files_gallery.select(
            fn=select_uploaded_file,
            inputs=[target_collection],
            outputs=[
                target_collection,
                target_upload_status,
            ],
        )

        remove_source_button.click(
            fn=remove_uploaded_file,
            inputs=[source_collection],
            outputs=[
                source_collection,
                source_files_gallery,
                source_upload_status,
            ],
        )

        remove_target_button.click(
            fn=remove_uploaded_file,
            inputs=[target_collection],
            outputs=[
                target_collection,
                target_files_gallery,
                target_upload_status,
            ],
        )

        analyse_button.click(
            fn=analyse_ui,
            inputs=[
                source_collection,
                target_collection,
                studio_state,
                confidence_threshold,
            ],
            outputs=[
                studio_state,
                source_faces_gallery,
                target_preview_gallery_component,
                target_faces_gallery,
                selected_source_text,
                selected_target_text,
                status_output,
            ],
        )

        source_faces_gallery.select(
            fn=select_source_face,
            inputs=[studio_state],
            outputs=[
                studio_state,
                selected_source_text,
            ],
        )

        target_faces_gallery.select(
            fn=select_target_face,
            inputs=[studio_state],
            outputs=[
                studio_state,
                selected_target_text,
            ],
        )

        assign_button.click(
            fn=assign_selected_face,
            inputs=[studio_state],
            outputs=[
                studio_state,
                target_faces_gallery,
                status_output,
            ],
        )

        assign_all_button.click(
            fn=assign_source_to_all,
            inputs=[studio_state],
            outputs=[
                studio_state,
                target_faces_gallery,
                status_output,
            ],
        )

        skip_button.click(
            fn=skip_selected_face,
            inputs=[studio_state],
            outputs=[
                studio_state,
                target_faces_gallery,
                status_output,
            ],
        )

        process_button.click(
            fn=process_ui,
            inputs=[
                studio_state,
                controls["swap_model"],
                controls["enhance_faces"],
                controls["enhancement_weight"],
                controls["upscale_image"],
                controls["upscale_factor"],
                controls["tile_size"],
            ],
            outputs=[
                result_gallery,
                download_archive,
                status_output,
            ],
        )

        clear_button.click(
            fn=clear_temporary_files,
            inputs=[],
            outputs=[status_output],
        )

    return demo
