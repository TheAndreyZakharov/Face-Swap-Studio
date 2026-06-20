"use strict";

const state = {
    sessionId: null,
    session: null,
    models: [],
    busy: false,
};

const elements = {};

document.addEventListener(
    "DOMContentLoaded",
    initializeApplication,
);


async function initializeApplication() {
    collectElements();
    bindEvents();

    updateRangeOutputs();

    try {
        await checkConnection();
        await createSession();
        await loadModels();

        setStatus(
            "Ready. Add source and target images.",
        );
    } catch (error) {
        handleError(error);
    }
}


function collectElements() {
    const ids = [
        "connection-status",
        "reset-session-button",
        "status-message",

        "source-input",
        "target-input",
        "source-dropzone",
        "target-dropzone",

        "source-count",
        "target-count",
        "source-image-list",
        "target-image-list",

        "confidence-input",
        "confidence-output",
        "detect-button",

        "source-face-count",
        "target-face-count",
        "source-face-grid",
        "target-face-grid",

        "mapping-list",
        "clear-mappings-button",

        "model-select",
        "model-description",
        "model-list",

        "enhance-input",
        "enhancement-weight-input",
        "enhancement-weight-output",

        "upscale-input",
        "upscale-factor-select",
        "tile-size-select",

        "generate-button",

        "result-preview",
        "result-placeholder",
        "result-image",
        "download-button",

        "loading-overlay",
        "loading-title",
        "loading-message",

        "image-dialog",
        "dialog-close-button",
        "dialog-image",
        "dialog-caption",

        "toast-region",
    ];

    for (const id of ids) {
        elements[id] = document.getElementById(id);
    }
}


function bindEvents() {
    elements["source-input"].addEventListener(
        "change",
        event => {
            uploadImages(
                "sources",
                event.target.files,
            );
        },
    );

    elements["target-input"].addEventListener(
        "change",
        event => {
            uploadImages(
                "targets",
                event.target.files,
            );
        },
    );

    setupDropzone(
        elements["source-dropzone"],
        elements["source-input"],
        files => uploadImages(
            "sources",
            files,
        ),
    );

    setupDropzone(
        elements["target-dropzone"],
        elements["target-input"],
        files => uploadImages(
            "targets",
            files,
        ),
    );

    elements["confidence-input"].addEventListener(
        "input",
        updateRangeOutputs,
    );

    elements["enhancement-weight-input"].addEventListener(
        "input",
        updateRangeOutputs,
    );

    elements["detect-button"].addEventListener(
        "click",
        detectFaces,
    );

    elements["clear-mappings-button"].addEventListener(
        "click",
        clearMappings,
    );

    elements["model-select"].addEventListener(
        "change",
        updateSelectedModelDescription,
    );

    elements["generate-button"].addEventListener(
        "click",
        generateImage,
    );

    elements["reset-session-button"].addEventListener(
        "click",
        resetSession,
    );

    elements["result-preview"].addEventListener(
        "click",
        () => {
            const source = elements["result-image"].src;

            if (source) {
                openImageDialog(
                    source,
                    "Generated result",
                );
            }
        },
    );

    elements["dialog-close-button"].addEventListener(
        "click",
        closeImageDialog,
    );

    elements["image-dialog"].addEventListener(
        "click",
        event => {
            if (
                event.target
                === elements["image-dialog"]
            ) {
                closeImageDialog();
            }
        },
    );

    document.addEventListener(
        "keydown",
        event => {
            if (event.key === "Escape") {
                closeImageDialog();
            }
        },
    );

    document
        .querySelectorAll(
            "[data-scroll-target]",
        )
        .forEach(button => {
            button.addEventListener(
                "click",
                () => {
                    const target = document.getElementById(
                        button.dataset.scrollTarget,
                    );

                    if (target) {
                        target.scrollIntoView({
                            behavior: "smooth",
                            block: "start",
                        });
                    }
                },
            );
        });
}


function setupDropzone(
    dropzone,
    input,
    onFiles,
) {
    const stopEvent = event => {
        event.preventDefault();
        event.stopPropagation();
    };

    for (
        const eventName
        of [
            "dragenter",
            "dragover",
            "dragleave",
            "drop",
        ]
    ) {
        dropzone.addEventListener(
            eventName,
            stopEvent,
        );
    }

    for (
        const eventName
        of [
            "dragenter",
            "dragover",
        ]
    ) {
        dropzone.addEventListener(
            eventName,
            () => {
                dropzone.classList.add(
                    "is-dragging",
                );
            },
        );
    }

    for (
        const eventName
        of [
            "dragleave",
            "drop",
        ]
    ) {
        dropzone.addEventListener(
            eventName,
            () => {
                dropzone.classList.remove(
                    "is-dragging",
                );
            },
        );
    }

    dropzone.addEventListener(
        "drop",
        event => {
            const files = event.dataTransfer.files;

            if (files.length > 0) {
                onFiles(files);
            }
        },
    );

    input.addEventListener(
        "click",
        () => {
            input.value = "";
        },
    );
}


async function apiRequest(
    path,
    options = {},
) {
    const response = await fetch(
        path,
        options,
    );

    if (!response.ok) {
        let message = (
            `Request failed with status `
            + response.status
        );

        try {
            const body = await response.json();

            if (body.detail) {
                if (
                    typeof body.detail
                    === "string"
                ) {
                    message = body.detail;
                } else {
                    message = JSON.stringify(
                        body.detail,
                    );
                }
            }
        } catch {
            const text = await response.text();

            if (text) {
                message = text;
            }
        }

        throw new Error(message);
    }

    const contentType = (
        response.headers.get(
            "content-type",
        )
        || ""
    );

    if (
        contentType.includes(
            "application/json",
        )
    ) {
        return response.json();
    }

    return response;
}


async function checkConnection() {
    await apiRequest(
        "/api/health",
    );

    elements["connection-status"].textContent = (
        "Connected"
    );

    elements["connection-status"].classList.add(
        "is-connected",
    );

    elements["connection-status"].classList.remove(
        "is-disconnected",
    );
}


async function createSession() {
    const response = await apiRequest(
        "/api/sessions",
        {
            method: "POST",
        },
    );

    state.sessionId = response.session_id;

    await refreshSession();
}


async function refreshSession() {
    if (!state.sessionId) {
        return;
    }

    state.session = await apiRequest(
        `/api/sessions/${state.sessionId}`,
    );

    renderSession();
}


async function loadModels() {
    state.models = await apiRequest(
        "/api/models",
    );

    renderModels();
}


async function uploadImages(
    kind,
    files,
) {
    if (
        state.busy
        || !files
        || files.length === 0
    ) {
        return;
    }

    const formData = new FormData();

    for (const file of files) {
        formData.append(
            "files",
            file,
            file.name,
        );
    }

    await runBusyOperation(
        kind === "sources"
            ? "Adding source images"
            : "Adding target images",
        "Preparing uploaded files.",
        async () => {
            state.session = await apiRequest(
                (
                    `/api/sessions/`
                    + `${state.sessionId}/`
                    + kind
                ),
                {
                    method: "POST",
                    body: formData,
                },
            );

            renderSession();

            setStatus(
                kind === "sources"
                    ? "Source images added."
                    : "Target images added.",
            );
        },
    );

    elements[
        kind === "sources"
            ? "source-input"
            : "target-input"
    ].value = "";
}


async function selectTargetImage(
    targetImageId,
) {
    if (state.busy) {
        return;
    }

    await runBusyOperation(
        "Selecting target image",
        "Resetting previous face detection.",
        async () => {
            state.session = await apiRequest(
                (
                    `/api/sessions/`
                    + `${state.sessionId}/`
                    + "target-selection"
                ),
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify({
                        target_image_id:
                            targetImageId,
                    }),
                },
            );

            renderSession();

            setStatus(
                "Target image selected. Run face detection.",
            );
        },
    );
}


async function detectFaces() {
    if (state.busy) {
        return;
    }

    await runBusyOperation(
        "Detecting faces",
        "Analysing source identities and the selected target image.",
        async () => {
            state.session = await apiRequest(
                (
                    `/api/sessions/`
                    + `${state.sessionId}/detect`
                ),
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify({
                        confidence_threshold:
                            Number(
                                elements[
                                    "confidence-input"
                                ].value,
                            ),
                    }),
                },
            );

            renderSession();

            setStatus(
                (
                    "Detection complete. "
                    + `${state.session.source_faces.length} `
                    + "source face(s), "
                    + `${state.session.target_faces.length} `
                    + "target face(s)."
                ),
            );

            document
                .getElementById(
                    "mapping-section",
                )
                .scrollIntoView({
                    behavior: "smooth",
                    block: "start",
                });
        },
    );
}


async function updateMapping(
    targetFaceIndex,
    sourceFaceIndex,
) {
    if (
        state.busy
        || !state.session
    ) {
        return;
    }

    const mappings = {
        ...state.session.mappings,
    };

    mappings[
        String(
            targetFaceIndex,
        )
    ] = sourceFaceIndex;

    const payload = {
        mappings: Object.entries(
            mappings,
        ).map(
            ([
                targetIndex,
                sourceIndex,
            ]) => ({
                target_face_index:
                    Number(targetIndex),
                source_face_index:
                    sourceIndex === null
                        ? null
                        : Number(sourceIndex),
            }),
        ),
    };

    try {
        state.session = await apiRequest(
            (
                `/api/sessions/`
                + `${state.sessionId}/mappings`
            ),
            {
                method: "PUT",
                headers: {
                    "Content-Type":
                        "application/json",
                },
                body: JSON.stringify(
                    payload,
                ),
            },
        );

        renderMappings();
        renderResult();

        setStatus(
            sourceFaceIndex === null
                ? (
                    `Target face `
                    + `${targetFaceIndex + 1} `
                    + "will remain unchanged."
                )
                : (
                    `Target face `
                    + `${targetFaceIndex + 1} `
                    + "will use source face "
                    + `${sourceFaceIndex + 1}.`
                ),
        );
    } catch (error) {
        handleError(error);
        renderMappings();
    }
}


async function clearMappings() {
    if (
        !state.session
        || !state.session.analysis_completed
    ) {
        return;
    }

    const payload = {
        mappings:
            state.session.target_faces.map(
                face => ({
                    target_face_index:
                        face.index,
                    source_face_index:
                        null,
                }),
            ),
    };

    try {
        state.session = await apiRequest(
            (
                `/api/sessions/`
                + `${state.sessionId}/mappings`
            ),
            {
                method: "PUT",
                headers: {
                    "Content-Type":
                        "application/json",
                },
                body: JSON.stringify(
                    payload,
                ),
            },
        );

        renderMappings();
        renderResult();

        setStatus(
            "All target faces will remain unchanged.",
        );
    } catch (error) {
        handleError(error);
    }
}


async function generateImage() {
    if (state.busy) {
        return;
    }

    const modelId = (
        elements["model-select"].value
    );

    if (!modelId) {
        showToast(
            "Select an available model.",
            true,
        );

        return;
    }

    await runBusyOperation(
        "Generating image",
        "The selected model is processing the configured replacements.",
        async () => {
            state.session = await apiRequest(
                (
                    `/api/sessions/`
                    + `${state.sessionId}/generate`
                ),
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify({
                        model_id:
                            modelId,
                        enhance_faces:
                            elements[
                                "enhance-input"
                            ].checked,
                        enhancement_weight:
                            Number(
                                elements[
                                    "enhancement-weight-input"
                                ].value,
                            ),
                        upscale_image:
                            elements[
                                "upscale-input"
                            ].checked,
                        upscale_factor:
                            Number(
                                elements[
                                    "upscale-factor-select"
                                ].value,
                            ),
                        tile_size:
                            Number(
                                elements[
                                    "tile-size-select"
                                ].value,
                            ),
                    }),
                },
            );

            renderSession();

            setStatus(
                "Generation complete.",
            );

            document
                .getElementById(
                    "generation-section",
                )
                .scrollIntoView({
                    behavior: "smooth",
                    block: "start",
                });
        },
    );
}


async function resetSession() {
    if (state.busy) {
        return;
    }

    await runBusyOperation(
        "Clearing session",
        "Removing temporary images and mappings.",
        async () => {
            if (state.sessionId) {
                try {
                    await apiRequest(
                        (
                            `/api/sessions/`
                            + state.sessionId
                        ),
                        {
                            method: "DELETE",
                        },
                    );
                } catch {
                    // A missing previous session is harmless.
                }
            }

            state.sessionId = null;
            state.session = null;

            await createSession();

            setStatus(
                "Session cleared. Add new images.",
            );
        },
    );
}


function renderSession() {
    renderUploadedImages();
    renderDetectedFaces();
    renderMappings();
    renderResult();
}


function renderUploadedImages() {
    if (!state.session) {
        return;
    }

    elements["source-count"].textContent = (
        String(
            state.session.source_images.length,
        )
    );

    elements["target-count"].textContent = (
        String(
            state.session.target_images.length,
        )
    );

    renderUploadedImageList(
        elements["source-image-list"],
        state.session.source_images,
        false,
    );

    renderUploadedImageList(
        elements["target-image-list"],
        state.session.target_images,
        true,
    );
}


function renderUploadedImageList(
    container,
    images,
    selectable,
) {
    container.replaceChildren();

    if (images.length === 0) {
        container.append(
            createEmptyInline(
                selectable
                    ? "No target images"
                    : "No source images",
            ),
        );

        return;
    }

    for (const image of images) {
        const button = document.createElement(
            "button",
        );

        button.type = "button";
        button.className = "upload-thumbnail";

        if (image.selected) {
            button.classList.add(
                "is-selected",
            );
        }

        const preview = document.createElement(
            "img",
        );

        preview.src = image.url;
        preview.alt = image.name;
        preview.loading = "lazy";

        const label = document.createElement(
            "span",
        );

        label.className = (
            "upload-thumbnail-label"
        );

        label.textContent = image.name;

        button.append(
            preview,
            label,
        );

        button.addEventListener(
            "dblclick",
            () => {
                openImageDialog(
                    image.url,
                    image.name,
                );
            },
        );

        button.addEventListener(
            "click",
            () => {
                if (selectable) {
                    selectTargetImage(
                        image.id,
                    );
                } else {
                    openImageDialog(
                        image.url,
                        image.name,
                    );
                }
            },
        );

        container.append(
            button,
        );
    }
}


function renderDetectedFaces() {
    if (!state.session) {
        return;
    }

    elements[
        "source-face-count"
    ].textContent = String(
        state.session.source_faces.length,
    );

    elements[
        "target-face-count"
    ].textContent = String(
        state.session.target_faces.length,
    );

    renderFaceGrid(
        elements["source-face-grid"],
        state.session.source_faces,
        "No source faces",
    );

    renderFaceGrid(
        elements["target-face-grid"],
        state.session.target_faces,
        "No target faces",
    );
}


function renderFaceGrid(
    container,
    faces,
    emptyTitle,
) {
    container.replaceChildren();

    if (faces.length === 0) {
        container.append(
            createEmptyState(
                emptyTitle,
                "Add images and run detection.",
            ),
        );

        return;
    }

    for (const face of faces) {
        const button = document.createElement(
            "button",
        );

        button.type = "button";
        button.className = "face-card";

        const image = document.createElement(
            "img",
        );

        image.src = face.image_url;
        image.alt = face.label;
        image.loading = "lazy";

        const label = document.createElement(
            "span",
        );

        label.className = "face-card-label";
        label.textContent = face.label;

        button.append(
            image,
            label,
        );

        button.addEventListener(
            "click",
            () => {
                openImageDialog(
                    face.image_url,
                    face.label,
                );
            },
        );

        container.append(
            button,
        );
    }
}


function renderMappings() {
    const container = elements["mapping-list"];

    container.replaceChildren();

    if (
        !state.session
        || !state.session.analysis_completed
        || state.session.target_faces.length === 0
    ) {
        container.append(
            createEmptyState(
                "No mappings available",
                "Detect faces to configure replacements.",
                true,
            ),
        );

        return;
    }

    for (
        const targetFace
        of state.session.target_faces
    ) {
        const row = document.createElement(
            "div",
        );

        row.className = "mapping-row";

        const targetBlock = document.createElement(
            "div",
        );

        targetBlock.className = "mapping-target";

        const targetImage = document.createElement(
            "img",
        );

        targetImage.src = targetFace.image_url;
        targetImage.alt = targetFace.label;

        targetImage.addEventListener(
            "click",
            () => {
                openImageDialog(
                    targetFace.image_url,
                    targetFace.label,
                );
            },
        );

        const targetCopy = document.createElement(
            "div",
        );

        const targetTitle = document.createElement(
            "strong",
        );

        targetTitle.textContent = (
            `Target face `
            + `${targetFace.index + 1}`
        );

        const targetCaption = document.createElement(
            "span",
        );

        targetCaption.textContent = (
            "Detected target identity"
        );

        targetCopy.append(
            targetTitle,
            targetCaption,
        );

        targetBlock.append(
            targetImage,
            targetCopy,
        );

        const arrow = document.createElement(
            "div",
        );

        arrow.className = "mapping-arrow";
        arrow.textContent = "→";

        const sourceBlock = document.createElement(
            "div",
        );

        sourceBlock.className = (
            "mapping-select-wrap"
        );

        const sourcePreview = document.createElement(
            "div",
        );

        sourcePreview.className = (
            "mapping-source-preview"
        );

        const select = document.createElement(
            "select",
        );

        const unchangedOption = (
            document.createElement(
                "option",
            )
        );

        unchangedOption.value = "";
        unchangedOption.textContent = (
            "Leave unchanged"
        );

        select.append(
            unchangedOption,
        );

        for (
            const sourceFace
            of state.session.source_faces
        ) {
            const option = document.createElement(
                "option",
            );

            option.value = String(
                sourceFace.index,
            );

            option.textContent = (
                `Source face `
                + `${sourceFace.index + 1}`
            );

            select.append(
                option,
            );
        }

        const currentMapping = (
            state.session.mappings[
                String(
                    targetFace.index,
                )
            ]
        );

        select.value = (
            currentMapping === null
            || currentMapping === undefined
                ? ""
                : String(currentMapping)
        );

        updateMappingPreview(
            sourcePreview,
            currentMapping,
        );

        select.addEventListener(
            "change",
            () => {
                const value = select.value;

                const sourceIndex = (
                    value === ""
                        ? null
                        : Number(value)
                );

                updateMappingPreview(
                    sourcePreview,
                    sourceIndex,
                );

                updateMapping(
                    targetFace.index,
                    sourceIndex,
                );
            },
        );

        sourceBlock.append(
            sourcePreview,
            select,
        );

        row.append(
            targetBlock,
            arrow,
            sourceBlock,
        );

        container.append(
            row,
        );
    }
}


function updateMappingPreview(
    container,
    sourceIndex,
) {
    container.replaceChildren();

    if (
        sourceIndex === null
        || sourceIndex === undefined
    ) {
        container.textContent = "—";
        return;
    }

    const sourceFace = (
        state.session.source_faces.find(
            face => (
                face.index
                === Number(sourceIndex)
            ),
        )
    );

    if (!sourceFace) {
        container.textContent = "—";
        return;
    }

    const image = document.createElement(
        "img",
    );

    image.src = sourceFace.image_url;
    image.alt = sourceFace.label;

    image.addEventListener(
        "click",
        () => {
            openImageDialog(
                sourceFace.image_url,
                sourceFace.label,
            );
        },
    );

    container.append(
        image,
    );
}


function renderModels() {
    const select = elements["model-select"];
    const list = elements["model-list"];

    select.replaceChildren();
    list.replaceChildren();

    const availableModels = (
        state.models.filter(
            model => model.available,
        )
    );

    for (const model of state.models) {
        const option = document.createElement(
            "option",
        );

        option.value = model.id;
        option.textContent = (
            model.available
                ? model.name
                : `${model.name} — unavailable`
        );

        option.disabled = !model.available;

        select.append(
            option,
        );

        const item = document.createElement(
            "article",
        );

        item.className = "model-item";

        const heading = document.createElement(
            "div",
        );

        heading.className = (
            "model-item-heading"
        );

        const name = document.createElement(
            "strong",
        );

        name.textContent = model.name;

        const status = document.createElement(
            "span",
        );

        status.className = (
            "status-chip "
            + (
                model.available
                    ? "is-ready"
                    : "is-unavailable"
            )
        );

        status.textContent = (
            model.available
                ? "Available"
                : "Unavailable"
        );

        heading.append(
            name,
            status,
        );

        const description = document.createElement(
            "p",
        );

        description.textContent = (
            model.description
        );

        item.append(
            heading,
            description,
        );

        list.append(
            item,
        );
    }

    const preferredIds = [
        "inswapper_128",
        "simswap_512",
        "ghost_unet_3blocks",
        "ghost2_head",
    ];

    let selectedModel = null;

    for (const modelId of preferredIds) {
        selectedModel = availableModels.find(
            model => model.id === modelId,
        );

        if (selectedModel) {
            break;
        }
    }

    if (
        !selectedModel
        && availableModels.length > 0
    ) {
        selectedModel = availableModels[0];
    }

    if (selectedModel) {
        select.value = selectedModel.id;
    }

    updateSelectedModelDescription();
}


function updateSelectedModelDescription() {
    const model = state.models.find(
        item => (
            item.id
            === elements["model-select"].value
        ),
    );

    elements["model-description"].textContent = (
        model
            ? model.description
            : "Select an available model."
    );
}


function renderResult() {
    if (
        state.session
        && state.session.result_url
    ) {
        elements["result-image"].src = (
            state.session.result_url
            + `?v=${Date.now()}`
        );

        elements["result-image"].hidden = false;
        elements["result-placeholder"].hidden = true;

        elements["result-preview"].disabled = false;

        elements["download-button"].href = (
            state.session.download_url
        );

        elements["download-button"].classList.remove(
            "is-hidden",
        );

        return;
    }

    elements["result-image"].removeAttribute(
        "src",
    );

    elements["result-image"].hidden = true;
    elements["result-placeholder"].hidden = false;

    elements["result-preview"].disabled = true;

    elements["download-button"].classList.add(
        "is-hidden",
    );

    elements["download-button"].removeAttribute(
        "href",
    );
}


function createEmptyInline(
    text,
) {
    const element = document.createElement(
        "div",
    );

    element.className = "empty-inline";
    element.textContent = text;

    return element;
}


function createEmptyState(
    title,
    description,
    wide = false,
) {
    const element = document.createElement(
        "div",
    );

    element.className = "empty-state";

    if (wide) {
        element.classList.add(
            "empty-state-wide",
        );
    }

    const titleElement = document.createElement(
        "strong",
    );

    titleElement.textContent = title;

    const descriptionElement = (
        document.createElement(
            "span",
        )
    );

    descriptionElement.textContent = description;

    element.append(
        titleElement,
        descriptionElement,
    );

    return element;
}


function updateRangeOutputs() {
    elements["confidence-output"].value = (
        Number(
            elements["confidence-input"].value,
        ).toFixed(2)
    );

    elements[
        "enhancement-weight-output"
    ].value = (
        Number(
            elements[
                "enhancement-weight-input"
            ].value,
        ).toFixed(2)
    );
}


function openImageDialog(
    source,
    caption,
) {
    elements["dialog-image"].src = source;
    elements["dialog-caption"].textContent = (
        caption
        || ""
    );

    if (
        !elements["image-dialog"].open
    ) {
        elements["image-dialog"].showModal();
    }
}


function closeImageDialog() {
    if (elements["image-dialog"].open) {
        elements["image-dialog"].close();
    }

    elements["dialog-image"].removeAttribute(
        "src",
    );
}


async function runBusyOperation(
    title,
    message,
    operation,
) {
    if (state.busy) {
        return;
    }

    state.busy = true;

    elements["loading-title"].textContent = title;
    elements["loading-message"].textContent = message;

    elements["loading-overlay"].classList.remove(
        "is-hidden",
    );

    elements["loading-overlay"].setAttribute(
        "aria-hidden",
        "false",
    );

    try {
        await operation();
    } catch (error) {
        handleError(error);
    } finally {
        state.busy = false;

        elements["loading-overlay"].classList.add(
            "is-hidden",
        );

        elements["loading-overlay"].setAttribute(
            "aria-hidden",
            "true",
        );
    }
}


function setStatus(
    message,
) {
    elements["status-message"].textContent = message;
}


function showToast(
    message,
    isError = false,
) {
    const toast = document.createElement(
        "div",
    );

    toast.className = "toast";

    if (isError) {
        toast.classList.add(
            "is-error",
        );
    }

    toast.textContent = message;

    elements["toast-region"].append(
        toast,
    );

    window.setTimeout(
        () => {
            toast.remove();
        },
        5000,
    );
}


function handleError(
    error,
) {
    console.error(error);

    const message = (
        error instanceof Error
            ? error.message
            : String(error)
    );

    setStatus(
        `Error: ${message}`,
    );

    showToast(
        message,
        true,
    );
}