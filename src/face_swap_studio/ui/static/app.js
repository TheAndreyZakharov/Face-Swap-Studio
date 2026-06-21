"use strict";

const state = {
    sessionId: null,
    session: null,
    models: [],
    busy: false,
    theme: "dark",
    resultIndex: 0,
};

const elements = {};

document.addEventListener(
    "DOMContentLoaded",
    initializeApplication,
);


async function initializeApplication() {
    applySavedTheme();
    collectElements();
    bindEvents();
    updateThemeToggle();
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
        "theme-toggle-button",
        "theme-toggle-label",
        "status-message",

        "source-input",
        "target-input",
        "source-dropzone",
        "target-dropzone",

        "source-count",
        "target-count",
        "source-image-list",
        "target-image-list",
        "target-select-all-button",
        "target-clear-selection-button",

        "confidence-input",
        "confidence-output",
        "detect-button",

        "source-face-count",
        "target-face-count",
        "source-face-grid",
        "target-face-grid",

        "mapping-list",
        "mapping-quick-source-list",
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
        "result-previous-button",
        "result-next-button",
        "result-counter",
        "download-button",
        "generated-result-list",

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

    elements["target-select-all-button"].addEventListener(
        "click",
        selectAllTargetImages,
    );

    elements["target-clear-selection-button"].addEventListener(
        "click",
        clearTargetImageSelection,
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

    elements["theme-toggle-button"].addEventListener(
        "click",
        toggleTheme,
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

    elements["result-previous-button"].addEventListener(
        "click",
        event => {
            event.stopPropagation();

            moveResultPreview(
                -1,
            );
        },
    );

    elements["result-next-button"].addEventListener(
        "click",
        event => {
            event.stopPropagation();

            moveResultPreview(
                1,
            );
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


function applySavedTheme() {
    const savedTheme = localStorage.getItem(
        "face-swap-studio-theme",
    );

    state.theme = (
        savedTheme === "light"
            ? "light"
            : "dark"
    );

    document.documentElement.dataset.theme = (
        state.theme
    );
}


function toggleTheme() {
    state.theme = (
        state.theme === "dark"
            ? "light"
            : "dark"
    );

    document.documentElement.dataset.theme = (
        state.theme
    );

    localStorage.setItem(
        "face-swap-studio-theme",
        state.theme,
    );

    updateThemeToggle();
}


function updateThemeToggle() {
    if (!elements["theme-toggle-button"]) {
        return;
    }

    elements["theme-toggle-button"].classList.toggle(
        "is-light",
        state.theme === "light",
    );

    elements["theme-toggle-label"].textContent = (
        state.theme === "light"
            ? "Light"
            : "Dark"
    );
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
                    ? "Source images added. Run detection again."
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


async function deleteUploadedImage(
    kind,
    imageId,
) {
    if (
        state.busy
        || !state.session
    ) {
        return;
    }

    await runBusyOperation(
        kind === "sources"
            ? "Removing source image"
            : "Removing target image",
        "Updating the current session.",
        async () => {
            state.session = await apiRequest(
                (
                    `/api/sessions/`
                    + `${state.sessionId}/`
                    + `${kind}/`
                    + imageId
                ),
                {
                    method: "DELETE",
                },
            );

            renderSession();

            setStatus(
                kind === "sources"
                    ? "Source image removed. Detection was reset."
                    : "Target image removed.",
            );
        },
    );
}


async function selectTargetImage(
    targetImageId,
) {
    if (state.busy) {
        return;
    }

    await runBusyOperation(
        "Selecting target image",
        "Loading target-specific detection and mappings.",
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

            if (state.session.analysis_completed) {
                setStatus(
                    "Target image selected. Existing mappings loaded.",
                );
            } else {
                setStatus(
                    "Target image selected. Run face detection for this target.",
                );
            }
        },
    );
}

async function toggleTargetImageSelection(
    targetImageId,
) {
    if (
        state.busy
        || !state.session
    ) {
        return;
    }

    const selectedIds = new Set(
        state.session.selected_target_image_ids
        || [],
    );

    const shouldInclude = !selectedIds.has(
        targetImageId
    );

    if (shouldInclude) {
        selectedIds.add(
            targetImageId
        );
    } else {
        selectedIds.delete(
            targetImageId
        );
    }

    await updateTargetSelectionSet(
        Array.from(
            selectedIds,
        ),
        targetImageId,
        shouldInclude
            ? "Target image added for generation."
            : "Target image removed from generation.",
    );
}

async function updateTargetSelectionSet(
    targetImageIds,
    activeTargetImageId,
    statusMessage,
) {
    if (
        state.busy
        || !state.session
    ) {
        return;
    }

    await runBusyOperation(
        "Updating target selection",
        "Changing target images selected for generation.",
        async () => {
            if (activeTargetImageId) {
                await apiRequest(
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
                                activeTargetImageId,
                        }),
                    },
                );
            }

            state.session = await apiRequest(
                (
                    `/api/sessions/`
                    + `${state.sessionId}/`
                    + "target-batch-selection"
                ),
                {
                    method: "PUT",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify({
                        target_image_ids:
                            targetImageIds,
                    }),
                },
            );

            renderSession();

            setStatus(
                statusMessage,
            );
        },
    );
}


async function selectAllTargetImages() {
    if (
        state.busy
        || !state.session
    ) {
        return;
    }

    const targetIds = state.session.target_images.map(
        image => image.id,
    );

    if (targetIds.length === 0) {
        return;
    }

    await updateTargetSelectionSet(
        targetIds,
        state.session.active_target_image_id
        || targetIds[0],
        "All target images selected for generation.",
    );
}


async function clearTargetImageSelection() {
    if (
        state.busy
        || !state.session
    ) {
        return;
    }

    await updateTargetSelectionSet(
        [],
        state.session.active_target_image_id,
        "Target generation selection cleared.",
    );
}


async function detectFaces() {
    if (state.busy) {
        return;
    }

    await runBusyOperation(
        "Detecting faces",
        "Analysing source identities and selected target images.",
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

            const detectedTargetCount = (
                state.session.target_analyses
                || []
            ).reduce(
                (total, analysis) => (
                    total
                    + (
                        analysis.analysis_completed
                            ? analysis.target_faces.length
                            : 0
                    )
                ),
                0,
            );

            setStatus(
                (
                    "Detection complete for selected targets. "
                    + `${state.session.source_faces.length} `
                    + "source face(s), "
                    + `${detectedTargetCount} `
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
    targetImageId,
    targetFaceIndex,
    sourceFaceIndex,
) {
    if (
        state.busy
        || !state.session
    ) {
        return;
    }

    const analysis = findTargetAnalysis(
        targetImageId,
    );

    if (!analysis) {
        return;
    }

    const mappings = {
        ...analysis.mappings,
    };

    mappings[
        String(
            targetFaceIndex,
        )
    ] = (
        sourceFaceIndex === undefined
            ? null
            : sourceFaceIndex
    );

    const payload = {
        target_image_id:
            targetImageId,
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
                    || sourceIndex === undefined
                    || sourceIndex === ""
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
            || sourceFaceIndex === undefined
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
        await refreshSession();
    }
}

async function applySourceFaceToAllSelectedTargets(
    sourceFaceIndex,
) {
    if (
        state.busy
        || !state.session
    ) {
        return;
    }

    const targetIds = selectedTargetIdsForMappings();

    const readyAnalyses = targetIds
        .map(targetImageId => findTargetAnalysis(
            targetImageId,
        ))
        .filter(analysis => (
            analysis
            && analysis.analysis_completed
            && analysis.target_faces.length > 0
        ));

    if (readyAnalyses.length === 0) {
        showToast(
            "Run detection for selected targets first.",
            true,
        );

        return;
    }

    await runBusyOperation(
        "Applying source face",
        "Assigning one source face to every detected target face.",
        async () => {
            for (const analysis of readyAnalyses) {
                const payload = {
                    target_image_id:
                        analysis.target_image_id,
                    mappings:
                        analysis.target_faces.map(
                            face => ({
                                target_face_index:
                                    face.index,
                                source_face_index:
                                    sourceFaceIndex,
                            }),
                        ),
                };

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
            }

            renderSession();

            setStatus(
                (
                    `Source face ${sourceFaceIndex + 1} `
                    + "was applied to every detected face "
                    + "in selected targets."
                ),
            );
        },
    );
}


function renderMappingQuickActions() {
    const container = elements[
        "mapping-quick-source-list"
    ];

    container.replaceChildren();

    if (
        !state.session
        || state.session.source_faces.length === 0
    ) {
        const empty = document.createElement(
            "div",
        );

        empty.className = (
            "mapping-quick-empty"
        );

        empty.textContent = (
            "Detect source faces to enable quick assignment."
        );

        container.append(
            empty,
        );

        return;
    }

    const header = document.createElement(
        "div",
    );

    header.className = (
        "mapping-quick-header"
    );

    const title = document.createElement(
        "strong",
    );

    title.textContent = (
        "Apply one source to all target faces"
    );

    const subtitle = document.createElement(
        "span",
    );

    subtitle.textContent = (
        "Click a source face below to assign it to every detected face in selected targets."
    );

    header.append(
        title,
        subtitle,
    );

    const grid = document.createElement(
        "div",
    );

    grid.className = (
        "mapping-quick-source-grid"
    );

    for (
        const sourceFace
        of state.session.source_faces
    ) {
        const button = document.createElement(
            "button",
        );

        button.type = "button";
        button.className = (
            "mapping-quick-source-button"
        );

        const image = document.createElement(
            "img",
        );

        image.src = sourceFace.image_url;
        image.alt = sourceFace.label;
        image.loading = "lazy";

        const label = document.createElement(
            "span",
        );

        label.textContent = (
            `Source ${sourceFace.index + 1}`
        );

        button.append(
            image,
            label,
        );

        button.addEventListener(
            "click",
            () => {
                applySourceFaceToAllSelectedTargets(
                    sourceFace.index,
                );
            },
        );

        grid.append(
            button,
        );
    }

    container.append(
        header,
        grid,
    );
}

async function clearMappings() {
    if (
        state.busy
        || !state.session
    ) {
        return;
    }

    const activeTargetId = (
        state.session.active_target_image_id
    );

    if (!activeTargetId) {
        return;
    }

    const analysis = findTargetAnalysis(
        activeTargetId,
    );

    if (
        !analysis
        || !analysis.analysis_completed
    ) {
        return;
    }

    const payload = {
        target_image_id:
            activeTargetId,
        mappings:
            analysis.target_faces.map(
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
            "All target faces for the active target will remain unchanged.",
        );
    } catch (error) {
        handleError(error);
        await refreshSession();
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
        "Generating selected targets",
        "The selected model is processing every checked target image.",
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
                (
                    "Generation complete. "
                    + `${state.session.generated_results.length} `
                    + "result image(s) available."
                ),
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
        "Removing temporary images, mappings and generated results.",
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
    renderMappingQuickActions();
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
        "sources",
    );

    renderUploadedImageList(
        elements["target-image-list"],
        state.session.target_images,
        "targets",
    );
}


function renderUploadedImageList(
    container,
    images,
    kind,
) {
    container.replaceChildren();

    if (images.length === 0) {
        container.append(
            createEmptyInline(
                kind === "targets"
                    ? "No target images"
                    : "No source images",
            ),
        );

        return;
    }

    for (const image of images) {
        const card = document.createElement(
            "article",
        );

        card.className = "upload-thumbnail";

        if (image.included) {
            card.classList.add(
                "is-included",
            );
        }

        const previewButton = document.createElement(
            "button",
        );

        previewButton.type = "button";
        previewButton.className = (
            "upload-thumbnail-preview"
        );

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

        previewButton.append(
            preview,
            label,
        );

        previewButton.addEventListener(
            "dblclick",
            event => {
                event.stopPropagation();

                openImageDialog(
                    image.url,
                    image.name,
                );
            },
        );

        previewButton.addEventListener(
            "click",
            event => {
                event.stopPropagation();

                openImageDialog(
                    image.url,
                    image.name,
                );
            },
        );

        const removeButton = document.createElement(
            "button",
        );

        removeButton.type = "button";
        removeButton.className = (
            "upload-thumbnail-remove"
        );

        removeButton.setAttribute(
            "aria-label",
            `Remove ${image.name}`,
        );

        removeButton.textContent = "×";

        removeButton.addEventListener(
            "click",
            event => {
                event.stopPropagation();

                deleteUploadedImage(
                    kind,
                    image.id,
                );
            },
        );

        card.append(
            previewButton,
            removeButton,
        );

        if (kind === "targets") {
            const selectionButton = document.createElement(
                "button",
            );

            selectionButton.type = "button";
            selectionButton.className = (
                "target-toggle-button"
            );

            if (image.included) {
                selectionButton.classList.add(
                    "is-included",
                );
            }

            selectionButton.textContent = (
                image.included
                    ? "Remove from generation"
                    : "Add for generation"
            );

            selectionButton.addEventListener(
                "click",
                event => {
                    event.stopPropagation();

                    toggleTargetImageSelection(
                        image.id,
                    );
                },
            );

            card.append(
                selectionButton,
            );
        }

        container.append(
            card,
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

    const detectedTargetCount = (
        state.session.target_analyses
        || []
    ).reduce(
        (total, analysis) => (
            total
            + (
                analysis.analysis_completed
                    ? analysis.target_faces.length
                    : 0
            )
        ),
        0,
    );

    elements[
        "target-face-count"
    ].textContent = String(
        detectedTargetCount,
    );

    renderFaceGrid(
        elements["source-face-grid"],
        state.session.source_faces,
        "No source faces",
    );

    renderTargetFaceGroups();
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

function renderTargetFaceGroups() {
    const container = elements["target-face-grid"];

    container.replaceChildren();

    if (!state.session) {
        return;
    }

    const analyses = (
        state.session.target_analyses
        || []
    ).filter(
        analysis => (
            analysis.analysis_completed
            && analysis.target_faces.length > 0
        ),
    );

    if (analyses.length === 0) {
        container.append(
            createEmptyState(
                "No target faces",
                "Select target images and run detection.",
            ),
        );

        return;
    }

    for (const analysis of analyses) {
        const group = document.createElement(
            "section",
        );

        group.className = "target-face-group";

        const heading = document.createElement(
            "div",
        );

        heading.className = "target-face-group-heading";

        const title = document.createElement(
            "strong",
        );

        title.textContent = analysis.target_image_name;

        const count = document.createElement(
            "span",
        );

        count.textContent = (
            `${analysis.target_faces.length} face(s)`
        );

        heading.append(
            title,
            count,
        );

        const grid = document.createElement(
            "div",
        );

        grid.className = "target-face-group-grid";

        for (const face of analysis.target_faces) {
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
                        `${analysis.target_image_name} · ${face.label}`,
                    );
                },
            );

            grid.append(
                button,
            );
        }

        group.append(
            heading,
            grid,
        );

        container.append(
            group,
        );
    }
}

function findTargetImage(
    targetImageId,
) {
    if (!state.session) {
        return null;
    }

    return (
        state.session.target_images.find(
            image => image.id === targetImageId,
        )
        || null
    );
}


function findTargetAnalysis(
    targetImageId,
) {
    if (!state.session) {
        return null;
    }

    return (
        (
            state.session.target_analyses
            || []
        ).find(
            analysis => (
                analysis.target_image_id
                === targetImageId
            ),
        )
        || null
    );
}


function selectedTargetIdsForMappings() {
    if (!state.session) {
        return [];
    }

    return (
        state.session.selected_target_image_ids
        || []
    );
}

function renderMappings() {
    const container = elements["mapping-list"];

    container.replaceChildren();

    if (!state.session) {
        container.append(
            createEmptyState(
                "No mappings available",
                "Add target images first.",
                true,
            ),
        );

        return;
    }

    const targetIds = selectedTargetIdsForMappings();

    if (targetIds.length === 0) {
        container.append(
            createEmptyState(
                "No targets selected",
                "Click a target image to add the green border and include it in generation.",
                true,
            ),
        );

        return;
    }

    let renderedAnySection = false;

    for (const targetImageId of targetIds) {
        const targetImage = findTargetImage(
            targetImageId,
        );

        if (!targetImage) {
            continue;
        }

        const analysis = findTargetAnalysis(
            targetImageId,
        );

        const section = document.createElement(
            "section",
        );

        section.className = "mapping-target-section";


        const header = document.createElement(
            "div",
        );

        header.className = (
            "mapping-target-section-header"
        );

        const image = document.createElement(
            "img",
        );

        image.src = targetImage.url;
        image.alt = targetImage.name;

        const copy = document.createElement(
            "div",
        );

        const title = document.createElement(
            "strong",
        );

        title.textContent = targetImage.name;

        const subtitle = document.createElement(
            "span",
        );

        subtitle.textContent = (
            analysis
            && analysis.analysis_completed
                ? (
                    `${analysis.target_faces.length} `
                    + "target face(s) detected"
                )
                : "Detection is not available for this target yet"
        );

        copy.append(
            title,
            subtitle,
        );


        header.append(
            image,
            copy,
        );

        header.addEventListener(
            "click",
            () => {
                selectTargetImage(
                    targetImageId,
                );
            },
        );

        section.append(
            header,
        );

        if (
            !analysis
            || !analysis.analysis_completed
            || analysis.target_faces.length === 0
        ) {
            section.append(
                createEmptyState(
                    "Detection required",
                    "Click this target image above, then run Detect active target.",
                    true,
                ),
            );

            container.append(
                section,
            );

            renderedAnySection = true;
            continue;
        }

        for (
            const targetFace
            of analysis.target_faces
        ) {
            section.append(
                createMappingRow(
                    targetImageId,
                    targetFace,
                    analysis.mappings,
                ),
            );
        }

        container.append(
            section,
        );

        renderedAnySection = true;
    }

    if (!renderedAnySection) {
        container.append(
            createEmptyState(
                "No mappings available",
                "Select target images and run detection.",
                true,
            ),
        );
    }
}

function createMappingRow(
    targetImageId,
    targetFace,
    mappings,
) {
    const row = document.createElement(
        "div",
    );

    row.className = "mapping-row mapping-row-visual";

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
        "Choose a source face"
    );

    targetCopy.append(
        targetTitle,
        targetCaption,
    );

    targetBlock.append(
        targetImage,
        targetCopy,
    );

    const sourcePicker = document.createElement(
        "div",
    );

    sourcePicker.className = "source-face-picker";

    const currentMapping = (
        mappings?.[
            String(
                targetFace.index,
            )
        ]
        ?? null
    );

    const unchangedButton = document.createElement(
        "button",
    );

    unchangedButton.type = "button";
    unchangedButton.className = (
        "source-face-option source-face-option-empty"
    );

    if (
        currentMapping === null
        || currentMapping === undefined
    ) {
        unchangedButton.classList.add(
            "is-selected",
        );
    }

    unchangedButton.textContent = "Keep";

    unchangedButton.addEventListener(
        "click",
        () => {
            updateMapping(
                targetImageId,
                targetFace.index,
                null,
            );
        },
    );

    sourcePicker.append(
        unchangedButton,
    );

    for (
        const sourceFace
        of state.session.source_faces
    ) {
        const option = document.createElement(
            "button",
        );

        option.type = "button";
        option.className = "source-face-option";

        if (
            currentMapping !== null
            && currentMapping !== undefined
            && Number(currentMapping) === sourceFace.index
        ) {
            option.classList.add(
                "is-selected",
            );
        }

        const image = document.createElement(
            "img",
        );

        image.src = sourceFace.image_url;
        image.alt = sourceFace.label;
        image.loading = "lazy";

        const label = document.createElement(
            "span",
        );

        label.textContent = (
            `Source ${sourceFace.index + 1}`
        );

        option.append(
            image,
            label,
        );

        option.addEventListener(
            "click",
            () => {
                updateMapping(
                    targetImageId,
                    targetFace.index,
                    sourceFace.index,
                );
            },
        );

        sourcePicker.append(
            option,
        );
    }

    row.append(
        targetBlock,
        sourcePicker,
    );

    return row;
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
        "hyperswap_1a_256",
        "hyperswap_1b_256",
        "uniface_256",
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

async function deleteGeneratedResult(
    targetImageId,
) {
    if (
        state.busy
        || !state.session
    ) {
        return;
    }

    await runBusyOperation(
        "Removing generated result",
        "Deleting this result from the current session.",
        async () => {
            state.session = await apiRequest(
                (
                    `/api/sessions/`
                    + `${state.sessionId}/`
                    + "results/"
                    + targetImageId
                ),
                {
                    method: "DELETE",
                },
            );

            clampResultIndex();
            renderSession();

            setStatus(
                "Generated result removed.",
            );
        },
    );
}

function currentGeneratedResults() {
    return (
        state.session?.generated_results
        || []
    );
}


function clampResultIndex() {
    const results = currentGeneratedResults();

    if (results.length === 0) {
        state.resultIndex = 0;
        return;
    }

    if (state.resultIndex < 0) {
        state.resultIndex = results.length - 1;
    }

    if (state.resultIndex >= results.length) {
        state.resultIndex = 0;
    }
}


function moveResultPreview(
    direction,
) {
    const results = currentGeneratedResults();

    if (results.length <= 1) {
        return;
    }

    state.resultIndex += direction;

    clampResultIndex();
    renderResult();
}


function setResultPreviewByTarget(
    targetImageId,
) {
    const results = currentGeneratedResults();

    const index = results.findIndex(
        result => (
            result.target_image_id
            === targetImageId
        ),
    );

    if (index >= 0) {
        state.resultIndex = index;
        renderResult();
    }
}

function renderResult() {
    const generatedResults = currentGeneratedResults();

    elements["generated-result-list"].replaceChildren();

    if (generatedResults.length > 0) {
        clampResultIndex();

        const activeResult = (
            generatedResults[
                state.resultIndex
            ]
        );

        elements["result-image"].src = (
            activeResult.url
            + `?v=${Date.now()}`
        );

        elements["result-image"].hidden = false;
        elements["result-placeholder"].hidden = true;

        elements["result-preview"].disabled = false;
        
        elements["result-preview"].setAttribute(
            "aria-disabled",
            "false",
        );

        elements["download-button"].href = (
            `/api/sessions/${state.sessionId}/download`
        );

        elements["download-button"].classList.remove(
            "is-hidden",
        );

        elements["download-button"].textContent = (
            generatedResults.length > 1
                ? "Download archive"
                : "Download active"
        );

        elements["result-counter"].textContent = (
            `${state.resultIndex + 1} / ${generatedResults.length}`
        );

        elements["result-previous-button"].hidden = (
            generatedResults.length <= 1
        );

        elements["result-next-button"].hidden = (
            generatedResults.length <= 1
        );

        elements["result-counter"].hidden = (
            generatedResults.length <= 1
        );

        for (const result of generatedResults) {
            const card = document.createElement(
                "article",
            );

            card.className = "generated-result-card";

            if (
                result.target_image_id
                === activeResult.target_image_id
            ) {
                card.classList.add(
                    "is-active",
                );
            }

            const removeButton = document.createElement(
                "button",
            );

            removeButton.type = "button";
            removeButton.className = (
                "generated-result-remove"
            );

            removeButton.setAttribute(
                "aria-label",
                `Remove ${result.target_image_name}`,
            );

            removeButton.textContent = "×";

            removeButton.addEventListener(
                "click",
                event => {
                    event.stopPropagation();

                    deleteGeneratedResult(
                        result.target_image_id,
                    );
                },
            );

            const imageButton = document.createElement(
                "button",
            );

            imageButton.type = "button";
            imageButton.className = (
                "generated-result-preview"
            );

            const image = document.createElement(
                "img",
            );

            image.src = (
                result.url
                + `?v=${Date.now()}`
            );

            image.alt = result.target_image_name;
            image.loading = "lazy";

            imageButton.append(
                image,
            );

            imageButton.addEventListener(
                "click",
                () => {
                    openImageDialog(
                        image.src,
                        result.target_image_name,
                    );
                },
            );

            const footer = document.createElement(
                "div",
            );

            footer.className = (
                "generated-result-footer"
            );

            const name = document.createElement(
                "button",
            );

            name.type = "button";
            name.textContent = (
                result.target_image_name
            );

            name.addEventListener(
                "click",
                () => {
                    setResultPreviewByTarget(
                        result.target_image_id,
                    );
                },
            );

            const download = document.createElement(
                "a",
            );

            download.href = result.download_url;
            download.download = "";
            download.textContent = "Download";

            footer.append(
                name,
                download,
            );

            card.append(
                removeButton,
                imageButton,
                footer,
            );

            elements["generated-result-list"].append(
                card,
            );
        }

        return;
    }

    elements["result-image"].removeAttribute(
        "src",
    );

    elements["result-image"].hidden = true;
    elements["result-placeholder"].hidden = false;

    elements["result-preview"].disabled = true;
    
    elements["result-preview"].setAttribute(
        "aria-disabled",
        "true",
    );

    elements["download-button"].classList.add(
        "is-hidden",
    );

    elements["download-button"].removeAttribute(
        "href",
    );

    elements["download-button"].textContent = (
        "Download active"
    );
    
    elements["result-previous-button"].hidden = true;
    elements["result-next-button"].hidden = true;
    elements["result-counter"].hidden = true;
    elements["result-counter"].textContent = "";
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