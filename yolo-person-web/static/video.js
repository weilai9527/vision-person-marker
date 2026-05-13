const form = document.getElementById("videoForm");
const input = document.getElementById("videoInput");
const fileName = document.getElementById("videoFileName");
const fileMeta = document.getElementById("videoFileMeta");
const submitButton = document.getElementById("videoSubmitButton");
const progressFill = document.getElementById("progressFill");
const progressPercent = document.getElementById("progressPercent");
const progressText = document.getElementById("progressText");
const videoResult = document.getElementById("videoResult");
const historyList = document.getElementById("videoHistoryList");
const refreshButton = document.getElementById("refreshVideoHistory");
const dropzone = document.getElementById("videoDropzone");
const uploadVideoPreview = document.getElementById("uploadVideoPreview");
const uploadPreviewVideo = document.getElementById("uploadPreviewVideo");
const uploadPreviewCaption = document.getElementById("uploadPreviewCaption");
const videoStats = document.getElementById("videoStats");
const currentPersons = document.getElementById("currentPersons");
const sumCount = document.getElementById("sumCount");
const pageConfig = document.body.dataset || {};
const apiBase = pageConfig.videoApiBase || "/api/video";
const sharedApiBase = pageConfig.sharedVideoApiBase || "/api/video";
const detectionTarget = pageConfig.detectionTarget || "person";
const allowedVideoExtensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"];
const maxVideoSizeMB = 500;
let progressTimer = null;
let previewTimer = null;
let previewToken = 0;
let previewFrameObjectUrl = null;
let uploadPreviewObjectUrl = null;

const text = {
    noVideo: "\u8fd8\u6ca1\u6709\u9009\u62e9\u89c6\u9891",
    selected: "\u5df2\u9009\u62e9",
    streamPreview: "\u517c\u5bb9\u9884\u89c8",
    download: "\u4e0b\u8f7d\u7ed3\u679c",
    annotationVideo: "\u62c9\u6846\u6807\u6ce8",
    play: "\u64ad\u653e",
    pause: "\u6682\u505c",
    currentPersons: "\u5f53\u524d\u753b\u9762\u4eba\u6570",
    usingStream: "\u6b63\u5728\u4f7f\u7528\u517c\u5bb9\u9884\u89c8",
    cannotPreview: "\u89c6\u9891\u8fd8\u4e0d\u80fd\u9884\u89c8",
    noRecords: "\u6682\u65e0\u89c6\u9891\u8bb0\u5f55\u3002",
    frame: "\u5e27",
    preview: "\u9884\u89c8",
    delete: "\u5220\u9664",
    confirmDelete: "\u786e\u5b9a\u5220\u9664\u8fd9\u6761\u89c6\u9891\u8bb0\u5f55\u5417\uff1f",
    waiting: "\u7b49\u5f85\u4e0a\u4f20\u89c6\u9891",
    failed: "\u5904\u7406\u5931\u8d25",
    uploading: "\u6b63\u5728\u4e0a\u4f20...",
    uploadFailed: "\u4e0a\u4f20\u5931\u8d25",
    starting: "\u5df2\u4e0a\u4f20\uff0c\u6b63\u5728\u5f00\u59cb\u5904\u7406...",
    previewAlt: "\u89c6\u9891\u9884\u89c8",
    maxSize: "\u6700\u5927 500MB",
    unsupportedType: "\u683c\u5f0f\u4e0d\u652f\u6301\uff0c\u8bf7\u9009\u62e9 MP4\u3001AVI\u3001MOV\u3001MKV\u3001WEBM \u6216 FLV",
    tooLarge: "\u6587\u4ef6\u8fc7\u5927\uff0c\u4e0d\u80fd\u8d85\u8fc7 500MB",
    uploadPreview: "\u4e0a\u4f20\u9884\u89c8",
};

const endpoints = {
    history: `${apiBase}/history`,
    upload: `${apiBase}/upload`,
    progress: (recordId) => `${sharedApiBase}/progress/${recordId}`,
    result: (recordId) => `${sharedApiBase}/result/${recordId}`,
    frame: (recordId) => `${sharedApiBase}/frame/${recordId}`,
    download: (recordId) => `${sharedApiBase}/download/${recordId}`,
    delete: (recordId) => `${sharedApiBase}/delete/${recordId}`,
};

if (detectionTarget === "vehicle") {
    Object.assign(text, {
        currentPersons: "\u5f53\u524d\u753b\u9762\u8f66\u8f86\u6570",
        noRecords: "\u6682\u65e0\u8f66\u8f86\u89c6\u9891\u8bb0\u5f55\u3002",
        waiting: "\u7b49\u5f85\u4e0a\u4f20\u8f66\u8f86\u89c6\u9891",
        starting: "\u5df2\u4e0a\u4f20\uff0c\u6b63\u5728\u5f00\u59cb\u8f66\u8f86\u68c0\u6d4b...",
        annotationVideo: "\u8f66\u8f86\u62c9\u6846\u6807\u6ce8",
    });
}

function formatFileSize(bytes) {
    if (!bytes) {
        return "0 MB";
    }
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function isAllowedVideo(file) {
    const lowerName = file.name.toLowerCase();
    return allowedVideoExtensions.some((extension) => lowerName.endsWith(extension));
}

function clearUploadPreview() {
    uploadPreviewVideo.pause();
    uploadPreviewVideo.removeAttribute("src");
    uploadPreviewCaption.textContent = "";
    uploadVideoPreview.hidden = true;
    dropzone.classList.remove("has-preview");
    if (uploadPreviewObjectUrl) {
        URL.revokeObjectURL(uploadPreviewObjectUrl);
        uploadPreviewObjectUrl = null;
    }
}

function renderUploadPreview(file) {
    clearUploadPreview();
    if (!file) {
        return;
    }
    uploadPreviewObjectUrl = URL.createObjectURL(file);
    uploadPreviewVideo.src = uploadPreviewObjectUrl;
    uploadPreviewCaption.textContent = `${text.uploadPreview}: ${file.name}`;
    uploadVideoPreview.hidden = false;
    dropzone.classList.add("has-preview");
}

function validateVideoFile(file) {
    if (!file) {
        return "";
    }
    if (!isAllowedVideo(file)) {
        return text.unsupportedType;
    }
    if (file.size > maxVideoSizeMB * 1024 * 1024) {
        return text.tooLarge;
    }
    return "";
}

function setFileState(file, errorMessage = "") {
    dropzone.classList.toggle("is-selected", Boolean(file) && !errorMessage);
    dropzone.classList.toggle("is-error", Boolean(errorMessage));
    fileMeta.classList.toggle("error", Boolean(errorMessage));

    if (!file) {
        fileName.textContent = text.noVideo;
        fileMeta.textContent = text.maxSize;
        submitButton.disabled = true;
        clearUploadPreview();
        return;
    }

    fileName.textContent = `${text.selected}: ${file.name}`;
    fileMeta.textContent = errorMessage || `${formatFileSize(file.size)} / ${text.maxSize}`;
    submitButton.disabled = Boolean(errorMessage);
    if (errorMessage) {
        clearUploadPreview();
    } else {
        renderUploadPreview(file);
    }
}

function setProgress(value, message) {
    const percent = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
    progressFill.style.width = `${percent}%`;
    progressPercent.textContent = `${percent}%`;
    progressText.textContent = message;
}

function setStats(stats = {}) {
    currentPersons.textContent = stats.current_person_count || 0;
    if (sumCount && (Object.keys(stats).length === 0 || "unique_count" in stats || "total_count" in stats)) {
        sumCount.textContent = stats.unique_count || stats.total_count || 0;
    }
}

function showStats() {
    videoStats.hidden = false;
}

function hideStats() {
    videoStats.hidden = true;
    setStats();
}

function stopPreviewPlayback() {
    previewToken += 1;
    if (previewTimer) {
        clearTimeout(previewTimer);
        previewTimer = null;
    }
    if (previewFrameObjectUrl) {
        URL.revokeObjectURL(previewFrameObjectUrl);
        previewFrameObjectUrl = null;
    }
}

function buildFrameUrl(frameUrl, frameIndex) {
    const separator = frameUrl.includes("?") ? "&" : "?";
    return `${frameUrl}${separator}frame=${frameIndex}&t=${Date.now()}`;
}

function setPreviewImage(image, blob) {
    const previousObjectUrl = previewFrameObjectUrl;
    const nextObjectUrl = URL.createObjectURL(blob);
    previewFrameObjectUrl = nextObjectUrl;
    image.onload = () => {
        if (previousObjectUrl) {
            URL.revokeObjectURL(previousObjectUrl);
        }
    };
    image.src = nextObjectUrl;
}

async function toggleFullscreen(element) {
    if (!element) {
        return;
    }

    try {
        if (document.fullscreenElement || document.webkitFullscreenElement) {
            if (document.exitFullscreen) {
                await document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            }
            return;
        }

        if (element.requestFullscreen) {
            await element.requestFullscreen();
        } else if (element.webkitRequestFullscreen) {
            element.webkitRequestFullscreen();
        }
    } catch {
        // Browser can deny fullscreen if the user gesture is interrupted.
    }
}

function startFramePreview(frameUrl, previewImage, playButton) {
    stopPreviewPlayback();
    const token = previewToken;
    let frameIndex = 0;
    let isPaused = false;
    let isLoading = false;

    const setButtonState = () => {
        playButton.textContent = isPaused ? text.play : text.pause;
        playButton.setAttribute("aria-pressed", isPaused ? "true" : "false");
    };

    const renderNextFrame = async () => {
        if (token !== previewToken || isPaused || isLoading) {
            return;
        }

        try {
            isLoading = true;
            const response = await fetch(buildFrameUrl(frameUrl, frameIndex), { cache: "no-store" });
            if (!response.ok) {
                throw new Error("Frame request failed");
            }
            if (token !== previewToken || isPaused) {
                return;
            }

            const totalFrames = Number(response.headers.get("X-Frame-Total") || 0);
            const fps = Number(response.headers.get("X-Fps") || 25);
            const personCount = Number(response.headers.get("X-Detection-Count") || response.headers.get("X-Person-Count") || 0);
            const blob = await response.blob();
            if (token !== previewToken || isPaused) {
                return;
            }

            setPreviewImage(previewImage, blob);
            setStats({ current_person_count: personCount });
            frameIndex = totalFrames > 0 ? (frameIndex + 1) % totalFrames : frameIndex + 1;

            if (!isPaused) {
                const delay = Math.min(Math.max(1000 / Math.max(fps, 1), 40), 160);
                previewTimer = setTimeout(renderNextFrame, delay);
            }
        } catch {
            if (!isPaused) {
                previewTimer = setTimeout(renderNextFrame, 500);
            }
        } finally {
            isLoading = false;
        }
    };

    playButton.addEventListener("click", () => {
        isPaused = !isPaused;
        if (isPaused && previewTimer) {
            clearTimeout(previewTimer);
            previewTimer = null;
        }
        setButtonState();
        if (!isPaused) {
            renderNextFrame();
        }
    });

    setButtonState();
    renderNextFrame();
}

function renderPlayer(result, recordId) {
    stopPreviewPlayback();
    setStats(result);
    showStats();
    videoResult.innerHTML = `
        <section class="preview-player fullscreen-preview" title="双击全屏">
            <h3>${text.annotationVideo}</h3>
            <img id="streamPreviewImage" class="stream-preview" alt="${text.previewAlt}">
        </section>
        <div class="record-actions">
            <button class="button-secondary" type="button" id="previewPlayButton">${text.pause}</button>
            <a class="button-secondary" href="${endpoints.download(recordId)}">${text.download}</a>
        </div>
    `;

    const playerPanel = videoResult.querySelector(".preview-player");
    const streamImage = document.getElementById("streamPreviewImage");
    const playButton = document.getElementById("previewPlayButton");

    playerPanel.addEventListener("dblclick", (event) => {
        event.preventDefault();
        toggleFullscreen(playerPanel);
    });
    startFramePreview(result.frame_url || result.stream_url, streamImage, playButton);
    setProgress(100, result.filename ? `${text.streamPreview}: ${result.filename}` : text.usingStream);
    videoResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function previewVideo(recordId) {
    const response = await fetch(endpoints.result(recordId));
    const result = await response.json();
    if (!result.success) {
        alert(result.error || text.cannotPreview);
        return;
    }
    renderPlayer(result, recordId);
}

function bindDropzone() {
    input.addEventListener("change", () => {
        const file = input.files[0];
        setFileState(file, validateVideoFile(file));
    });

    uploadPreviewVideo.addEventListener("click", (event) => event.stopPropagation());
    uploadPreviewVideo.addEventListener("mousedown", (event) => event.stopPropagation());
    uploadPreviewVideo.addEventListener("touchstart", (event) => event.stopPropagation());

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "dragend", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (event) => {
        const files = event.dataTransfer.files;
        if (files && files.length > 0) {
            const file = files[0];
            const errorMessage = validateVideoFile(file);
            if (!errorMessage) {
                input.files = files;
            } else {
                input.value = "";
            }
            setFileState(file, errorMessage);
        }
    });
}

async function loadHistory() {
    const response = await fetch(endpoints.history);
    const data = await response.json();
    if (!data.records.length) {
        historyList.textContent = text.noRecords;
        return;
    }

    historyList.innerHTML = data.records.map((record) => `
        <article class="video-record">
            <div>
                <strong>${record.original_filename}</strong>
                <p>${record.uploaded_at} · ${record.status} · ${record.processed_frames || 0}/${record.total_frames || 0} ${text.frame}${record.total_count ? ` · 总数${record.total_count}` : ""}</p>
            </div>
            <div class="record-actions">
                ${record.has_result ? `<button class="button-secondary" type="button" data-preview-video="${record.id}">${text.preview}</button>` : ""}
                ${record.has_result ? `<a class="button-secondary" href="${endpoints.download(record.id)}">${text.download}</a>` : ""}
                <button class="button-danger" type="button" data-delete-video="${record.id}">${text.delete}</button>
            </div>
        </article>
    `).join("");

    document.querySelectorAll("[data-preview-video]").forEach((button) => {
        button.addEventListener("click", () => previewVideo(button.dataset.previewVideo));
    });

    document.querySelectorAll("[data-delete-video]").forEach((button) => {
        button.addEventListener("click", async () => {
            if (!confirm(text.confirmDelete)) {
                return;
            }
            await fetch(endpoints.delete(button.dataset.deleteVideo), { method: "DELETE" });
            stopPreviewPlayback();
            hideStats();
            videoResult.innerHTML = "";
            setProgress(0, text.waiting);
            loadHistory();
        });
    });
}

async function pollProgress(recordId) {
    const response = await fetch(endpoints.progress(recordId));
    const progress = await response.json();
    setProgress(progress.progress || 0, progress.message || progress.status);
    setStats(progress);
    if (progress.status === "processing" && Number(progress.current_frame || 0) > 0) {
        showStats();
    }
    if (progress.status === "completed") {
        clearInterval(progressTimer);
        submitButton.disabled = false;
        await previewVideo(recordId);
        loadHistory();
    } else if (progress.status === "failed") {
        clearInterval(progressTimer);
        submitButton.disabled = false;
        setProgress(progress.progress || 0, progress.message || text.failed);
        loadHistory();
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const selectedFile = input.files[0];
    const fileError = validateVideoFile(selectedFile);
    if (!selectedFile || fileError) {
        setFileState(selectedFile, fileError);
        return;
    }
    submitButton.disabled = true;
    stopPreviewPlayback();
    hideStats();
    videoResult.innerHTML = "";
    setProgress(0, text.uploading);

    const formData = new FormData();
    formData.append("video", selectedFile);
    const response = await fetch(endpoints.upload, { method: "POST", body: formData });
    const data = await response.json();
    if (!data.success) {
        submitButton.disabled = false;
        setProgress(0, data.error || text.uploadFailed);
        return;
    }
    setProgress(0, text.starting);
    progressTimer = setInterval(() => pollProgress(data.record_id), 1000);
    pollProgress(data.record_id);
});

refreshButton.addEventListener("click", loadHistory);
bindDropzone();
loadHistory();
