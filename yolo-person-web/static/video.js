const form = document.getElementById("videoForm");
const input = document.getElementById("videoInput");
const fileName = document.getElementById("videoFileName");
const submitButton = document.getElementById("videoSubmitButton");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const videoResult = document.getElementById("videoResult");
const historyList = document.getElementById("videoHistoryList");
const refreshButton = document.getElementById("refreshVideoHistory");
const dropzone = document.getElementById("videoDropzone");
let progressTimer = null;

const text = {
    noVideo: "\u8fd8\u6ca1\u6709\u9009\u62e9\u89c6\u9891",
    selected: "\u5df2\u9009\u62e9",
    streamPreview: "\u517c\u5bb9\u9884\u89c8",
    nativePlayer: "\u539f\u751f\u64ad\u653e\u5668",
    nativePlay: "\u539f\u751f\u64ad\u653e",
    download: "\u4e0b\u8f7d\u7ed3\u679c",
    usingStream: "\u6b63\u5728\u4f7f\u7528\u517c\u5bb9\u9884\u89c8",
    usingNative: "\u6b63\u5728\u4f7f\u7528\u539f\u751f\u64ad\u653e\u5668",
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
};

function setProgress(value, message) {
    progressFill.style.width = `${value}%`;
    progressText.textContent = message;
}

function renderPlayer(url, streamUrl, recordId, filename) {
    videoResult.innerHTML = `
        <img id="streamPreviewImage" class="stream-preview" src="${streamUrl}?t=${Date.now()}" alt="${text.previewAlt}">
        <video id="nativePreviewVideo" controls preload="metadata" src="${url}" hidden></video>
        <div class="record-actions">
            <button class="button-secondary" type="button" id="nativePreviewButton">${text.nativePlayer}</button>
            <button class="button-secondary" type="button" id="compatPreviewButton">${text.streamPreview}</button>
            <a class="button-secondary" href="/api/video/download/${recordId}">${text.download}</a>
        </div>
    `;

    const nativeVideo = document.getElementById("nativePreviewVideo");
    const streamImage = document.getElementById("streamPreviewImage");
    const nativeButton = document.getElementById("nativePreviewButton");
    const compatButton = document.getElementById("compatPreviewButton");

    const showStream = () => {
        nativeVideo.hidden = true;
        nativeVideo.pause();
        streamImage.src = `${streamUrl}?t=${Date.now()}`;
        streamImage.hidden = false;
        setProgress(100, filename ? `${text.streamPreview}: ${filename}` : text.usingStream);
    };

    const showNativeVideo = () => {
        streamImage.hidden = true;
        streamImage.removeAttribute("src");
        nativeVideo.hidden = false;
        nativeVideo.load();
        nativeVideo.play().catch(() => showStream());
        setProgress(100, filename ? `${text.nativePlay}: ${filename}` : text.usingNative);
    };

    nativeVideo.addEventListener("error", showStream);
    nativeButton.addEventListener("click", showNativeVideo);
    compatButton.addEventListener("click", showStream);
    setProgress(100, filename ? `${text.streamPreview}: ${filename}` : text.usingStream);
    videoResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function previewVideo(recordId) {
    const response = await fetch(`/api/video/result/${recordId}`);
    const result = await response.json();
    if (!result.success) {
        alert(result.error || text.cannotPreview);
        return;
    }
    renderPlayer(result.url, result.stream_url, recordId, result.filename);
}

function bindDropzone() {
    input.addEventListener("change", () => {
        fileName.textContent = input.files[0] ? `${text.selected}: ${input.files[0].name}` : text.noVideo;
    });

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
            input.files = files;
            fileName.textContent = `${text.selected}: ${files[0].name}`;
        }
    });
}

async function loadHistory() {
    const response = await fetch("/api/video/history");
    const data = await response.json();
    if (!data.records.length) {
        historyList.textContent = text.noRecords;
        return;
    }

    historyList.innerHTML = data.records.map((record) => `
        <article class="video-record">
            <div>
                <strong>${record.original_filename}</strong>
                <p>${record.uploaded_at} · ${record.status} · ${record.processed_frames || 0}/${record.total_frames || 0} ${text.frame}</p>
            </div>
            <div class="record-actions">
                ${record.has_result ? `<button class="button-secondary" type="button" data-preview-video="${record.id}">${text.preview}</button>` : ""}
                ${record.has_result ? `<a class="button-secondary" href="/api/video/download/${record.id}">${text.download}</a>` : ""}
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
            await fetch(`/api/video/delete/${button.dataset.deleteVideo}`, { method: "DELETE" });
            videoResult.innerHTML = "";
            setProgress(0, text.waiting);
            loadHistory();
        });
    });
}

async function pollProgress(recordId) {
    const response = await fetch(`/api/video/progress/${recordId}`);
    const progress = await response.json();
    setProgress(progress.progress || 0, progress.message || progress.status);
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
    if (!input.files[0]) {
        return;
    }
    submitButton.disabled = true;
    videoResult.innerHTML = "";
    setProgress(0, text.uploading);

    const formData = new FormData();
    formData.append("video", input.files[0]);
    const response = await fetch("/api/video/upload", { method: "POST", body: formData });
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
