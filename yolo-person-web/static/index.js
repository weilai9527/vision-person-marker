const openApiButton = document.getElementById("openApiButton");
const closeApiButton = document.getElementById("closeApiButton");
const cancelApiButton = document.getElementById("cancelApiButton");
const apiModal = document.getElementById("apiModal");
const provider = document.getElementById("provider");
const apiUrl = document.getElementById("apiUrl");
const apiKey = document.getElementById("apiKey");
const model = document.getElementById("model");
const dropzone = document.getElementById("dropzone");
const input = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const form = document.getElementById("uploadForm");
const submitButton = document.getElementById("submitButton");
const uploadImagePreview = document.getElementById("uploadImagePreview");
const uploadPreviewImage = document.getElementById("uploadPreviewImage");

function openApiModal() {
    apiModal.classList.add("open");
    apiModal.setAttribute("aria-hidden", "false");
    apiKey.focus();
}

function closeApiModal() {
    apiModal.classList.remove("open");
    apiModal.setAttribute("aria-hidden", "true");
    openApiButton.focus();
}

function updatePreview(file) {
    if (!file) {
        fileName.textContent = "还没有选择图片";
        dropzone.classList.remove("has-preview");
        uploadImagePreview.hidden = true;
        uploadPreviewImage.removeAttribute("src");
        return;
    }

    fileName.textContent = `已选择: ${file.name}`;
    const imageUrl = URL.createObjectURL(file);
    uploadPreviewImage.onload = () => URL.revokeObjectURL(imageUrl);
    uploadPreviewImage.src = imageUrl;
    uploadImagePreview.hidden = false;
    dropzone.classList.add("has-preview");
}

if (openApiButton) {
    openApiButton.addEventListener("click", openApiModal);
    closeApiButton.addEventListener("click", closeApiModal);
    cancelApiButton.addEventListener("click", closeApiModal);

    provider.addEventListener("change", () => {
        const selectedProvider = provider.options[provider.selectedIndex];
        apiUrl.value = selectedProvider.dataset.url || apiUrl.value;
        model.value = selectedProvider.dataset.model || model.value;
    });

    apiModal.addEventListener("click", (event) => {
        if (event.target === apiModal) {
            closeApiModal();
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && apiModal.classList.contains("open")) {
            closeApiModal();
        }
    });
}

if (input) {
    input.addEventListener("change", () => updatePreview(input.files[0]));

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
        if (!files || files.length === 0) {
            return;
        }
        input.files = files;
        updatePreview(files[0]);
    });

    form.addEventListener("submit", () => {
        submitButton.disabled = true;
        submitButton.textContent = "检测中...";
    });
}
