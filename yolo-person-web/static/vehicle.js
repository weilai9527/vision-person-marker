const dropzone = document.getElementById("vehicleDropzone");
const input = document.getElementById("vehicleImageInput");
const fileName = document.getElementById("vehicleFileName");
const form = document.getElementById("vehicleForm");
const submitButton = document.getElementById("vehicleSubmitButton");
const previewEmpty = document.getElementById("vehiclePreviewEmpty");
const previewImage = document.getElementById("vehiclePreviewImage");

function updatePreview(file) {
    if (!file) {
        fileName.textContent = "\u8fd8\u6ca1\u6709\u9009\u62e9\u56fe\u7247";
        previewImage.removeAttribute("src");
        previewImage.style.display = "none";
        previewEmpty.style.display = "grid";
        dropzone.classList.remove("is-selected");
        return;
    }

    fileName.textContent = `\u5df2\u9009\u62e9: ${file.name}`;
    dropzone.classList.add("is-selected");
    const imageUrl = URL.createObjectURL(file);
    previewImage.onload = () => URL.revokeObjectURL(imageUrl);
    previewImage.src = imageUrl;
    previewImage.style.display = "block";
    previewEmpty.style.display = "none";
}

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
    submitButton.textContent = "\u68c0\u6d4b\u4e2d...";
});
