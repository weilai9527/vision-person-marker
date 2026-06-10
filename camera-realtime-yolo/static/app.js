(function () {
    const video = document.getElementById("cameraVideo");
    const detectVideo = document.getElementById("detectVideo");
    const overlay = document.getElementById("overlay");
    const overlayContext = overlay.getContext("2d");
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const monitorButton = document.getElementById("monitorButton");
    const targetButton = document.getElementById("targetButton");
    const recordButton = document.getElementById("recordButton");
    const cameraSelect = document.getElementById("cameraSelect");
    const cameraPlaceholder = document.getElementById("cameraPlaceholder");
    const detectPlaceholder = document.getElementById("detectPlaceholder");
    const statusText = document.getElementById("statusText");
    const statusDot = document.getElementById("statusDot");
    const countText = document.getElementById("countText");
    const countLabel = document.getElementById("countLabel");
    const detectMeta = document.getElementById("detectMeta");
    const targetPill = document.getElementById("targetPill");
    const toast = document.getElementById("toast");
    const recordPanel = document.getElementById("recordPanel");
    const recordPreview = document.getElementById("recordPreview");
    const recordDownload = document.getElementById("recordDownload");
    const recordMeta = document.getElementById("recordMeta");

    const captureCanvas = document.createElement("canvas");
    const captureContext = captureCanvas.getContext("2d", { willReadFrequently: true });

    let stream = null;
    let devices = [];
    let target = "person";
    let monitoring = false;
    let detecting = false;
    let detectTimer = null;
    let toastTimer = null;
    let lastDetection = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let recording = false;
    let recordStartedAt = 0;
    let recordTimer = null;
    let recordUrl = null;

    function isSecureCameraContext() {
        const host = window.location.hostname;
        return window.location.protocol === "https:" || host === "localhost" || host === "127.0.0.1" || host === "::1";
    }

    function showToast(message, type = "") {
        window.clearTimeout(toastTimer);
        toast.textContent = message;
        toast.className = `toast ${type} show`;
        toastTimer = window.setTimeout(() => toast.classList.remove("show"), 3200);
    }

    function setStatus(text, type = "idle") {
        statusText.textContent = text;
        statusDot.className = `status-dot ${type}`;
    }

    function setPlaceholders(visible) {
        cameraPlaceholder.classList.toggle("hidden", !visible);
        detectPlaceholder.classList.toggle("hidden", !visible);
    }

    function updateTargetUi() {
        const isPerson = target === "person";
        targetButton.textContent = isPerson ? "切换车辆" : "切换人员";
        targetPill.textContent = isPerson ? "人员检测" : "车辆检测";
        countLabel.textContent = isPerson ? "当前人数" : "当前车辆数";
    }

    function resizeOverlay() {
        const rect = overlay.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.round(rect.width * ratio));
        const height = Math.max(1, Math.round(rect.height * ratio));
        if (overlay.width !== width || overlay.height !== height) {
            overlay.width = width;
            overlay.height = height;
        }
        overlayContext.setTransform(ratio, 0, 0, ratio, 0, 0);
        return rect;
    }

    function clearDetections() {
        const rect = resizeOverlay();
        overlayContext.clearRect(0, 0, rect.width, rect.height);
        lastDetection = null;
        countText.textContent = "0";
    }

    function drawDetections(detection) {
        const rect = resizeOverlay();
        overlayContext.clearRect(0, 0, rect.width, rect.height);
        if (!detection || !Array.isArray(detection.boxes) || detection.boxes.length === 0) {
            return;
        }

        const refWidth = detectVideo.videoWidth || detection.width;
        const refHeight = detectVideo.videoHeight || detection.height;
        const scale = Math.min(rect.width / refWidth, rect.height / refHeight);
        const drawWidth = refWidth * scale;
        const drawHeight = refHeight * scale;
        const offsetX = (rect.width - drawWidth) / 2;
        const offsetY = (rect.height - drawHeight) / 2;
        const color = target === "person" ? "#00d46a" : "#2f8cff";

        overlayContext.lineWidth = 2;
        overlayContext.font = "13px Microsoft YaHei, Arial, sans-serif";
        overlayContext.textBaseline = "top";

        detection.boxes.forEach((box, index) => {
            const boxX1 = box.x1 * (refWidth / detection.width);
            const boxY1 = box.y1 * (refHeight / detection.height);
            const boxX2 = box.x2 * (refWidth / detection.width);
            const boxY2 = box.y2 * (refHeight / detection.height);
            const x = offsetX + boxX1 * scale;
            const y = offsetY + boxY1 * scale;
            const width = (boxX2 - boxX1) * scale;
            const height = (boxY2 - boxY1) * scale;
            const label = `${index + 1} ${(box.conf || 0).toFixed(2)}`;
            const labelWidth = overlayContext.measureText(label).width + 12;

            overlayContext.strokeStyle = color;
            overlayContext.fillStyle = target === "person" ? "rgba(0, 212, 106, 0.14)" : "rgba(47, 140, 255, 0.14)";
            overlayContext.strokeRect(x, y, width, height);
            overlayContext.fillRect(x, y, width, height);
            overlayContext.fillStyle = color;
            overlayContext.fillRect(x, Math.max(0, y - 20), labelWidth, 20);
            overlayContext.fillStyle = "#ffffff";
            overlayContext.fillText(label, x + 6, Math.max(0, y - 17));
        });
    }

    function captureFrameBlob() {
        const sourceWidth = video.videoWidth;
        const sourceHeight = video.videoHeight;
        if (!sourceWidth || !sourceHeight) {
            return Promise.resolve(null);
        }

        const maxSide = 960;
        const scale = Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight));
        captureCanvas.width = Math.max(1, Math.round(sourceWidth * scale));
        captureCanvas.height = Math.max(1, Math.round(sourceHeight * scale));
        captureContext.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

        return new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.85));
    }

    async function detectFrame() {
        if (!monitoring || detecting || !stream) {
            return;
        }

        detecting = true;
        detectMeta.textContent = "正在检测当前画面...";
        try {
            const blob = await captureFrameBlob();
            if (!blob) {
                detectMeta.textContent = "等待摄像头画面稳定";
                return;
            }

            const formData = new FormData();
            formData.append("frame", blob, "camera-frame.jpg");
            formData.append("target", target);

            const response = await fetch("/api/detect", {
                method: "POST",
                body: formData,
                cache: "no-store",
            });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || "检测失败");
            }

            lastDetection = data;
            countText.textContent = String(data.count || 0);
            detectMeta.textContent = `YOLO 实时检测，用时 ${data.elapsed_ms || 0} ms`;
            drawDetections(data);
        } catch (error) {
            detectMeta.textContent = error.message || "实时检测失败";
        } finally {
            detecting = false;
            if (monitoring && stream) {
                detectTimer = window.setTimeout(detectFrame, 220);
            }
        }
    }

    function startMonitoring() {
        if (!stream || monitoring) {
            return;
        }
        monitoring = true;
        monitorButton.textContent = "暂停检测";
        detectFrame();
    }

    function stopMonitoring(resetView = true) {
        monitoring = false;
        detecting = false;
        window.clearTimeout(detectTimer);
        detectTimer = null;
        monitorButton.textContent = "实时检测";
        if (resetView) {
            clearDetections();
            detectMeta.textContent = "未开始实时检测";
        }
    }

    function toggleMonitoring() {
        if (monitoring) {
            stopMonitoring(false);
            detectMeta.textContent = "实时检测已暂停";
        } else {
            startMonitoring();
        }
    }

    function toggleTarget() {
        target = target === "person" ? "vehicle" : "person";
        updateTargetUi();
        clearDetections();
        detectMeta.textContent = `已切换为${target === "person" ? "人员" : "车辆"}检测`;
    }

    function formatDuration(totalSeconds) {
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
    }

    function updateRecordButton() {
        const supported = Boolean(window.MediaRecorder);
        recordButton.disabled = !stream || !supported;
        recordButton.textContent = recording ? `停止录像 ${formatDuration(Math.floor((Date.now() - recordStartedAt) / 1000))}` : "开始录像";
        recordButton.classList.toggle("danger", recording);
        recordButton.classList.toggle("secondary", !recording);
    }

    function getRecordingMimeType() {
        const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
        return types.find((type) => window.MediaRecorder && MediaRecorder.isTypeSupported(type)) || "";
    }

    function clearRecordingUrl() {
        if (recordUrl) {
            URL.revokeObjectURL(recordUrl);
        }
        recordUrl = null;
    }

    function finishRecording() {
        window.clearInterval(recordTimer);
        recordTimer = null;
        recording = false;
        updateRecordButton();
        if (recordedChunks.length === 0) {
            recordMeta.textContent = "没有生成有效录像";
            mediaRecorder = null;
            return;
        }

        clearRecordingUrl();
        const blob = new Blob(recordedChunks, { type: recordedChunks[0].type || "video/webm" });
        const duration = Math.max(1, Math.round((Date.now() - recordStartedAt) / 1000));
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        recordUrl = URL.createObjectURL(blob);
        recordPreview.src = recordUrl;
        recordDownload.href = recordUrl;
        recordDownload.download = `camera-recording-${timestamp}.webm`;
        recordMeta.textContent = `时长 ${formatDuration(duration)}，大小 ${(blob.size / 1024 / 1024).toFixed(2)} MB`;
        recordPanel.hidden = false;
        mediaRecorder = null;
    }

    function startRecording() {
        if (!stream || recording) {
            return;
        }
        if (!window.MediaRecorder) {
            showToast("当前浏览器不支持录像", "error");
            return;
        }

        try {
            const mimeType = getRecordingMimeType();
            recordedChunks = [];
            clearRecordingUrl();
            mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
            mediaRecorder.addEventListener("dataavailable", (event) => {
                if (event.data && event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            });
            mediaRecorder.addEventListener("stop", finishRecording);
            mediaRecorder.start(1000);
            recording = true;
            recordStartedAt = Date.now();
            recordPanel.hidden = true;
            updateRecordButton();
            recordTimer = window.setInterval(updateRecordButton, 500);
            showToast("录像已开始", "success");
        } catch (error) {
            recording = false;
            updateRecordButton();
            showToast(error.message || "录像启动失败", "error");
        }
    }

    function stopRecording(showNotice = true) {
        window.clearInterval(recordTimer);
        recordTimer = null;
        if (!mediaRecorder || mediaRecorder.state === "inactive") {
            recording = false;
            updateRecordButton();
            return;
        }
        mediaRecorder.stop();
        if (showNotice) {
            showToast("录像已停止", "success");
        }
    }

    function toggleRecording() {
        if (recording) {
            stopRecording(true);
        } else {
            startRecording();
        }
    }

    function renderDeviceOptions() {
        cameraSelect.innerHTML = "";
        if (devices.length === 0) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "未检测到摄像头";
            cameraSelect.appendChild(option);
            cameraSelect.disabled = true;
            return;
        }

        devices.forEach((device, index) => {
            const option = document.createElement("option");
            option.value = device.deviceId;
            option.textContent = device.label || `摄像头 ${index + 1}`;
            cameraSelect.appendChild(option);
        });
        cameraSelect.disabled = false;
    }

    async function refreshDevices() {
        try {
            const allDevices = await navigator.mediaDevices.enumerateDevices();
            devices = allDevices.filter((device) => device.kind === "videoinput");
            renderDeviceOptions();
            setStatus(devices.length > 0 ? `就绪，检测到 ${devices.length} 个摄像头` : "未检测到摄像头", devices.length > 0 ? "idle" : "error");
        } catch (error) {
            setStatus("设备检测失败", "error");
            showToast(error.message || "无法读取摄像头设备列表", "error");
        }
    }

    async function startCamera() {
        if (stream) {
            return;
        }

        setStatus("正在请求权限...", "idle");
        const selectedDeviceId = cameraSelect.value;
        const videoConstraint = selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true;

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: videoConstraint,
                audio: false,
            });

            video.srcObject = stream;
            detectVideo.srcObject = stream;
            await Promise.all([video.play(), detectVideo.play()]);

            setPlaceholders(false);
            startButton.disabled = true;
            stopButton.disabled = false;
            monitorButton.disabled = false;
            targetButton.disabled = false;
            recordButton.disabled = !window.MediaRecorder;
            cameraSelect.disabled = true;
            setStatus("摄像头已开启", "active");
            showToast("摄像头已开启", "success");

            await refreshDevices();
            cameraSelect.disabled = true;
            const activeDeviceId = stream.getVideoTracks()[0].getSettings().deviceId;
            if (activeDeviceId) {
                cameraSelect.value = activeDeviceId;
            }
            stream.getVideoTracks()[0].addEventListener("ended", stopCamera);
            startMonitoring();
        } catch (error) {
            handleCameraError(error);
        }
    }

    function stopCurrentStream() {
        stopRecording(false);
        stopMonitoring(true);
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
        }
        stream = null;
        video.srcObject = null;
        detectVideo.srcObject = null;
        updateRecordButton();
    }

    function stopCamera() {
        stopCurrentStream();
        setPlaceholders(true);
        startButton.disabled = devices.length === 0;
        stopButton.disabled = true;
        monitorButton.disabled = true;
        targetButton.disabled = true;
        recordButton.disabled = true;
        cameraSelect.disabled = devices.length === 0;
        setStatus(devices.length > 0 ? "已停止" : "未检测到摄像头", devices.length > 0 ? "idle" : "error");
    }

    function handleCameraError(error) {
        const messages = {
            NotAllowedError: ["权限被拒绝", "请在浏览器设置中允许本网站访问摄像头"],
            PermissionDeniedError: ["权限被拒绝", "请在浏览器设置中允许本网站访问摄像头"],
            NotFoundError: ["未找到摄像头", "未检测到可用摄像头设备"],
            DevicesNotFoundError: ["未找到摄像头", "未检测到可用摄像头设备"],
            NotReadableError: ["摄像头被占用", "摄像头可能正在被其他应用使用"],
            TrackStartError: ["摄像头被占用", "摄像头可能正在被其他应用使用"],
            OverconstrainedError: ["设备不可用", "当前摄像头不满足浏览器要求"],
        };
        const fallback = ["访问失败", error.message || "摄像头访问失败"];
        const message = messages[error.name] || fallback;
        stopCamera();
        setStatus(message[0], "error");
        showToast(message[1], "error");
    }

    async function init() {
        updateTargetUi();
        resizeOverlay();
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            startButton.disabled = true;
            recordButton.disabled = true;
            cameraSelect.disabled = true;
            setStatus("浏览器不支持", "error");
            showToast("请使用新版 Chrome、Edge、Firefox 或 Safari", "error");
            return;
        }

        if (!isSecureCameraContext()) {
            startButton.disabled = true;
            recordButton.disabled = true;
            cameraSelect.disabled = true;
            setStatus("需要安全环境", "error");
            showToast("摄像头功能需要在 localhost 或 HTTPS 下使用", "error");
            return;
        }

        await refreshDevices();
        startButton.disabled = devices.length === 0;
        navigator.mediaDevices.addEventListener("devicechange", async () => {
            await refreshDevices();
            if (!stream) {
                startButton.disabled = devices.length === 0;
                cameraSelect.disabled = devices.length === 0;
            }
        });
    }

    startButton.addEventListener("click", startCamera);
    stopButton.addEventListener("click", stopCamera);
    monitorButton.addEventListener("click", toggleMonitoring);
    targetButton.addEventListener("click", toggleTarget);
    recordButton.addEventListener("click", toggleRecording);
    window.addEventListener("resize", () => {
        if (lastDetection) {
            drawDetections(lastDetection);
        } else {
            resizeOverlay();
        }
    });
    window.addEventListener("beforeunload", () => {
        clearRecordingUrl();
        stopCurrentStream();
    });

    init();
})();
