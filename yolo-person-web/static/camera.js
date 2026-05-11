(function () {
    const video = document.getElementById("cameraVideo");
    const startButton = document.getElementById("startCameraButton");
    const stopButton = document.getElementById("stopCameraButton");
    const cameraSelect = document.getElementById("cameraSelect");
    const statusText = document.getElementById("cameraStatusText");
    const statusDot = document.getElementById("cameraStatusDot");
    const placeholder = document.getElementById("cameraPlaceholder");
    const hint = document.getElementById("cameraHint");
    const toast = document.getElementById("cameraToast");

    let stream = null;
    let devices = [];
    let toastTimer = null;

    function isSecureCameraContext() {
        const host = window.location.hostname;
        return window.location.protocol === "https:" || host === "localhost" || host === "127.0.0.1" || host === "::1";
    }

    function showToast(message, type) {
        window.clearTimeout(toastTimer);
        toast.textContent = message;
        toast.className = `toast ${type} show`;
        toastTimer = window.setTimeout(() => {
            toast.classList.remove("show");
        }, 3200);
    }

    function setStatus(text, type) {
        statusText.textContent = text;
        statusDot.className = `camera-status-dot is-${type}`;
    }

    function setPlaceholder(visible) {
        placeholder.classList.toggle("is-hidden", !visible);
    }

    function stopCurrentStream() {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
        }
        stream = null;
        video.srcObject = null;
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

            if (devices.length > 0) {
                setStatus(`就绪，检测到 ${devices.length} 个摄像头`, "idle");
            } else {
                setStatus("未检测到摄像头", "error");
            }
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
            await video.play();

            setPlaceholder(false);
            startButton.disabled = true;
            stopButton.disabled = false;
            cameraSelect.disabled = true;
            hint.textContent = "摄像头正在预览中。";
            showToast("摄像头已开启", "success");

            await refreshDevices();
            cameraSelect.disabled = true;
            setStatus("摄像头已开启", "active");

            const activeDeviceId = stream.getVideoTracks()[0].getSettings().deviceId;
            if (activeDeviceId) {
                cameraSelect.value = activeDeviceId;
            }

            stream.getVideoTracks()[0].addEventListener("ended", stopCamera);
        } catch (error) {
            handleCameraError(error);
        }
    }

    function stopCamera() {
        stopCurrentStream();
        setPlaceholder(true);
        startButton.disabled = devices.length === 0;
        stopButton.disabled = true;
        cameraSelect.disabled = devices.length === 0;
        hint.textContent = "浏览器会在开始时请求摄像头权限。";
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
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            startButton.disabled = true;
            cameraSelect.disabled = true;
            setStatus("浏览器不支持", "error");
            showToast("请使用最新版 Chrome、Edge、Firefox 或 Safari", "error");
            return;
        }

        if (!isSecureCameraContext()) {
            startButton.disabled = true;
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
    window.addEventListener("beforeunload", stopCurrentStream);

    init();
})();
