(function () {
    const video = document.getElementById("cameraVideo");
    const powerButton = document.getElementById("cameraPowerButton");
    const cameraSelect = document.getElementById("cameraSelect");
    const yoloModelSelect = document.getElementById("cameraYoloModelSelect");
    const currentYoloModelText = document.getElementById("cameraCurrentYoloModel");
    const statusText = document.getElementById("cameraStatusText");
    const statusDot = document.getElementById("cameraStatusDot");
    const placeholder = document.getElementById("cameraPlaceholder");
    const detectPlaceholder = document.getElementById("cameraDetectPlaceholder");
    const resultPlaceholder = document.getElementById("cameraResultPlaceholder");
    const hint = document.getElementById("cameraHint");
    const toast = document.getElementById("globalToast");
    const recognitionCanvas = document.getElementById("cameraRecognitionFrame");
    const resultCanvas = document.getElementById("cameraResultCanvas");
    const monitorButton = document.getElementById("monitorCameraButton");
    const targetToggleButton = document.getElementById("targetToggleButton");
    const recordButton = document.getElementById("recordCameraButton");
    const personCountText = document.getElementById("cameraPersonCount");
    const totalCountLabel = document.getElementById("cameraTotalLabel");
    const totalCountText = document.getElementById("cameraTotalCount");
    const detectMeta = document.getElementById("cameraDetectMeta");
    const recordPanel = document.getElementById("cameraRecordPanel");
    const recordPreview = document.getElementById("cameraRecordPreview");
    const recordDownload = document.getElementById("cameraRecordDownload");
    const recordMeta = document.getElementById("cameraRecordMeta");

    let stream = null;
    let devices = [];
    let toastTimer = null;
    let monitoring = false;
    let detecting = false;
    let detectTimer = null;
    let lastDetection = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let recording = false;
    let recordStartedAt = 0;
    let recordTimer = null;
    let recordUrl = null;
    let detectionTarget = "both";
    let lastRequestMs = 0;
    let resultAnimationFrameId = null;
    let sessionTotalCount = 0;
    let cameraSessionId = "";
    let shouldResetBackendTracking = false;
    let peerConnection = null;
    let webRTCResultTimer = null;
    let webRTCActive = false;
    let lastWebRTCSequence = 0;
    let webRTCFallback = false;
    let nextTrackId = 1;
    let trackedTargets = [];
    const minDetectDelay = 180;
    const maxDetectDelay = 900;
    const cameraFrameMaxSide = 1280;
    const detectTimeoutMs = 10000;
    const trackIouThreshold = 0.24;
    const trackCenterRatio = 0.32;
    const trackMaxMisses = 4;
    const captureCanvas = document.createElement("canvas");
    const captureContext = captureCanvas.getContext("2d", { willReadFrequently: true });
    const recognitionContext = recognitionCanvas.getContext("2d");
    const resultContext = resultCanvas.getContext("2d");

    function getSelectedYoloModel() {
        return yoloModelSelect ? yoloModelSelect.value : "";
    }

    function syncCurrentYoloModel() {
        if (yoloModelSelect && currentYoloModelText) {
            currentYoloModelText.textContent = yoloModelSelect.value;
        }
    }

    function isSecureCameraContext() {
        const host = window.location.hostname;
        return window.location.protocol === "https:" || host === "localhost" || host === "127.0.0.1" || host === "::1";
    }

    function showToast(message, type) {
        if (!toast) {
            return;
        }
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
        if (visible) {
            detectPlaceholder.classList.remove("is-hidden");
            resultPlaceholder.classList.remove("is-hidden");
        }
    }

    function updateMonitorButton() {
        monitorButton.disabled = !stream;
        monitorButton.textContent = monitoring ? "暂停检测" : "实时检测";
    }

    function updatePowerButton() {
        if (!powerButton) {
            return;
        }
        powerButton.disabled = !stream && devices.length === 0;
        powerButton.textContent = stream ? "停止" : "开始";
        powerButton.classList.toggle("button-danger", Boolean(stream));
    }

    function updateTargetToggleButton() {
        targetToggleButton.disabled = !stream;
        if (detectionTarget === "both") {
            targetToggleButton.textContent = "仅检测人员";
            document.getElementById("cameraCountLabel").textContent = "当前目标";
            if (totalCountLabel) totalCountLabel.textContent = "累计目标";
            document.getElementById("cameraFeedChip").textContent = "人员+车辆";
        } else if (detectionTarget === "person") {
            targetToggleButton.textContent = "仅检测车辆";
            document.getElementById("cameraCountLabel").textContent = "当前人数";
            if (totalCountLabel) totalCountLabel.textContent = "累计人数";
            document.getElementById("cameraFeedChip").textContent = "人员检测";
        } else {
            targetToggleButton.textContent = "合并检测";
            document.getElementById("cameraCountLabel").textContent = "当前车辆数";
            if (totalCountLabel) totalCountLabel.textContent = "累计车辆数";
            document.getElementById("cameraFeedChip").textContent = "车辆检测";
        }
    }

    function toggleTarget() {
        if (detectionTarget === "both") {
            detectionTarget = "person";
        } else if (detectionTarget === "person") {
            detectionTarget = "vehicle";
        } else {
            detectionTarget = "both";
        }
        updateTargetToggleButton();
        clearDetections();
        resetSessionTotal();
        const label = detectionTarget === "both" ? "人员+车辆合并" : (detectionTarget === "person" ? "人员" : "车辆");
        detectMeta.textContent = `已切换为${label}检测`;
        if (monitoring && stream && webRTCActive) {
            startWebRTCDetection().catch(() => {
                webRTCFallback = true;
                detectFrame();
            });
        }
    }

    function resetSessionTotal() {
        sessionTotalCount = 0;
        nextTrackId = 1;
        trackedTargets = [];
        shouldResetBackendTracking = true;
        if (totalCountText) totalCountText.textContent = "0";
    }

    function boxArea(box) {
        return Math.max(0, Number(box.x2) - Number(box.x1)) * Math.max(0, Number(box.y2) - Number(box.y1));
    }

    function boxIntersection(first, second) {
        const left = Math.max(Number(first.x1), Number(second.x1));
        const top = Math.max(Number(first.y1), Number(second.y1));
        const right = Math.min(Number(first.x2), Number(second.x2));
        const bottom = Math.min(Number(first.y2), Number(second.y2));
        return Math.max(0, right - left) * Math.max(0, bottom - top);
    }

    function boxIou(first, second) {
        const intersection = boxIntersection(first, second);
        const union = boxArea(first) + boxArea(second) - intersection;
        return union > 0 ? intersection / union : 0;
    }

    function centerDistanceRatio(first, second) {
        const firstCx = (Number(first.x1) + Number(first.x2)) / 2;
        const firstCy = (Number(first.y1) + Number(first.y2)) / 2;
        const secondCx = (Number(second.x1) + Number(second.x2)) / 2;
        const secondCy = (Number(second.y1) + Number(second.y2)) / 2;
        const distance = Math.hypot(firstCx - secondCx, firstCy - secondCy);
        const firstDiag = Math.max(1, Math.hypot(Number(first.x2) - Number(first.x1), Number(first.y2) - Number(first.y1)));
        const secondDiag = Math.max(1, Math.hypot(Number(second.x2) - Number(second.x1), Number(second.y2) - Number(second.y1)));
        return distance / Math.max(firstDiag, secondDiag);
    }

    function trackMatchScore(track, box) {
        if (track.target !== (box.detection_target || "person")) {
            return -1;
        }
        const iou = boxIou(track.box, box);
        const distanceRatio = centerDistanceRatio(track.box, box);
        if (iou < trackIouThreshold && distanceRatio > trackCenterRatio) {
            return -1;
        }
        return iou + Math.max(0, trackCenterRatio - distanceRatio);
    }

    function updateTrackedTargets(detection) {
        if (!detection || !Array.isArray(detection.boxes)) {
            return 0;
        }

        trackedTargets.forEach((track) => {
            track.missed += 1;
            track.matched = false;
        });

        let newTrackCount = 0;
        detection.boxes.forEach((box) => {
            let bestTrack = null;
            let bestScore = -1;
            trackedTargets.forEach((track) => {
                if (track.matched) {
                    return;
                }
                const score = trackMatchScore(track, box);
                if (score > bestScore) {
                    bestScore = score;
                    bestTrack = track;
                }
            });

            if (bestTrack) {
                bestTrack.box = { ...box };
                bestTrack.missed = 0;
                bestTrack.matched = true;
                box.track_id = bestTrack.id;
            } else {
                const track = {
                    id: nextTrackId,
                    target: box.detection_target || "person",
                    box: { ...box },
                    missed: 0,
                    matched: true,
                };
                nextTrackId += 1;
                trackedTargets.push(track);
                box.track_id = track.id;
                newTrackCount += 1;
            }
        });

        trackedTargets = trackedTargets.filter((track) => track.missed <= trackMaxMisses);
        return newTrackCount;
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
        recordButton.classList.toggle("button-danger", recording);
        recordButton.classList.toggle("button-secondary", !recording);
    }

    function getRecordingMimeType() {
        const types = [
            "video/webm;codecs=vp9",
            "video/webm;codecs=vp8",
            "video/webm",
            "video/mp4",
        ];
        return types.find((type) => window.MediaRecorder && MediaRecorder.isTypeSupported(type)) || "";
    }

    function clearRecordingUrl() {
        if (recordUrl) {
            URL.revokeObjectURL(recordUrl);
        }
        recordUrl = null;
    }

    function resizeCanvas(canvas, context) {
        const rect = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.round(rect.width * ratio));
        const height = Math.max(1, Math.round(rect.height * ratio));

        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }

        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        return rect;
    }

    function drawFrameToCanvas(canvas, context, sourceCanvas) {
        const rect = resizeCanvas(canvas, context);
        context.clearRect(0, 0, rect.width, rect.height);

        const sourceWidth = sourceCanvas.videoWidth || sourceCanvas.width;
        const sourceHeight = sourceCanvas.videoHeight || sourceCanvas.height;
        if (!sourceWidth || !sourceHeight) {
            return { rect, scale: 1, offsetX: 0, offsetY: 0, drawWidth: 0, drawHeight: 0 };
        }

        const scale = Math.min(rect.width / sourceWidth, rect.height / sourceHeight);
        const drawWidth = sourceWidth * scale;
        const drawHeight = sourceHeight * scale;
        const offsetX = (rect.width - drawWidth) / 2;
        const offsetY = (rect.height - drawHeight) / 2;
        context.drawImage(sourceCanvas, 0, 0, sourceWidth, sourceHeight, offsetX, offsetY, drawWidth, drawHeight);
        return { rect, scale, offsetX, offsetY, drawWidth, drawHeight };
    }

    function renderRecognitionFrame() {
        if (recognitionCanvas.closest("[hidden]")) {
            return;
        }
        drawFrameToCanvas(recognitionCanvas, recognitionContext, captureCanvas);
        detectPlaceholder.classList.add("is-hidden");
    }

    function clearDetections() {
        let rect = resizeCanvas(recognitionCanvas, recognitionContext);
        recognitionContext.clearRect(0, 0, rect.width, rect.height);
        rect = resizeCanvas(resultCanvas, resultContext);
        resultContext.clearRect(0, 0, rect.width, rect.height);
        lastDetection = null;
        personCountText.textContent = "0";
        resetSessionTotal();
    }

    function drawDetectionBoxes(detection, frame) {
        if (!detection || !detection.boxes || !detection.boxes.length) {
            return;
        }

        const detectionWidth = detection.width || captureCanvas.width || video.videoWidth;
        const detectionHeight = detection.height || captureCanvas.height || video.videoHeight;
        if (!detectionWidth || !detectionHeight || !frame.drawWidth || !frame.drawHeight) {
            return;
        }

        const scaleX = frame.drawWidth / detectionWidth;
        const scaleY = frame.drawHeight / detectionHeight;
        resultContext.lineWidth = 2;
        resultContext.font = "13px Microsoft YaHei, Arial, sans-serif";
        resultContext.textBaseline = "top";
        detection.boxes.forEach((box, index) => {
            const isVehicle = box.detection_target === "vehicle";
            const strokeColor = isVehicle ? "#2f8cff" : "#00d46a";
            const labelColor = isVehicle ? "#1765d8" : "#00a656";
            const fillColor = isVehicle ? "rgba(47, 140, 255, 0.16)" : "rgba(0, 212, 106, 0.16)";
            const x = frame.offsetX + box.x1 * scaleX;
            const y = frame.offsetY + box.y1 * scaleY;
            const width = (box.x2 - box.x1) * scaleX;
            const height = (box.y2 - box.y1) * scaleY;
            const confidence = Number.isFinite(Number(box.conf)) ? Number(box.conf).toFixed(2) : "";
            const labelPrefix = isVehicle ? (box.class_name || "vehicle") : "person";
            const labelId = box.track_id ? `#${box.track_id}` : `${index + 1}`;
            const label = confidence ? `${labelId} ${labelPrefix} ${confidence}` : `${labelId} ${labelPrefix}`;
            const labelWidth = resultContext.measureText(label).width + 12;

            resultContext.strokeStyle = strokeColor;
            resultContext.fillStyle = fillColor;
            resultContext.strokeRect(x, y, width, height);
            resultContext.fillRect(x, y, width, height);

            resultContext.fillStyle = labelColor;
            resultContext.fillRect(x, Math.max(0, y - 20), labelWidth, 20);
            resultContext.fillStyle = "#ffffff";
            resultContext.fillText(label, x + 6, Math.max(0, y - 17));
        });
    }

    function drawDetections(detection) {
        const source = stream && video.videoWidth ? video : captureCanvas;
        const frame = drawFrameToCanvas(resultCanvas, resultContext, source);
        resultPlaceholder.classList.add("is-hidden");
        drawDetectionBoxes(detection, frame);
    }

    function renderResultLoop() {
        if (!stream) {
            resultAnimationFrameId = null;
            return;
        }

        const frame = drawFrameToCanvas(resultCanvas, resultContext, video);
        resultPlaceholder.classList.add("is-hidden");
        drawDetectionBoxes(lastDetection, frame);
        resultAnimationFrameId = window.requestAnimationFrame(renderResultLoop);
    }

    function startResultLoop() {
        if (!resultAnimationFrameId) {
            resultAnimationFrameId = window.requestAnimationFrame(renderResultLoop);
        }
    }

    function stopResultLoop() {
        if (resultAnimationFrameId) {
            window.cancelAnimationFrame(resultAnimationFrameId);
            resultAnimationFrameId = null;
        }
    }

    function captureFrameBlob() {
        const sourceWidth = video.videoWidth;
        const sourceHeight = video.videoHeight;

        if (!sourceWidth || !sourceHeight) {
            return Promise.resolve(null);
        }

        const scale = Math.min(1, cameraFrameMaxSide / Math.max(sourceWidth, sourceHeight));
        captureCanvas.width = Math.max(1, Math.round(sourceWidth * scale));
        captureCanvas.height = Math.max(1, Math.round(sourceHeight * scale));
        captureContext.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        renderRecognitionFrame();

        return new Promise((resolve) => {
            captureCanvas.toBlob(resolve, "image/jpeg", 0.85);
        });
    }

    function stopMonitoring(resetView = true) {
        monitoring = false;
        window.clearTimeout(detectTimer);
        detectTimer = null;
        detecting = false;
        stopWebRTCDetection();
        updateMonitorButton();

        if (resetView) {
            clearDetections();
            detectMeta.textContent = "未开始实时检测";
        }
    }

    function getNextDetectDelay(elapsedMs) {
        const adaptiveDelay = Math.round((elapsedMs || lastRequestMs || 0) * 0.25);
        return Math.max(minDetectDelay, Math.min(maxDetectDelay, adaptiveDelay));
    }

    function applyDetectionResult(data, requestStartedAt) {
        lastDetection = data;
        shouldResetBackendTracking = false;
        lastRequestMs = Math.round(performance.now() - requestStartedAt);
        const count = data.count !== undefined ? data.count : (data.person_count || 0);
        personCountText.textContent = String(count);
        const newTrackCount = data.new_track_count !== undefined ? data.new_track_count : updateTrackedTargets(data);
        sessionTotalCount = data.total_unique_count !== undefined ? data.total_unique_count : (sessionTotalCount + newTrackCount);
        if (totalCountText) totalCountText.textContent = String(sessionTotalCount);
        const rawCount = data.raw_count !== undefined ? `，原始 ${data.raw_count}` : "";
        const breakdown = data.detection_target === "both"
            ? `，人 ${data.person_count || 0} / 车 ${data.vehicle_count || 0}`
            : "";
        const transportInfo = data.transport === "webrtc" ? "，WebRTC" : "，HTTP";
        const trackerInfo = data.tracker_used ? "，跟踪开启" : (data.tracker_fallback ? "，跟踪回退" : "");
        const uniqueInfo = `，新增 ${newTrackCount} / 已去重 ${sessionTotalCount}`;
        detectMeta.textContent = `YOLO ${data.elapsed_ms || 0} ms / 总 ${lastRequestMs} ms${breakdown}${uniqueInfo}${transportInfo}${trackerInfo}${rawCount}`;
        drawDetections(data);
    }

    async function detectFrame() {
        if (!monitoring || detecting || !stream) {
            return;
        }

        detecting = true;
        const requestStartedAt = performance.now();
        detectMeta.textContent = lastDetection ? "后台更新检测中，保留上一帧标注..." : "正在检测当前画面...";

        try {
            const blob = await captureFrameBlob();
            if (!blob) {
                detectMeta.textContent = "等待摄像头画面稳定";
                return;
            }

            const formData = new FormData();
            formData.append("frame", blob, "camera-frame.jpg");
            formData.append("target", detectionTarget);
            formData.append("session_id", cameraSessionId);
            formData.append("model_name", getSelectedYoloModel());
            if (shouldResetBackendTracking) {
                formData.append("reset_tracking", "1");
            }
            const controller = new AbortController();
            const timeoutId = window.setTimeout(() => controller.abort(), detectTimeoutMs);

            let response;
            let data;
            try {
                response = await fetch("/api/camera/detect", {
                    method: "POST",
                    body: formData,
                    cache: "no-store",
                    signal: controller.signal,
                });
                data = await response.json();
            } finally {
                window.clearTimeout(timeoutId);
            }

            if (!response.ok || !data.success) {
                throw new Error(data.error || "检测失败");
            }

            applyDetectionResult(data, requestStartedAt);
        } catch (error) {
            lastRequestMs = Math.round(performance.now() - requestStartedAt);
            detectMeta.textContent = error.name === "AbortError" ? "本轮检测超时，已跳过" : (error.message || "实时检测失败");
        } finally {
            detecting = false;
            if (monitoring && stream) {
                detectTimer = window.setTimeout(detectFrame, getNextDetectDelay(lastRequestMs));
            }
        }
    }

    async function startMonitoring() {
        if (!stream || monitoring) {
            return;
        }

        monitoring = true;
        updateMonitorButton();
        try {
            const started = await startWebRTCDetection();
            if (!started) {
                webRTCFallback = true;
                detectFrame();
            }
        } catch (error) {
            webRTCFallback = true;
            await stopWebRTCDetection();
            if (monitoring && stream) {
                detectMeta.textContent = "WebRTC 不可用，已回退 HTTP 检测";
                detectFrame();
            }
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

    function stopCurrentStream() {
        stopRecording(false);
        stopMonitoring(true);
        stopResultLoop();
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
        }
        stream = null;
        video.srcObject = null;
        updateMonitorButton();
        updateRecordButton();
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
        const mimeType = recordedChunks[0].type || "video/webm";
        const blob = new Blob(recordedChunks, { type: mimeType });
        recordUrl = URL.createObjectURL(blob);
        const duration = Math.max(1, Math.round((Date.now() - recordStartedAt) / 1000));
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");

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
        cameraSessionId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
        resetSessionTotal();
        const selectedDeviceId = cameraSelect.value;
        const videoConstraint = {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 30 },
        };
        if (selectedDeviceId) {
            videoConstraint.deviceId = { exact: selectedDeviceId };
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: videoConstraint,
                audio: false,
            });

            video.srcObject = stream;
            await video.play();

            setPlaceholder(false);
            startResultLoop();
            updatePowerButton();
            monitorButton.disabled = false;
            targetToggleButton.disabled = false;
            recordButton.disabled = !window.MediaRecorder;
            cameraSelect.disabled = true;
            hint.textContent = "摄像头正在实时检测中。";
            showToast("摄像头已开启", "success");

            await refreshDevices();
            cameraSelect.disabled = true;
            setStatus("摄像头已开启", "active");

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

    function stopCamera() {
        stopCurrentStream();
        setPlaceholder(true);
        updatePowerButton();
        monitorButton.disabled = true;
        targetToggleButton.disabled = true;
        recordButton.disabled = true;
        cameraSelect.disabled = devices.length === 0;
        hint.textContent = "浏览器会在开始时请求摄像头权限。";
        setStatus(devices.length > 0 ? "已停止" : "未检测到摄像头", devices.length > 0 ? "idle" : "error");
    }

    function togglePowerCamera() {
        if (stream) {
            stopCamera();
        } else {
            startCamera();
        }
    }

    async function pollWebRTCResult() {
        if (!webRTCActive || !cameraSessionId) {
            return;
        }

        const requestStartedAt = performance.now();
        try {
            const response = await fetch(`/api/camera/webrtc/result?session_id=${encodeURIComponent(cameraSessionId)}`, {
                cache: "no-store",
            });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || "WebRTC result failed");
            }
            if (data.pending || !data.sequence || data.sequence === lastWebRTCSequence) {
                return;
            }
            lastWebRTCSequence = data.sequence;
            applyDetectionResult(data, requestStartedAt);
        } catch (error) {
            detectMeta.textContent = error.message || "WebRTC result failed";
        }
    }

    async function stopWebRTCDetection() {
        webRTCActive = false;
        window.clearInterval(webRTCResultTimer);
        webRTCResultTimer = null;
        lastWebRTCSequence = 0;

        if (peerConnection) {
            peerConnection.getSenders().forEach((sender) => {
                if (sender.track) {
                    peerConnection.removeTrack(sender);
                }
            });
            peerConnection.close();
            peerConnection = null;
        }

        if (cameraSessionId) {
            try {
                await fetch("/api/camera/webrtc/stop", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: cameraSessionId }),
                    cache: "no-store",
                });
            } catch (error) {
                // Best-effort cleanup only.
            }
        }
    }

    async function startWebRTCDetection() {
        if (!window.RTCPeerConnection || !stream) {
            return false;
        }

        await stopWebRTCDetection();
        peerConnection = new RTCPeerConnection();
        stream.getVideoTracks().forEach((track) => {
            peerConnection.addTrack(track, stream);
        });

        const offer = await peerConnection.createOffer({
            offerToReceiveAudio: false,
            offerToReceiveVideo: false,
        });
        await peerConnection.setLocalDescription(offer);
        await waitForIceGathering(peerConnection);

        const response = await fetch("/api/camera/webrtc/offer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sdp: peerConnection.localDescription.sdp,
                type: peerConnection.localDescription.type,
                target: detectionTarget,
                session_id: cameraSessionId,
                reset_tracking: shouldResetBackendTracking,
                model_name: getSelectedYoloModel(),
            }),
            cache: "no-store",
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || "WebRTC offer failed");
        }

        await peerConnection.setRemoteDescription(data.answer);
        shouldResetBackendTracking = false;
        webRTCActive = true;
        webRTCFallback = false;
        detectMeta.textContent = "WebRTC 实时检测已启动";
        webRTCResultTimer = window.setInterval(pollWebRTCResult, 120);
        return true;
    }

    function waitForIceGathering(connection) {
        if (connection.iceGatheringState === "complete") {
            return Promise.resolve();
        }
        return new Promise((resolve) => {
            const timeoutId = window.setTimeout(resolve, 3000);
            connection.addEventListener("icegatheringstatechange", () => {
                if (connection.iceGatheringState === "complete") {
                    window.clearTimeout(timeoutId);
                    resolve();
                }
            });
        });
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
            powerButton.disabled = true;
            recordButton.disabled = true;
            cameraSelect.disabled = true;
            setStatus("浏览器不支持", "error");
            showToast("请使用最新版 Chrome、Edge、Firefox 或 Safari", "error");
            return;
        }

        if (!isSecureCameraContext()) {
            powerButton.disabled = true;
            recordButton.disabled = true;
            cameraSelect.disabled = true;
            setStatus("需要安全环境", "error");
            showToast("摄像头功能需要在 localhost 或 HTTPS 下使用", "error");
            return;
        }

        await refreshDevices();
        updatePowerButton();
        updateTargetToggleButton();
        syncCurrentYoloModel();

        navigator.mediaDevices.addEventListener("devicechange", async () => {
            await refreshDevices();
            if (!stream) {
                updatePowerButton();
                cameraSelect.disabled = devices.length === 0;
            }
        });
    }

    powerButton.addEventListener("click", togglePowerCamera);
    monitorButton.addEventListener("click", toggleMonitoring);
    targetToggleButton.addEventListener("click", toggleTarget);
    recordButton.addEventListener("click", toggleRecording);
    if (yoloModelSelect) {
        yoloModelSelect.addEventListener("change", async () => {
            syncCurrentYoloModel();
            resetTrackingState(true);
            shouldResetBackendTracking = true;
            if (monitoring && stream) {
                await stopWebRTCDetection();
                detectMeta.textContent = "模型已切换，正在重新检测...";
                startMonitoring();
            }
        });
    }
    window.addEventListener("beforeunload", () => {
        clearRecordingUrl();
        stopCurrentStream();
    });
    window.addEventListener("resize", () => {
        if (lastDetection) {
            drawDetections(lastDetection);
        } else {
            resizeCanvas(recognitionCanvas, recognitionContext);
            resizeCanvas(resultCanvas, resultContext);
        }
    });

    init();
})();
