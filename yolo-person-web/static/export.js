var ExportPage = (function () {
    var state = {
        exportType: "",
        typeLabel: "",
        detectionLabel: "",
        mediaLabel: "",
        currentTaskId: null,
        pollTimer: null,
    };

    function init(options) {
        state.exportType = options.exportType || "";
        state.typeLabel = options.typeLabel || "";
        state.detectionLabel = options.detectionLabel || "";
        state.mediaLabel = options.mediaLabel || "";

        bindEvents();
        loadExportHistory();
    }

    function $(id) {
        return document.getElementById(id);
    }

    function bindEvents() {
        var startBtn = $("startExportBtn");
        if (startBtn) {
            startBtn.addEventListener("click", startExport);
        }

        var selectAll = $("selectAllFields");
        if (selectAll) {
            selectAll.addEventListener("click", function () {
                var checks = document.querySelectorAll(".field-select");
                checks.forEach(function (cb) { cb.checked = true; });
            });
        }

        var deselectAll = $("deselectAllFields");
        if (deselectAll) {
            deselectAll.addEventListener("click", function () {
                var checks = document.querySelectorAll(".field-select");
                checks.forEach(function (cb) { cb.checked = false; });
            });
        }

        var resetBtn = $("resetExportBtn");
        if (resetBtn) {
            resetBtn.addEventListener("click", resetExport);
        }

        var refreshBtn = $("refreshHistoryBtn");
        if (refreshBtn) {
            refreshBtn.addEventListener("click", loadExportHistory);
        }
    }

    function getSelectedFields() {
        var checks = document.querySelectorAll(".field-select:checked");
        var fields = [];
        checks.forEach(function (cb) { fields.push(cb.value); });
        return fields;
    }

    function getParams() {
        var startDate = $("startDate") ? $("startDate").value : "";
        var endDate = $("endDate") ? $("endDate").value : "";
        var format = $("formatSelect") ? $("formatSelect").value : "xlsx";
        var fields = getSelectedFields();

        return {
            type: state.exportType,
            format: format,
            fields: fields,
            start_date: startDate,
            end_date: endDate,
        };
    }

    function validateParams(params) {
        if (params.fields.length === 0) {
            showToast("请至少选择一个导出字段", "error");
            return false;
        }
        if (params.start_date && params.end_date && params.start_date > params.end_date) {
            showToast("起始日期不能晚于结束日期", "error");
            return false;
        }
        return true;
    }

    function startExport() {
        var params = getParams();
        if (!validateParams(params)) return;

        var startBtn = $("startExportBtn");
        startBtn.disabled = true;
        startBtn.textContent = "正在创建导出任务...";

        var progressPanel = $("exportProgressPanel");
        if (progressPanel) progressPanel.hidden = false;

        var progressStatus = $("exportProgressStatus");
        if (progressStatus) progressStatus.hidden = true;

        var progressActions = $("exportProgressActions");
        if (progressActions) progressActions.hidden = true;

        updateProgress(0, "正在创建导出任务...");

        fetch("/api/export/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        })
            .then(function (res) { return res.json(); })
            .then(function (data) {
                if (data.success) {
                    state.currentTaskId = data.task_id;
                    startBtn.textContent = "导出中...";
                    pollProgress(data.task_id);
                } else {
                    handleExportError(data.error || "创建导出任务失败");
                    startBtn.disabled = false;
                    startBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 开始导出';
                }
            })
            .catch(function (err) {
                handleExportError("网络请求失败: " + err.message);
                startBtn.disabled = false;
                startBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 开始导出';
            });
    }

    function pollProgress(taskId) {
        if (state.pollTimer) {
            clearInterval(state.pollTimer);
        }

        state.pollTimer = setInterval(function () {
            fetch("/api/export/progress/" + taskId)
                .then(function (res) { return res.json(); })
                .then(function (data) {
                    if (data.success) {
                        var task = data.task;
                        updateProgress(task.progress, task.message);

                        if (task.status === "completed") {
                            clearInterval(state.pollTimer);
                            state.pollTimer = null;
                            handleExportComplete(task);
                        } else if (task.status === "failed") {
                            clearInterval(state.pollTimer);
                            state.pollTimer = null;
                            handleExportError(task.error || task.message || "导出失败");
                        }
                    }
                })
                .catch(function () { });
        }, 1500);
    }

    function updateProgress(percent, message) {
        var fill = $("exportProgressFill");
        var pct = $("exportProgressPercent");
        var msg = $("exportProgressMessage");
        if (fill) fill.style.width = percent + "%";
        if (pct) pct.textContent = percent + "%";
        if (msg) msg.textContent = message || "";
    }

    function handleExportComplete(task) {
        var startBtn = $("startExportBtn");
        startBtn.disabled = false;
        startBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 开始导出';

        var progressStatus = $("exportProgressStatus");
        var statusIcon = $("exportStatusIcon");
        var statusText = $("exportStatusText");
        if (progressStatus) {
            progressStatus.className = "progress-status success";
            progressStatus.hidden = false;
        }
        if (statusIcon) {
            statusIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
        }
        if (statusText) {
            var sizeInfo = task.file_size ? " (" + formatFileSize(task.file_size) + ")" : "";
            statusText.textContent = "导出完成！共导出 " + task.exported_records + " 条记录" + sizeInfo;
        }

        var downloadBtn = $("downloadExportBtn");
        if (downloadBtn) {
            downloadBtn.href = "/api/export/download/" + state.currentTaskId;
        }

        var progressActions = $("exportProgressActions");
        if (progressActions) progressActions.hidden = false;

        updateProgress(100, "导出完成");

        showToast("导出完成，共 " + task.exported_records + " 条记录", "success");
        loadExportHistory();
    }

    function handleExportError(errorMessage) {
        var progressStatus = $("exportProgressStatus");
        var statusIcon = $("exportStatusIcon");
        var statusText = $("exportStatusText");
        if (progressStatus) {
            progressStatus.className = "progress-status error";
            progressStatus.hidden = false;
        }
        if (statusIcon) {
            statusIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
        }
        if (statusText) {
            statusText.textContent = errorMessage || "导出失败，请重试";
        }

        var progressActions = $("exportProgressActions");
        if (progressActions) progressActions.hidden = false;

        var resetBtn = $("resetExportBtn");
        if (resetBtn) resetBtn.textContent = "重试";

        var downloadBtn = $("downloadExportBtn");
        if (downloadBtn) downloadBtn.style.display = "none";

        showToast(errorMessage || "导出失败", "error");
    }

    function resetExport() {
        var progressPanel = $("exportProgressPanel");
        if (progressPanel) progressPanel.hidden = true;

        var progressStatus = $("exportProgressStatus");
        if (progressStatus) progressStatus.hidden = true;

        var progressActions = $("exportProgressActions");
        if (progressActions) progressActions.hidden = true;

        var downloadBtn = $("downloadExportBtn");
        if (downloadBtn) downloadBtn.style.display = "";

        var resetBtn = $("resetExportBtn");
        if (resetBtn) resetBtn.textContent = "重新导出";

        var startBtn = $("startExportBtn");
        startBtn.disabled = false;
        startBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> 开始导出';

        updateProgress(0, "准备中...");
        state.currentTaskId = null;

        if (state.pollTimer) {
            clearInterval(state.pollTimer);
            state.pollTimer = null;
        }
    }

    function loadExportHistory() {
        var tbody = $("exportHistoryBody");
        if (!tbody) return;

        fetch("/api/export/history")
            .then(function (res) { return res.json(); })
            .then(function (data) {
                if (!data.success) return;
                var history = data.history || [];
                if (history.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" class="empty-cell">暂无导出记录</td></tr>';
                    return;
                }

                var typeLabels = {
                    "image_pedestrian": "图片行人",
                    "image_vehicle": "图片车辆",
                    "video_pedestrian": "视频行人",
                    "video_vehicle": "视频车辆",
                };

                var html = "";
                history.forEach(function (task) {
                    var typeLabel = typeLabels[task.type] || task.type;
                    var statusClass = task.status;
                    var statusLabel = getStatusLabel(task.status);
                    var canDownload = task.status === "completed" && task.file_name;

                    html += "<tr>";
                    html += "<td><code>" + (task.id ? task.id.substring(0, 8) : "-") + "</code></td>";
                    html += "<td>" + typeLabel + "</td>";
                    html += "<td>" + (task.file_name ? getFormatLabel(task.file_name) : "-") + "</td>";
                    html += '<td><span class="status-tag ' + statusClass + '">' + statusLabel + "</span></td>";
                    html += "<td>" + (task.exported_records || 0) + "</td>";
                    html += "<td>" + (task.created_at || "-") + "</td>";
                    html += "<td>";
                    if (canDownload) {
                        html += '<a href="/api/export/download/' + task.id + '" class="history-action-btn">下载</a>';
                    } else {
                        html += '<button class="history-action-btn" disabled>--</button>';
                    }
                    html += "</td>";
                    html += "</tr>";
                });
                tbody.innerHTML = html;
            })
            .catch(function () { });
    }

    function getStatusLabel(status) {
        var map = {
            "pending": "等待中",
            "processing": "处理中",
            "completed": "已完成",
            "failed": "失败",
        };
        return map[status] || status;
    }

    function getFormatLabel(filename) {
        if (!filename) return "-";
        var ext = filename.split(".").pop().toLowerCase();
        var map = { "xlsx": "Excel", "csv": "CSV", "json": "JSON" };
        return map[ext] || ext.toUpperCase();
    }

    function formatFileSize(bytes) {
        if (!bytes) return "0 B";
        var units = ["B", "KB", "MB", "GB"];
        var i = 0;
        var size = bytes;
        while (size >= 1024 && i < units.length - 1) {
            size /= 1024;
            i++;
        }
        return size.toFixed(i > 0 ? 1 : 0) + " " + units[i];
    }

    function showToast(message, type) {
        if (window.Dashboard && window.Dashboard.showToast) {
            window.Dashboard.showToast(message, type);
        }
    }

    return {
        init: init,
    };
})();
