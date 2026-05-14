var HistoryPage = (function () {
    "use strict";

    var state = {
        type: "image_pedestrian",
        typeLabel: "",
        detectionLabel: "",
        mediaLabel: "",
        isImage: true,
        page: 1,
        perPage: 20,
        total: 0,
        pages: 0,
        keyword: "",
        startDate: "",
        endDate: "",
        apiProvider: "",
        statusFilter: "",
        sortBy: "uploaded_at",
        sortOrder: "desc",
        selectedIds: [],
        loading: false,
    };

    var els = {};

    function qs(sel, ctx) { return (ctx || document).querySelector(sel); }

    function qsa(sel, ctx) { return Array.from((ctx || document).querySelectorAll(sel)); }

    function debounce(fn, ms) {
        var timer = null;
        return function () {
            var args = arguments;
            var ctx = this;
            if (timer) clearTimeout(timer);
            timer = setTimeout(function () { fn.apply(ctx, args); }, ms);
        };
    }

    function getApiUrl() {
        return state.isImage ? "/api/history/image/records" : "/api/history/video/records";
    }

    function getDetailUrl(id) {
        return state.isImage ? "/api/history/image/detail/" + id : "/api/history/video/detail/" + id;
    }

    function getBatchDeleteUrl() {
        return state.isImage ? "/api/history/image/batch-delete" : "/api/history/video/batch-delete";
    }

    function formatDate(raw) {
        if (!raw) return "-";
        return raw;
    }

    function escapeHtml(text) {
        if (!text) return "";
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }

    function formatSize(w, h) {
        if (!w || !h) return "-";
        return w + "\u00d7" + h;
    }

    function statusTag(status) {
        var labels = {
            completed: "\u5df2\u5b8c\u6210",
            processing: "\u5904\u7406\u4e2d",
            pending: "\u7b49\u5f85\u4e2d",
            failed: "\u5931\u8d25",
        };
        var cls = status || "pending";
        var label = labels[status] || status || "\u672a\u77e5";
        return '<span class="status-tag ' + cls + '">' + label + "</span>";
    }

    function providerLabel(val) {
        var map = {
            "local_yolo": "\u672c\u5730 YOLO",
            "local_yolo_vehicle": "\u672c\u5730 YOLO \u8f66\u8f86",
            "gpt-4o-mini": "GPT-4o-mini",
            "dashscope_qwen": "\u901a\u4e49\u5343\u95ee",
            "moonshot": "Kimi",
        };
        return map[val] || val || "-";
    }

    function renderImageRow(rec) {
        var checked = state.selectedIds.indexOf(rec.id) !== -1;
        var thumb = rec.original_image_path
            ? '<img src="/' + rec.original_image_path + '" alt="" class="filename-icon" onerror="this.style.display=\'none\'" />'
            : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="filename-icon" style="padding:6px"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>';
        return '<tr class="' + (checked ? "selected" : "") + '">'
            + '<td class="col-check"><input type="checkbox" class="row-check" value="' + rec.id + '" ' + (checked ? "checked" : "") + " /></td>"
            + '<td class="col-id">' + rec.id + "</td>"
            + '<td class="col-filename"><div class="filename-cell">' + thumb + '<span class="filename-text" title="' + escapeHtml(rec.original_filename) + '">' + escapeHtml(rec.original_filename) + "</span></div></td>"
            + '<td class="col-date">' + formatDate(rec.uploaded_at) + "</td>"
            + '<td class="col-size">' + formatSize(rec.original_width, rec.original_height) + "</td>"
            + '<td class="col-count">' + rec.detection_count + "</td>"
            + '<td class="col-provider">' + (rec.detections && rec.detections[0] ? providerLabel(rec.detections[0].api_provider) : "-") + "</td>"
            + '<td class="col-actions"><button class="action-btn detail-btn" data-id="' + rec.id + '">\u8be6\u60c5</button></td>'
            + "</tr>";
    }

    function renderVideoRow(rec) {
        var checked = state.selectedIds.indexOf(rec.id) !== -1;
        return '<tr class="' + (checked ? "selected" : "") + '">'
            + '<td class="col-check"><input type="checkbox" class="row-check" value="' + rec.id + '" ' + (checked ? "checked" : "") + " /></td>"
            + '<td class="col-id">' + rec.id + "</td>"
            + '<td class="col-filename"><div class="filename-cell"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="filename-icon" style="padding:6px"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg><span class="filename-text" title="' + escapeHtml(rec.original_filename) + '">' + escapeHtml(rec.original_filename) + "</span></div></td>"
            + '<td class="col-date">' + formatDate(rec.uploaded_at) + "</td>"
            + '<td class="col-status">' + statusTag(rec.status) + "</td>"
            + '<td class="col-frames">' + (rec.total_frames || 0) + "</td>"
            + '<td class="col-count">' + (rec.total_persons || 0) + "</td>"
            + '<td class="col-actions"><button class="action-btn detail-btn" data-id="' + rec.id + '">\u8be6\u60c5</button></td>'
            + "</tr>";
    }

    function renderPagination() {
        var info = "\u5171 " + state.total + " \u6761\u8bb0\u5f55\uff0c\u7b2c " + state.page + "/" + state.pages + " \u9875";
        els.paginationInfo.textContent = info;

        if (state.pages <= 1) {
            els.paginationControls.innerHTML = "";
            return;
        }

        var html = "";
        var p = state.page;
        var total = state.pages;

        html += '<button class="page-btn" data-page="' + (p - 1) + '" ' + (p <= 1 ? "disabled" : "") + ">\u2039</button>";

        var start = Math.max(1, p - 2);
        var end = Math.min(total, p + 2);
        if (start > 1) {
            html += '<button class="page-btn" data-page="1">1</button>';
            if (start > 2) html += '<button class="page-btn" disabled>...</button>';
        }
        for (var i = start; i <= end; i++) {
            html += '<button class="page-btn' + (i === p ? " active" : "") + '" data-page="' + i + '">' + i + "</button>";
        }
        if (end < total) {
            if (end < total - 1) html += '<button class="page-btn" disabled>...</button>';
            html += '<button class="page-btn" data-page="' + total + '">' + total + "</button>";
        }

        html += '<button class="page-btn" data-page="' + (p + 1) + '" ' + (p >= total ? "disabled" : "") + ">\u203a</button>";

        els.paginationControls.innerHTML = html;
    }

    function renderTable(records) {
        if (!records || records.length === 0) {
            els.historyBody.innerHTML = '<tr><td colspan="8" class="empty-cell">\u6682\u65e0\u8bb0\u5f55</td></tr>';
            return;
        }
        var rows = "";
        for (var i = 0; i < records.length; i++) {
            rows += state.isImage ? renderImageRow(records[i]) : renderVideoRow(records[i]);
        }
        els.historyBody.innerHTML = rows;
    }

    function updateStats() {
        var totalEl = document.getElementById("totalCount");
        var pageEl = document.getElementById("currentPage");
        if (totalEl) totalEl.textContent = state.total;
        if (pageEl) pageEl.textContent = state.page + "/" + state.pages;
    }

    function updateBatchBtn() {
        var btn = els.batchDeleteBtn;
        if (!btn) return;
        var count = state.selectedIds.length;
        btn.disabled = count === 0;
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>\u5220\u9664\u9009\u4e2d'
            + (count > 0 ? " (" + count + ")" : "");
    }

    function loadRecords() {
        if (state.loading) return;
        state.loading = true;
        els.historyBody.innerHTML = '<tr><td colspan="8" class="empty-cell">\u52a0\u8f7d\u4e2d...</td></tr>';

        var params = new URLSearchParams();
        params.set("page", state.page);
        params.set("per_page", state.perPage);
        params.set("sort_by", state.sortBy);
        params.set("sort_order", state.sortOrder);
        params.set("keyword", state.keyword);
        params.set("start_date", state.startDate);
        params.set("end_date", state.endDate);

        if (state.isImage) {
            var typeMap = { image_pedestrian: "pedestrian", image_vehicle: "vehicle" };
            params.set("type", typeMap[state.type] || "all");
            params.set("api_provider", state.apiProvider);
        } else {
            var targetMap = { video_pedestrian: "person", video_vehicle: "vehicle" };
            params.set("target", targetMap[state.type] || "person");
            params.set("status", state.statusFilter);
        }

        fetch(getApiUrl() + "?" + params.toString())
            .then(function (r) { return r.json(); })
            .then(function (data) {
                state.loading = false;
                if (data.success) {
                    state.total = data.total || 0;
                    state.pages = data.pages || 0;
                    renderTable(data.records || []);
                    renderPagination();
                    updateStats();
                    state.selectedIds = [];
                    if (els.selectAllCheck) els.selectAllCheck.checked = false;
                    updateBatchBtn();
                } else {
                    els.historyBody.innerHTML = '<tr><td colspan="8" class="empty-cell">\u52a0\u8f7d\u5931\u8d25</td></tr>';
                }
            })
            .catch(function () {
                state.loading = false;
                els.historyBody.innerHTML = '<tr><td colspan="8" class="empty-cell">\u7f51\u7edc\u9519\u8bef</td></tr>';
            });
    }

    function goToPage(page) {
        if (page < 1 || page > state.pages) return;
        state.page = page;
        loadRecords();
    }

    function openDetail(id) {
        var modal = els.detailModal;
        var body = els.detailModalBody;
        if (!modal || !body) return;

        modal.hidden = false;
        body.innerHTML = '<div class="detail-loading">\u52a0\u8f7d\u4e2d...</div>';

        fetch(getDetailUrl(id))
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.success && data.record) {
                    body.innerHTML = state.isImage
                        ? buildImageDetailHtml(data.record)
                        : buildVideoDetailHtml(data.record);
                } else {
                    body.innerHTML = '<div class="detail-error">\u83b7\u53d6\u8be6\u60c5\u5931\u8d25</div>';
                }
            })
            .catch(function () {
                body.innerHTML = '<div class="detail-error">\u7f51\u7edc\u9519\u8bef</div>';
            });
    }

    function closeDetail() {
        if (els.detailModal) els.detailModal.hidden = true;
    }

    function buildImageDetailHtml(rec) {
        var html = "";

        html += '<div class="detail-section">';
        html += '<h4 class="detail-section-title">\u57fa\u672c\u4fe1\u606f</h4>';
        html += '<div class="detail-grid">';
        html += field("ID", rec.id);
        html += field("\u6587\u4ef6\u540d", rec.original_filename);
        html += field("\u4e0a\u4f20\u65f6\u95f4", rec.uploaded_at);
        html += field("\u539f\u59cb\u5c3a\u5bf8", formatSize(rec.original_width, rec.original_height));
        html += field("\u6a21\u578b\u5c3a\u5bf8", formatSize(rec.model_width, rec.model_height));
        html += field("\u68c0\u6d4b\u6b21\u6570", rec.detection_count);
        html += "</div></div>";

        if (rec.original_image_path) {
            html += '<div class="detail-section">';
            html += '<h4 class="detail-section-title">\u539f\u59cb\u56fe\u50cf</h4>';
            html += '<img src="/' + rec.original_image_path + '" alt="original" class="detail-image-preview" onerror="this.style.display=\'none\'" />';
            html += "</div>";
        }

        if (rec.detections && rec.detections.length > 0) {
            for (var i = 0; i < rec.detections.length; i++) {
                var d = rec.detections[i];
                html += '<div class="detail-section">';
                html += '<h4 class="detail-section-title">\u68c0\u6d4b\u8bb0\u5f55 #' + (i + 1) + "</h4>";
                html += '<div class="detail-grid">';
                html += field("\u68c0\u6d4b\u65f6\u95f4", d.detected_at);
                html += field("\u68c0\u6d4b\u65b9\u5f0f", providerLabel(d.api_provider));
                html += field("\u6a21\u578b\u540d\u79f0", d.model_name || "-");
                html += field("\u68c0\u6d4b\u6570\u91cf", d.person_count + " \u4e2a" + (state.type === "image_pedestrian" ? "\u884c\u4eba" : "\u8f66\u8f86"));
                html += "</div>";

                if (d.result_image_path) {
                    html += '<div style="margin-top:10px">';
                    html += '<img src="/' + d.result_image_path + '" alt="result" class="detail-image-preview" onerror="this.style.display=\'none\'" />';
                    html += '<div class="detection-actions">';
                    html += '<a href="/' + d.result_image_path + '" target="_blank" class="action-btn">\u67e5\u770b\u539f\u56fe</a>';
                    html += "</div></div>";
                }

                if (d.analysis_text) {
                    html += '<div style="margin-top:10px">';
                    html += '<h5 style="font-size:12px;color:var(--text-secondary);margin:0 0 6px 0;font-weight:600;">LLM \u5206\u6790\u7ed3\u679c</h5>';
                    html += '<div class="detail-analysis-box">' + escapeHtml(d.analysis_text) + "</div></div>";
                }

                html += "</div>";
            }
        }

        return html;
    }

    function buildVideoDetailHtml(rec) {
        var html = "";

        html += '<div class="detail-section">';
        html += '<h4 class="detail-section-title">\u57fa\u672c\u4fe1\u606f</h4>';
        html += '<div class="detail-grid">';
        html += field("ID", rec.id);
        html += field("\u6587\u4ef6\u540d", rec.original_filename);
        html += field("\u4e0a\u4f20\u65f6\u95f4", rec.uploaded_at);
        html += field("\u68c0\u6d4b\u76ee\u6807", rec.detection_target === "person" ? "\u884c\u4eba" : "\u8f66\u8f86");
        html += field("\u72b6\u6001", statusTag(rec.status));
        html += field("\u5206\u8fa8\u7387", formatSize(rec.video_width, rec.video_height));
        html += "</div></div>";

        html += '<div class="detail-section">';
        html += '<h4 class="detail-section-title">\u5904\u7406\u53c2\u6570</h4>';
        html += '<div class="detail-grid">';
        html += field("\u603b\u5e27\u6570", rec.total_frames || 0);
        html += field("\u5df2\u5904\u7406\u5e27", rec.processed_frames || 0);
        html += field("FPS", rec.fps || 0);
        html += field("\u65f6\u957f(\u79d2)", rec.duration || 0);
        html += "</div></div>";

        html += '<div class="detail-section">';
        html += '<h4 class="detail-section-title">\u68c0\u6d4b\u7ed3\u679c</h4>';
        html += '<div class="detail-grid">';
        html += field("\u6700\u5927\u540c\u5e27\u6570", rec.total_persons || 0);
        html += field("\u53bb\u91cd\u603b\u6570", rec.unique_count || 0);
        html += field("\u7d2f\u8ba1\u603b\u6570", rec.sum_count || 0);
        html += field("\u5e73\u5747\u7f6e\u4fe1\u5ea6", rec.avg_confidence || 0);
        html += "</div></div>";

        if (rec.has_result && rec.processed_video_path) {
            html += '<div class="detection-actions">';
            html += '<a href="/' + rec.processed_video_path + '" target="_blank" class="action-btn">\u4e0b\u8f7d\u5904\u7406\u540e\u89c6\u9891</a>';
            html += "</div>";
        }

        if (rec.error_message) {
            html += '<div class="detail-section">';
            html += '<h4 class="detail-section-title">\u9519\u8bef\u4fe1\u606f</h4>';
            html += '<div class="detail-error">' + escapeHtml(rec.error_message) + "</div></div>";
        }

        return html;
    }

    function field(label, value) {
        return '<div class="detail-field"><span class="detail-label">' + label + '</span><span class="detail-value">' + value + "</span></div>";
    }

    function bindEvents() {
        els.selectAllCheck.addEventListener("change", function () {
            var checked = this.checked;
            qsa(".row-check", els.historyBody).forEach(function (cb) {
                cb.checked = checked;
                var id = parseInt(cb.value, 10);
                var idx = state.selectedIds.indexOf(id);
                if (checked && idx === -1) {
                    state.selectedIds.push(id);
                } else if (!checked && idx !== -1) {
                    state.selectedIds.splice(idx, 1);
                }
            });
            qsa("tr", els.historyBody).forEach(function (tr) {
                tr.classList.toggle("selected", checked);
            });
            updateBatchBtn();
        });

        els.historyBody.addEventListener("change", function (e) {
            if (e.target.classList.contains("row-check")) {
                var id = parseInt(e.target.value, 10);
                var idx = state.selectedIds.indexOf(id);
                if (e.target.checked && idx === -1) {
                    state.selectedIds.push(id);
                } else if (!e.target.checked && idx !== -1) {
                    state.selectedIds.splice(idx, 1);
                }
                var tr = e.target.closest("tr");
                if (tr) tr.classList.toggle("selected", e.target.checked);
                var allChecked = qsa(".row-check", els.historyBody).every(function (cb) { return cb.checked; });
                els.selectAllCheck.checked = allChecked;
                updateBatchBtn();
            }
        });

        els.historyBody.addEventListener("click", function (e) {
            var btn = e.target.closest(".detail-btn");
            if (btn) {
                var id = parseInt(btn.getAttribute("data-id"), 10);
                openDetail(id);
            }
        });

        els.paginationControls.addEventListener("click", function (e) {
            var btn = e.target.closest(".page-btn");
            if (btn && !btn.disabled && !btn.classList.contains("active")) {
                var page = parseInt(btn.getAttribute("data-page"), 10);
                if (!isNaN(page)) goToPage(page);
            }
        });

        var debouncedSearch = debounce(function () {
            state.keyword = els.keywordInput.value.trim();
            state.page = 1;
            loadRecords();
        }, 400);

        els.keywordInput.addEventListener("input", debouncedSearch);

        els.startDateFilter.addEventListener("change", function () {
            state.startDate = this.value;
            state.page = 1;
            loadRecords();
        });

        els.endDateFilter.addEventListener("change", function () {
            state.endDate = this.value;
            state.page = 1;
            loadRecords();
        });

        var apiFilter = document.getElementById("apiProviderFilter");
        if (apiFilter) {
            apiFilter.addEventListener("change", function () {
                state.apiProvider = this.value;
                state.page = 1;
                loadRecords();
            });
        }

        var statusFilter = document.getElementById("statusFilter");
        if (statusFilter) {
            statusFilter.addEventListener("change", function () {
                state.statusFilter = this.value;
                state.page = 1;
                loadRecords();
            });
        }

        els.refreshBtn.addEventListener("click", function () {
            state.page = 1;
            loadRecords();
        });

        els.batchDeleteBtn.addEventListener("click", function () {
            if (state.selectedIds.length === 0) return;
            if (!confirm("\u786e\u5b9a\u5220\u9664\u9009\u4e2d\u7684 " + state.selectedIds.length + " \u6761\u8bb0\u5f55\u5417\uff1f\u6b64\u64cd\u4f5c\u4e0d\u53ef\u64a4\u9500\u3002")) return;

            var btn = els.batchDeleteBtn;
            btn.disabled = true;
            btn.textContent = "\u5220\u9664\u4e2d...";

            fetch(getBatchDeleteUrl(), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ids: state.selectedIds }),
            })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.success) {
                        state.selectedIds = [];
                        updateBatchBtn();
                        loadRecords();
                    } else {
                        alert("\u5220\u9664\u5931\u8d25\uff1a" + (data.error || "\u672a\u77e5\u9519\u8bef"));
                        btn.disabled = false;
                        updateBatchBtn();
                    }
                })
                .catch(function () {
                    alert("\u7f51\u7edc\u9519\u8bef\uff0c\u5220\u9664\u5931\u8d25");
                    btn.disabled = false;
                    updateBatchBtn();
                });
        });

        els.detailModalClose.addEventListener("click", closeDetail);

        els.detailModal.addEventListener("click", function (e) {
            if (e.target === els.detailModal) closeDetail();
        });

        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape") closeDetail();
        });
    }

    function init(opts) {
        if (!opts) opts = {};
        state.type = opts.type || "image_pedestrian";
        state.typeLabel = opts.typeLabel || "";
        state.detectionLabel = opts.detectionLabel || "";
        state.mediaLabel = opts.mediaLabel || "";
        state.isImage = opts.isImage !== undefined ? opts.isImage : true;

        var pageEl = document.getElementById("historyPage");
        if (!pageEl) return;

        els.keywordInput = document.getElementById("keywordInput");
        els.startDateFilter = document.getElementById("startDateFilter");
        els.endDateFilter = document.getElementById("endDateFilter");
        els.selectAllCheck = document.getElementById("selectAllCheck");
        els.historyBody = document.getElementById("historyBody");
        els.paginationInfo = document.getElementById("paginationInfo");
        els.paginationControls = document.getElementById("paginationControls");
        els.refreshBtn = document.getElementById("refreshBtn");
        els.batchDeleteBtn = document.getElementById("batchDeleteBtn");
        els.detailModal = document.getElementById("detailModal");
        els.detailModalBody = document.getElementById("detailModalBody");
        els.detailModalClose = document.getElementById("detailModalClose");

        if (!els.historyBody) return;

        bindEvents();
        loadRecords();
    }

    return { init: init };
})();
