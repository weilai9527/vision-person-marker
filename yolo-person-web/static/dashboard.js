(function () {
    var sidebar = document.getElementById("sidebar");
    var dashboard = document.getElementById("dashboard");
    var sidebarToggle = document.getElementById("sidebarToggle");
    var topbarMenuBtn = document.getElementById("topbarMenuBtn");
    var mobileOverlay = document.getElementById("mobileOverlay");

    function isMobileLayout() {
        return window.matchMedia && window.matchMedia("(max-width: 768px)").matches;
    }

    function openMobileSidebar() {
        dashboard.classList.add("mobile-sidebar-open");
    }

    function closeMobileSidebar() {
        dashboard.classList.remove("mobile-sidebar-open");
    }

    function toggleSidebar() {
        if (isMobileLayout()) {
            dashboard.classList.toggle("mobile-sidebar-open");
            return;
        }
        dashboard.classList.toggle("sidebar-collapsed");
        var isCollapsed = dashboard.classList.contains("sidebar-collapsed");
        localStorage.setItem("sidebarCollapsed", isCollapsed ? "true" : "false");
    }

    if (sidebarToggle) {
        sidebarToggle.addEventListener("click", toggleSidebar);
    }
    if (topbarMenuBtn) {
        topbarMenuBtn.addEventListener("click", toggleSidebar);
    }
    if (mobileOverlay) {
        mobileOverlay.addEventListener("click", closeMobileSidebar);
    }

    var savedState = localStorage.getItem("sidebarCollapsed");
    if (savedState === "true" && !isMobileLayout()) {
        dashboard.classList.add("sidebar-collapsed");
    }
    window.addEventListener("resize", function () {
        if (isMobileLayout()) {
            dashboard.classList.remove("sidebar-collapsed");
        } else {
            closeMobileSidebar();
            if (localStorage.getItem("sidebarCollapsed") === "true") {
                dashboard.classList.add("sidebar-collapsed");
            }
        }
    });

    var apiModal = document.getElementById("apiModal");
    var openApiButton = document.getElementById("openApiButton");
    var closeApiButton = document.getElementById("closeApiButton");
    var cancelApiButton = document.getElementById("cancelApiButton");
    var provider = document.getElementById("provider");
    var apiUrl = document.getElementById("apiUrl");
    var apiKey = document.getElementById("apiKey");
    var model = document.getElementById("model");

    function openApiModal() {
        if (!apiModal) return;
        apiModal.classList.add("open");
        apiModal.setAttribute("aria-hidden", "false");
        if (apiKey) apiKey.focus();
    }

    function closeApiModal() {
        if (!apiModal) return;
        apiModal.classList.remove("open");
        apiModal.setAttribute("aria-hidden", "true");
    }

    if (openApiButton) {
        openApiButton.addEventListener("click", openApiModal);
    }
    if (closeApiButton) {
        closeApiButton.addEventListener("click", closeApiModal);
    }
    if (cancelApiButton) {
        cancelApiButton.addEventListener("click", closeApiModal);
    }

    if (provider) {
        provider.addEventListener("change", function () {
            var selectedProvider = provider.options[provider.selectedIndex];
            if (selectedProvider.dataset.url) {
                apiUrl.value = selectedProvider.dataset.url;
            }
            if (selectedProvider.dataset.model) {
                model.value = selectedProvider.dataset.model;
            }
        });
    }

    if (apiModal) {
        apiModal.addEventListener("click", function (event) {
            if (event.target === apiModal) {
                closeApiModal();
            }
        });
        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && apiModal.classList.contains("open")) {
                closeApiModal();
            }
        });
    }

    function showToast(message, type) {
        var toast = document.getElementById("globalToast");
        if (!toast) return;
        toast.textContent = message;
        toast.className = "toast " + (type || "success") + " show";
        setTimeout(function () {
            toast.classList.remove("show");
        }, 3200);
    }

    window.Dashboard = {
        showToast: showToast,
        openApiModal: openApiModal,
        closeApiModal: closeApiModal
    };

    var chartInstances = {};

    function initCharts() {
        var charts = document.querySelectorAll("[data-chart]");
        charts.forEach(function (canvas) {
            var type = canvas.dataset.chart;
            var labels = [];
            var data = [];
            try {
                labels = JSON.parse(canvas.dataset.labels || "[]");
                data = JSON.parse(canvas.dataset.values || "[]");
            } catch (e) {
                return;
            }

            var baseConfig = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            font: { family: "'Microsoft YaHei', 'PingFang SC', sans-serif" },
                            padding: 16,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: "rgba(26, 31, 46, 0.92)",
                        titleFont: { family: "'Microsoft YaHei', 'PingFang SC', sans-serif", size: 13 },
                        bodyFont: { family: "'Microsoft YaHei', 'PingFang SC', sans-serif", size: 12 },
                        padding: 12,
                        cornerRadius: 8
                    }
                }
            };

            var configs = {
                line: {
                    type: "line",
                    data: {
                        labels: labels,
                        datasets: [{
                            label: "检测数量",
                            data: data,
                            borderColor: "#0f766e",
                            backgroundColor: "rgba(15, 118, 110, 0.08)",
                            borderWidth: 2.5,
                            pointBackgroundColor: "#0f766e",
                            pointBorderColor: "#fff",
                            pointBorderWidth: 2,
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            fill: true,
                            tension: 0.35
                        }]
                    },
                    options: {
                        ...baseConfig,
                        scales: {
                            x: {
                                grid: { display: false },
                                ticks: { font: { size: 11 } }
                            },
                            y: {
                                beginAtZero: true,
                                grid: { color: "rgba(0,0,0,0.04)" },
                                ticks: { font: { size: 11 }, stepSize: 1 }
                            }
                        }
                    }
                },
                bar: {
                    type: "bar",
                    data: {
                        labels: labels,
                        datasets: [{
                            label: "检测数量",
                            data: data,
                            backgroundColor: [
                                "rgba(15, 118, 110, 0.78)",
                                "rgba(2, 132, 199, 0.78)",
                                "rgba(234, 88, 12, 0.78)",
                                "rgba(124, 58, 237, 0.78)",
                                "rgba(220, 38, 38, 0.78)"
                            ],
                            borderColor: [
                                "#0f766e", "#0284c7", "#ea580c", "#7c3aed", "#dc2626"
                            ],
                            borderWidth: 1,
                            borderRadius: 4,
                            barPercentage: 0.6
                        }]
                    },
                    options: {
                        ...baseConfig,
                        scales: {
                            x: {
                                grid: { display: false },
                                ticks: { font: { size: 11 } }
                            },
                            y: {
                                beginAtZero: true,
                                grid: { color: "rgba(0,0,0,0.04)" },
                                ticks: { font: { size: 11 }, stepSize: 1 }
                            }
                        }
                    }
                },
                pie: {
                    type: "pie",
                    data: {
                        labels: labels,
                        datasets: [{
                            data: data,
                            backgroundColor: [
                                "rgba(15, 118, 110, 0.85)",
                                "rgba(2, 132, 199, 0.85)",
                                "rgba(234, 88, 12, 0.85)",
                                "rgba(124, 58, 237, 0.85)",
                                "rgba(220, 38, 38, 0.85)"
                            ],
                            borderColor: "#ffffff",
                            borderWidth: 2
                        }]
                    },
                    options: {
                        ...baseConfig,
                        cutout: "55%",
                        plugins: {
                            ...baseConfig.plugins,
                            legend: {
                                ...baseConfig.plugins.legend,
                                position: "bottom"
                            }
                        }
                    }
                },
                doughnut: {
                    type: "doughnut",
                    data: {
                        labels: labels,
                        datasets: [{
                            data: data,
                            backgroundColor: [
                                "rgba(15, 118, 110, 0.85)",
                                "rgba(234, 88, 12, 0.85)",
                                "rgba(100, 116, 139, 0.35)"
                            ],
                            borderColor: "#ffffff",
                            borderWidth: 2
                        }]
                    },
                    options: {
                        ...baseConfig,
                        cutout: "70%",
                        plugins: {
                            ...baseConfig.plugins,
                            legend: {
                                ...baseConfig.plugins.legend,
                                position: "bottom"
                            }
                        }
                    }
                }
            };

            var config = configs[type];
            if (!config) return;

            if (chartInstances[canvas.id]) {
                chartInstances[canvas.id].destroy();
            }
            chartInstances[canvas.id] = new Chart(canvas.getContext("2d"), config);
        });
    }

    if (typeof Chart !== "undefined") {
        document.addEventListener("DOMContentLoaded", initCharts);
    }

    var navToggles = document.querySelectorAll(".nav-toggle");
    navToggles.forEach(function (toggle) {
        toggle.addEventListener("click", function (e) {
            e.stopPropagation();
            var targetId = this.dataset.target;
            var submenu = document.getElementById(targetId);
            if (!submenu) return;

            var isOpen = submenu.classList.contains("open");
            if (isOpen) {
                submenu.classList.remove("open");
                this.classList.remove("open");
                this.setAttribute("aria-expanded", "false");
            } else {
                submenu.classList.add("open");
                this.classList.add("open");
                this.setAttribute("aria-expanded", "true");
            }
        });
    });

    var currentEndpoint = document.querySelector("meta[name='current-endpoint']");
    if (currentEndpoint) {
        var endpoint = currentEndpoint.content;
        var subItems = document.querySelectorAll(".nav-sub-item.active");
        if (subItems.length > 0) {
            var parentGroup = subItems[0].closest(".nav-group");
            if (parentGroup) {
                var toggle = parentGroup.querySelector(".nav-toggle");
                var submenu = parentGroup.querySelector(".nav-submenu");
                if (toggle && submenu) {
                    submenu.classList.add("open");
                    toggle.classList.add("open");
                    toggle.setAttribute("aria-expanded", "true");
                }
            }
        }
    }
})();
