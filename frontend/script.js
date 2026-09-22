// =========================================================
// API URL
// =========================================================
//
// When running locally:
// http://127.0.0.1:5000
//
// When deployed:
// automatically uses the Render URL
//

const API_URL =
    window.location.protocol === "file:"
        ? "http://127.0.0.1:5000"
        : window.location.origin;


// =========================================================
// GLOBAL VARIABLES
// =========================================================

let powerChart = null;


// =========================================================
// DOM ELEMENTS
// =========================================================

const dateSelector =
    document.getElementById("dateSelector");

const modeBadge =
    document.getElementById("modeBadge");

const dataRange =
    document.getElementById("dataRange");

const currentDate =
    document.getElementById("currentDate");

const lastUpdated =
    document.getElementById("lastUpdated");

const currentPower =
    document.getElementById("currentPower");

const predictedPower =
    document.getElementById("predictedPower");

const irradiation =
    document.getElementById("irradiation");

const temperature =
    document.getElementById("temperature");

const dailyYield =
    document.getElementById("dailyYield");

const efficiency =
    document.getElementById("efficiency");

const systemStatus =
    document.getElementById("systemStatus");

const alertCount =
    document.getElementById("alertCount");

const recommendation =
    document.getElementById("recommendation");

const alerts =
    document.getElementById("alerts");


// =========================================================
// LOAD DATASET DATE RANGE
// =========================================================

async function loadDateInfo() {

    try {

        const response = await fetch(
            `${API_URL}/api/date-info`
        );

        const result = await response.json();

        if (result.status === "success") {

            dataRange.textContent =
                `${result.min_date} → ${result.max_date}`;

            // Default to today's date.
            // If today's date is outside the dataset,
            // the application automatically enters
            // AI Forecast Mode.

            const today =
                new Date()
                    .toISOString()
                    .split("T")[0];

            dateSelector.value = today;

            await loadDashboard(today);
            await loadChart(today);
        }

    } catch (error) {

        console.error(
            "Date information error:",
            error
        );

        showConnectionError();
    }
}


// =========================================================
// LOAD DASHBOARD
// =========================================================

async function loadDashboard(
    selectedDate
) {

    try {

        const response = await fetch(
            `${API_URL}/api/dashboard?date=${selectedDate}`
        );

        const result = await response.json();

        if (result.status !== "success") {

            console.error(
                result.message
            );

            return;
        }

        const d = result.data;


        // =================================================
        // MODE
        // =================================================

        if (d.mode === "forecast") {

            modeBadge.textContent =
                "AI Forecast";

            modeBadge.className =
                "mode-badge forecast";

        } else {

            modeBadge.textContent =
                "Historical Data";

            modeBadge.className =
                "mode-badge historical";
        }


        // =================================================
        // HEADER
        // =================================================

        currentDate.textContent =
            selectedDate;

        lastUpdated.textContent =
            d.date_time || "--";


        // =================================================
        // POWER
        // =================================================

        if (d.actual_power !== null) {

            currentPower.textContent =
                `${Number(d.actual_power).toFixed(2)} kW`;

        } else {

            currentPower.textContent =
                "Forecast";
        }


        if (d.predicted_power !== null) {

            predictedPower.textContent =
                `${Number(d.predicted_power).toFixed(2)} kW`;

        } else {

            predictedPower.textContent =
                "--";
        }


        // =================================================
        // IRRADIATION
        // =================================================

        if (d.irradiation !== null) {

            irradiation.textContent =
                `${Number(d.irradiation).toFixed(3)} kW/m²`;

        } else {

            irradiation.textContent =
                "--";
        }


        // =================================================
        // TEMPERATURE
        // =================================================

        if (d.ambient_temperature !== null) {

            temperature.textContent =
                `${Number(
                    d.ambient_temperature
                ).toFixed(1)} °C`;

        } else {

            temperature.textContent =
                "--";
        }


        // =================================================
        // DAILY YIELD
        // =================================================

        if (d.daily_yield !== null) {

            dailyYield.textContent =
                `${Number(
                    d.daily_yield
                ).toFixed(2)} kWh`;

        } else {

            dailyYield.textContent =
                "--";
        }


        // =================================================
        // EFFICIENCY
        // =================================================

        if (d.efficiency !== null) {

            efficiency.textContent =
                `${Number(
                    d.efficiency
                ).toFixed(2)} %`;

        } else {

            efficiency.textContent =
                "N/A";
        }


        // =================================================
        // SYSTEM STATUS
        // =================================================

        systemStatus.textContent =
            d.system_status || "--";


        // =================================================
        // ALERT COUNT
        // =================================================

        alertCount.textContent =
            d.alert_count ?? 0;


        // =================================================
        // RECOMMENDATION
        // =================================================

        recommendation.textContent =
            d.recommendation ||
            "No recommendation available.";


        // =================================================
        // ALERTS
        // =================================================

        renderAlerts(d);

    } catch (error) {

        console.error(
            "Dashboard error:",
            error
        );

        showConnectionError();
    }
}


// =========================================================
// RENDER ALERTS
// =========================================================

function renderAlerts(data) {

    if (!alerts) {
        return;
    }


    // Forecast mode
    if (data.mode === "forecast") {

        alerts.innerHTML = `
            <div class="alert-item">
                <div class="alert-icon">AI</div>
                <div>
                    <strong>Forecast Mode</strong>
                    <p>
                        This date is outside the historical
                        dataset. The system is using an
                        AI-based forecast from historical
                        solar patterns.
                    </p>
                </div>
            </div>
        `;

        return;
    }


    // No alerts
    if (!data.alert_count) {

        alerts.innerHTML = `
            <div class="alert-item normal">
                <div class="alert-icon">✓</div>
                <div>
                    <strong>System Normal</strong>
                    <p>
                        No significant performance
                        anomalies detected.
                    </p>
                </div>
            </div>
        `;

        return;
    }


    // Alert detected
    alerts.innerHTML = `
        <div class="alert-item warning">
            <div class="alert-icon">!</div>
            <div>
                <strong>Performance Alert</strong>
                <p>
                    The system detected a deviation
                    between actual and predicted
                    power generation.
                </p>
            </div>
        </div>
    `;
}


// =========================================================
// LOAD CHART
// =========================================================

async function loadChart(
    selectedDate
) {

    try {

        const response = await fetch(
            `${API_URL}/api/chart?date=${selectedDate}`
        );

        const result = await response.json();

        if (result.status !== "success") {

            console.error(
                result.message
            );

            return;
        }

        const chartData =
            result.data;


        const labels =
            chartData.map(
                item => item.time
            );

        const actualValues =
            chartData.map(
                item => item.actual
            );

        const predictedValues =
            chartData.map(
                item => item.predicted
            );


        const canvas =
            document.getElementById(
                "powerChart"
            );

        if (!canvas) {
            return;
        }


        const ctx =
            canvas.getContext("2d");


        // Destroy previous chart
        if (powerChart) {

            powerChart.destroy();

            powerChart = null;
        }


        // =================================================
        // HISTORICAL CHART
        // =================================================

        if (result.mode === "historical") {

            powerChart =
                new Chart(
                    ctx,
                    {
                        type: "line",

                        data: {
                            labels: labels,

                            datasets: [
                                {
                                    label:
                                        "Actual Power",

                                    data:
                                        actualValues,

                                    borderWidth: 2,

                                    tension: 0.3,

                                    pointRadius: 0
                                },

                                {
                                    label:
                                        "AI Predicted Power",

                                    data:
                                        predictedValues,

                                    borderWidth: 2,

                                    tension: 0.3,

                                    pointRadius: 0
                                }
                            ]
                        },

                        options: {
                            responsive: true,

                            maintainAspectRatio:
                                false,

                            interaction: {
                                mode: "index",
                                intersect: false
                            },

                            plugins: {
                                legend: {
                                    display: true
                                }
                            },

                            scales: {

                                x: {
                                    title: {
                                        display: true,
                                        text: "Time"
                                    }
                                },

                                y: {
                                    title: {
                                        display: true,
                                        text:
                                            "Power (kW)"
                                    },

                                    beginAtZero: true
                                }
                            }
                        }
                    }
                );

            return;
        }


        // =================================================
        // FORECAST CHART
        // =================================================

        powerChart =
            new Chart(
                ctx,
                {
                    type: "line",

                    data: {
                        labels: labels,

                        datasets: [
                            {
                                label:
                                    "AI Forecast",

                                data:
                                    predictedValues,

                                borderWidth: 2,

                                tension: 0.3,

                                pointRadius: 0
                            }
                        ]
                    },

                    options: {
                        responsive: true,

                        maintainAspectRatio:
                            false,

                        interaction: {
                            mode: "index",
                            intersect: false
                        },

                        plugins: {
                            legend: {
                                display: true
                            }
                        },

                        scales: {

                            x: {
                                title: {
                                    display: true,
                                    text: "Time"
                                }
                            },

                            y: {
                                title: {
                                    display: true,
                                    text:
                                        "Forecast Power (kW)"
                                },

                                beginAtZero: true
                            }
                        }
                    }
                }
            );

    } catch (error) {

        console.error(
            "Chart error:",
            error
        );
    }
}


// =========================================================
// DATE CHANGE
// =========================================================

if (dateSelector) {

    dateSelector.addEventListener(
        "change",
        async function () {

            const selectedDate =
                this.value;

            if (!selectedDate) {
                return;
            }

            await loadDashboard(
                selectedDate
            );

            await loadChart(
                selectedDate
            );
        }
    );
}


// =========================================================
// CONNECTION ERROR
// =========================================================

function showConnectionError() {

    if (systemStatus) {

        systemStatus.textContent =
            "Backend Offline";
    }

    if (recommendation) {

        recommendation.textContent =
            "Unable to connect to the Flask backend. "
            + "Please make sure the server is running.";
    }
}


// =========================================================
// INITIALIZE
// =========================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        loadDateInfo();

    }
);