"use strict";


const BANDS = [
    "160m",
    "80m",
    "40m",
    "30m",
    "20m",
    "17m",
    "15m",
    "12m",
    "10m",
];


function byId(id) {
    return document.getElementById(id);
}


function setText(id, value) {
    const element = byId(id);

    if (element) {
        element.textContent = value;
    }
}


function formatNumber(value, decimals = 1) {
    if (value === null || value === undefined) {
        return "--";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "--";
    }

    return number.toFixed(decimals);
}


function formatFrequency(value) {
    if (
        value === null ||
        value === undefined
    ) {
        return "";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "";
    }

    return `${number.toFixed(3)} MHz`;
}


/* ============================================================
   SPACE WEATHER
   ============================================================ */

function updateSpaceWeather(weather) {
    if (!weather) {
        return;
    }

    setText(
        "sfi",
        weather.solar_flux_10_7 ?? "--"
    );

    setText(
        "k-index",
        weather.australian_k_index ??
        weather.planetary_k_index ??
        "--"
    );

    setText(
        "a-index",
        weather.a_index ?? "--"
    );

    setText(
        "xray",
        weather.xray_flux ?? "--"
    );
}


/* ============================================================
   INDEPENDENT BAND CARDS
   ============================================================ */

function updateBandCards(report) {

    const forecasts =
        report.hap_band_forecast || {};

    const currentHour =
        Number(report.current_utc_hour ?? 0);

    for (const band of BANDS) {

        const card =
            byId(`band-card-${band}`);

        const score =
            byId(`score-${band}`);

        const status =
            byId(`status-${band}`);

        const bar =
            byId(`bar-${band}`);

        const bandForecast =
            forecasts[band] || {};

        const prediction =
            bandForecast[String(currentHour)] ??
            bandForecast[currentHour];

        if (
            !prediction ||
            prediction.error
        ) {

            if (score) {
                score.textContent = "--/49";
            }

            if (status) {
                status.textContent =
                    "HAP data unavailable";
            }

            if (bar) {
                bar.style.width = "0%";
            }

            if (card) {
                card.classList.remove(
                    "hap-supported"
                );

                card.classList.add(
                    "hap-unavailable"
                );
            }

            continue;
        }

        const gridPoints =
            Number(
                prediction.grid_points ?? 49
            );

        const supportedGridPoints =
            Number(
                prediction.supported_grid_points ?? 0
            );

        const baseSupported =
            Boolean(
                prediction.supported
            );

        if (score) {
            score.textContent =
                `${supportedGridPoints}/${gridPoints}`;
        }

        if (status) {

            status.textContent =
                baseSupported
                    ? "Supported at Nelson"
                    : "Not supported at Nelson";
        }

        if (bar) {

            const percentage =
                gridPoints > 0
                    ? (
                        supportedGridPoints /
                        gridPoints
                    ) * 100
                    : 0;

            bar.style.width =
                `${Math.max(
                    0,
                    Math.min(
                        100,
                        percentage
                    )
                )}%`;

            bar.title =
                `${supportedGridPoints}/${gridPoints} ` +
                `regional HAP grid points`;
        }

        if (card) {

            card.classList.toggle(
                "hap-supported",
                baseSupported
            );

            card.classList.toggle(
                "hap-not-supported",
                !baseSupported
            );

            card.classList.remove(
                "hap-unavailable"
            );
        }
    }
}


/* ============================================================
   CURRENT UNIVERSAL HAP
   ============================================================ */

function updateCurrentHAP(report) {

    setText(
        "current-band",
        report.current_band || "--"
    );

    if (
        report.current_frequency_mhz !== null &&
        report.current_frequency_mhz !== undefined
    ) {
        setText(
            "current-frequency",
            `(${formatFrequency(
                report.current_frequency_mhz
            )})`
        );
    } else {
        setText(
            "current-frequency",
            ""
        );
    }

    if (
        report.current_support !== null &&
        report.current_support !== undefined
    ) {
        setText(
            "current-support",
            `${report.current_support}/49`
        );
    } else {
        setText(
            "current-support",
            "--"
        );
    }
}


/* ============================================================
   NEXT TRANSITION
   ============================================================ */

function updateNextTransition(report) {

    if (
        report.next_transition_utc !== null &&
        report.next_transition_utc !== undefined
    ) {

        setText(
            "next-transition",
            `${String(
                report.next_transition_utc
            ).padStart(2, "0")} UTC`
        );

    } else {

        setText(
            "next-transition",
            "No transition available"
        );
    }

    setText(
        "next-band",
        report.next_transition_band || "--"
    );

    if (
        report.next_transition_frequency_mhz !==
        null &&
        report.next_transition_frequency_mhz !==
        undefined
    ) {

        setText(
            "next-frequency",
            formatFrequency(
                report.next_transition_frequency_mhz
            )
        );

    } else {

        setText(
            "next-frequency",
            "--"
        );
    }
}


/* ============================================================
   UNIVERSAL 24-HOUR FORECAST
   ============================================================ */

function updateUniversalForecast(report) {

    const container =
        byId("hap-timeline");

    if (!container) {
        return;
    }

    const forecast =
        report.hap_forecast || {};

    const hours = [];

    for (let hour = 0; hour < 24; hour++) {

        const item =
            forecast[String(hour)] ??
            forecast[hour] ??
            {};

        hours.push({
            hour,
            band: item.band || "—",
            support:
                item.support !== null &&
                item.support !== undefined
                    ? item.support
                    : null
        });
    }

    const rows = [
        hours.slice(0, 12),
        hours.slice(12, 24)
    ];

    const transitions = [];

    for (let i = 1; i < hours.length; i++) {

        if (
            hours[i - 1].band !== "—" &&
            hours[i].band !== "—" &&
            hours[i - 1].band !== hours[i].band
        ) {
            transitions.push({
                hour: hours[i].hour,
                from: hours[i - 1].band,
                to: hours[i].band
            });
        }
    }

    container.innerHTML = `
        <div class="hap-timeline">

            <div class="hap-timeline-header">
                <div class="hap-timeline-title">
                    24-hour recommendation
                </div>

                <div class="hap-timeline-subtitle">
                    SWS recommended band at Nelson
                </div>
            </div>

            <div class="hap-hours">

                ${rows.map(row => `
                    <div class="hap-hour-row">

                        ${row.map(item => `
                            <div
                                class="hap-hour-cell"
                                title="${String(item.hour).padStart(2, "0")}:00 UTC · ${item.band}${item.support !== null ? ` · ${item.support} local pixels` : ""}"
                            >

                                <div class="hap-hour-label">
                                    ${String(item.hour).padStart(2, "0")}
                                </div>

                                <div class="hap-hour-band">
                                    ${item.band}
                                </div>

                                ${
                                    item.support !== null
                                        ? `
                                            <div class="hap-hour-support">
                                                ${item.support}/24
                                            </div>
                                        `
                                        : ""
                                }

                            </div>
                        `).join("")}

                    </div>
                `).join("")}

            </div>

            ${
                transitions.length
                    ? `
                        <div class="hap-transitions">

                            ${transitions.map(t => `
                                <div class="hap-transition">

                                    <span class="hap-transition-time">
                                        ${String(t.hour).padStart(2, "0")}:00 UTC
                                    </span>

                                    <span class="hap-transition-arrow">
                                        ${t.from} → ${t.to}
                                    </span>

                                </div>
                            `).join("")}

                        </div>
                    `
                    : ""
            }

            <div class="hap-timeline-legend">

                <span>
                    <strong>UTC hour</strong>
                </span>

                <span>
                    <strong>Band</strong>
                    SWS recommendation
                </span>

                <span>
                    <strong>Pixels</strong>
                    local HAP support
                </span>

            </div>

        </div>
    `;
}



/* ============================================================
   INDEPENDENT 24-HOUR BAND FORECAST
   ============================================================ */

function updateIndependentBandForecast(
    report
) {

    const container =
        byId("hap-band-forecast");

    if (!container) {
        return;
    }

    const forecasts =
        report.hap_band_forecast || {};

    const currentHour =
        Number(report.current_utc_hour ?? 0);

    container.innerHTML = "";

    // Hour header
    const header =
        document.createElement("div");

    header.className =
        "independent-band-row independent-band-header";

    const headerTitle =
        document.createElement("div");

    headerTitle.className =
        "independent-band-title";

    headerTitle.textContent =
        "BAND";

    header.appendChild(
        headerTitle
    );

    const headerHours =
        document.createElement("div");

    headerHours.className =
        "independent-band-hours";

    for (let hour = 0; hour < 24; hour++) {

        const cell =
            document.createElement("div");

        cell.className =
            "independent-band-hour header-hour";

        if (hour === currentHour) {
            cell.classList.add(
                "current-hour"
            );
        }

        cell.textContent =
            String(hour).padStart(2, "0");

        cell.title =
            `${String(hour).padStart(2, "0")} UTC`;

        headerHours.appendChild(
            cell
        );
    }

    header.appendChild(
        headerHours
    );

    container.appendChild(
        header
    );

    // Band rows
    for (const band of BANDS) {

        const bandForecast =
            forecasts[band] || {};

        const row =
            document.createElement("div");

        row.className =
            "independent-band-row";

        const title =
            document.createElement("div");

        title.className =
            "independent-band-title";

        title.textContent =
            band;

        row.appendChild(
            title
        );

        const hours =
            document.createElement("div");

        hours.className =
            "independent-band-hours";

        for (let hour = 0; hour < 24; hour++) {

            const prediction =
                bandForecast[String(hour)] ??
                bandForecast[hour];

            const cell =
                document.createElement("div");

            cell.className =
                "independent-band-hour";

            if (hour === currentHour) {
                cell.classList.add(
                    "current-hour"
                );
            }

            if (
                prediction &&
                !prediction.error
            ) {

                const regional =
                    `${prediction.supported_grid_points ?? 0}` +
                    `/${prediction.grid_points ?? 49}`;

                if (prediction.supported) {

                    cell.classList.add(
                        "supported"
                    );

                    cell.title =
                        `${band} ${String(hour).padStart(2, "0")} UTC: ` +
                        `SUPPORTED at Nelson; ` +
                        `${regional} regional grid points`;

                } else {

                    cell.classList.add(
                        "unsupported"
                    );

                    cell.title =
                        `${band} ${String(hour).padStart(2, "0")} UTC: ` +
                        `not supported at Nelson; ` +
                        `${regional} regional grid points`;
                }

            } else {

                cell.classList.add(
                    "unavailable"
                );

                cell.title =
                    `${band} ${String(hour).padStart(2, "0")} UTC: unavailable`;
            }

            cell.textContent =
                String(
                    prediction?.supported_grid_points ?? ""
                );

            hours.appendChild(
                cell
            );
        }

        row.appendChild(
            hours
        );

        container.appendChild(
            row
        );
    }
}


/* ============================================================
   LOCATION
   ============================================================ */

function updateLocation(report) {

    setText(
        "tx-location",
        report.tx_location ||
        report.location_name ||
        "Nelson"
    );

    setText(
        "next-tx-location",
        report.tx_location ||
        report.location_name ||
        "Nelson"
    );

    setText(
        "rx-location",
        report.rx_location ||
        "Australia"
    );

    setText(
        "next-rx-location",
        report.rx_location ||
        "Australia"
    );
}


/* ============================================================
   MAIN REPORT
   ============================================================ */

async function loadDashboard() {

    try {

        const response =
            await fetch(
                "/api/propagation",
                {
                    cache: "no-store"
                }
            );

        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}`
            );
        }

        const responseData =
            await response.json();

        if (
            responseData.status !== "ok" ||
            !responseData.report
        ) {
            throw new Error(
                responseData.error ||
                "Propagation API returned no report"
            );
        }

        const report =
            responseData.report;

        updateBandCards(
            report
        );

        updateCurrentHAP(
            report
        );

        updateNextTransition(
            report
        );

        updateUniversalForecast(
            report
        );

        updateIndependentBandForecast(
            report
        );

        updateLocation(
            report
        );

        updateSpaceWeather(
            report.space_weather
        );

        if (report.generated_local) {
            setText(
                "last-update",
                report.generated_local
            );
        }

    } catch (error) {

        console.error(
            "RadioPathwayTool dashboard error:",
            error
        );

        setText(
            "last-update",
            "Update failed"
        );

        for (const band of BANDS) {

            setText(
                `score-${band}`,
                "--"
            );

            setText(
                `status-${band}`,
                "Unavailable"
            );
        }
    }
}


/* ============================================================
   START
   ============================================================ */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        loadDashboard();

        // Keep the existing 5-minute report cadence.
        setInterval(
            loadDashboard,
            5 * 60 * 1000
        );
    }
);
