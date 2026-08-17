const API_BASE =
    window.AIRSENSE_API_BASE ||
    "https://airsense-india.onrender.com";

let trendChart = null;
let monthlyChart = null;

async function getJSON(path) {
    const response = await fetch(`${API_BASE}${path}`);
    if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
    }
    return response.json();
}

async function initDashboard() {
    try {
        const [stats, cities, ranking, monthly] = await Promise.all([
            getJSON("/statistics"),
            getJSON("/cities"),
            getJSON("/city-ranking"),
            getJSON("/monthly-trend")
        ]);

        document.getElementById("avgAqi").textContent = stats.average_aqi;
        document.getElementById("highAqi").textContent = stats.highest_aqi;
        document.getElementById("cityCount").textContent = stats.cities;
        document.getElementById("rowCount").textContent =
            stats.rows.toLocaleString();

        const select = document.getElementById("trendCity");
        select.innerHTML = cities.cities
            .map(city => `<option value="${escapeHtml(city)}">${escapeHtml(city)}</option>`)
            .join("");

        if (cities.cities.includes("Delhi")) {
            select.value = "Delhi";
        }

        select.addEventListener("change", () => loadCityTrend(select.value));

        renderMonthly(monthly.data);
        renderRanking(ranking.data);

        await loadCityTrend(select.value);
    } catch (error) {
        console.error(error);
        alert("Unable to load dashboard data. Make sure the API is running.");
    }
}

async function loadCityTrend(city) {
    try {
        const result = await getJSON(
            `/trend/${encodeURIComponent(city)}?limit=5000`
        );

        document.getElementById("trendTitle").textContent =
            `${city} AQI trend`;

        renderTrend(result.data);
    } catch (error) {
        console.error(error);
    }
}

function renderTrend(data) {
    const ctx = document.getElementById("trendChart");

    if (trendChart) trendChart.destroy();

    trendChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.map(x => x.date),
            datasets: [{
                label: "AQI",
                data: data.map(x => x.aqi),
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: "index"
            },
            plugins: {
                legend: {display: false}
            },
            scales: {
                x: {
                    ticks: {maxTicksLimit: 10}
                }
            }
        }
    });
}

function renderMonthly(data) {
    const ctx = document.getElementById("monthlyChart");

    if (monthlyChart) monthlyChart.destroy();

    monthlyChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: data.map(x => `Month ${x.month}`),
            datasets: [{
                label: "Average AQI",
                data: data.map(x => x.average_aqi),
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {display: false}
            }
        }
    });
}

function renderRanking(data) {
    const body = document.getElementById("rankingBody");

    body.innerHTML = data.map((row, index) => `
        <tr>
            <td>${index + 1}</td>
            <td><strong>${escapeHtml(row.city)}</strong></td>
            <td>${row.average_aqi}</td>
            <td>${row.median_aqi}</td>
            <td>${row.max_aqi}</td>
            <td>${row.observations.toLocaleString()}</td>
        </tr>
    `).join("");
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

initDashboard();
