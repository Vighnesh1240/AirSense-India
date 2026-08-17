const API_BASE =
    window.AIRSENSE_API_BASE ||
    "https://airsense-india.onrender.com";

const form = document.getElementById("predictionForm");
const citySelect = document.getElementById("city");
const dateInput = document.getElementById("date");

async function loadCities() {
    try {
        const response = await fetch(`${API_BASE}/cities`);
        if (!response.ok) throw new Error("Unable to load cities.");

        const data = await response.json();

        citySelect.innerHTML = data.cities
            .map(city => `<option value="${escapeHtml(city)}">${escapeHtml(city)}</option>`)
            .join("");

        if (data.cities.includes("Delhi")) {
            citySelect.value = "Delhi";
        }
    } catch (error) {
        citySelect.innerHTML = `<option value="Delhi">Delhi</option>`;
        console.error(error);
    }
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function numberValue(id) {
    return Number(document.getElementById(id).value);
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const button = form.querySelector("button");
    button.disabled = true;
    button.querySelector("span:first-child").textContent = "Predicting...";

    const payload = {
        City: citySelect.value,
        Date: dateInput.value,

        PM2_5: numberValue("PM2_5"),
        PM10: numberValue("PM10"),
        NO: numberValue("NO"),
        NO2: numberValue("NO2"),
        NOx: numberValue("NOx"),
        NH3: numberValue("NH3"),
        CO: numberValue("CO"),
        SO2: numberValue("SO2"),
        O3: numberValue("O3"),
        Benzene: numberValue("Benzene"),
        Toluene: numberValue("Toluene"),
        Xylene: numberValue("Xylene")
    };

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        renderResult(data.prediction);
    } catch (error) {
        alert(error.message);
        console.error(error);
    } finally {
        button.disabled = false;
        button.querySelector("span:first-child").textContent = "Generate AQI prediction";
    }
});

function renderResult(result) {
    document.getElementById("resultEmpty").classList.add("hidden");
    document.getElementById("resultContent").classList.remove("hidden");

    document.getElementById("aqiValue").textContent = result.aqi;
    document.getElementById("aqiCategory").textContent = result.category;
    document.getElementById("riskValue").textContent = result.risk;
    document.getElementById("riskMessage").textContent = result.message;
    document.getElementById("dominantPollutant").textContent =
        result.dominant_pollutant;

    const list = document.getElementById("recommendations");
    list.innerHTML = result.recommendations
        .map(item => `<li>${escapeHtml(item)}</li>`)
        .join("");
}

loadCities();

dateInput.value = new Date().toISOString().split("T")[0];
