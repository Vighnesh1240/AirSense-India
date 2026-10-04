# 🌫️ AirSense India

**AI-powered Air Quality Index (AQI) prediction and analytics for Indian cities.**

AirSense India estimates the AQI from pollutant measurements using a machine-learning regression pipeline, serves predictions through a **FastAPI** backend, and presents them in a clean web frontend with an interactive analytics dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green)
![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7)

---

## 📌 Table of Contents

1. [Features](#-features)
2. [Project Structure](#-project-structure)
3. [Dataset](#-dataset)
4. [Machine Learning Pipeline](#-machine-learning-pipeline)
5. [Model Results](#-model-results)
6. [Getting Started](#-getting-started)
7. [API Reference](#-api-reference)
8. [Frontend](#-frontend)
9. [Testing](#-testing)
10. [Deployment](#-deployment)
11. [Limitations](#-limitations)
12. [Future Work](#-future-work)

---

## ✨ Features

- **AQI prediction** from 12 pollutant concentrations, city, and date
- **AQI category, health-risk level, advisory message, and recommendations** for each prediction (based on the Indian AQI categories)
- **Dominant pollutant indicator** for quick interpretation
- **Analytics dashboard** with city-wise AQI trends, monthly seasonality, and a city ranking table
- **Leak-free ML pipeline** – imputation and encoding are fitted on the training split only
- **Time-based train/test split** (train < 2020, test ≥ 2020) to mimic real forecasting conditions
- **Five models compared** – Linear Regression, Random Forest, Extra Trees, Gradient Boosting, XGBoost
- **REST API** with input validation (Pydantic), CORS support, and automated tests
- **One-click deployment** to Render via `render.yaml`

---

## 📁 Project Structure

```
AirSense-India/
├── app/
│   ├── main.py              # FastAPI application and endpoints
│   └── schemas.py           # Pydantic request/response models
├── src/
│   ├── preprocess.py        # Data cleaning and processed-dataset creation
│   ├── features.py          # Feature engineering + sklearn-compatible transformer
│   ├── train.py             # Model training, comparison, and export
│   ├── predict.py           # Inference wrapper used by the API
│   └── aqi_utils.py         # AQI categories, risk levels, recommendations
├── frontend/
│   ├── index.html           # Prediction page
│   ├── dashboard.html       # Analytics dashboard
│   ├── script.js            # Prediction page logic
│   ├── dashboard.js         # Dashboard logic (Chart.js)
│   └── style.css
├── notebooks/
│   ├── eda.ipynb                    # Exploratory data analysis
│   ├── feature_engineering.ipynb    # Feature experiments
│   └── model_training.ipynb         # Model experiments
├── data/
│   ├── raw/city_day.csv            # Original dataset
│   └── processed/city_day_cleaned.csv
├── models/
│   ├── aqi_pipeline.pkl            # Final trained pipeline (used by the API)
│   ├── aqi_model.pkl
│   └── model_metadata.json         # Model name, metrics, features
├── results/
│   ├── model_comparison.csv
│   ├── city_wise_evaluation.csv
│   └── feature_importance.csv
├── tests/
│   └── test_api.py
├── main.py                  # Entry point (exposes `app` for uvicorn)
├── requirements.txt
├── render.yaml              # Render deployment config
└── runtime.txt / .python-version
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `data/raw/city_day.csv` |
| Granularity | Daily, per city |
| Coverage | 26 Indian cities |
| Period | 1 Jan 2015 – 1 Jul 2020 |
| Raw rows | ~29,500 |
| Target | `AQI` |
| Inputs | `PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene` |

The dataset is the widely used *Air Quality Data in India* collection (CPCB station data aggregated by city and day).

---

## 🧠 Machine Learning Pipeline

```
Raw row (City, Date, 12 pollutants)
        │
        ▼
 AQIFeatureBuilder      → date features, season, ratios, aggregates
        │
        ▼
 ColumnTransformer
   ├─ numeric:      median imputation
   └─ categorical:  most-frequent imputation + One-Hot Encoding (City, Season)
        │
        ▼
 Regressor (best model by RMSE)  →  predicted AQI
```

### Data cleaning (`src/preprocess.py`)
- Parse dates, drop invalid dates and duplicate rows
- Strip whitespace in city names, coerce pollutant columns to numeric
- Negative pollutant values are treated as invalid (set to missing)
- Drop rows with no AQI (no supervised target)
- Remove duplicate `City + Date` records

> Missing-value **imputation is deliberately not done here**. It happens inside the sklearn pipeline and is fitted only on the training split, which avoids data leakage.

### Engineered features (`src/features.py`)

| Group | Features |
|---|---|
| Date | `Year, Month, Day, DayOfWeek, DayOfYear, Quarter, IsWeekend` |
| Season | `Winter` (Dec–Feb), `Summer` (Mar–May), `Monsoon` (Jun–Sep), `PostMonsoon` (Oct–Nov) |
| Pollution ratios | `PM_Ratio = PM2.5 / PM10`, `NOx_NO2_Ratio = NOx / NO2` |
| Aggregates | `Particulate_Load` (PM2.5 + PM10), `Gas_Pollution` (NO, NO2, NOx, NH3, SO2, O3), `Total_Pollution` |

The same `engineer_features()` function is used in both training and inference, so there is no train/serve skew.

### Training setup (`src/train.py`)
- **Split:** train on dates **before 2020-01-01** (20,429 rows), test on **2020-01-01 onward** (4,421 rows)
- **Metrics:** MAE, RMSE, R²
- The best model by RMSE is saved to `models/aqi_pipeline.pkl`, and its metadata to `models/model_metadata.json`

---

## 🏆 Model Results

Evaluated on the held-out 2020 test set:

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|
| **Gradient Boosting** ⭐ | **16.22** | **26.15** | **0.902** |
| Random Forest | 16.14 | 26.28 | 0.901 |
| XGBoost | 17.79 | 27.73 | 0.890 |
| Extra Trees | 18.37 | 31.38 | 0.859 |
| Linear Regression | 23.77 | 39.71 | 0.774 |

**Selected model:** Gradient Boosting (300 estimators, learning rate 0.05, max depth 5).

### Most influential features

| Rank | Feature | Importance |
|---|---|---|
| 1 | PM2.5 | 0.470 |
| 2 | CO | 0.367 |
| 3 | NO | 0.033 |
| 4 | PM10 | 0.021 |
| 5 | Particulate_Load | 0.008 |

PM2.5 and CO together account for roughly **84%** of the model's predictive importance.

### City-wise performance
Per-city MAE is stored in `results/city_wise_evaluation.csv`. The model performs best for **Bengaluru (MAE ≈ 6.3)**, **Thiruvananthapuram (≈ 8.0)** and **Mumbai (≈ 8.1)**, and worst for **Ahmedabad (≈ 45.2)**.

---

## 🚀 Getting Started

### Prerequisites
- Python **3.10**
- `pip`

### 1. Clone and set up

```bash
git clone https://github.com/<your-username>/AirSense-India.git
cd AirSense-India

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. (Optional) Re-train the model

A trained model is already included in `models/`. To rebuild it from the dataset:

```bash
python -m src.train
```

This will regenerate `results/model_comparison.csv`, `models/aqi_pipeline.pkl`, and `models/model_metadata.json`.

### 3. Run the API

```bash
uvicorn main:app --reload
```

- API: http://127.0.0.1:8000
- Interactive docs (Swagger): http://127.0.0.1:8000/docs

### 4. Open the frontend

The frontend points to the hosted API by default. To use your local API, set the base URL before the scripts load (for example, add this line in the `<head>` of both HTML files):

```html
<script>window.AIRSENSE_API_BASE = "http://127.0.0.1:8000";</script>
```

Then serve the folder:

```bash
cd frontend
python -m http.server 5500
```

Visit http://127.0.0.1:5500.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API name, version, status |
| GET | `/health` | Health check and model-loaded flag |
| GET | `/model-info` | Model name, metrics, and expected features |
| GET | `/cities` | List of supported cities |
| POST | `/predict` | Predict AQI from pollutant readings |
| GET | `/statistics` | Dataset-level summary statistics |
| GET | `/trend/{city}?limit=365` | Recent AQI time series for a city (limit 1–5000) |
| GET | `/city-ranking` | Cities ranked by average AQI |
| GET | `/monthly-trend` | Average AQI by calendar month |

### Example: `POST /predict`

**Request**

```json
{
  "City": "Delhi",
  "Date": "2020-06-15",
  "PM2_5": 80,
  "PM10": 120,
  "NO": 20,
  "NO2": 40,
  "NOx": 45,
  "NH3": 15,
  "CO": 1.2,
  "SO2": 15,
  "O3": 60,
  "Benzene": 3,
  "Toluene": 8,
  "Xylene": 2
}
```

> Note: the API field is `PM2_5` (underscore) because `.` is not valid in a field name. It is mapped to `PM2.5` internally.

**Response**

```json
{
  "success": true,
  "prediction": {
    "aqi": 123.4,
    "category": "Moderate",
    "risk": "Moderate",
    "message": "Sensitive individuals may experience discomfort.",
    "dominant_pollutant": "PM10",
    "recommendations": [
      "Sensitive individuals should reduce prolonged outdoor exertion.",
      "Consider limiting exposure during visible pollution episodes."
    ]
  }
}
```

*(Values above are illustrative.)*

**cURL**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"City":"Delhi","Date":"2020-06-15","PM2_5":80,"PM10":120,"NO":20,"NO2":40,"NOx":45,"NH3":15,"CO":1.2,"SO2":15,"O3":60,"Benzene":3,"Toluene":8,"Xylene":2}'
```

### AQI categories used

| AQI | Category | Risk |
|---|---|---|
| 0 – 50 | Good | Low |
| 51 – 100 | Satisfactory | Low to Moderate |
| 101 – 200 | Moderate | Moderate |
| 201 – 300 | Poor | High |
| 301 – 400 | Very Poor | Very High |
| 401+ | Severe | Severe |

---

## 🖥️ Frontend

- **Predict page (`index.html`)** – enter a city, date, and pollutant values to get the predicted AQI, category, risk level, dominant pollutant, and health recommendations.
- **Dashboard (`dashboard.html`)** – KPI cards (average AQI, highest AQI, number of cities, observations), a city-wise AQI trend chart, a monthly AQI chart, and a city ranking table. Charts are built with [Chart.js](https://www.chartjs.org/).

The frontend is plain HTML, CSS, and JavaScript, so there is no build step.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

Tests cover the root, health, cities, statistics, and prediction endpoints. The prediction test is skipped automatically if the trained model file is missing.

---

## ☁️ Deployment

The project is configured for **Render** using `render.yaml`:

```yaml
services:
  - type: web
    name: airsense-india-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

1. Push the repository to GitHub.
2. In Render, create a **New Web Service** (or **Blueprint**) from the repo.
3. Render installs the dependencies and starts the API.
4. Host the `frontend/` folder on any static host (GitHub Pages, Netlify, Vercel) and set `window.AIRSENSE_API_BASE` to your Render URL.

> The free Render tier sleeps when idle, so the first request after inactivity can take a while.

---

## ⚠️ Limitations

- Predictions are **estimates from pollutant readings**, not official AQI values, and not a forecast of future air quality.
- The dataset ends in **July 2020**, so recent pollution patterns are not captured. The 2020 test period was also affected by COVID-19 lockdowns.
- Accuracy varies by city (for example, Ahmedabad has a much higher error than Bengaluru).
- The "dominant pollutant" is a simple concentration-based indicator used for UI explanation, not an official AQI sub-index.
- This tool is for educational and informational purposes and should not replace official CPCB / SAFAR advisories.

---

## 🔮 Future Work

- Add live data from official CPCB or open air-quality APIs
- Forecast AQI for upcoming days using time-series models (lag features, LSTM, Prophet)
- Add weather features such as temperature, humidity, and wind
- Hyperparameter tuning and cross-validation by time
- Model explainability with SHAP
- Containerize with Docker and add CI/CD

---

## 🛠️ Tech Stack

**Backend:** Python, FastAPI, Uvicorn, Pydantic
**ML / Data:** scikit-learn, XGBoost, pandas, NumPy, joblib
**Visualization:** Chart.js, Matplotlib, Seaborn
**Frontend:** HTML, CSS, JavaScript
**Tools:** Jupyter, pytest, Render

---

## 👤 Author

**Vighnesh**
BTech CSE (AI/ML), Jawaharlal Nehru Government Engineering College, Sundernagar

---

## 📄 License

Add a license of your choice (for example, MIT) before publishing. Create a `LICENSE` file in the repo root.
