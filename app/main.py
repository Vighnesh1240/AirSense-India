import json
import os
import sys
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas import AQIRequest
from src.predict import predict_aqi


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "city_day.csv")
PROCESSED_PATH = os.path.join(
    BASE_DIR, "data", "processed", "city_day_cleaned.csv"
)
METADATA_PATH = os.path.join(
    BASE_DIR, "models", "model_metadata.json"
)

app = FastAPI(
    title="AirSense India API",
    description="AI-powered AQI prediction and air-quality analytics API.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@lru_cache(maxsize=1)
def get_data():
    path = PROCESSED_PATH if os.path.exists(PROCESSED_PATH) else DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])


@app.get("/")
def root():
    return {
        "name": "AirSense India API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    model_exists = os.path.exists(
        os.path.join(BASE_DIR, "models", "aqi_pipeline.pkl")
    )

    return {
        "status": "healthy",
        "model_loaded": model_exists
    }


@app.get("/model-info")
def model_info():
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found. Train the model first."
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/cities")
def cities():
    df = get_data()
    return {
        "cities": sorted(df["City"].dropna().unique().tolist())
    }


@app.post("/predict")
def predict(request: AQIRequest):
    try:
        result = predict_aqi(request.model_dump())
        return {
            "success": True,
            "prediction": result
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.get("/statistics")
def statistics():
    df = get_data()

    aqi = pd.to_numeric(df["AQI"], errors="coerce").dropna()

    city_means = (
        df.dropna(subset=["AQI"])
        .groupby("City")["AQI"]
        .mean()
        .sort_values(ascending=False)
    )

    return {
        "rows": int(len(df)),
        "cities": int(df["City"].nunique()),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "average_aqi": round(float(aqi.mean()), 2),
        "highest_aqi": round(float(aqi.max()), 2),
        "lowest_aqi": round(float(aqi.min()), 2),
        "highest_average_city": (
            city_means.index[0] if not city_means.empty else None
        )
    }


@app.get("/trend/{city}")
def city_trend(
    city: str,
    limit: int = Query(default=365, ge=1, le=5000)
):
    df = get_data()

    city_df = (
        df[df["City"].str.lower() == city.lower()]
        .dropna(subset=["AQI"])
        .sort_values("Date")
        .tail(limit)
    )

    if city_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No AQI records found for city: {city}"
        )

    return {
        "city": city,
        "data": [
            {
                "date": row.Date.strftime("%Y-%m-%d"),
                "aqi": round(float(row.AQI), 2)
            }
            for row in city_df.itertuples()
        ]
    }


@app.get("/city-ranking")
def city_ranking():
    df = get_data()

    ranking = (
        df.dropna(subset=["AQI"])
        .groupby("City")["AQI"]
        .agg(["mean", "median", "max", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    return {
        "data": [
            {
                "city": row.City,
                "average_aqi": round(float(row["mean"]), 2),
                "median_aqi": round(float(row["median"]), 2),
                "max_aqi": round(float(row["max"]), 2),
                "observations": int(row["count"])
            }
            for _, row in ranking.iterrows()
        ]
    }


@app.get("/monthly-trend")
def monthly_trend():
    df = get_data().dropna(subset=["AQI"]).copy()

    grouped = (
        df.assign(Month=df["Date"].dt.month)
        .groupby("Month")["AQI"]
        .mean()
        .reset_index()
    )

    return {
        "data": [
            {
                "month": int(row.Month),
                "average_aqi": round(float(row.AQI), 2)
            }
            for row in grouped.itertuples()
        ]
    }
