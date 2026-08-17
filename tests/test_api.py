import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_cities():
    response = client.get("/cities")
    assert response.status_code == 200
    assert "cities" in response.json()


def test_statistics():
    response = client.get("/statistics")
    assert response.status_code == 200
    assert "average_aqi" in response.json()


@pytest.mark.skipif(
    not os.path.exists("models/aqi_pipeline.pkl"),
    reason="Train model before prediction API test."
)
def test_prediction():
    payload = {
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

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "aqi" in data["prediction"]
    assert "category" in data["prediction"]
