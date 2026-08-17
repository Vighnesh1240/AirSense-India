import os
import sys
from typing import Dict

import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.aqi_utils import (
    find_dominant_pollutant,
    get_aqi_info,
    get_recommendations
)


MODEL_PATH = "models/aqi_pipeline.pkl"


class AQIPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run `python -m src.train` first."
            )
        self.pipeline = joblib.load(model_path)

    def predict(self, data: Dict) -> Dict:
        row = dict(data)

        # API uses PM2_5; model dataset uses PM2.5.
        if "PM2_5" in row:
            row["PM2.5"] = row.pop("PM2_5")

        df = pd.DataFrame([row])

        prediction = float(self.pipeline.predict(df)[0])
        prediction = max(0.0, prediction)

        info = get_aqi_info(prediction)

        pollutant_values = {
            key: row.get(key)
            for key in [
                "PM2.5", "PM10", "NO", "NO2", "NOx",
                "NH3", "CO", "SO2", "O3",
                "Benzene", "Toluene", "Xylene"
            ]
        }

        dominant = find_dominant_pollutant(pollutant_values)

        return {
            "aqi": round(prediction, 2),
            "category": info["category"],
            "risk": info["risk"],
            "message": info["message"],
            "dominant_pollutant": dominant,
            "recommendations": get_recommendations(prediction)
        }


_predictor = None


def get_predictor() -> AQIPredictor:
    global _predictor

    if _predictor is None:
        _predictor = AQIPredictor()

    return _predictor


def predict_aqi(data: Dict) -> Dict:
    return get_predictor().predict(data)
