import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

CATEGORICAL_FEATURES = ["City", "Season"]

ENGINEERED_NUMERIC_FEATURES = [
    *POLLUTANTS,
    "Year", "Month", "Day", "DayOfWeek", "DayOfYear",
    "Quarter", "IsWeekend",
    "PM_Ratio", "NOx_NO2_Ratio",
    "Particulate_Load", "Gas_Pollution", "Total_Pollution"
]

MODEL_FEATURES = [
    "City",
    *ENGINEERED_NUMERIC_FEATURES,
    "Season"
]


def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "PostMonsoon"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the exact features used by both training and inference."""
    out = df.copy()

    if "Date" not in out.columns:
        raise ValueError("Input must contain a Date column.")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    if out["Date"].isna().any():
        raise ValueError("One or more Date values are invalid.")

    # Ensure expected pollutant columns exist.
    for col in POLLUTANTS:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Date-derived features
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Day"] = out["Date"].dt.day
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["DayOfYear"] = out["Date"].dt.dayofyear
    out["Quarter"] = out["Date"].dt.quarter
    out["IsWeekend"] = (out["DayOfWeek"] >= 5).astype(int)
    out["Season"] = out["Month"].apply(get_season)

    # Pollution-derived features.
    out["PM_Ratio"] = out["PM2.5"] / (out["PM10"] + 1e-6)
    out["NOx_NO2_Ratio"] = out["NOx"] / (out["NO2"] + 1e-6)
    out["Particulate_Load"] = out["PM2.5"] + out["PM10"]

    gas_cols = ["NO", "NO2", "NOx", "NH3", "SO2", "O3"]
    out["Gas_Pollution"] = out[gas_cols].sum(axis=1, min_count=1)
    out["Total_Pollution"] = out[POLLUTANTS].sum(axis=1, min_count=1)

    out = out.replace([np.inf, -np.inf], np.nan)

    return out


class AQIFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer.
    It converts raw rows containing City, Date and pollutants
    into the exact engineered features used by the model.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        out = engineer_features(X)

        if "City" not in out.columns:
            raise ValueError("Input must contain City.")

        return out[MODEL_FEATURES]
