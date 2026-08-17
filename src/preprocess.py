import os
import sys
import numpy as np
import pandas as pd

# Allow direct script execution: python src/preprocess.py
if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import POLLUTANTS, engineer_features

RAW_PATH = "data/raw/city_day.csv"
PROCESSED_PATH = "data/processed/city_day_cleaned_2.csv"


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset-level cleaning used for EDA, reproducibility and saved
    processed data.

    Model-time missing-value imputation is intentionally NOT performed here.
    The final sklearn pipeline fits imputers on the training split only.
    """
    out = df.copy()

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    out = out.drop_duplicates()

    if "City" in out.columns:
        out["City"] = out["City"].astype(str).str.strip()

    for col in POLLUTANTS + ["AQI"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Negative pollutant concentrations are invalid.
    for col in POLLUTANTS:
        if col in out.columns:
            out.loc[out[col] < 0, col] = np.nan

    # AQI is the supervised target. Rows without a target cannot train a regressor.
    out = out.dropna(subset=["AQI"])

    # Sort for reproducibility.
    out = out.sort_values(["City", "Date"]).reset_index(drop=True)

    # Duplicate City-Date observations: retain first after sorting.
    out = out.drop_duplicates(["City", "Date"], keep="first")

    return out


def prepare_processed_dataset(
    input_path: str = RAW_PATH,
    output_path: str = PROCESSED_PATH
) -> pd.DataFrame:
    df = load_raw_data(input_path)
    cleaned = clean_data(df)
    engineered = engineer_features(cleaned)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    engineered.to_csv(output_path, index=False)

    return engineered


if __name__ == "__main__":
    result = prepare_processed_dataset()
    print(f"Processed dataset shape: {result.shape}")
    print(f"Saved to: {PROCESSED_PATH}")
