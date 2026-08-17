import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import AQIFeatureBuilder, POLLUTANTS, ENGINEERED_NUMERIC_FEATURES


RAW_PATH = "data/raw/city_day.csv"
PROCESSED_PATH = "data/processed/city_day_cleaned.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

TARGET = "AQI"
TEST_START = "2020-01-01"


def load_training_data() -> pd.DataFrame:
    if os.path.exists(PROCESSED_PATH):
        df = pd.read_csv(PROCESSED_PATH)
    else:
        from src.preprocess import prepare_processed_dataset
        df = prepare_processed_dataset(RAW_PATH, PROCESSED_PATH)

    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_preprocessor() -> ColumnTransformer:
    categorical_features = ["City", "Season"]

    numeric_features = [
        col for col in ENGINEERED_NUMERIC_FEATURES
        if col not in categorical_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])


def get_models():
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

    return models


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
        "predictions": pred
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_training_data()

    train_df = df[df["Date"] < TEST_START].copy()
    test_df = df[df["Date"] >= TEST_START].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Time split produced an empty train/test set. "
            "Check the dataset date range."
        )

    raw_input_columns = ["City", "Date", *POLLUTANTS]

    X_train = train_df[raw_input_columns]
    X_test = test_df[raw_input_columns]
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    models = get_models()
    results = []
    trained = {}

    for name, estimator in models.items():
        print(f"\nTraining: {name}")

        pipeline = Pipeline([
            ("feature_builder", AQIFeatureBuilder()),
            ("preprocessor", build_preprocessor()),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        results.append({
            "Model": name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"]
        })

        trained[name] = pipeline

        print(
            f"MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"R2={metrics['R2']:.4f}"
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(["RMSE", "MAE"])
        .reset_index(drop=True)
    )

    results_df.to_csv(
        os.path.join(RESULTS_DIR, "model_comparison.csv"),
        index=False
    )

    best_name = results_df.iloc[0]["Model"]
    best_pipeline = trained[best_name]

    model_path = os.path.join(MODEL_DIR, "aqi_pipeline.pkl")
    joblib.dump(best_pipeline, model_path)

    metadata = {
        "model_name": best_name,
        "target": TARGET,
        "test_start": TEST_START,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": float(results_df.iloc[0]["MAE"]),
        "rmse": float(results_df.iloc[0]["RMSE"]),
        "r2": float(results_df.iloc[0]["R2"]),
        "features": raw_input_columns
    }

    with open(
        os.path.join(MODEL_DIR, "model_metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
