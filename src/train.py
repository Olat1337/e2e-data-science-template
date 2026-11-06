"""
Model training and evaluation pipeline.
Generates baseline synthetic tabular data (or loads raw data),
trains an ensemble model, evaluates performance, and exports model artifacts.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def generate_sample_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Generates synthetic environmental / urban metrics data for demonstration."""
    np.random.seed(random_state)

    temperature = np.random.uniform(-5.0, 32.0, n_samples)
    humidity = np.random.uniform(20.0, 95.0, n_samples)
    wind_speed = np.random.uniform(0.5, 18.0, n_samples)
    traffic_index = np.random.uniform(10.0, 100.0, n_samples)
    hour_of_day = np.random.randint(0, 24, n_samples)

    # Synthetic target with realistic domain noise: Air Quality Index (AQI PM2.5)
    # Higher traffic + low wind + low temp = higher pollution
    base_aqi = (
        0.5 * traffic_index
        - 1.8 * wind_speed
        + 0.3 * (30 - temperature)
        + 0.15 * humidity
        + 10.0
    )
    noise = np.random.normal(0, 4.0, n_samples)
    target_aqi = np.clip(base_aqi + noise, 5.0, 180.0)

    df = pd.DataFrame(
        {
            "temperature_c": np.round(temperature, 1),
            "humidity_pct": np.round(humidity, 1),
            "wind_speed_kmh": np.round(wind_speed, 1),
            "traffic_density": np.round(traffic_index, 1),
            "hour_of_day": hour_of_day,
            "target_aqi": np.round(target_aqi, 1),
        }
    )
    return df


def run_pipeline():
    print("[1/4] Loading and preparing data...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "raw" / "dataset.csv"
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_sample_dataset()
        df.to_csv(raw_path, index=False)
        print(f"      Created baseline dataset: {raw_path}")
    else:
        df = pd.read_csv(raw_path)
        print(f"      Loaded existing dataset: {raw_path}")

    feature_cols = [
        "temperature_c",
        "humidity_pct",
        "wind_speed_kmh",
        "traffic_density",
        "hour_of_day",
    ]
    target_col = "target_aqi"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[2/4] Training set size: {len(X_train)} | Test set size: {len(X_test)}")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("[3/4] Evaluating model...")
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"      Mean Absolute Error (MAE) : {mae:.2f}")
    print(f"      Root Mean Squared Error   : {rmse:.2f}")
    print(f"      R-squared (R2 score)      : {r2:.4f}")

    print("[4/4] Serializing model artifact...")
    artifact_path = MODELS_DIR / "model_pipeline.joblib"
    artifact_payload = {
        "pipeline": pipeline,
        "feature_names": feature_cols,
        "metrics": {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)},
    }
    joblib.dump(artifact_payload, artifact_path)
    print(f"      Pipeline artifact saved successfully to: {artifact_path}")


if __name__ == "__main__":
    run_pipeline()
