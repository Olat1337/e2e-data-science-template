"""
Model training and evaluation pipeline.
Generates a baseline synthetic dataset (Laptop Specifications -> Price),
trains an ensemble regression pipeline, evaluates performance, and exports artifacts.
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


def generate_baseline_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Generates synthetic laptop hardware specs and market price data."""
    np.random.seed(random_state)

    ram_gb = np.random.choice([8, 16, 32, 64], size=n_samples, p=[0.35, 0.45, 0.15, 0.05])
    cpu_cores = np.random.choice([4, 6, 8, 12, 16], size=n_samples, p=[0.2, 0.35, 0.3, 0.1, 0.05])
    storage_gb = np.random.choice([256, 512, 1024, 2048], size=n_samples, p=[0.25, 0.45, 0.25, 0.05])
    weight_kg = np.random.uniform(1.1, 2.8, size=n_samples)
    has_dedicated_gpu = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

    # Formula for synthetic price with realistic noise
    base_price = (
        300
        + (ram_gb * 18)
        + (cpu_cores * 45)
        + (storage_gb * 0.35)
        + (has_dedicated_gpu * 450)
        - (weight_kg * 40)
    )
    noise = np.random.normal(0, 50, n_samples)
    price_usd = np.round(np.clip(base_price + noise, 350, 4500), 2)

    return pd.DataFrame(
        {
            "ram_gb": ram_gb,
            "cpu_cores": cpu_cores,
            "storage_gb": storage_gb,
            "weight_kg": np.round(weight_kg, 2),
            "has_dedicated_gpu": has_dedicated_gpu,
            "price_usd": price_usd,
        }
    )


def run_pipeline():
    print("[1/4] Preparing directories and dataset...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "raw" / "dataset.csv"
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_baseline_dataset()
        df.to_csv(raw_path, index=False)
        print(f"      Created baseline dataset: {raw_path}")
    else:
        df = pd.read_csv(raw_path)
        print(f"      Loaded existing dataset: {raw_path}")

    feature_cols = ["ram_gb", "cpu_cores", "storage_gb", "weight_kg", "has_dedicated_gpu"]
    target_col = "price_usd"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[2/4] Training model (Train: {len(X_train)}, Test: {len(X_test)})...")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("[3/4] Evaluating metrics...")
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"      MAE  : ${mae:.2f}")
    print(f"      RMSE : ${rmse:.2f}")
    print(f"      R²   : {r2:.4f}")

    print("[4/4] Serializing model pipeline...")
    artifact_path = MODELS_DIR / "model_pipeline.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": feature_cols,
            "metrics": {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)},
        },
        artifact_path,
    )
    print(f"      Artifact saved to: {artifact_path}")


if __name__ == "__main__":
    run_pipeline()
