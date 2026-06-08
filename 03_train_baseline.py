"""Baseline trainer using final_dataset1.csv plus image_vectors.npy.

Use this only when the original train.csv is not available. It avoids the
broken vector strings inside the CSV and reconstructs image features from the
real image embedding file. Text vectors need raw row alignment, so they are not
included in this fallback baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(
        100
        * np.mean(
            np.divide(
                np.abs(y_true - y_pred),
                denominator,
                out=np.zeros_like(y_true, dtype=float),
                where=denominator != 0,
            )
        )
    )


def load_image_vectors(path: Path) -> dict[int, np.ndarray]:
    vectors = np.load(path, allow_pickle=True).item()
    return {int(key): np.asarray(value, dtype=np.float32) for key, value in vectors.items()}


def train_baseline(final_csv: Path, image_vectors_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(final_csv)
    image_vectors = load_image_vectors(image_vectors_path)
    df = df[df["sample_id"].astype(int).isin(image_vectors)].copy()

    numeric = np.column_stack(
        [
            df["value"].to_numpy(dtype=np.float32),
            df["unit_norm_te"].to_numpy(dtype=np.float32),
            np.log1p(df["value"].to_numpy(dtype=np.float32)),
        ]
    )
    images = np.vstack([image_vectors[int(sample_id)] for sample_id in df["sample_id"]])
    features = np.hstack([numeric, images])
    target = df["price"].to_numpy(dtype=np.float32)
    log_target = np.log1p(target)

    scaler = StandardScaler()
    features[:, :3] = scaler.fit_transform(features[:, :3])

    (
        x_train,
        x_valid,
        y_train_log,
        y_valid_log,
        y_train,
        y_valid,
        ids_train,
        ids_valid,
    ) = train_test_split(
        features,
        log_target,
        target,
        df["sample_id"].to_numpy(),
        test_size=0.2,
        random_state=42,
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.01,
        random_state=42,
    )
    model.fit(x_train, y_train_log)
    pred = np.clip(np.expm1(model.predict(x_valid)), 0.01, None)

    metrics = {
        "rows_train": int(len(y_train)),
        "rows_valid": int(len(y_valid)),
        "features": int(features.shape[1]),
        "mae": float(mean_absolute_error(y_valid, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, pred))),
        "smape": smape(y_valid, pred),
        "target_transform": "log1p(price), then expm1 prediction and clip to >= 0.01",
        "note": "Fallback baseline excludes text vectors because final_dataset1.csv contains shortened vector strings.",
    }

    pd.DataFrame(
        {
            "sample_id": ids_valid,
            "actual_price": y_valid,
            "predicted_price": pred,
            "absolute_error": np.abs(y_valid - pred),
        }
    ).to_csv(output_dir / "baseline_validation_predictions.csv", index=False)

    with (output_dir / "baseline_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    joblib.dump({"model": model, "scaler": scaler}, output_dir / "scalar_image_baseline.joblib")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fallback scalar+image baseline.")
    parser.add_argument("--final-csv", type=Path, required=True, help="Path to final_dataset1.csv")
    parser.add_argument("--image-vectors", type=Path, required=True, help="Path to image_vectors.npy")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline(args.final_csv, args.image_vectors, args.output_dir)
