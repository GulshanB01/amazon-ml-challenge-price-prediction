"""Train XGBoost/LightGBM ensemble on prepared Amazon ML Challenge features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.divide(
        np.abs(y_true - y_pred),
        denominator,
        out=np.zeros_like(y_true, dtype=float),
        where=denominator != 0,
    )
    return float(100 * np.mean(ratio))


def train_ensemble(
    features_path: Path,
    target_path: Path,
    sample_ids_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    features = sparse.load_npz(features_path)
    target = np.load(target_path)
    sample_ids = np.load(sample_ids_path)
    log_target = np.log1p(target)

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
        sample_ids,
        test_size=test_size,
        random_state=random_state,
    )

    xgb_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb_model.fit(x_train, y_train_log)

    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    lgb_model.fit(x_train, y_train_log)

    xgb_pred = np.clip(np.expm1(xgb_model.predict(x_valid)), 0.01, None)
    lgb_pred = np.clip(np.expm1(lgb_model.predict(x_valid)), 0.01, None)
    ensemble_pred = 0.5 * xgb_pred + 0.5 * lgb_pred

    metrics = {
        "rows_train": int(x_train.shape[0]),
        "rows_valid": int(x_valid.shape[0]),
        "features": int(features.shape[1]),
        "mae": float(mean_absolute_error(y_valid, ensemble_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, ensemble_pred))),
        "smape": smape(y_valid, ensemble_pred),
        "target_transform": "log1p(price), then expm1 prediction and clip to >= 0.01",
    }

    predictions = pd.DataFrame(
        {
            "sample_id": ids_valid,
            "actual_price": y_valid,
            "xgb_pred": xgb_pred,
            "lgb_pred": lgb_pred,
            "ensemble_pred": ensemble_pred,
            "absolute_error": np.abs(y_valid - ensemble_pred),
        }
    )
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    joblib.dump(xgb_model, output_dir / "xgb_model.joblib")
    joblib.dump(lgb_model, output_dir / "lgb_model.joblib")

    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Amazon ML Challenge ensemble model.")
    parser.add_argument("--features", type=Path, required=True, help="Path to features_sparse.npz")
    parser.add_argument("--target", type=Path, required=True, help="Path to target.npy")
    parser.add_argument("--sample-ids", type=Path, required=True, help="Path to sample_ids.npy")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for model outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ensemble(
        features_path=args.features,
        target_path=args.target,
        sample_ids_path=args.sample_ids,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
