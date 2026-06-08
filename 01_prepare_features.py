"""Build clean feature artifacts for the Amazon ML Challenge project.

The original notebooks saved vector columns into CSV files. Pandas shortened
those arrays with ellipses, so this script keeps embeddings in numeric matrix
format and writes model-ready artifacts.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


UNIT_MAPPING = {
    "fl": "fluid_ounce",
    "fluid": "fluid_ounce",
    "fluid ounce": "fluid_ounce",
    "ml": "milliliter",
    "millilitre": "milliliter",
    "mililitro": "milliliter",
    "ltr": "liter",
    "liters": "liter",
    "oz": "ounce",
    "ounces": "ounce",
    "ounce": "ounce",
    "lb": "pound",
    "lbs": "pound",
    "pounds": "pound",
    "gram": "gram",
    "grams": "gram",
    "gramm": "gram",
    "gr": "gram",
    "kg": "kilogram",
    "k": "kilogram",
    "count": "count",
    "ct": "count",
    "each": "count",
    "piece": "count",
    "pack": "pack",
    "packs": "pack",
    "box": "box",
    "boxes": "box",
    "bottle": "bottle",
    "bottles": "bottle",
    "jar": "jar",
    "can": "can",
    "bag": "bag",
    "bucket": "bucket",
    "pouch": "pouch",
    "carton": "carton",
    "sq": "square_unit",
    "foot": "square_unit",
    "cm": "centimeter",
    "in": "inch",
    "case": "case",
    "unit?": "unit",
    "paper": "paper",
    "capsule": "capsule",
}


def parse_product(text: Any) -> dict[str, Any]:
    """Parse Amazon catalog_content into structured fields."""
    text = "" if pd.isna(text) else str(text)
    item_name = re.search(r"Item Name:\s*(.*)", text)
    description = re.search(r"Product Description:\s*(.*)", text)
    bullet_points = re.findall(r"Bullet Point \d+:\s*(.*)", text)
    value = re.search(r"Value:\s*([\d.]+)", text)
    unit = re.search(r"Unit:\s*([^\n\r]+)", text)

    return {
        "item_name": item_name.group(1).strip() if item_name else "",
        "product_description": description.group(1).strip() if description else "",
        "bullet_points": [point.strip() for point in bullet_points],
        "value": float(value.group(1)) if value else np.nan,
        "unit": unit.group(1).strip() if unit else "unknown",
    }


def extract_features(parsed: dict[str, Any]) -> dict[str, Any]:
    bullets = parsed["bullet_points"]
    combined_text = " ".join(
        part
        for part in [
            parsed["item_name"],
            parsed["product_description"],
            " ".join(bullets),
        ]
        if part
    )

    return {
        "item_name_length": len(parsed["item_name"]),
        "num_bullets": len(bullets),
        "avg_bullet_len": float(np.mean([len(point) for point in bullets])) if bullets else 0.0,
        "has_description": int(bool(parsed["product_description"])),
        "value": parsed["value"],
        "unit_norm": normalize_unit(parsed["unit"]),
        "combined_text": combined_text,
    }


def normalize_unit(unit: Any) -> str:
    if pd.isna(unit) or str(unit).strip().lower() in {"none", "", "nan"}:
        return "unknown"
    unit_clean = str(unit).strip().lower()
    return UNIT_MAPPING.get(unit_clean, unit_clean)


def is_numeric_text(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def target_encode_cv(
    df: pd.DataFrame,
    column: str,
    target_col: str = "price",
    smoothing: float = 10.0,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """Leakage-safe target encoding using KFold out-of-fold means."""
    encoded = pd.Series(index=df.index, dtype=float)
    global_mean = np.log1p(df[target_col]).mean()
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, valid_idx in kfold.split(df):
        train = df.iloc[train_idx]
        valid = df.iloc[valid_idx]
        category_mean = train.groupby(column)[target_col].apply(lambda x: np.log1p(x).mean())
        counts = train.groupby(column)[target_col].count()
        smooth = (category_mean * counts + global_mean * smoothing) / (counts + smoothing)
        encoded.iloc[valid_idx] = valid[column].map(smooth).fillna(global_mean)

    return encoded.fillna(global_mean)


def remove_price_outliers(df: pd.DataFrame, column: str = "price") -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 0.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)].copy()


def load_text_vectors(path: Path) -> sparse.csr_matrix:
    with path.open("rb") as file:
        vectors = pickle.load(file)
    if sparse.issparse(vectors):
        return vectors.tocsr()
    return sparse.csr_matrix(np.asarray(vectors))


def load_image_vectors(path: Path) -> dict[int, np.ndarray]:
    image_vectors = np.load(path, allow_pickle=True).item()
    return {int(key): np.asarray(value, dtype=np.float32) for key, value in image_vectors.items()}


def build_feature_artifacts(
    raw_csv: Path,
    text_vectors_path: Path,
    image_vectors_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv)
    required = {"sample_id", "catalog_content", "price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {sorted(missing)}")

    text_vectors = load_text_vectors(text_vectors_path)
    if text_vectors.shape[0] != len(df):
        raise ValueError(
            "text_vec.pkl row count does not match raw CSV rows. "
            f"Got {text_vectors.shape[0]} vectors for {len(df)} rows."
        )

    image_vectors = load_image_vectors(image_vectors_path)
    image_dim = len(next(iter(image_vectors.values())))

    parsed = df["catalog_content"].apply(parse_product)
    feature_df = pd.DataFrame(parsed.apply(extract_features).tolist(), index=df.index)
    modeling_df = pd.concat(
        [df[["sample_id", "price"]].copy(), feature_df],
        axis=1,
    )

    valid_mask = (
        modeling_df["unit_norm"].ne("unknown")
        & ~modeling_df["unit_norm"].apply(is_numeric_text)
        & modeling_df["sample_id"].astype(int).isin(image_vectors)
        & modeling_df["value"].notna()
        & modeling_df["price"].notna()
    )
    modeling_df = modeling_df.loc[valid_mask].reset_index(drop=False).rename(columns={"index": "raw_row"})
    modeling_df = remove_price_outliers(modeling_df, "price").reset_index(drop=True)
    modeling_df["unit_norm_te"] = target_encode_cv(modeling_df, "unit_norm")
    modeling_df["log_value"] = np.log1p(modeling_df["value"])
    modeling_df["value_per_name_char"] = modeling_df["value"] / modeling_df[
        "item_name_length"
    ].clip(lower=1)

    raw_rows = modeling_df["raw_row"].to_numpy()
    selected_text = text_vectors[raw_rows]
    selected_images = np.vstack(
        [image_vectors[int(sample_id)] for sample_id in modeling_df["sample_id"]]
    ).astype(np.float32)

    numeric_cols = [
        "value",
        "item_name_length",
        "num_bullets",
        "avg_bullet_len",
        "has_description",
        "unit_norm_te",
        "log_value",
        "value_per_name_char",
    ]
    scaler = StandardScaler()
    numeric_matrix = scaler.fit_transform(modeling_df[numeric_cols]).astype(np.float32)
    unit_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    unit_matrix = unit_encoder.fit_transform(modeling_df[["unit_norm"]]).astype(np.float32)

    features = sparse.hstack(
        [
            sparse.csr_matrix(numeric_matrix),
            unit_matrix,
            selected_text.astype(np.float32),
            sparse.csr_matrix(selected_images),
        ],
        format="csr",
    )

    sparse.save_npz(output_dir / "features_sparse.npz", features)
    np.save(output_dir / "target.npy", modeling_df["price"].to_numpy(dtype=np.float32))
    np.save(output_dir / "sample_ids.npy", modeling_df["sample_id"].to_numpy(dtype=np.int64))
    modeling_df.drop(columns=["combined_text"]).to_csv(output_dir / "modeling_dataset.csv", index=False)

    feature_names = (
        numeric_cols
        + [f"unit_{unit}" for unit in unit_encoder.categories_[0]]
        + [f"text_tfidf_{idx}" for idx in range(selected_text.shape[1])]
        + [f"image_{idx}" for idx in range(image_dim)]
    )
    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=2)

    summary = {
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "text_features": int(selected_text.shape[1]),
        "image_features": int(image_dim),
        "price_min": float(modeling_df["price"].min()),
        "price_max": float(modeling_df["price"].max()),
    }
    with (output_dir / "build_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Amazon ML Challenge features.")
    parser.add_argument("--raw-csv", type=Path, required=True, help="Path to original train.csv")
    parser.add_argument("--text-vectors", type=Path, required=True, help="Path to text_vec.pkl")
    parser.add_argument("--image-vectors", type=Path, required=True, help="Path to image_vectors.npy")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for clean artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_feature_artifacts(
        raw_csv=args.raw_csv,
        text_vectors_path=args.text_vectors,
        image_vectors_path=args.image_vectors,
        output_dir=args.output_dir,
    )
