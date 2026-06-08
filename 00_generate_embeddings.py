"""Generate text and image embeddings for the Amazon ML Challenge pipeline.

Inputs:
    train.csv with at least:
    - sample_id
    - catalog_content
    - image_link

Outputs:
    - text_vec.pkl: TF-IDF sparse matrix for catalog_content
    - tfidf_vectorizer.joblib: fitted TF-IDF vectorizer
    - image_vectors.npy: dictionary mapping sample_id -> image embedding
    - embedding_summary.json: basic run summary
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def generate_text_embeddings(
    df: pd.DataFrame,
    output_dir: Path,
    max_features: int = 5000,
    min_df: int = 2,
    ngram_max: int = 2,
) -> None:
    """Create TF-IDF vectors from product catalog text."""
    text = df["catalog_content"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, ngram_max),
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    text_vectors = vectorizer.fit_transform(text)

    with (output_dir / "text_vec.pkl").open("wb") as file:
        pickle.dump(text_vectors, file)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")


def download_image(url: str, timeout: int = 15) -> Image.Image | None:
    """Download one image URL and return an RGB PIL image."""
    if not isinstance(url, str) or not url.strip():
        return None

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def batched(items: list[tuple[int, str]], batch_size: int) -> Iterable[list[tuple[int, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_image_model():
    """Load EfficientNetB0 feature extractor lazily.

    TensorFlow is imported inside this function so text embedding generation can
    still run in lightweight environments.
    """
    from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    return model, preprocess_input, img_to_array


def fetch_batch(batch: list[tuple[int, str]], workers: int) -> list[tuple[int, Image.Image | None]]:
    results: list[tuple[int, Image.Image | None]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(download_image, image_url): sample_id
            for sample_id, image_url in batch
        }
        for future in as_completed(future_map):
            sample_id = future_map[future]
            try:
                image = future.result()
            except Exception:
                image = None
            results.append((sample_id, image))
    return results


def generate_image_embeddings(
    df: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 32,
    workers: int = 8,
    image_size: int = 224,
) -> None:
    """Download product images and generate EfficientNetB0 embeddings."""
    model, preprocess_input, img_to_array = load_image_model()
    items = list(zip(df["sample_id"].astype(int), df["image_link"].astype(str)))
    image_vectors: dict[int, np.ndarray] = {}

    for batch in tqdm(list(batched(items, batch_size)), desc="Image embedding batches"):
        downloaded = fetch_batch(batch, workers=workers)
        valid_ids: list[int] = []
        arrays: list[np.ndarray] = []

        for sample_id, image in downloaded:
            if image is None:
                continue
            image = image.resize((image_size, image_size))
            arrays.append(img_to_array(image))
            valid_ids.append(sample_id)

        if not arrays:
            continue

        batch_array = preprocess_input(np.asarray(arrays, dtype=np.float32))
        embeddings = model.predict(batch_array, verbose=0)
        for sample_id, embedding in zip(valid_ids, embeddings):
            image_vectors[int(sample_id)] = embedding.astype(np.float32)

    np.save(output_dir / "image_vectors.npy", image_vectors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text and image embeddings.")
    parser.add_argument("--raw-csv", type=Path, required=True, help="Path to train.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for embedding artifacts")
    parser.add_argument("--skip-text", action="store_true", help="Skip TF-IDF text embeddings")
    parser.add_argument("--skip-images", action="store_true", help="Skip image embeddings")
    parser.add_argument("--max-text-features", type=int, default=5000)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--download-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw_csv)
    required = {"sample_id", "catalog_content", "image_link"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {sorted(missing)}")

    if not args.skip_text:
        generate_text_embeddings(
            df,
            args.output_dir,
            max_features=args.max_text_features,
        )

    if not args.skip_images:
        generate_image_embeddings(
            df,
            args.output_dir,
            batch_size=args.image_batch_size,
            workers=args.download_workers,
        )

    summary = {
        "rows": int(len(df)),
        "text_embeddings": not args.skip_text,
        "image_embeddings": not args.skip_images,
        "max_text_features": int(args.max_text_features),
    }
    with (args.output_dir / "embedding_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
