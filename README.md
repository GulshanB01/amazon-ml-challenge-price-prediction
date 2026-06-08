# Amazon ML Challenge - Clean ML Pipeline

This folder contains a cleaned, reusable version of the notebook workflow.

## Project Summary

The project predicts product price from Amazon catalog data using:

- Parsed catalog text fields
- Normalized product quantity/unit features
- Log-transformed quantity features
- One-hot and target-encoded unit features
- TF-IDF text embeddings
- Image embeddings
- Cross-validated target encoding for product units
- XGBoost/LightGBM regression ensemble
- Log-price training with positive prediction clipping for lower SMAPE

## Why This Version Is Cleaner

The original notebooks store `text_vec` and `image_vec` inside CSV files as shortened string representations like `[0. 0. ... 0.]`. Those strings cannot be reliably converted back into vectors. This version keeps embeddings as real arrays/sparse matrices and saves final training data as `.npz` / `.npy` files.

## Recommended Input Files

Place your original files in one folder, for example:

```text
C:\Users\user\Data science\submit
```

Recommended files:

- `train.csv` - original Amazon ML Challenge training data
- `image_vectors.npy` - dictionary mapping `sample_id` to image embedding
- `text_vec.pkl` - TF-IDF sparse matrix created from `catalog_content`
- `processed_dataset.csv` or `final_dataset1.csv` - optional, useful for reference

## Full Workflow

Run the files in this order:

```text
00_generate_embeddings.py  ->  creates text_vec.pkl and image_vectors.npy
01_prepare_features.py     ->  builds clean model-ready matrices
02_train_ensemble.py       ->  trains XGBoost + LightGBM ensemble
03_train_baseline.py       ->  optional fallback if raw text alignment is unavailable
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Generate Embeddings

Start from the original challenge training file:

```bash
python 00_generate_embeddings.py ^
  --raw-csv "C:\Users\user\Data science\submit\train.csv" ^
  --output-dir "C:\Users\user\Data science\submit\clean_artifacts"
```

This creates:

- `text_vec.pkl` - TF-IDF text embedding matrix
- `tfidf_vectorizer.joblib` - fitted text vectorizer
- `image_vectors.npy` - EfficientNetB0 image embeddings mapped by `sample_id`
- `embedding_summary.json`

If you already have text embeddings and only want image embeddings:

```bash
python 00_generate_embeddings.py ^
  --raw-csv "C:\Users\user\Data science\submit\train.csv" ^
  --output-dir "C:\Users\user\Data science\submit\clean_artifacts" ^
  --skip-text
```

## Build Features

Best option, if `train.csv` is available:

```bash
python 01_prepare_features.py ^
  --raw-csv "C:\Users\user\Data science\submit\train.csv" ^
  --text-vectors "C:\Users\user\Data science\submit\clean_artifacts\text_vec.pkl" ^
  --image-vectors "C:\Users\user\Data science\submit\clean_artifacts\image_vectors.npy" ^
  --output-dir "C:\Users\user\Data science\submit\clean_artifacts"
```

This creates:

- `features_sparse.npz`
- `target.npy`
- `sample_ids.npy`
- `feature_names.json`
- `modeling_dataset.csv`

## Train Model

```bash
python 02_train_ensemble.py ^
  --features "C:\Users\user\Data science\submit\clean_artifacts\features_sparse.npz" ^
  --target "C:\Users\user\Data science\submit\clean_artifacts\target.npy" ^
  --sample-ids "C:\Users\user\Data science\submit\clean_artifacts\sample_ids.npy" ^
  --output-dir "C:\Users\user\Data science\submit\clean_artifacts"
```

Outputs:

- `validation_predictions.csv`
- `metrics.json`
- `xgb_model.joblib`
- `lgb_model.joblib`

The trainer is tuned for SMAPE improvement by training on `log1p(price)`,
converting predictions back with `expm1`, and clipping predictions to positive
values. This avoids negative price predictions, which were one reason the
notebook SMAPE stayed around 60%.

## Fallback Baseline

If you only have `final_dataset1.csv` and `image_vectors.npy`, run:

```bash
python 03_train_baseline.py ^
  --final-csv "C:\Users\user\Data science\submit\final_dataset1.csv" ^
  --image-vectors "C:\Users\user\Data science\submit\image_vectors.npy" ^
  --output-dir "C:\Users\user\Data science\submit\clean_artifacts"
```

This baseline uses scalar columns and real image vectors. It skips the CSV text-vector column because that column was saved in a shortened, non-parseable format.

## Resume Description

Amazon ML Challenge | Multimodal Price Prediction

- Built a multimodal ML pipeline using product catalog text, structured attributes, and image embeddings.
- Parsed noisy product descriptions, normalized units, engineered numeric/log features, and applied cross-validated target encoding.
- Combined TF-IDF text vectors, image embeddings, and structured features for regression-based price prediction.
- Trained and evaluated log-target XGBoost/LightGBM ensemble models using MAE, RMSE, and SMAPE.
