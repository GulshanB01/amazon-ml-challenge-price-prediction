# Amazon ML Challenge - Multimodal Price Prediction

This project is a cleaned and structured machine learning pipeline for the Amazon ML Challenge. It predicts product prices using a combination of product catalog text, product images, quantity/unit information, and ensemble regression models.

## Project Overview

Product prices are predicted from multiple data sources:

- Product catalog text from `catalog_content`
- Product image URLs from `image_link`
- Product quantity/value information
- Product unit information such as ounce, count, gram, ml, kg
- TF-IDF text embeddings
- EfficientNetB0 image embeddings
- XGBoost and LightGBM regression models

The model is evaluated using:

- MAE
- RMSE
- SMAPE

Best validation result from the cleaned full pipeline:

```text
SMAPE: 51.09%
MAE:   7.90
RMSE:  11.72
```

## Why This Project Was Refactored

The original work was spread across multiple notebooks and some vector columns were saved inside CSV files as shortened strings like:

```text
[0. 0. ... 0.]
```

Those values cannot be reliably converted back into real vectors. This cleaned version stores embeddings and model-ready features in proper formats:

- `.pkl` for TF-IDF sparse text vectors
- `.npy` for image embeddings and targets
- `.npz` for sparse model feature matrices
- `.joblib` for trained models

## Project Structure

```text
amazon_ml_challenge_clean/
│
├── 00_generate_embeddings.py      # Generates text and image embeddings
├── 01_prepare_features.py         # Builds clean model-ready features
├── 02_train_ensemble.py           # Trains XGBoost + LightGBM ensemble
├── 03_train_baseline.py           # Fallback model when raw text alignment is unavailable
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Input Files

Recommended input files:

```text
train.csv
text_vec.pkl
image_vectors.npy
processed_dataset.csv
final_dataset1.csv
```

Expected columns in `train.csv`:

```text
sample_id
catalog_content
image_link
price
```

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

## Workflow

Run the files in this order:

```text
00_generate_embeddings.py
        ↓
01_prepare_features.py
        ↓
02_train_ensemble.py
        ↓
metrics.json / validation_predictions.csv
```

## Step 1: Generate Embeddings

This step creates text and image embeddings from the raw dataset.

```bash
python 00_generate_embeddings.py ^
  --raw-csv "C:\Users\user\Data science\train.csv" ^
  --output-dir "C:\Users\user\Data science\submit\clean_artifacts"
```

Outputs:

```text
text_vec.pkl
tfidf_vectorizer.joblib
image_vectors.npy
embedding_summary.json
```

What happens:

- `catalog_content` is converted into TF-IDF text vectors.
- Images are downloaded from `image_link`.
- EfficientNetB0 generates image embeddings.
- Image embeddings are mapped using `sample_id`.

If text embeddings already exist and only image embeddings are needed:

```bash
python 00_generate_embeddings.py ^
  --raw-csv "C:\Users\user\Data science\train.csv" ^
  --output-dir "C:\Users\user\Data science\submit\clean_artifacts" ^
  --skip-text
```

## Step 2: Prepare Features

This step parses product details, cleans units, removes outliers, and combines all features.

```bash
python 01_prepare_features.py ^
  --raw-csv "C:\Users\user\Data science\train.csv" ^
  --text-vectors "C:\Users\user\Data science\submit\text_vec.pkl" ^
  --image-vectors "C:\Users\user\Data science\submit\image_vectors.npy" ^
  --output-dir "C:\Users\user\Data science\submit\clean_artifacts_full"
```

Outputs:

```text
features_sparse.npz
target.npy
sample_ids.npy
feature_names.json
modeling_dataset.csv
build_summary.json
```

Feature engineering includes:

- Parsing item name, description, bullet points, value, and unit
- Unit normalization
- Log-transformed value features
- Unit target encoding
- Unit one-hot encoding
- TF-IDF text features
- EfficientNetB0 image features
- Price outlier removal

## Step 3: Train Ensemble Model

This step trains XGBoost and LightGBM regressors.

```bash
python 02_train_ensemble.py ^
  --features "C:\Users\user\Data science\submit\clean_artifacts_full\features_sparse.npz" ^
  --target "C:\Users\user\Data science\submit\clean_artifacts_full\target.npy" ^
  --sample-ids "C:\Users\user\Data science\submit\clean_artifacts_full\sample_ids.npy" ^
  --output-dir "C:\Users\user\Data science\submit\clean_artifacts_full"
```

Outputs:

```text
metrics.json
validation_predictions.csv
xgb_model.joblib
lgb_model.joblib
```

Training strategy:

- Train on `log1p(price)` instead of raw price
- Convert predictions back using `expm1`
- Clip predictions to positive values
- Average XGBoost and LightGBM predictions

This helps reduce SMAPE and avoids negative price predictions.

## Fallback Baseline

Use this only when the original `train.csv` is not available.

```bash
python 03_train_baseline.py ^
  --final-csv "C:\Users\user\Data science\submit\final_dataset1.csv" ^
  --image-vectors "C:\Users\user\Data science\submit\image_vectors.npy" ^
  --output-dir "C:\Users\user\Data science\submit\clean_artifacts"
```

The fallback model uses scalar features and image embeddings. It does not use text vectors from `final_dataset1.csv` because those vectors were saved in a shortened, non-parseable format.

Fallback result:

```text
SMAPE: 55.84%
```

## Models Used

The project uses:

- TF-IDF Vectorizer for text feature extraction
- EfficientNetB0 for image embeddings
- XGBoost Regressor for price prediction
- LightGBM Regressor for price prediction
- XGBoost + LightGBM ensemble for final prediction
- HistGradientBoostingRegressor for fallback baseline

## Evaluation Metric

The main metric is SMAPE:

```text
SMAPE = 100 * mean(|actual - predicted| / ((|actual| + |predicted|) / 2))
```

Lower SMAPE is better.

SMAPE is useful here because product prices vary across different ranges, and percentage-based error gives a more balanced view than raw error alone.

## Results

Cleaned full pipeline:

```text
Rows:     68,233
Features: 6,335
SMAPE:    51.09%
MAE:      7.90
RMSE:     11.72
```

Fallback baseline:

```text
SMAPE: 55.84%
```

## Key Learnings

This project helped demonstrate:

- End-to-end machine learning pipeline development
- Feature engineering from noisy product catalog text
- Multimodal learning using text and image data
- Sparse matrix handling for large feature sets
- Regression model training and evaluation
- SMAPE-focused model improvement
- Refactoring notebooks into clean Python scripts

## Resume Summary

**Amazon ML Challenge | Multimodal Product Price Prediction**

Built an end-to-end multimodal ML pipeline to predict Amazon product prices using catalog text, product images, and structured quantity/unit features. Generated TF-IDF text embeddings and EfficientNetB0 image embeddings, engineered unit-aware features, and trained a log-target XGBoost-LightGBM ensemble evaluated with SMAPE, MAE, and RMSE.

