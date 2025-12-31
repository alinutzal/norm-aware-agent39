# Norm-Aware Agent: ML Pipelines

This project demonstrates valid, reproducible, and verifiable ML workflows specifically designed for agentic coding environments. It focuses on the **Bank Marketing** dataset (Tabular ML) but includes workflows for Deep Learning and HPO Sweeps.

## Quick Start

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Explore Workflows**
    For detailed step-by-step commands, refer to the workflows in `.agent/workflows/`:
    -   [Tabular ML Pipeline](.agent/workflows/tabular_pipeline.md)
    -   [Deep Learning Pipeline](.agent/workflows/dl_pipeline.md)
    -   [HPO Sweep](.agent/workflows/hpo_sweep.md)

## Tabular ML Pipeline (Bank Marketing)

The tabular pipeline is fully implemented for the UCI Bank Marketing dataset.

### 1. Data Setup
The data is located in `data/raw/`. The pipeline expects `data/raw/latest.csv` (which is a copy of `bank-full.csv`).
If you need to download it again:
```bash
wget -O data/raw/bank.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
unzip data/raw/bank.zip -d data/raw
cp data/raw/bank-full.csv data/raw/latest.csv
```

### 2. Validation
Validate the schema and data integrity.
```bash
python src/validate_data.py --input data/raw/latest.csv --schema config/schema.yaml
```

### 3. Feature Engineering
Process the raw CSV into `train` and `test` Parquet files (80/20 split).
```bash
python src/build_features.py --input data/raw/latest.csv --output data/processed/ --seed 42
```
*Outputs: `data/processed/train.parquet`, `data/processed/test.parquet`*

### 4. Training
Train a **CatBoost** model on the training set. The script internally splits 20% for validation to monitor early stopping.
```bash
python src/train.py --input data/processed/train.parquet --config config/model_params.yaml --output models/model_v1.pkl
```

### 5. Evaluation
Evaluate the model on the held-out test set metrics (AUC, Accuracy, F1).
```bash
python src/evaluate.py --model models/model_v1.pkl --test-set data/processed/test.parquet --output reports/metrics.json
```

## Project Structure
```
.
├── .agent/workflows/   # Doc files describing command sequences
├── config/             # YAML configurations
│   ├── schema.yaml     # Data validation schema
│   └── model_params.yaml # CatBoost hyperparameters
├── data/
│   ├── raw/            # Original CSVs
│   └── processed/      # Parquet files (train/test)
├── src/
│   ├── validate_data.py
│   ├── build_features.py
│   ├── train.py
│   └── evaluate.py
└── requirements.txt
```
