---
description: Run the Tabular ML Pipeline (Data Validation -> Training -> Eval)
---

This workflow defines a reproducible pipeline for tabular data, addressing common pain points around data consistency and model reproducibility.

1. **Environment Clean & Setup**
   Ensure we are running in a clean state to avoid leaking global state.
   ```bash
   # Remove any cached pyc files
   find . -name "*.pyc" -delete
   ```

2. **Data Validation**
   Validate input data schema and distributions before training to prevent silent failures.
   // turbo
   ```bash
   python src/validate_data.py --input data/raw/latest.csv --schema config/schema.yaml
   ```

3. **Feature Engineering**
   Generate features and split into train/test sets (80/20) with a fixed seed.
   // turbo
   ```bash
   python src/build_features.py --input data/raw/latest.csv --output data/processed/ --seed 42
   ```

4. **Model Training**
   Train the model using the training set (further split for validation inside the script).
   // turbo
   ```bash
   python src/train.py --input data/processed/train.parquet --config config/model_params.yaml --output models/model_v1.pkl
   ```

5. **Evaluation**
   Run evaluation on the held-out test set.
   // turbo
   ```bash
   python src/evaluate.py --model models/model_v1.pkl --test-set data/processed/test.parquet --output reports/metrics.json
   ```
