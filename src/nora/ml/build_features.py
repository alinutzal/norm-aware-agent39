"""Feature engineering and data preprocessing"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def build_features(input_path, output_path, seed=42):
    """Build features from raw data"""
    np.random.seed(seed)

    print(f"Reading data from {input_path}...")
    try:
        df = pd.read_csv(input_path, sep=";")
    except Exception:
        df = pd.read_csv(input_path, sep=",")

    print(f"Initial shape: {df.shape}")

    # 1. Binary encoding
    binary_cols = ["default", "housing", "loan", "y"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})

    # 2. Month Mapping
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    if "month" in df.columns:
        df["month"] = df["month"].map(month_map)

    # 3. One-Hot Encoding
    categorical_cols = ["job", "marital", "education", "contact", "poutcome"]
    # Filter only those present
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 4. Shuffle and Split Data
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["y"] if "y" in df.columns else None,
    )

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Determine output paths
    if output_path.endswith(".parquet"):
        output_dir = os.path.dirname(output_path)
    else:
        output_dir = output_path

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    print(f"Saving train to {train_path}...")
    train.to_parquet(train_path, index=False)

    print(f"Saving test to {test_path}...")
    test.to_parquet(test_path, index=False)

    print("Success.")
    return train_path, test_path
