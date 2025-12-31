import argparse
import yaml
import pandas as pd
import pickle
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(input_path, config_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Assuming 'y' is the target. 'y' was mapped to 1/0 in build_features.py
    X = df.drop(columns=['y'])
    y = df['y']
    
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)
    if 'params' in config:
        params = config['params']
    else:
        params = config

    # Split for validation (from the training set)
    # We take 20% of the training data for validation to monitor early stopping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
    
    print("Training CatBoost model...")
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
    
    # Validation metrics
    preds_proba = model.predict_proba(X_val)[:, 1]
    preds_class = model.predict(X_val)
    
    auc = roc_auc_score(y_val, preds_proba)
    acc = accuracy_score(y_val, preds_class)
    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    
    print(f"Saving model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Saving as pickle as requested, though catboost has its own saver.
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
        
    print("Success.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    train(args.input, args.config, args.output)

if __name__ == "__main__":
    main()
