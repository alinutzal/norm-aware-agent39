import argparse
import pandas as pd
import pickle
import json
import os
import catboost # Required for unpickling
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def evaluate(model_path, test_path, output_path):
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    print(f"Loading test set from {test_path}...")
    df_test = pd.read_parquet(test_path)
    
    X_test = df_test.drop(columns=['y'])
    y_test = df_test['y']
    
    print("Predicting...")
    # Support both sklearn-style and catboost
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback if model doesn't support probability
        y_prob = model.predict(X_test)

    y_pred = model.predict(X_test)
    
    print("Calculating metrics...")
    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "dataset_size": len(df_test)
    }
    
    print(f"Metrics: {metrics}")
    
    print(f"Saving report to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Success.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    evaluate(args.model, args.test_set, args.output)

if __name__ == "__main__":
    main()
