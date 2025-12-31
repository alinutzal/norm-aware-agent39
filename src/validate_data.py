import argparse
import pandas as pd
import yaml
import sys
import os

def load_schema(schema_path):
    with open(schema_path, "r") as f:
        return yaml.safe_load(f)

def validate_data(df, schema):
    valid = True
    messages = []

    # 1. Check for missing columns
    expected_columns = {col["name"] for col in schema["columns"]}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        messages.append(f"ERROR: Missing columns: {missing_columns}")
        valid = False

    # 2. Check individual columns
    for col_spec in schema["columns"]:
        col_name = col_spec["name"]
        if col_name not in df.columns:
            continue
        
        series = df[col_name]
        
        # Check dtype (rough check)
        expected_dtype = col_spec.get("dtype")
        if expected_dtype == "int64":
            if not pd.api.types.is_integer_dtype(series):
                 messages.append(f"ERROR: Column '{col_name}' expected int64, got {series.dtype}")
                 valid = False
        
        # Check allowed values (categorical)
        if "allowed_values" in col_spec:
            invalid_values = series[~series.isin(col_spec["allowed_values"])].unique()
            if len(invalid_values) > 0:
                messages.append(f"ERROR: Column '{col_name}' contains invalid values: {invalid_values[:5]}...")
                valid = False
        
        # Check ranges
        if "min" in col_spec:
            if series.min() < col_spec["min"]:
                messages.append(f"ERROR: Column '{col_name}' has values < {col_spec['min']}")
                valid = False
        if "max" in col_spec:
            if series.max() > col_spec["max"]:
                messages.append(f"ERROR: Column '{col_name}' has values > {col_spec['max']}")
                valid = False

    return valid, messages

def main():
    parser = argparse.ArgumentParser(description="Validate Bank Marketing Dataset")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--schema", required=True, help="Path to schema YAML file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
        
    if not os.path.exists(args.schema):
        print(f"Error: Schema file not found: {args.schema}")
        sys.exit(1)

    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input, sep=";") # bank-marketing usually uses semi-colon
    except Exception as e:
        # Fallback to comma if semicolon fails or just fail
        try:
             df = pd.read_csv(args.input, sep=",")
        except:
             print(f"Error reading CSV: {e}")
             sys.exit(1)

    print(f"Loading schema from {args.schema}...")
    schema = load_schema(args.schema)
    
    print("Validating data...")
    valid, messages = validate_data(df, schema)
    
    if valid:
        print("SUCCESS: Data validation passed.")
        sys.exit(0)
    else:
        print("FAILURE: Data validation failed.")
        for msg in messages:
            print(msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
