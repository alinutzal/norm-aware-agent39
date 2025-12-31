"""Data validation and quality checks"""

import pandas as pd
import yaml
import sys
import os


def load_schema(schema_path):
    """Load data schema from YAML"""
    with open(schema_path, "r") as f:
        return yaml.safe_load(f)


def validate_data(df, schema):
    """Validate dataframe against schema"""
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
                messages.append(
                    f"ERROR: Column '{col_name}' expected int64, got {series.dtype}"
                )
                valid = False

        # Check allowed values (categorical)
        if "allowed_values" in col_spec:
            invalid_values = series[
                ~series.isin(col_spec["allowed_values"])
            ].unique()
            if len(invalid_values) > 0:
                messages.append(
                    f"ERROR: Column '{col_name}' contains invalid values: {invalid_values[:5]}..."
                )
                valid = False

        # Check ranges
        if "min" in col_spec:
            if series.min() < col_spec["min"]:
                messages.append(
                    f"ERROR: Column '{col_name}' has values < {col_spec['min']}"
                )
                valid = False
        if "max" in col_spec:
            if series.max() > col_spec["max"]:
                messages.append(
                    f"ERROR: Column '{col_name}' has values > {col_spec['max']}"
                )
                valid = False

    return valid, messages


def validate_data_file(input_path, schema_path):
    """Validate data file against schema"""
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False

    if not os.path.exists(schema_path):
        print(f"Error: Schema file not found: {schema_path}")
        return False

    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path, sep=";")
    except Exception:
        try:
            df = pd.read_csv(input_path, sep=",")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False

    print(f"Loading schema from {schema_path}...")
    schema = load_schema(schema_path)

    print("Validating data...")
    valid, messages = validate_data(df, schema)

    if valid:
        print("SUCCESS: Data validation passed.")
        return True
    else:
        print("FAILURE: Data validation failed.")
        for msg in messages:
            print(msg)
        return False
