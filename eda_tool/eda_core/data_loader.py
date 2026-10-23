import pandas as pd
import os

def load_csv(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: input file not found at {filepath}")
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except UnicodeDecodeError:
            raise ValueError("Error: could not decode file with utf-8 or latin-1 encoding")
    if df.shape[0] == 0:
        raise ValueError("Input file contains no data rows")
    return df

def validate_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")
    if df[target].isna().all():
        raise ValueError(f"Target column '{target}' is entirely missing")
