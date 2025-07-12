import pandas as pd
import numpy as np

def run_eda(df: pd.DataFrame, target_col: str):
    print(f"Data shape: {df.shape}")
    print("\nMissing value summary:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)
    print("\nCorrelation matrix (numerical features):")
    print(df.select_dtypes(include=[np.number]).corr())
    print(f"\nTarget distribution for '{target_col}':")
    print(df[target_col].value_counts() if df[target_col].dtype == 'object' or len(df[target_col].unique()) < 20 else df[target_col].describe()) 