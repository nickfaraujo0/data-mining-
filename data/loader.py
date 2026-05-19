# =============================================================================
#  data/loader.py
#  STEP 1 — Load the dataset and display initial exploration info
# =============================================================================

import pandas as pd
import numpy as np


def load_and_explore(filepath: str) -> pd.DataFrame:
    """
    Load the CSV dataset and print an exploratory summary.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g. 'car data.csv').

    Returns
    -------
    pd.DataFrame
        The raw (unprocessed) dataframe.
    """
    print("\n📂  Loading dataset...")
    df = pd.read_csv(filepath)

    print("\n" + "─" * 65)
    print("  SECTION 1 — DATA EXPLORATION")
    print("─" * 65)

    # First 5 rows
    print("\n▶  First 5 Rows of the Dataset:")
    print(df.head().to_string(index=True))

    # Shape & dtypes
    print("\n▶  Dataset Info:")
    print(f"   Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    df.info()

    # Missing values
    print("\n▶  Missing Values in Each Column:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅  No missing values found. Dataset is clean.")
    else:
        print(missing[missing > 0])

    # Statistical summary (numeric columns only)
    print("\n▶  Statistical Summary:")
    print(df.describe().round(2).to_string())

    return df
