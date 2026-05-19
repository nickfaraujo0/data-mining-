# =============================================================================
#  data/preprocessor.py
#  STEP 2 — Handle missing values and encode categorical columns
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the dataframe:
      - Fill missing numeric values with their column median.
      - Fill missing categorical values with their column mode.
      - Label-encode the categorical feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe returned by loader.load_and_explore().

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe ready for analysis and modelling.
    """
    print("\n" + "─" * 65)
    print("  SECTION 2 — DATA PREPROCESSING")
    print("─" * 65)

    # ── 2a. Handle missing values ─────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"   🔧  Filled missing values in '{col}' with median.")

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"   🔧  Filled missing values in '{col}' with mode.")

    print("   ✅  Missing value handling complete.")

    # ── 2b. Label Encoding ────────────────────────────────────────────────────
    #  LabelEncoder converts text labels into numeric values, e.g.:
    #  Petrol → 2,  Diesel → 1,  CNG → 0  (alphabetical order)
    le = LabelEncoder()
    categorical_features = ["Fuel_Type", "Seller_Type", "Transmission"]

    print(f"\n▶  Encoding Categorical Columns: {categorical_features}")
    for col in categorical_features:
        original_classes = df[col].unique()
        df[col] = le.fit_transform(df[col])
        print(f"   [{col}]  Original classes: {original_classes}")

    print("\n   ✅  Label Encoding complete.")
    print("\n▶  Dataset After Preprocessing (first 5 rows):")
    print(df.head().to_string(index=True))

    return df
