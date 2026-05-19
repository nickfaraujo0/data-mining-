# =============================================================================
#  analysis/similarity.py
#  STEP 3 — Similarity & Dissimilarity Analysis
#           • Covariance  (measures how two variables change together)
#           • Euclidean Distance  (measures dissimilarity between two records)
# =============================================================================

import numpy as np
import pandas as pd


def run_similarity_analysis(df: pd.DataFrame) -> dict:
    """
    Compute and print:
      1. Covariance between Selling_Price and Present_Price.
      2. Euclidean Distance between the first two records.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe.

    Returns
    -------
    dict
        {
            'covariance' : float,
            'euclidean_distance' : float
        }
    """
    print("\n" + "─" * 65)
    print("  SECTION 3 — SIMILARITY & DISSIMILARITY ANALYSIS")
    print("─" * 65)

    # ── 3a. Covariance ────────────────────────────────────────────────────────
    #  Covariance indicates the direction of the linear relationship between
    #  two variables. A positive value means both rise together; a negative
    #  value means one rises as the other falls.
    cov_matrix = df[["Selling_Price", "Present_Price"]].cov()
    cov_value = cov_matrix.loc["Selling_Price", "Present_Price"]

    print(f"\n▶  Covariance between Selling_Price and Present_Price:")
    print(f"   Covariance Value = {cov_value:.4f}")

    if cov_value > 0:
        print("   📊  Interpretation: POSITIVE covariance → As the present")
        print("       (showroom) price of a car increases, the selling")
        print("       (resale) price also tends to increase. This indicates")
        print("       a direct / positive relationship between the two prices.")
    else:
        print("   📊  Interpretation: NEGATIVE covariance → Inverse relationship.")

    # ── 3b. Euclidean Distance ────────────────────────────────────────────────
    #  Euclidean Distance = sqrt( Σ (xi – yi)² )
    #  It is the straight-line distance between two data points in feature space.
    #  A smaller distance → the two cars are more similar in pricing.
    print(f"\n▶  Euclidean Distance between Car-0 and Car-1:")
    car_0 = df[["Selling_Price", "Present_Price"]].iloc[0].values
    car_1 = df[["Selling_Price", "Present_Price"]].iloc[1].values
    dist = np.sqrt(np.sum((car_0 - car_1) ** 2))

    print(f"   Car-0 → Selling: {car_0[0]}, Present: {car_0[1]}")
    print(f"   Car-1 → Selling: {car_1[0]}, Present: {car_1[1]}")
    print(f"   Euclidean Distance = {dist:.4f}")
    print("   📏  A smaller distance means the two cars are more similar")
    print("       in terms of their prices.")

    return {"covariance": cov_value, "euclidean_distance": dist}
