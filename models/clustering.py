# =============================================================================
#  models/clustering.py
#  STEP 5 — Cluster Analysis using K-Means
#
#  Theory:
#  K-Means partitions n observations into k clusters by minimising the
#  within-cluster sum of squared distances to the cluster centroid.
#  Each data point belongs to the cluster with the nearest centroid.
# =============================================================================

import pandas as pd
from sklearn.cluster import KMeans


def run_clustering(df: pd.DataFrame, n_clusters: int = 3) -> dict:
    """
    Apply K-Means clustering on Selling_Price and Present_Price.

    Parameters
    ----------
    df         : pd.DataFrame   Preprocessed dataframe.
    n_clusters : int            Number of clusters (default 3).

    Returns
    -------
    dict
        {
            'kmeans' : fitted KMeans object,
            'df'     : dataframe with 'Cluster' column added
        }
    """
    print("\n" + "─" * 65)
    print("  SECTION 5 — CLUSTER ANALYSIS (K-Means)")
    print("─" * 65)

    # ── 5a. Select features for clustering ───────────────────────────────────
    cluster_data = df[["Selling_Price", "Present_Price"]]

    # ── 5b. Fit K-Means ───────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(cluster_data)

    print(f"\n▶  KMeans Clustering with k={n_clusters} completed.")
    print(f"   Cluster Centers (Selling_Price, Present_Price):")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"   Cluster {i}: Selling = {center[0]:.2f}, Present = {center[1]:.2f}")

    print(f"\n▶  Cars per Cluster:")
    print(df["Cluster"].value_counts().sort_index().to_string())

    return {"kmeans": kmeans, "df": df}
