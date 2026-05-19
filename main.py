# =============================================================================
#          DATA MINING PROJECT - CAR PRICE ANALYSIS & PREDICTION
#          Subject  : Data Mining
#          Dataset  : car data.csv
#          Tools    : pandas, numpy, matplotlib, scikit-learn
#
#  Project Structure
#  ─────────────────
#  main.py                  ← you are here (orchestrator)
#  data/
#    loader.py              ← Step 1 : load & explore dataset
#    preprocessor.py        ← Step 2 : clean & encode data
#  analysis/
#    similarity.py          ← Step 3 : covariance & Euclidean distance
#  models/
#    classification.py      ← Step 4 : Decision Tree classifier
#    clustering.py          ← Step 5 : K-Means clustering
#  visualization/
#    plots.py               ← Step 6 : generate & save all charts
#  outputs/                 ← saved PNG charts (auto-created)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

# ── Step imports ──────────────────────────────────────────────────────────────
from data.loader          import load_and_explore
from data.preprocessor    import preprocess
from analysis.similarity  import run_similarity_analysis
from models.classification import run_classification
from models.clustering    import run_clustering
from visualization.plots  import generate_all_plots

# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  DATA MINING PROJECT — CAR PRICE ANALYSIS & PREDICTION")
print("=" * 65)

# Step 1 — Load & Explore
df = load_and_explore("car data.csv")

# Step 2 — Preprocess
df = preprocess(df)

# Step 3 — Similarity & Dissimilarity
similarity_results = run_similarity_analysis(df)

# Step 4 — Classification
clf_results = run_classification(df)

# Step 5 — Clustering
cluster_results = run_clustering(df, n_clusters=3)
df = cluster_results["df"]
kmeans = cluster_results["kmeans"]

# Step 6 — Visualisations (saved to outputs/)
generate_all_plots(df, kmeans, output_dir="outputs")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PROJECT SUMMARY")
print("=" * 65)
print(f"  Total Cars in Dataset     : {len(df)}")
print(f"  Features Used             : {len(clf_results['feature_cols'])}")
print(f"  Classification Accuracy   : {clf_results['accuracy'] * 100:.2f}%")
print(f"  KMeans Clusters Formed    : 3")
print(f"  Covariance (SP, PP)       : {similarity_results['covariance']:.4f}  (Positive → Direct Relationship)")
print(f"  Euclidean Distance (0,1)  : {similarity_results['euclidean_distance']:.4f}")
print("─" * 65)
print("  Charts saved to outputs/:")
print("    ✔  cluster_plot.png")
print("    ✔  price_category.png")
print("    ✔  price_scatter.png")
print("    ✔  fuel_distribution.png")
print("=" * 65)
print("  ✅  All tasks completed successfully.")
print("=" * 65)
