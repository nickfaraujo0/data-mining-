# =============================================================================
#  visualization/plots.py
#  STEP 6 — Generate and save all project charts
#
#  Charts produced:
#    1. KMeans cluster scatter plot       → outputs/cluster_plot.png
#    2. Price category bar chart          → outputs/price_category.png
#    3. Selling vs Present price scatter  → outputs/price_scatter.png
#    4. Fuel type pie chart               → outputs/fuel_distribution.png
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ── Dark GitHub-inspired theme ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
})

CLUSTER_COLORS = ["#58a6ff", "#3fb950", "#f78166"]
CLUSTER_LABELS = [
    "Economy (Cluster 0)",
    "Mid-Range (Cluster 1)",
    "Premium (Cluster 2)",
]


def generate_all_plots(df: pd.DataFrame, kmeans: KMeans, output_dir: str = "outputs") -> None:
    """
    Create and save the four project visualisations.

    Parameters
    ----------
    df         : pd.DataFrame   Dataframe with 'Cluster' and 'Price_Category' columns.
    kmeans     : KMeans         Fitted KMeans model (provides cluster_centers_).
    output_dir : str            Folder where PNGs will be saved (created if missing).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "─" * 65)
    print("  SECTION 6 — VISUALISATIONS")
    print("─" * 65)

    # ── Chart 1: KMeans Cluster Scatter ──────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    fig1.patch.set_facecolor("#0d1117")

    for cluster_id in range(3):
        mask = df["Cluster"] == cluster_id
        ax1.scatter(
            df.loc[mask, "Selling_Price"],
            df.loc[mask, "Present_Price"],
            c=CLUSTER_COLORS[cluster_id],
            label=CLUSTER_LABELS[cluster_id],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            s=65,
        )

    centers = kmeans.cluster_centers_
    ax1.scatter(
        centers[:, 0], centers[:, 1],
        c="yellow", marker="*", s=320,
        zorder=5, label="Cluster Centres", edgecolors="black", linewidths=0.8,
    )

    ax1.set_title("K-Means Clustering — Car Price Segments (k=3)",
                  fontsize=14, fontweight="bold", pad=14, color="#e6edf3")
    ax1.set_xlabel("Selling Price (in Lakhs ₹)", fontsize=11)
    ax1.set_ylabel("Present Price (in Lakhs ₹)", fontsize=11)
    ax1.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#e6edf3", fontsize=9)
    ax1.grid(True)
    fig1.tight_layout()
    path1 = os.path.join(output_dir, "cluster_plot.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
    print(f"   📊  Cluster scatter plot saved → {path1}")

    # ── Chart 2: Price Category Bar Chart ────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.patch.set_facecolor("#0d1117")

    cat_counts = df["Price_Category"].value_counts()
    bar_colors = ["#3fb950", "#f78166"]
    bars = ax2.bar(cat_counts.index, cat_counts.values,
                   color=bar_colors, edgecolor="white", linewidth=0.5, width=0.45)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f"{int(height)}", ha="center", va="bottom", fontsize=11,
                 color="#e6edf3", fontweight="bold")

    ax2.set_title("Price Category Distribution", fontsize=14,
                  fontweight="bold", pad=12, color="#e6edf3")
    ax2.set_xlabel("Category", fontsize=11)
    ax2.set_ylabel("Number of Cars", fontsize=11)
    ax2.grid(axis="y", alpha=0.5)
    fig2.tight_layout()
    path2 = os.path.join(output_dir, "price_category.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    print(f"   📊  Price category bar chart saved → {path2}")

    # ── Chart 3: Selling Price vs Present Price Scatter ───────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    fig3.patch.set_facecolor("#0d1117")

    ax3.scatter(df["Present_Price"], df["Selling_Price"],
                c="#58a6ff", alpha=0.65, edgecolors="white",
                linewidths=0.35, s=55)

    m, b = np.polyfit(df["Present_Price"], df["Selling_Price"], 1)
    x_line = np.linspace(df["Present_Price"].min(), df["Present_Price"].max(), 200)
    ax3.plot(x_line, m * x_line + b, color="#f78166", linewidth=2, label="Trend Line")

    ax3.set_title("Selling Price vs Present Price (with Trend Line)",
                  fontsize=14, fontweight="bold", pad=14, color="#e6edf3")
    ax3.set_xlabel("Present Price (in Lakhs ₹)", fontsize=11)
    ax3.set_ylabel("Selling Price (in Lakhs ₹)", fontsize=11)
    ax3.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#e6edf3", fontsize=9)
    ax3.grid(True)
    fig3.tight_layout()
    path3 = os.path.join(output_dir, "price_scatter.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
    print(f"   📊  Price scatter plot saved → {path3}")

    # ── Chart 4: Fuel Type Pie Chart ──────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(7, 6))
    fig4.patch.set_facecolor("#0d1117")
    ax4.set_facecolor("#161b22")

    fuel_map = {0: "Diesel", 1: "Petrol", 2: "CNG"}
    fuel_counts = df["Fuel_Type"].value_counts()
    fuel_labels = [fuel_map.get(i, f"Type {i}") for i in fuel_counts.index]
    pie_colors = ["#58a6ff", "#3fb950", "#d29922"]

    wedges, texts, autotexts = ax4.pie(
        fuel_counts.values,
        labels=fuel_labels,
        autopct="%1.1f%%",
        colors=pie_colors,
        startangle=140,
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 2},
    )
    for t in texts + autotexts:
        t.set_color("#e6edf3")

    ax4.set_title("Fuel Type Distribution", fontsize=14,
                  fontweight="bold", pad=12, color="#e6edf3")
    fig4.tight_layout()
    path4 = os.path.join(output_dir, "fuel_distribution.png")
    fig4.savefig(path4, dpi=150, bbox_inches="tight", facecolor=fig4.get_facecolor())
    print(f"   📊  Fuel type pie chart saved → {path4}")

    plt.show()
