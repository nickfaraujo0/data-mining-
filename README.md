# 🚗 Car Price Analysis & Prediction — Data Mining Project

A beginner-friendly Data Mining project in Python that performs data exploration,
preprocessing, similarity analysis, classification, and clustering on a car dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Sections Explained](#sections-explained)
- [Output](#output)
- [Libraries Used](#libraries-used)

---

## Overview

This project applies core Data Mining techniques to a real-world car dataset to:

| Step | Technique | Goal |
|------|-----------|------|
| 1 | Data Exploration | Understand the dataset shape, types, and statistics |
| 2 | Preprocessing | Handle missing values and encode categorical columns |
| 3 | Similarity Analysis | Compute covariance and Euclidean distance |
| 4 | Classification | Predict if a car is *Affordable* or *Expensive* using a Decision Tree |
| 5 | Clustering | Group cars into price segments using K-Means |
| 6 | Visualisation | Generate and save 4 charts to the `outputs/` folder |

---

## Project Structure

```
DM PROJECT/
├── main.py                    ← Orchestrator — runs all steps in order
├── car data.csv               ← Dataset
├── README.md                  ← You are here
│
├── data/
│   ├── loader.py              ← Step 1: Load & explore the dataset
│   └── preprocessor.py       ← Step 2: Clean & encode data
│
├── analysis/
│   └── similarity.py         ← Step 3: Covariance & Euclidean distance
│
├── models/
│   ├── classification.py     ← Step 4: Decision Tree classifier
│   └── clustering.py         ← Step 5: K-Means clustering
│
├── visualization/
│   └── plots.py              ← Step 6: Generate & save all charts
│
└── outputs/                  ← Auto-created — stores generated PNG charts
    ├── cluster_plot.png
    ├── price_category.png
    ├── price_scatter.png
    └── fuel_distribution.png
```

---

## Dataset

**File:** `car data.csv`

| Column | Description |
|--------|-------------|
| `Car_Name` | Name of the car model |
| `Year` | Manufacturing year |
| `Selling_Price` | Resale price (in Lakhs ₹) |
| `Present_Price` | Showroom / current price (in Lakhs ₹) |
| `Kms_Driven` | Total kilometres driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Seller_Type` | Dealer / Individual |
| `Transmission` | Manual / Automatic |
| `Owner` | Number of previous owners |

---

## Setup & Installation

### 1. Make sure Python 3 is installed
```bash
python3 --version
```

### 2. Create a virtual environment (first time only)
```bash
cd "/Users/nick/Documents/DM PROJECT "
python3 -m venv .venv
```

### 3. Install required libraries (first time only)
```bash
.venv/bin/pip install pandas numpy matplotlib scikit-learn
```

---

## How to Run

```bash
cd "/Users/nick/Documents/DM PROJECT "
.venv/bin/python main.py
```

The script will print results for all 6 sections and save 4 charts to the `outputs/` folder.

---

## Sections Explained

### Section 1 — Data Exploration (`data/loader.py`)
Loads the CSV and prints the first 5 rows, column info, missing value counts, and
statistical summary (mean, std, min/max, quartiles).

### Section 2 — Preprocessing (`data/preprocessor.py`)
- Fills missing **numeric** values with the column **median**
- Fills missing **categorical** values with the column **mode**
- Applies **Label Encoding** to convert text columns (`Fuel_Type`, `Seller_Type`,
  `Transmission`) into numbers so machine learning models can use them

### Section 3 — Similarity & Dissimilarity (`analysis/similarity.py`)
- **Covariance**: measures how `Selling_Price` and `Present_Price` vary together.
  A positive value means both increase together.
- **Euclidean Distance**: straight-line distance between Car-0 and Car-1 in
  price space. Smaller = more similar.

### Section 4 — Classification (`models/classification.py`)
- Creates a binary label: `Affordable` (≤ ₹5L) or `Expensive` (> ₹5L)
- Trains a **Decision Tree** (max depth = 4) on 80% of the data
- Evaluates on the remaining 20% — reports **accuracy** and a full
  **classification report** (precision, recall, F1-score)

### Section 5 — Clustering (`models/clustering.py`)
- Uses **K-Means (k=3)** on `Selling_Price` and `Present_Price`
- Groups cars into: **Economy**, **Mid-Range**, **Premium**
- Prints cluster centres and car counts per cluster

### Section 6 — Visualisations (`visualization/plots.py`)
Generates 4 dark-themed charts saved to `outputs/`:
1. **cluster_plot.png** — Scatter plot of K-Means clusters with centroids
2. **price_category.png** — Bar chart of Affordable vs Expensive counts
3. **price_scatter.png** — Selling Price vs Present Price with trend line
4. **fuel_distribution.png** — Pie chart of fuel type distribution

---

## Output

After a successful run you will see:

```
=================================================================
  PROJECT SUMMARY
=================================================================
  Total Cars in Dataset     : 301
  Features Used             : 7
  Classification Accuracy   : 91.80%
  KMeans Clusters Formed    : 3
  Covariance (SP, PP)       : 38.6193  (Positive → Direct Relationship)
  Euclidean Distance (0,1)  : 4.1908
=================================================================
  ✅  All tasks completed successfully.
=================================================================
```

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualisation |
| `scikit-learn` | Machine learning (Decision Tree, K-Means, LabelEncoder) |
