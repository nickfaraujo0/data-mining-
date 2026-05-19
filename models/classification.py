# =============================================================================
#  models/classification.py
#  STEP 4 — Classification using a Decision Tree
#
#  Theory:
#  A Decision Tree splits the data into branches based on feature thresholds,
#  building a tree structure where each leaf node holds a class label.
#  It is easy to interpret and works well for both numeric and categorical data.
# =============================================================================

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def run_classification(df: pd.DataFrame) -> dict:
    """
    Build and evaluate a Decision Tree classifier that predicts whether
    a car is 'Affordable' (Selling_Price ≤ 5 lakhs) or 'Expensive'.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe.

    Returns
    -------
    dict
        {
            'accuracy'   : float,
            'classifier' : trained DecisionTreeClassifier,
            'feature_cols': list[str]
        }
    """
    print("\n" + "─" * 65)
    print("  SECTION 4 — CLASSIFICATION (Decision Tree)")
    print("─" * 65)

    # ── 4a. Create binary target column ──────────────────────────────────────
    df["Price_Category"] = df["Selling_Price"].apply(
        lambda x: "Expensive" if x > 5 else "Affordable"
    )

    print(f"\n▶  Price Category Distribution:")
    print(df["Price_Category"].value_counts().to_string())

    # ── 4b. Feature / target split ────────────────────────────────────────────
    feature_cols = [
        "Present_Price", "Kms_Driven", "Fuel_Type",
        "Seller_Type", "Transmission", "Owner", "Year"
    ]
    X = df[feature_cols]
    y = df["Price_Category"]

    # ── 4c. Train-Test Split (80 % train / 20 % test) ─────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n▶  Train/Test Split:")
    print(f"   Training samples : {X_train.shape[0]}")
    print(f"   Testing  samples : {X_test.shape[0]}")

    # ── 4d. Train the Decision Tree ───────────────────────────────────────────
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    print("\n   ✅  Decision Tree model trained successfully.")

    # ── 4e. Evaluate ──────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n▶  Classification Results:")
    print(f"   Model Accuracy : {acc * 100:.2f}%")
    print("\n▶  Classification Report:")
    print(classification_report(y_test, y_pred))

    return {"accuracy": acc, "classifier": clf, "feature_cols": feature_cols}
