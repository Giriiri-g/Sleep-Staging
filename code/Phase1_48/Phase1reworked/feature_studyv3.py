import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score
)
from sklearn.utils import resample

# ======================================================
# CONFIGURATION
# ======================================================
CSV_PATH = r"csv-docs\mesa-sleep-dataset-0.8.0.csv"
TARGET_COLUMNS = ["insmnia5", "rstlesslgs5", "slpapnea5"]

ROW_MISSING_THRESHOLD = 0.5
N_BOOTSTRAPS = 100          # stability iterations
STABILITY_THRESHOLD = 0.3   # feature must appear in ≥30% of runs

RANDOM_STATE = 42

# ======================================================
# PREPROCESSING
# ======================================================
def preprocess_non_numeric(X):
    print("  → Handling non-numeric columns...")
    X = X.copy()

    for col in X.columns:
        if X[col].dtype == "object":
            # Try HH:MM:SS
            try:
                t = pd.to_datetime(X[col], format="%H:%M:%S", errors="raise")
                X[col] = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
                continue
            except:
                pass

            # Try numeric
            try:
                X[col] = pd.to_numeric(X[col])
                continue
            except:
                # Categorical
                X[col] = X[col].astype("category").cat.codes

    return X


def drop_nan_targets(X, y):
    mask = y.notna()
    dropped = (~mask).sum()
    print(f"  → Dropped {dropped} NaN targets")
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def drop_bad_rows(X, y):
    before = len(X)
    mask = X.isna().mean(axis=1) < ROW_MISSING_THRESHOLD
    X, y = X.loc[mask], y.loc[mask]
    print(f"  → Rows: {before} → {len(X)}")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def impute_and_scale(X):
    print("  → Dropping all-NaN columns")
    before = X.shape[1]

    # Drop columns with all NaN
    non_empty_cols = X.columns[X.notna().any()]
    X = X[non_empty_cols]

    after = X.shape[1]
    dropped = before - after
    if dropped > 0:
        print(f"    Dropped {dropped} all-NaN columns")

    print("  → Median imputation + scaling")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, X.columns.tolist()



# ======================================================
# STABILITY SELECTION (ElasticNet)
# ======================================================
def stability_selection_elasticnet(X, y, feature_names):
    print("  → Stability selection with ElasticNet")

    counts = Counter()
    n_samples = len(y)

    base_model = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.1, 0.3, 0.5, 0.7],
        Cs=10,
        class_weight="balanced",
        scoring="average_precision",
        max_iter=10000,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    print(f"    Total bootstraps: {N_BOOTSTRAPS}")
    for i in range(N_BOOTSTRAPS):
        # print(f"    Bootstrap {i + 1}/{N_BOOTSTRAPS}")
        Xb, yb = resample(
            X, y,
            n_samples=n_samples,
            stratify=y,
            random_state=RANDOM_STATE + i
        )

        base_model.fit(Xb, yb)

        coef = base_model.coef_.ravel()
        selected = np.array(feature_names)[coef != 0]

        counts.update(selected)

        if (i + 1) % 10 == 0:
            print(f"    Bootstrap {i + 1}/{N_BOOTSTRAPS}")

    min_count = int(STABILITY_THRESHOLD * N_BOOTSTRAPS)
    stable_features = [f for f, c in counts.items() if c >= min_count]

    print(f"    Stable features selected: {len(stable_features)}")

    return stable_features


# ======================================================
# FINAL MODEL + THRESHOLD TUNING
# ======================================================
def train_and_evaluate(X, y):
    model = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.1, 0.3, 0.5, 0.7],
        Cs=10,
        class_weight="balanced",
        scoring="average_precision",
        max_iter=10000,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    probs = cross_val_predict(
        model, X, y, cv=5, method="predict_proba"
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    best_idx = np.argmax(f1)
    best_f1 = f1[best_idx]
    best_threshold = thresholds[best_idx]
    pr_auc = average_precision_score(y, probs)

    print(f"    Best F1: {best_f1:.4f}")
    print(f"    PR-AUC: {pr_auc:.4f}")
    print(f"    Optimal threshold: {best_threshold:.3f}")

    return best_f1, pr_auc, best_threshold


# ======================================================
# MAIN PIPELINE
# ======================================================
def run_pipeline():
    df = pd.read_csv(CSV_PATH)

    for target in TARGET_COLUMNS:
        print("\n" + "=" * 60)
        print(f"🎯 Target: {target}")
        print("=" * 60)

        y = df[target]
        X = df.drop(columns=TARGET_COLUMNS)

        # Preprocessing
        X = preprocess_non_numeric(X)
        X, y = drop_nan_targets(X, y)
        X, y = drop_bad_rows(X, y)

        X_scaled, feature_names = impute_and_scale(X)


        # Stability selection
        stable_features = stability_selection_elasticnet(
            X_scaled, y, feature_names
        )

        # Final evaluation
        X_final = X_scaled[:, [feature_names.index(f) for f in stable_features]]
        f1, pr_auc, threshold = train_and_evaluate(X_final, y)

        print(f"\n✅ FINAL RESULT ({target})")
        print(f"Stable features: {len(stable_features)}")
        print(f"Best F1: {f1:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")

        pd.Series(stable_features).to_csv(
            f"{target}_stable_features.csv", index=False
        )


if __name__ == "__main__":
    run_pipeline()
