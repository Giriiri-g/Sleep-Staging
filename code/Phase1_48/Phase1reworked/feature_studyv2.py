import numpy as np
import pandas as pd
import collections

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    average_precision_score
)

from xgboost import XGBClassifier

# ======================================================
# CONFIGURATION (TUNED FOR INSOMNIA & RLS)
# ======================================================
CSV_PATH = r"csv-docs\mesa-sleep-dataset-0.8.0.csv"
TARGET_COLUMNS = ["insmnia5", "rstlesslgs5", "slpapnea5"]

COLUMN_MISSING_THRESHOLD = 0.8   # keep sparse columns
ROW_MISSING_THRESHOLD = 0.5

MI_TOP_K = 250
XGB_TOP_K = 80
STABILITY_FOLDS = 5
STABILITY_MIN_COUNT = 2

RANDOM_STATE = 42

# ======================================================
# PREPROCESSING
# ======================================================
def preprocess_non_numeric(X):
    print("  → Handling non-numeric columns...")
    for col in X.columns:
        if X[col].dtype == "object":
            try:
                t = pd.to_datetime(X[col], format="%H:%M:%S", errors="raise")
                X[col] = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
                continue
            except:
                pass

            try:
                X[col] = pd.to_numeric(X[col])
                continue
            except:
                X[col] = X[col].astype("category").cat.codes
    return X


def drop_nan_targets(X, y, target):
    mask = y.notna()
    print(f"  → Dropped {(~mask).sum()} NaN targets")
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def drop_sparse_columns(X):
    before = X.shape[1]
    keep = X.columns[X.isna().mean() < COLUMN_MISSING_THRESHOLD]
    X = X[keep]
    print(f"  → Columns: {before} → {X.shape[1]}")
    return X


def drop_bad_rows(X, y):
    before = len(X)
    keep = X.isna().mean(axis=1) < ROW_MISSING_THRESHOLD
    X, y = X.loc[keep], y.loc[keep]
    print(f"  → Rows: {before} → {len(X)}")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def impute(X):
    print("  → Median imputation")
    imp = SimpleImputer(strategy="median")
    return pd.DataFrame(imp.fit_transform(X), columns=X.columns)

# ======================================================
# FEATURE SELECTION
# ======================================================
def mutual_info_select(X, y):
    print("  → Mutual Information selection")
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    idx = np.argsort(mi)[-MI_TOP_K:]
    return X.iloc[:, idx]


def xgb_select(X, y):
    print("  → XGBoost importance selection")
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    return X[imp.sort_values(ascending=False).head(XGB_TOP_K).index]


def stability_selection(X, y):
    print("  → Stability selection")
    skf = StratifiedKFold(STABILITY_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    counts = collections.Counter()

    for train, _ in skf.split(X, y):
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )
        model.fit(X.iloc[train], y.iloc[train])
        top = np.argsort(model.feature_importances_)[-30:]
        counts.update(X.columns[top])

    features = [f for f, c in counts.items() if c >= STABILITY_MIN_COUNT]
    print(f"    Stable features: {len(features)}")
    return X[features]

# ======================================================
# FINAL MODEL & THRESHOLD TUNING
# ======================================================
def train_elasticnet(X, y):
    print("  → ElasticNet Logistic Regression + threshold tuning")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.5,
        max_iter=5000,
        class_weight="balanced"
    )

    probs = cross_val_predict(
        model, Xs, y, cv=5, method="predict_proba"
    )[:, 1]

    p, r, t = precision_recall_curve(y, probs)
    f1 = 2 * p * r / (p + r + 1e-9)
    best_idx = np.argmax(f1)

    best_threshold = t[best_idx]
    best_f1 = f1[best_idx]
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

        X = preprocess_non_numeric(X)
        X, y = drop_nan_targets(X, y, target)
        X = drop_sparse_columns(X)
        X, y = drop_bad_rows(X, y)
        X = impute(X)

        X = mutual_info_select(X, y)
        X = xgb_select(X, y)
        X = stability_selection(X, y)

        f1, pr_auc, thresh = train_elasticnet(X, y)

        print(f"\n✅ FINAL PERFORMANCE ({target})")
        print(f"Features used: {X.shape[1]}")
        print(f"Best F1: {f1:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")

        pd.Series(X.columns).to_csv(f"{target}_features_v2.csv", index=False)


if __name__ == "__main__":
    run_pipeline()
