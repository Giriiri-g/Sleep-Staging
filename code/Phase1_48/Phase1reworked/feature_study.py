import numpy as np
import pandas as pd
import collections
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# =========================
# CONFIGURATION
# =========================
CSV_PATH = r"csv-docs\mesa-sleep-dataset-0.8.0.csv"
TARGET_COLUMNS = [    'insmnia5',
    'rstlesslgs5',
    'slpapnea5']

MISSING_THRESHOLD = 0.8
MI_TOP_K = 250
CORR_THRESHOLD = 0.9
XGB_TOP_K = 40
STABILITY_FOLDS = 5
STABILITY_MIN_COUNT = 3
ABLATION_TOLERANCE = 0.002

RANDOM_STATE = 42

# =========================
# UTILITY FUNCTIONS
# =========================

from sklearn.impute import SimpleImputer



def drop_bad_rows(X, y, row_missing_threshold=0.5):
    print(f"  → Dropping rows with >{int(row_missing_threshold*100)}% missing values...")

    before = len(X)
    row_missing_ratio = X.isnull().mean(axis=1)
    mask = row_missing_ratio < row_missing_threshold

    X_clean = X.loc[mask].reset_index(drop=True)
    y_clean = y.loc[mask].reset_index(drop=True)

    after = len(X_clean)
    print(f"    Rows: {before} → {after}")

    return X_clean, y_clean



def impute_missing_features(X):
    print("  → Imputing remaining missing values (median)...")

    n_missing = X.isna().sum().sum()
    print(f"    Total missing values before: {n_missing}")

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )

    n_missing_after = X_imputed.isna().sum().sum()
    print(f"    Total missing values after: {n_missing_after}")

    return X_imputed


def drop_nan_targets(X, y, target_name):
    print(f"  → Checking target '{target_name}' for missing values...")

    before = len(y)
    mask = y.notna()

    X_clean = X.loc[mask].reset_index(drop=True)
    y_clean = y.loc[mask].reset_index(drop=True)

    after = len(y_clean)
    dropped = before - after

    print(f"    Dropped {dropped} rows with NaN target")
    print(f"    Samples: {before} → {after}")

    return X_clean, y_clean



def preprocess_non_numeric(X):
    print("  → Handling non-numeric columns...")

    for col in X.columns:
        if X[col].dtype == "object":

            # Try parsing as time (HH:MM:SS)
            try:
                parsed = pd.to_datetime(X[col], format="%H:%M:%S", errors="raise")
                X[col] = (
                    parsed.dt.hour * 3600 +
                    parsed.dt.minute * 60 +
                    parsed.dt.second
                )
                print(f"    Converted time column → seconds: {col}")
                continue
            except:
                pass

            # Try parsing as numeric
            try:
                X[col] = pd.to_numeric(X[col])
                print(f"    Converted string numeric column: {col}")
                continue
            except:
                pass

            # Otherwise: categorical encoding
            X[col] = X[col].astype("category").cat.codes
            print(f"    Label-encoded categorical column: {col}")

    return X


def drop_missing_and_constant(X, missing_threshold=None):
    if missing_threshold is None:
        missing_threshold = MISSING_THRESHOLD
    print("  → Dropping missing-heavy & constant features...")
    before = X.shape[1]

    missing_ratio = X.isnull().mean()
    X = X.loc[:, missing_ratio < missing_threshold]

    vt = VarianceThreshold(threshold=1e-5)
    X_reduced = vt.fit_transform(X)

    kept_columns = X.columns[vt.get_support()]
    after = len(kept_columns)

    print(f"    Features: {before} → {after}")
    return pd.DataFrame(X_reduced, columns=kept_columns)


def mutual_info_filter(X, y, k):
    print(f"  → Mutual Information filtering (top {k})...")
    before = X.shape[1]

    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    idx = np.argsort(mi)[-k:]

    after = len(idx)
    print(f"    Features: {before} → {after}")
    return X.iloc[:, idx]


def correlation_prune(X):
    print(f"  → Correlation pruning (|ρ| > {CORR_THRESHOLD})...")
    before = X.shape[1]

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]

    after = before - len(drop_cols)
    print(f"    Dropped {len(drop_cols)} correlated features")
    print(f"    Features: {before} → {after}")

    return X.drop(columns=drop_cols)


def xgb_importance(X, y, k):
    print(f"  → XGBoost importance ranking (top {k})...")
    before = X.shape[1]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    model.fit(X, y)

    imp = pd.Series(model.feature_importances_, index=X.columns)
    selected = imp.sort_values(ascending=False).head(k).index

    after = len(selected)
    print(f"    Features: {before} → {after}")

    return X[selected]


def l1_logistic_selection(X, y):
    print("  → L1 Logistic Regression (sparse selection)...")
    before = X.shape[1]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.1,
        max_iter=5000
    )
    model.fit(Xs, y)

    selected_idx = np.where(model.coef_[0] != 0)[0]
    after = len(selected_idx)

    print(f"    Features: {before} → {after}")
    return X.iloc[:, selected_idx]


def stability_selection(X, y):
    print(f"  → Stability selection ({STABILITY_FOLDS}-fold CV)...")
    print("    (this may take a while ⏳)")

    skf = StratifiedKFold(
        n_splits=STABILITY_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    counts = collections.Counter()

    for fold, (train, _) in enumerate(skf.split(X, y), 1):
        print(f"    Fold {fold}/{STABILITY_FOLDS}")

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )
        model.fit(X.iloc[train], y.iloc[train])

        top = np.argsort(model.feature_importances_)[-30:]
        counts.update(X.columns[top])

    stable_features = [f for f, c in counts.items() if c >= STABILITY_MIN_COUNT]
    print(f"    Stable features selected: {len(stable_features)}")

    return X[stable_features]


def backward_ablation(X, y):
    print("  → Backward ablation (CV-based)...")
    print("    (slow but worth it 🐢)")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    baseline = cross_val_score(model, X, y, cv=5, scoring="f1").mean()
    print(f"    Baseline CV F1: {baseline:.4f}")

    features = list(X.columns)

    for f in deepcopy(features):
        X_tmp = X.drop(columns=[f])
        score = cross_val_score(model, X_tmp, y, cv=5, scoring="f1").mean()

        if baseline - score < ABLATION_TOLERANCE:
            X = X_tmp
            baseline = score
            print(f"    ✔ Removed '{f}' | New F1: {baseline:.4f}")
        else:
            print(f"    ✘ Kept '{f}'")

    print(f"    Final features after ablation: {X.shape[1]}")
    return X, baseline


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline():
    print("\n📂 Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset shape: {df.shape}")

    results = {}

    for target in TARGET_COLUMNS:
        print("\n" + "=" * 60)
        print(f"🎯 Processing target: {target}")
        print("=" * 60)

        y = df[target]
        X = df.drop(columns=TARGET_COLUMNS)

        X = preprocess_non_numeric(X)
        X, y = drop_nan_targets(X, y, target)

        print(f"Initial feature count: {X.shape[1]}")

        X = drop_missing_and_constant(X, missing_threshold=0.8)

        # Drop bad rows instead of columns
        X, y = drop_bad_rows(X, y, row_missing_threshold=0.5)

        # Now impute safely
        X = impute_missing_features(X)
        X = mutual_info_filter(X, y, MI_TOP_K)

        X = correlation_prune(X)
        X = xgb_importance(X, y, XGB_TOP_K)
        X = l1_logistic_selection(X, y)
        X = stability_selection(X, y)

        X_final, score = backward_ablation(X, y)

        results[target] = {
            "features": list(X_final.columns),
            "cv_f1": score
        }

        print("\n✅ FINAL RESULT")
        print(f"Target: {target}")
        print(f"Selected features ({len(X_final.columns)}):")
        print(X_final.columns.tolist())
        print(f"Final CV F1: {score:.4f}")

        pd.Series(X_final.columns).to_csv(
            f"{target}_final_features.csv", index=False
        )

    print("\n🎉 Pipeline completed successfully.")
    return results


if __name__ == "__main__":
    run_pipeline()
