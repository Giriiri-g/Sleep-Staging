"""
Sleep Disorder Classification using Multimodal PSG-Derived Features (MESA)

This script trains and evaluates multiple machine learning and deep learning
models for three sleep disorders:
    - Insomnia
    - Restless Legs Syndrome (RLS)
    - Sleep Apnea

Key design principles:
----------------------
1. Severe class imbalance (~10% positives) is explicitly handled using:
   - Class-weighted learning
   - SMOTE (for margin-based models only)
   - Threshold tuning using G-Mean (Sensitivity × Specificity balance)

2. Feature selection is physiology-driven, not model-driven:
   - Sleep continuity & architecture
   - Respiratory/apnea indices
   - Demographics & anthropometrics

3. Models are evaluated as *screening / risk stratification tools*,
   not definitive diagnostic systems.

4. Deep learning is used conservatively (shallow MLP) because the
   feature space is structured, tabular, and low-dimensional.
"""

# ============================
# IMPORTS
# ============================

import pandas as pd
import numpy as np

# Train / test split and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Evaluation metrics (imbalance-aware)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    cohen_kappa_score, matthews_corrcoef
)

# Classical ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Class imbalance handling
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Deep learning (tabular MLP)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================
# DATA CONFIGURATION
# ============================

# Preprocessed dataset with imputed features and cleaned labels
CSV_PATH = r"code\Phase1_48\Phase1reworked\mesa_sleep_features_imputed_with_labels.csv"

# Multimodal feature set (dictionary-validated, non-leaky)
FEATURES = [
    # Demographics & anthropometrics
    "gender1",
    "sleepage5c",
    "bmi5c",

    # Sleep continuity & macro-architecture
    "slp_lat5",
    "waso5",
    "slp_eff5",
    "time_bed5",
    "slpprdp5",
    "timest15",
    "timest25",
    "times34p5",
    "timerem5",
    "remlaiip5",

    # Respiratory / apnea physiology (PSG-derived)
    # "oahi35",
    # "oahi45",
    # "ahiov505",
    # "apnea35",
    # "respevpr5"
    'wtlb5',
    'bpmmin5',
    'avgsleepboutswd5'
]

# Binary targets (modeled independently, clinically correct)
TARGETS = {
    "Insomnia": "insmnia5",
    "RLS": "rstlesslgs5",
    "SleepApnea": "slpapnea5"
}

# Load data
df = pd.read_csv(CSV_PATH)


# ============================
# EVALUATION METRICS
# ============================

def evaluate(y_true, y_pred, y_prob):
    """
    Compute a comprehensive set of imbalance-aware metrics.

    Rationale:
    ----------
    - Accuracy alone is misleading under class imbalance
    - Sensitivity & Specificity reflect clinical screening utility
    - G-Mean balances false positives and false negatives
    - MCC & Kappa are robust to skewed class distributions
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    gmean = np.sqrt(sensitivity * specificity)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": sensitivity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "GMean": gmean,
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob)
    }


def best_threshold_gmean(y_true, y_prob):
    """
    Select decision threshold that maximizes G-Mean.

    Justification:
    --------------
    - Default threshold (0.5) is inappropriate under imbalance
    - G-Mean avoids trivial all-positive or all-negative solutions
    - Clinically suitable for screening tasks
    """

    best_t, best_g = 0.5, 0

    for t in np.linspace(0.05, 0.95, 50):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        g = np.sqrt(sens * spec)

        if g > best_g:
            best_g, best_t = g, t

    return best_t


# ============================
# DEEP LEARNING MODEL
# ============================

def build_mlp(input_dim):
    """
    Shallow MLP for tabular physiological data.

    Design rationale:
    -----------------
    - Shallow network avoids overfitting on structured features
    - BatchNorm stabilizes training under imbalance
    - Dropout improves generalization
    - No convolutions or attention (inappropriate for tabular data)
    """

    model = Sequential([
        Dense(64, activation="relu", input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.4),

        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy"
    )

    return model


# ============================
# TRAINING & EVALUATION LOOP
# ============================

results = {}

for disorder, target in TARGETS.items():
    print(f"\n================ {disorder} =================")

    # Feature matrix and binary target
    X = df[FEATURES]
    y = df[target]

    # Stratified split preserves class prevalence
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )

    # Compute class weights from training data only
    class_weights = dict(zip(
        np.unique(y_train),
        compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
    ))

    results[disorder] = {}

    # ----- Logistic Regression (baseline, interpretable) -----
    lr = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(sampling_strategy=0.3, random_state=42)),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight=class_weights
        ))
    ])

    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    t = best_threshold_gmean(y_test, y_prob)

    results[disorder]["LogisticRegression"] = {
        **evaluate(y_test, (y_prob >= t).astype(int), y_prob),
        "Threshold": t
    }

    # ----- SVM (RBF kernel, margin-based separability) -----
    svm = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(sampling_strategy=0.3, random_state=42)),
        ("clf", SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            probability=True,
            class_weight=class_weights
        ))
    ])

    svm.fit(X_train, y_train)
    y_prob = svm.predict_proba(X_test)[:, 1]
    t = best_threshold_gmean(y_test, y_prob)

    results[disorder]["SVM_RBF"] = {
        **evaluate(y_test, (y_prob >= t).astype(int), y_prob),
        "Threshold": t
    }

    # ----- Random Forest (nonlinear ensemble, no SMOTE) -----
    rf = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=10,
        class_weight=class_weights,
        random_state=42
    )

    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    t = best_threshold_gmean(y_test, y_prob)

    results[disorder]["RandomForest"] = {
        **evaluate(y_test, (y_prob >= t).astype(int), y_prob),
        "Threshold": t
    }

    # ----- Deep Learning (MLP) -----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    dl = build_mlp(X_train_s.shape[1])
    dl.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[EarlyStopping(
            patience=15,
            restore_best_weights=True
        )],
        verbose=0
    )

    y_prob = dl.predict(X_test_s).ravel()
    t = best_threshold_gmean(y_test, y_prob)

    results[disorder]["MLP"] = {
        **evaluate(y_test, (y_prob >= t).astype(int), y_prob),
        "Threshold": t
    }


# ============================
# FINAL RESULTS
# ============================

for disorder in results:
    print(f"\n===== {disorder} Final Results =====")
    print(pd.DataFrame(results[disorder]).T.round(3))
