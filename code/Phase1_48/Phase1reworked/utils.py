import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    cohen_kappa_score, confusion_matrix
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_specificity(y_true, y_pred, num_classes=5):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificity = []

    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp + 1e-8))

    return float(np.mean(specificity))


def compute_metrics(y_true, y_pred, y_prob, num_classes=5):
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted")
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted")
    metrics["kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["specificity"] = compute_specificity(y_true, y_pred)

    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="weighted"
        )
        metrics["pr_auc"] = average_precision_score(
            y_true, y_prob, average="weighted"
        )
    except ValueError:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics
