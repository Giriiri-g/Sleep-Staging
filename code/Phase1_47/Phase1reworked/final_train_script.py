import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from torchinfo import summary

from enas_binary_classification import SharedModel, OP_NAMES
from cfs_dataset import create_binary_dataloaders

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "C:\Sleep-Staging\csv-docs\cfs_visit5_selected.csv"
DISEASE_COLUMN = "diadiag"

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
PATIENCE = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PREFIX = "diadiag_post_enas"

# ============================================================
# BEST DAG (FROM ENAS)
# ============================================================
BEST_DAG = {
    0: {"edges": [1, 0], "ops": ["conv_5", "sep_conv_3"]},
    1: {"edges": [1, 2], "ops": ["avg_pool", "dil_conv_3"]},
    2: {"edges": [1, 1], "ops": ["sep_conv_7", "avg_pool"]},
    3: {"edges": [3, 2], "ops": ["sep_conv_3", "sep_conv_5"]},
    4: {"edges": [1, 4], "ops": ["sep_conv_3", "sep_conv_5"]},
    5: {"edges": [6, 1], "ops": ["sep_conv_5", "avg_pool"]},
    6: {"edges": [2, 3], "ops": ["dil_conv_5", "sep_conv_3"]},
    7: {"edges": [1, 1], "ops": ["dil_conv_5", "sep_conv_7"]},
    8: {"edges": [5, 7], "ops": ["sep_conv_5", "avg_pool"]},
    9: {"edges": [7, 5], "ops": ["max_pool", "avg_pool"]},
}


OP_NAME_TO_IDX = {name: i for i, name in enumerate(OP_NAMES)}

def normalize_dag(dag):
    out = {}
    for k, v in dag.items():
        out[k] = {
            "edges": v["edges"],
            "ops": [OP_NAME_TO_IDX[o] for o in v["ops"]],
        }
    return out

BEST_DAG = normalize_dag(BEST_DAG)

# ============================================================
# DATA
# ============================================================
train_loader, val_loader, test_loader, stats = create_binary_dataloaders(
    csv_path=DATA_PATH,
    disease_column=DISEASE_COLUMN,
    batch_size=BATCH_SIZE,
    input_channels=7,
    input_length=3000,
    val_split=0.15,
    test_split=0.15,
    seed=42,
    normalization="zscore",
)

# ============================================================
# POS WEIGHT
# ============================================================
pos, neg = 0, 0
for _, y in train_loader:
    y = y.view(-1)
    pos += (y == 1).sum().item()
    neg += (y == 0).sum().item()

pos_weight = torch.tensor(neg / pos).to(DEVICE)
print(f"[INFO] pos_weight = {pos_weight.item():.3f}")

# ============================================================
# MODEL
# ============================================================
model = SharedModel(
    input_channels=7,
    input_length=3000,
    hidden_dim=48,
    num_nodes=10,
).to(DEVICE)


class FocalLossWithLogits(nn.Module):
    """
    Binary Focal Loss with logits.
    Supports class imbalance via pos_weight.
    """

    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()

        # BCE with logits (no reduction)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        # pt = prob of true class
        probas = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probas, 1 - probas)

        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

criterion = FocalLossWithLogits(alpha=1.0, gamma=2.0, pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


class ModelWrapper(nn.Module):
    def __init__(self, model, dag):
        super().__init__()
        self.model = model
        self.dag = dag

    def forward(self, x):
        return self.model(x, self.dag)


wrapped_model = ModelWrapper(model, BEST_DAG)
summary(
    wrapped_model,
    input_size=(1, 7, 3000),
    device=DEVICE.type,
)
# ============================================================
# UTILS
# ============================================================
def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1

# ============================================================
# TRAINING
# ============================================================
history = {
    "train_loss": [],
    "val_loss": [],
    "roc_auc": [],
    "pr_auc": [],
    "f1": [],
    "precision": [],
    "recall": [],
    "threshold": [],
}

best_pr_auc = -1
patience_ctr = 0

for epoch in range(EPOCHS):
    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.float().to(DEVICE).squeeze(-1)

        logits = model(x, BEST_DAG)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------- VALID ----------------
    model.eval()
    val_loss = 0
    logits_all, targets_all = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.float().to(DEVICE).squeeze(-1)

            logits = model(x, BEST_DAG)
            loss = criterion(logits, y)

            val_loss += loss.item()
            logits_all.append(logits.cpu())
            targets_all.append(y.cpu())

    val_loss /= len(val_loader)

    logits_all = torch.cat(logits_all).numpy()
    targets_all = torch.cat(targets_all).numpy()
    probs = torch.sigmoid(torch.tensor(logits_all)).numpy()

    roc_auc = roc_auc_score(targets_all, probs)
    pr_auc = average_precision_score(targets_all, probs)

    best_t, best_f1 = find_best_threshold(targets_all, probs)
    preds = (probs >= best_t).astype(int)

    prec = precision_score(targets_all, preds, zero_division=0)
    rec = recall_score(targets_all, preds)

    # ---------------- LOG ----------------
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["roc_auc"].append(roc_auc)
    history["pr_auc"].append(pr_auc)
    history["f1"].append(best_f1)
    history["precision"].append(prec)
    history["recall"].append(rec)
    history["threshold"].append(best_t)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
        f"ROC {roc_auc:.4f} | PR {pr_auc:.4f} | "
        f"F1* {best_f1:.4f} @ t={best_t:.2f}"
    )

    # ---------------- EARLY STOP ----------------
    if pr_auc > best_pr_auc:
        best_pr_auc = pr_auc
        patience_ctr = 0
        torch.save(model.state_dict(), f"{SAVE_PREFIX}_best_model.pth")
    else:
        patience_ctr += 1

    if patience_ctr >= PATIENCE:
        print("[INFO] Early stopping")
        break

# ============================================================
# TEST EVALUATION
# ============================================================
model.load_state_dict(torch.load(f"{SAVE_PREFIX}_best_model.pth"))
model.eval()

logits_all, targets_all = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.float().to(DEVICE).squeeze(-1)
        logits = model(x, BEST_DAG)
        logits_all.append(logits.cpu())
        targets_all.append(y.cpu())

logits_all = torch.cat(logits_all).numpy()
targets_all = torch.cat(targets_all).numpy()
probs = torch.sigmoid(torch.tensor(logits_all)).numpy()

best_epoch = np.argmax(history["pr_auc"])
best_t = history["threshold"][best_epoch]

preds = (probs >= best_t).astype(int)

print("\n=== TEST REPORT ===")
print(classification_report(targets_all, preds, digits=4))

# Confusion Matrix
cm = confusion_matrix(targets_all, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ROC / PR curves
fpr, tpr, _ = roc_curve(targets_all, probs)
prec_curve, rec_curve, _ = precision_recall_curve(targets_all, probs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr)
plt.title("ROC Curve")

plt.subplot(1, 2, 2)
plt.plot(rec_curve, prec_curve)
plt.title("Precision-Recall Curve")
plt.show()

# Metric history
with open(f"{SAVE_PREFIX}_history.json", "w") as f:
    json.dump(history, f, indent=2)
