import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from dataset import SleepEDFSequenceDataset
from model import SleepStagingModel
from losses import FocalLoss
from utils import set_seed, compute_metrics


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return compute_metrics(
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob)
    )


def train_fold(fold, train_idx, val_idx, dataset, device):
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    model = SleepStagingModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.25
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    criterion = FocalLoss(gamma=2)
    best_f1 = -1
    fold_logs = []

    for epoch in range(50):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["f1_weighted"])

        log = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            **metrics
        }

        fold_logs.append(log)

        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]
            torch.save(model.state_dict(), f"best_model_fold_{fold}.pt")

        print(f"[Fold {fold}] Epoch {epoch} | F1={metrics['f1_weighted']:.4f}")

    return fold_logs


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SleepEDFSequenceDataset(
        "processed_sleepedf/index.csv",
        window=10
    )

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_logs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        fold_logs = train_fold(fold, train_idx, val_idx, dataset, device)
        all_logs.extend(fold_logs)

    with open("training_metrics.json", "w") as f:
        json.dump(all_logs, f, indent=4)

    print("Training complete. Metrics saved.")


if __name__ == "__main__":
    main()
