import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from dataset import SleepEDFSequenceDataset, collate_fn
from model import SleepStagingModel
from losses import FocalLoss
from utils import set_seed, compute_metrics


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y, lengths in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            logits = model(x, lengths)

            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            mask = y != -1
            y_true.extend(y[mask].cpu().numpy())
            y_pred.extend(preds[mask].cpu().numpy())
            y_prob.extend(probs[mask].cpu().numpy())

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
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = SleepStagingModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.25
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    criterion = FocalLoss(gamma=2)

    best_f1 = -1
    fold_log = []

    for epoch in range(50):
        model.train()
        total_loss = 0

        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["f1_weighted"])

        log_entry = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            **val_metrics
        }

        fold_log.append(log_entry)

        if val_metrics["f1_weighted"] > best_f1:
            best_f1 = val_metrics["f1_weighted"]
            torch.save(
                model.state_dict(),
                f"best_model_fold_{fold}.pt"
            )

        print(f"[Fold {fold}] Epoch {epoch} | F1={val_metrics['f1_weighted']:.4f}")

    return fold_log


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SleepEDFSequenceDataset("processed_sleepedf/index.csv")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_logs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        fold_logs = train_fold(fold, train_idx, val_idx, dataset, device)
        all_logs.extend(fold_logs)

    with open("training_metrics.json", "w") as f:
        json.dump(all_logs, f, indent=4)

    print("Training complete. Metrics saved to training_metrics.json")


if __name__ == "__main__":
    main()
