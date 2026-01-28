import json
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

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
    print(f"[DEBUG] Starting train_fold for fold {fold}, train samples: {len(train_idx)}, val samples: {len(val_idx)}")
    start_time = time.time()

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=4,
        shuffle=False,
        num_workers=4
    )
    print(f"[DEBUG] DataLoaders created in {time.time() - start_time:.2f}s")

    model = SleepStagingModel().to(device)
    print(f"[DEBUG] Model created and moved to {device} in {time.time() - start_time:.2f}s")

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
    print(f"[DEBUG] Optimizer, scheduler, criterion initialized for fold {fold}.")
    for epoch in range(50):
        epoch_start = time.time()
        model.train()
        print(f"[DEBUG] Epoch {epoch} training started for fold {fold}.")
        total_loss = 0

        batch_times = []
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}")):
            batch_start = time.time()
            x, y = x.to(device), y.to(device)
            data_load_time = time.time() - batch_start

            optimizer.zero_grad()
            forward_start = time.time()
            logits = model(x)
            forward_time = time.time() - forward_start

            loss_start = time.time()
            loss = criterion(logits, y)
            loss_time = time.time() - loss_start

            backward_start = time.time()
            loss.backward()
            backward_time = time.time() - backward_start

            optimizer.step()
            total_loss += loss.item()

            batch_total = time.time() - batch_start
            batch_times.append(batch_total)
            if batch_idx % 100 == 0:  # Log every 100 batches
                print(f"[DEBUG] Batch {batch_idx}: DataLoad={data_load_time:.3f}s, Forward={forward_time:.3f}s, Loss={loss_time:.3f}s, Backward={backward_time:.3f}s, Total={batch_total:.3f}s")

        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        print(f"[DEBUG] Epoch {epoch} training completed in {time.time() - epoch_start:.2f}s, avg batch time: {avg_batch_time:.3f}s")

        val_start = time.time()
        metrics = evaluate(model, val_loader, device)
        val_time = time.time() - val_start
        print(f"[DEBUG] Validation completed in {val_time:.2f}s")

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
        window=5
    )
    print("Dataset loaded.")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_logs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Starting fold {fold}...")
        fold_logs = train_fold(fold, train_idx, val_idx, dataset, device)
        all_logs.extend(fold_logs)

    with open("training_metrics.json", "w") as f:
        json.dump(all_logs, f, indent=4)

    print("Training complete. Metrics saved.")


if __name__ == "__main__":
    main()
