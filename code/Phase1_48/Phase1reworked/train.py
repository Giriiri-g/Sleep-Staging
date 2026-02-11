import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split

from dataset import SleepEDFSequenceDataset
from model import SleepStagingModel
from losses import FocalLoss
from utils import set_seed, compute_metrics
import time

def logger(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def evaluate(model, loader, device):
    print("[EVAL] Starting evaluation...")
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    batch_count = 0


    with torch.no_grad():
        for x, y, padding_mask in loader:
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"[EVAL] Processed {batch_count}/{len(loader)} batches")
            x = x.to(device)
            y = y.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(x, padding_mask) # [1, T, C]
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            valid = ~padding_mask

            y_true.extend(y[valid].cpu().numpy())
            y_pred.extend(preds[valid].cpu().numpy())
            y_prob.extend(probs[valid].cpu().numpy())
    print(f"[EVAL] Finished forward passes.")
    print(f"[EVAL] Total valid samples: {len(y_true)}")

    print("[EVAL] Computing metrics...")

    return compute_metrics(
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob)
    )


def sleep_collate_fn(batch):
    """
    batch: list of (x, y)
      x: [T_i, 3000]
      y: [T_i]
    """

    lengths = [x.shape[0] for x, _ in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    # Allocate padded tensors
    x_padded = torch.zeros(batch_size, max_len, 3000)
    y_padded = torch.full((batch_size, max_len), -100)  # ignore index
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, (x, y) in enumerate(batch):
        T = x.shape[0]
        x_padded[i, :T] = x
        y_padded[i, :T] = y
        padding_mask[i, :T] = False  # False = valid token

    return x_padded, y_padded, padding_mask


def train_model(train_dataset, val_dataset, device):
    print(f"[DEBUG] Starting training, train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=sleep_collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=sleep_collate_fn,
        pin_memory=True,
        persistent_workers=True
    )


    model = SleepStagingModel().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger(f"Model initialized on {device}")
    logger(f"Total parameters: {total_params:,}")
    logger(f"Trainable parameters: {trainable_params:,}")


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    criterion = FocalLoss(gamma=2)
    scaler = torch.amp.GradScaler() # AMP scaler
    best_f1 = -1

    print(f"[DEBUG] Optimizer, scheduler, criterion initialized.")
    for epoch in range(50):
        model.train()
        print(f"[DEBUG] Epoch {epoch} training started.")
        total_loss = 0

        for batch_idx, (x, y, padding_mask) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                logits = model(x, padding_mask)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda" and batch_idx % 50 == 0:
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                logger(f"GPU Memory | Allocated: {mem_alloc:.1f} MB | Reserved: {mem_reserved:.1f} MB")
            total_loss += loss.item()
            if batch_idx % 20 == 0:
                logger(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")


        metrics = evaluate(model, val_loader, device)
        logger(f"Validation Results | F1: {metrics['f1_weighted']:.4f} | Acc: {metrics['accuracy']:.4f}")
        scheduler.step(metrics["f1_weighted"])
        log = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            **metrics
        }
        with open("training_metrics.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")

        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]
            torch.save(model.state_dict(), "best_model.pt")
        current_lr = optimizer.param_groups[0]["lr"]

        logger(
            f"Epoch {epoch} Completed | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"F1: {metrics['f1_weighted']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"LR: {current_lr:.6f}"
        )



    print("Training completed")

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full dataset to get file indices
    full_dataset = SleepEDFSequenceDataset("processed_sleepedf/index.csv")
    num_files = len(full_dataset.df)
    print(f"Total files: {num_files}")

    # Split at file level
    indices = list(range(num_files))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = SleepEDFSequenceDataset("processed_sleepedf/index.csv", file_indices=train_indices)
    val_dataset = SleepEDFSequenceDataset("processed_sleepedf/index.csv", file_indices=val_indices)

    print(f"Train files: {len(train_indices)}, Val files: {len(val_indices)}")

    print("Starting training...")
    train_model(train_dataset, val_dataset, device)

if __name__ == "__main__":
    main()
