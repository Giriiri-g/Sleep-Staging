"""
Comprehensive evaluation script for DARTS-discovered architecture.

Evaluates the best architecture on complete dataset (train + val + test)
and generates comprehensive visualizations for multi-label classification.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, ConcatDataset

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfs_dataset import create_cfs_dataloaders, CFSAilmentDataset, load_cfs_dataframe
from nas_search_space import Network, DARTS_OPS, ALL_OPS
from rl_search import architecture_to_model
from utils import (
    print_header,
    print_section,
    print_info,
    print_success,
    print_warning,
    print_key_value,
    suppress_warnings,
)

suppress_warnings()

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_best_architecture(results_path: Path) -> Tuple[Dict, Dict]:
    """Load best architecture and metadata from DARTS results JSON."""
    with open(results_path, "r") as f:
        results = json.load(f)
    
    best_arch = results.get("best_architecture", results.get("final_architecture", {}))
    metadata = {
        "best_val_acc": results.get("best_val_acc", 0.0),
        "best_efficiency": results.get("best_efficiency", {}),
        "final_efficiency": results.get("final_efficiency", {}),
    }
    
    return best_arch, metadata


def build_model_from_architecture(
    arch: Dict,
    input_channels: int,
    input_length: int,
    num_classes: int,
    init_channels: int,
    num_cells: int,
    num_nodes: int,
    device: torch.device,
) -> nn.Module:
    """Build and return model from architecture dictionary."""
    model = architecture_to_model(
        arch=arch,
        input_channels=input_channels,
        input_length=input_length,
        num_classes=num_classes,
        init_channels=init_channels,
        num_cells=num_cells,
        num_nodes=num_nodes,
    )
    model = model.to(device)
    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 0.025,
    momentum: float = 0.9,
    weight_decay: float = 3e-4,
    threshold: float = 0.5,
) -> Tuple[nn.Module, Dict]:
    """Train the model and return trained model with training history."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, target in train_loader:
            x = x.to(device)
            target = target.float().to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) >= threshold).float()
            per_sample = (preds == target).float().mean(dim=1)
            train_correct += per_sample.sum().item()
            train_total += per_sample.shape[0]

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, target in val_loader:
                x = x.to(device)
                target = target.float().to(device)

                logits = model(x)
                loss = criterion(logits, target)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) >= threshold).float()
                per_sample = (preds == target).float().mean(dim=1)
                val_correct += per_sample.sum().item()
                val_total += per_sample.shape[0]

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / max(1, train_total)
        val_acc = 100.0 * val_correct / max(1, val_total)

        history["train_loss"].append(train_loss_avg)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss_avg)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print_info(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%"
            )

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print_success(f"Training completed. Best Val Acc: {best_val_acc:.2f}%")
    return model, history


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    class_names: List[str] = None,
) -> Dict:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_logits = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.float().to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Overall metrics
    exact_match = (all_preds == all_targets).all(axis=1).mean()
    hamming = hamming_loss(all_targets, all_preds)
    subset_acc = exact_match

    # Per-class metrics
    num_classes = all_targets.shape[1]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    per_class_precision = precision_score(
        all_targets, all_preds, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        all_targets, all_preds, average=None, zero_division=0
    )
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    # Macro and micro averages
    macro_precision = precision_score(
        all_targets, all_preds, average="macro", zero_division=0
    )
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    micro_precision = precision_score(
        all_targets, all_preds, average="micro", zero_division=0
    )
    micro_recall = recall_score(all_targets, all_preds, average="micro", zero_division=0)
    micro_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)

    # AUC scores (per class and macro)
    try:
        per_class_auc = []
        for i in range(num_classes):
            if all_targets[:, i].sum() > 0:  # At least one positive sample
                auc = roc_auc_score(all_targets[:, i], all_logits[:, i])
                per_class_auc.append(auc)
            else:
                per_class_auc.append(0.0)
        macro_auc = np.mean(per_class_auc)
    except Exception as e:
        print_warning(f"Could not compute AUC: {e}")
        per_class_auc = [0.0] * num_classes
        macro_auc = 0.0

    # Average Precision (AP) scores
    try:
        per_class_ap = []
        for i in range(num_classes):
            if all_targets[:, i].sum() > 0:
                ap = average_precision_score(all_targets[:, i], all_logits[:, i])
                per_class_ap.append(ap)
            else:
                per_class_ap.append(0.0)
        macro_ap = np.mean(per_class_ap)
    except Exception as e:
        print_warning(f"Could not compute AP: {e}")
        per_class_ap = [0.0] * num_classes
        macro_ap = 0.0

    metrics = {
        "exact_match": exact_match,
        "subset_accuracy": subset_acc,
        "hamming_loss": hamming,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_auc": per_class_auc,
        "per_class_ap": per_class_ap,
        "all_logits": all_logits,
        "all_preds": all_preds,
        "all_targets": all_targets,
        "class_names": class_names,
    }

    return metrics


def plot_training_history(history: Dict, save_path: Path):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved training history plot: {save_path}")


def plot_roc_curves(metrics: Dict, save_path: Path, max_classes: int = 20):
    """Plot ROC curves for each class."""
    all_logits = metrics["all_logits"]
    all_targets = metrics["all_targets"]
    class_names = metrics["class_names"]
    num_classes = min(len(class_names), max_classes)

    fig, axes = plt.subplots(
        int(np.ceil(num_classes / 5)), 5, figsize=(20, 4 * np.ceil(num_classes / 5))
    )
    axes = axes.flatten() if num_classes > 1 else [axes]

    for i in range(num_classes):
        ax = axes[i]
        if all_targets[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(all_targets[:, i], all_logits[:, i])
            auc = metrics["per_class_auc"][i]
            ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
            ax.set_xlabel("False Positive Rate", fontsize=10)
            ax.set_ylabel("True Positive Rate", fontsize=10)
            ax.set_title(f"{class_names[i]}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No positive samples", ha="center", va="center")
            ax.set_title(f"{class_names[i]}", fontsize=11)

    # Hide unused subplots
    for i in range(num_classes, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved ROC curves: {save_path}")


def plot_pr_curves(metrics: Dict, save_path: Path, max_classes: int = 20):
    """Plot Precision-Recall curves for each class."""
    all_logits = metrics["all_logits"]
    all_targets = metrics["all_targets"]
    class_names = metrics["class_names"]
    num_classes = min(len(class_names), max_classes)

    fig, axes = plt.subplots(
        int(np.ceil(num_classes / 5)), 5, figsize=(20, 4 * np.ceil(num_classes / 5))
    )
    axes = axes.flatten() if num_classes > 1 else [axes]

    for i in range(num_classes):
        ax = axes[i]
        if all_targets[:, i].sum() > 0:
            precision, recall, _ = precision_recall_curve(
                all_targets[:, i], all_logits[:, i]
            )
            ap = metrics["per_class_ap"][i]
            ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
            ax.set_xlabel("Recall", fontsize=10)
            ax.set_ylabel("Precision", fontsize=10)
            ax.set_title(f"{class_names[i]}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No positive samples", ha="center", va="center")
            ax.set_title(f"{class_names[i]}", fontsize=11)

    for i in range(num_classes, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved PR curves: {save_path}")


def plot_class_performance(metrics: Dict, save_path: Path):
    """Plot bar chart of per-class performance metrics."""
    class_names = metrics["class_names"]
    precision = metrics["per_class_precision"]
    recall = metrics["per_class_recall"]
    f1 = metrics["per_class_f1"]
    auc = metrics["per_class_auc"]

    x = np.arange(len(class_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.5), 6))
    ax.bar(x - 1.5 * width, precision, width, label="Precision", alpha=0.8)
    ax.bar(x - 0.5 * width, recall, width, label="Recall", alpha=0.8)
    ax.bar(x + 0.5 * width, f1, width, label="F1", alpha=0.8)
    ax.bar(x + 1.5 * width, auc, width, label="AUC", alpha=0.8)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Performance Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved class performance plot: {save_path}")


def plot_prediction_heatmap(metrics: Dict, save_path: Path, max_samples: int = 100):
    """Plot heatmap of predictions vs ground truth."""
    all_preds = metrics["all_preds"]
    all_targets = metrics["all_targets"]
    class_names = metrics["class_names"]

    # Sample if too many
    if len(all_preds) > max_samples:
        indices = np.random.choice(len(all_preds), max_samples, replace=False)
        all_preds = all_preds[indices]
        all_targets = all_targets[indices]

    # Create combined matrix: first half is predictions, second half is targets
    combined = np.hstack([all_preds, all_targets])
    labels = [f"Pred_{name}" for name in class_names] + [
        f"True_{name}" for name in class_names
    ]

    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.8), max(8, len(all_preds) * 0.1)))
    sns.heatmap(
        combined.T,
        cmap="RdYlGn",
        cbar_kws={"label": "Label"},
        xticklabels=False,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Predictions vs Ground Truth (Sample)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Classes", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved prediction heatmap: {save_path}")


def plot_confusion_matrix_per_class(metrics: Dict, save_path: Path, max_classes: int = 20):
    """Plot confusion matrices for each class (binary classification per class)."""
    all_preds = metrics["all_preds"]
    all_targets = metrics["all_targets"]
    class_names = metrics["class_names"]
    num_classes = min(len(class_names), max_classes)

    n_cols = 5
    n_rows = int(np.ceil(num_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if num_classes > 1 else [axes]

    for i in range(num_classes):
        ax = axes[i]
        cm = confusion_matrix(all_targets[:, i], all_preds[:, i])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
        )
        ax.set_title(f"{class_names[i]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("True", fontsize=10)
        ax.set_xlabel("Pred", fontsize=10)

    for i in range(num_classes, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved confusion matrices: {save_path}")


def save_metrics_report(metrics: Dict, save_path: Path, split_name: str = ""):
    """Save detailed metrics report to text file."""
    with open(save_path, "w") as f:
        f.write(f"=== Multi-Label Classification Metrics {split_name} ===\n\n")
        f.write("Overall Metrics:\n")
        f.write(f"  Exact Match Accuracy: {metrics['exact_match']:.4f}\n")
        f.write(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}\n")
        f.write(f"  Hamming Loss: {metrics['hamming_loss']:.4f}\n\n")

        f.write("Macro-Averaged Metrics:\n")
        f.write(f"  Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"  Recall: {metrics['macro_recall']:.4f}\n")
        f.write(f"  F1-Score: {metrics['macro_f1']:.4f}\n")
        f.write(f"  AUC-ROC: {metrics['macro_auc']:.4f}\n")
        f.write(f"  Average Precision: {metrics['macro_ap']:.4f}\n\n")

        f.write("Micro-Averaged Metrics:\n")
        f.write(f"  Precision: {metrics['micro_precision']:.4f}\n")
        f.write(f"  Recall: {metrics['micro_recall']:.4f}\n")
        f.write(f"  F1-Score: {metrics['micro_f1']:.4f}\n\n")

        f.write("Per-Class Metrics:\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12} {'AP':<12}\n")
        f.write("-" * 80 + "\n")
        for i, name in enumerate(metrics["class_names"]):
            f.write(
                f"{name:<20} {metrics['per_class_precision'][i]:<12.4f} "
                f"{metrics['per_class_recall'][i]:<12.4f} "
                f"{metrics['per_class_f1'][i]:<12.4f} "
                f"{metrics['per_class_auc'][i]:<12.4f} "
                f"{metrics['per_class_ap'][i]:<12.4f}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DARTS-discovered architecture with comprehensive visualizations"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="efficient_darts_results/efficient_darts_results.json",
        help="Path to DARTS results JSON file",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="csv-docs/cfs_visit5_selected.csv",
        help="Path to CFS CSV file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to trained model checkpoint (optional, will train if not provided)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="darts_evaluation_results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=7,
        help="Number of input channels",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=3000,
        help="Input sequence length",
    )
    parser.add_argument(
        "--init_channels",
        type=int,
        default=24,
        help="Initial channels in architecture",
    )
    parser.add_argument(
        "--num_cells",
        type=int,
        default=3,
        help="Number of cells in architecture",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=2,
        help="Number of nodes per cell",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=50,
        help="Number of epochs to train (if checkpoint not provided)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for multi-label predictions",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only evaluate (requires checkpoint)",
    )

    args = parser.parse_args()

    print_header("DARTS Architecture Evaluation")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("Configuration")
    print_key_value("Device", device)
    print_key_value("Results path", args.results_path)
    print_key_value("CSV path", args.csv_path)
    print_key_value("Output directory", output_dir)

    # Load best architecture
    print_section("Loading Architecture")
    results_path = Path(args.results_path)
    if not results_path.exists():
        print_warning(f"Results file not found: {results_path}")
        return

    best_arch, metadata = load_best_architecture(results_path)
    print_success(f"Loaded best architecture (Val Acc: {metadata['best_val_acc']:.2f}%)")

    # Load data
    print_section("Loading Data")
    df = load_cfs_dataframe(args.csv_path)
    num_classes = len([col for col in df.columns if col != "path"])

    # Create dataloaders for train/val/test
    train_loader, val_loader, test_loader, stats = create_cfs_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        input_channels=args.input_channels,
        input_length=args.input_length,
        val_split=100,  # Use same split as training
        test_split=0.2,
        num_workers=0,
        seed=42,
        channel_names=None,  # Use preprocessed tensors
        target_sample_rate=128.0,
        normalization="zscore",
    )

    print_key_value("Train samples", stats["train"])
    print_key_value("Val samples", stats["val"])
    print_key_value("Test samples", stats["test"])
    print_key_value("Number of classes", num_classes)

    # Get class names from CSV
    class_names = [col for col in df.columns if col != "path"]

    # Build model
    print_section("Building Model")
    model = build_model_from_architecture(
        arch=best_arch,
        input_channels=args.input_channels,
        input_length=args.input_length,
        num_classes=num_classes,
        init_channels=args.init_channels,
        num_cells=args.num_cells,
        num_nodes=args.num_nodes,
        device=device,
    )
    print_success("Model built successfully")

    # Train or load checkpoint
    history = None
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        print_section("Loading Checkpoint")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        print_success(f"Loaded checkpoint from {args.checkpoint_path}")
    elif not args.skip_training:
        print_section("Training Model")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.train_epochs,
            threshold=args.threshold,
        )
        # Save trained model
        torch.save(model.state_dict(), output_dir / "trained_model.pth")
        print_success(f"Saved trained model to {output_dir / 'trained_model.pth'}")
    else:
        print_warning("Skipping training but no checkpoint provided!")

    # Create combined dataloader for full dataset evaluation
    print_section("Creating Combined Dataset")
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset if test_loader else None

    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    if test_dataset:
        combined_dataset = ConcatDataset([combined_dataset, test_dataset])

    combined_loader = DataLoader(
        combined_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Evaluate on all splits
    print_section("Evaluation")

    splits = {
        "Train": train_loader,
        "Validation": val_loader,
        "Test": test_loader if test_loader else None,
        "Complete": combined_loader,
    }

    all_metrics = {}

    for split_name, loader in splits.items():
        if loader is None:
            continue
        print_info(f"Evaluating on {split_name} set...")
        metrics = evaluate_model(
            model=model,
            dataloader=loader,
            device=device,
            threshold=args.threshold,
            class_names=class_names,
        )
        all_metrics[split_name] = metrics

        print_key_value(f"{split_name} - Exact Match", f"{metrics['exact_match']:.4f}")
        print_key_value(f"{split_name} - Macro F1", f"{metrics['macro_f1']:.4f}")
        print_key_value(f"{split_name} - Macro AUC", f"{metrics['macro_auc']:.4f}")

    # Generate visualizations
    print_section("Generating Visualizations")

    # Training history
    if history:
        plot_training_history(history, output_dir / "training_history.png")

    # For each split, generate visualizations
    for split_name, metrics in all_metrics.items():
        split_dir = output_dir / split_name.lower().replace(" ", "_")
        split_dir.mkdir(exist_ok=True)

        # ROC curves
        plot_roc_curves(metrics, split_dir / "roc_curves.png")

        # PR curves
        plot_pr_curves(metrics, split_dir / "pr_curves.png")

        # Class performance
        plot_class_performance(metrics, split_dir / "class_performance.png")

        # Confusion matrices
        plot_confusion_matrix_per_class(metrics, split_dir / "confusion_matrices.png")

        # Prediction heatmap
        plot_prediction_heatmap(metrics, split_dir / "prediction_heatmap.png")

        # Metrics report
        save_metrics_report(metrics, split_dir / "metrics_report.txt", split_name)

    # Save summary
    summary = {
        "architecture": best_arch,
        "metadata": metadata,
        "config": {
            "input_channels": args.input_channels,
            "input_length": args.input_length,
            "num_classes": num_classes,
            "init_channels": args.init_channels,
            "num_cells": args.num_cells,
            "num_nodes": args.num_nodes,
        },
        "metrics_summary": {
            split: {
                "exact_match": float(m["exact_match"]),
                "macro_f1": float(m["macro_f1"]),
                "macro_auc": float(m["macro_auc"]),
                "macro_ap": float(m["macro_ap"]),
            }
            for split, m in all_metrics.items()
        },
    }

    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print_success(f"\nEvaluation complete! Results saved to {output_dir}")
    print_info(f"  - Training history: {output_dir / 'training_history.png'}")
    for split_name in all_metrics.keys():
        print_info(f"  - {split_name} visualizations: {output_dir / split_name.lower().replace(' ', '_')}")


if __name__ == "__main__":
    main()

