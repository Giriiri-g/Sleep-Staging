"""
Evaluate a trained DARTS model (multi-label, 20 classes) and generate plots:
- Metrics: precision/recall/F1 (macro/micro/per-class), exact match, hamming
- Confusion matrices (per class)
- ROC and PR curves (per class)
- Prediction heatmap sample
- Accuracy/loss curves (if history available)
- DARTS search process plots from results/logs (acc/loss + FLOPs/MACs/Params)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from torch.utils.data import DataLoader

# Local imports
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfs_dataset import CFSAilmentDataset, load_cfs_dataframe  # type: ignore
from nas_search_space import Network, DARTS_OPS, ALL_OPS  # type: ignore
from utils import (
    print_header,
    print_section,
    print_info,
    print_warning,
    print_success,
)


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def build_model_from_arch(
    arch: Dict,
    input_channels: int,
    input_length: int,
    num_classes: int,
    init_channels: int,
    num_cells: int,
    num_nodes: int,
    device: torch.device,
    search_space: Optional[List[str]] = None,
) -> nn.Module:
    """
    Build a Network and set arch_params to the discrete architecture.
    Uses ALL_OPS by default to match checkpoints saved from evaluate_darts_architecture.py
    """
    if search_space is None:
        search_space = ALL_OPS  # Use ALL_OPS to match saved checkpoints
    
    model = Network(
        input_channels=input_channels,
        input_length=input_length,
        num_classes=num_classes,
        init_channels=init_channels,
        num_cells=num_cells,
        num_nodes=num_nodes,
        search_space=search_space,
    ).to(device)

    alpha = torch.zeros_like(model.arch_params)
    for cell_idx in range(num_cells):
        if cell_idx not in arch:
            continue
        cell_arch = arch[cell_idx]
        edge_idx = 0
        for node_idx in range(num_nodes):
            if node_idx < len(cell_arch):
                for prev_node, op_name in cell_arch[node_idx]:
                    if op_name in search_space:
                        op_idx = search_space.index(op_name)
                        alpha[cell_idx, edge_idx, op_idx] = 100.0
                    edge_idx += 1
            else:
                edge_idx += node_idx + 2
    model._arch_params.data = alpha
    return model


def make_dataloader(csv_path: Path, batch_size: int, input_channels: int, input_length: int) -> DataLoader:
    df = load_cfs_dataframe(str(csv_path))
    dataset = CFSAilmentDataset(
        dataframe=df,
        indices=list(range(len(df))),
        input_channels=input_channels,
        input_length=input_length,
        channel_names=None,
        target_sample_rate=128.0,
        normalization="zscore",
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float, class_names: List[str]) -> Dict:
    model.eval()
    logits_all, preds_all, targets_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            logits_all.append(logits.cpu().numpy())
            preds_all.append(preds.cpu().numpy())
            targets_all.append(y.cpu().numpy())
    logits_all = np.vstack(logits_all)
    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)

    # Ensure binary
    preds_all = (preds_all > 0.5).astype(int)
    targets_all = (targets_all > 0.5).astype(int)

    # Metrics
    macro_p = precision_score(targets_all, preds_all, average="macro", zero_division=0)
    macro_r = recall_score(targets_all, preds_all, average="macro", zero_division=0)
    macro_f1 = f1_score(targets_all, preds_all, average="macro", zero_division=0)
    micro_p = precision_score(targets_all, preds_all, average="micro", zero_division=0)
    micro_r = recall_score(targets_all, preds_all, average="micro", zero_division=0)
    micro_f1 = f1_score(targets_all, preds_all, average="micro", zero_division=0)
    exact_match = (preds_all == targets_all).all(axis=1).mean()
    h_loss = hamming_loss(targets_all, preds_all)

    # Per-class
    per_p = precision_score(targets_all, preds_all, average=None, zero_division=0)
    per_r = recall_score(targets_all, preds_all, average=None, zero_division=0)
    per_f1 = f1_score(targets_all, preds_all, average=None, zero_division=0)

    # AUC / AP
    per_auc, per_ap = [], []
    for i in range(targets_all.shape[1]):
        if targets_all[:, i].sum() > 0:
            try:
                per_auc.append(roc_auc_score(targets_all[:, i], logits_all[:, i]))
            except Exception:
                per_auc.append(0.0)
            try:
                per_ap.append(average_precision_score(targets_all[:, i], logits_all[:, i]))
            except Exception:
                per_ap.append(0.0)
        else:
            per_auc.append(0.0)
            per_ap.append(0.0)
    macro_auc = float(np.mean(per_auc)) if per_auc else 0.0
    macro_ap = float(np.mean(per_ap)) if per_ap else 0.0

    return {
        "logits": logits_all,
        "preds": preds_all,
        "targets": targets_all,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
        "exact_match": exact_match,
        "hamming": h_loss,
        "per_p": per_p,
        "per_r": per_r,
        "per_f1": per_f1,
        "per_auc": per_auc,
        "per_ap": per_ap,
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "class_names": class_names,
    }


def plot_confusions(metrics: Dict, out_dir: Path):
    preds = metrics["preds"]
    targets = metrics["targets"]
    class_names = metrics["class_names"]
    n = len(class_names)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()
    for i in range(n):
        cm = confusion_matrix(targets[:, i], preds[:, i])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"], ax=axes[i])
        axes[i].set_title(class_names[i], fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    out_path = out_dir / "confusion_matrices.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved {out_path}")


def plot_roc_pr(metrics: Dict, out_dir: Path):
    logits = metrics["logits"]
    targets = metrics["targets"]
    class_names = metrics["class_names"]
    n = len(class_names)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    # ROC
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()
    for i in range(n):
        ax = axes[i]
        if targets[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(targets[:, i], logits[:, i])
            auc = metrics["per_auc"][i]
            ax.plot(fpr, tpr, label=f"AUC={auc:.3f}", linewidth=2)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_title(class_names[i], fontsize=10)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No positives", ha="center", va="center")
            ax.set_title(class_names[i], fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    out_path = out_dir / "roc_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved {out_path}")

    # PR
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()
    for i in range(n):
        ax = axes[i]
        if targets[:, i].sum() > 0:
            precision, recall, _ = precision_recall_curve(targets[:, i], logits[:, i])
            ap = metrics["per_ap"][i]
            ax.plot(recall, precision, label=f"AP={ap:.3f}", linewidth=2)
            ax.set_title(class_names[i], fontsize=10)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No positives", ha="center", va="center")
            ax.set_title(class_names[i], fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    out_path = out_dir / "pr_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved {out_path}")


def plot_prediction_heatmap(metrics: Dict, out_dir: Path, max_samples: int = 100):
    preds = metrics["preds"]
    targets = metrics["targets"]
    class_names = metrics["class_names"]
    if len(preds) > max_samples:
        idx = np.random.choice(len(preds), max_samples, replace=False)
        preds = preds[idx]
        targets = targets[idx]
    combined = np.hstack([preds, targets])
    labels = [f"P_{c}" for c in class_names] + [f"T_{c}" for c in class_names]
    fig, ax = plt.subplots(figsize=(max(12, len(class_names)), max(6, len(preds) * 0.1)))
    sns.heatmap(combined.T, cmap="RdYlGn", xticklabels=False, yticklabels=labels, ax=ax)
    ax.set_title("Predictions vs Ground Truth (sample)", fontsize=12)
    out_path = out_dir / "prediction_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print_success(f"Saved {out_path}")


def plot_history_from_results(results_path: Path, out_dir: Path):
    """
    Plot DARTS search history and efficiency (FLOPs/MACs/Params) if present.
    - Uses 'history' from results JSON for train/val losses/acc
    - Uses logs_epoch_*.json (if available) for efficiency over epochs
    """
    if not results_path.exists():
        return
    with open(results_path, "r") as f:
        res = json.load(f)
    hist = res.get("history", {})
    epochs = range(1, len(hist.get("train_loss", [])) + 1)

    # Train/val loss/acc
    if epochs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, hist.get("train_loss", []), label="Train Loss")
        axes[0].plot(epochs, hist.get("val_loss", []), label="Val Loss")
        axes[0].set_title("DARTS: Loss")
        axes[0].legend()
        axes[1].plot(epochs, hist.get("train_acc", []), label="Train Acc")
        axes[1].plot(epochs, hist.get("val_acc", []), label="Val Acc")
        axes[1].set_title("DARTS: Accuracy")
        axes[1].legend()
        plt.tight_layout()
        out_path = out_dir / "darts_history.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print_success(f"Saved {out_path}")

    # Efficiency over epochs from logs
    log_files = sorted(glob.glob(str(results_path.parent / "logs_epoch_*.json")))
    if log_files:
        epochs_e, flops, macs, params = [], [], [], []
        for lf in log_files:
            with open(lf, "r") as f:
                d = json.load(f)
                epochs_e.append(d.get("epoch", None))
                eff = d.get("current_efficiency", {})
                flops.append(eff.get("flops", None))
                macs.append(eff.get("macs", None))
                params.append(eff.get("params", None))
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(epochs_e, flops, label="FLOPs")
        ax.plot(epochs_e, macs, label="MACs")
        ax.plot(epochs_e, params, label="Params")
        ax.set_xlabel("Epoch")
        ax.set_title("DARTS Efficiency over Epochs")
        ax.legend()
        plt.tight_layout()
        out_path = out_dir / "darts_efficiency.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print_success(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved DARTS model with visualizations")
    parser.add_argument("--results_path", type=str,
                        default="efficient_darts_results/efficient_darts_results.json",
                        help="Path to DARTS results JSON (best_architecture inside)")
    parser.add_argument("--checkpoint_path", type=str,
                        default="darts_evaluation_results/trained_model.pth",
                        help="Path to trained model state_dict")
    parser.add_argument("--csv_path", type=str, default="csv-docs/cfs_visit5_selected.csv",
                        help="CSV with paths to preprocessed tensors")
    parser.add_argument("--output_dir", type=str, default="final_eval_results",
                        help="Directory to save evaluation outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--input_channels", type=int, default=7)
    parser.add_argument("--input_length", type=int, default=3000)
    parser.add_argument("--init_channels", type=int, default=24)
    parser.add_argument("--num_cells", type=int, default=3)
    parser.add_argument("--num_nodes", type=int, default=2)
    args = parser.parse_args()

    print_header("Evaluate Saved DARTS Model")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_section("Config")
    print_info(f"Device: {device}")
    print_info(f"Results JSON: {args.results_path}")
    print_info(f"Checkpoint: {args.checkpoint_path}")
    print_info(f"CSV: {args.csv_path}")

    # Load architecture
    with open(args.results_path, "r") as f:
        res = json.load(f)
    arch = res.get("best_architecture", res.get("final_architecture", {}))
    if not arch:
        print_warning("No architecture found in results file.")
        return
    # class names from CSV
    df = load_cfs_dataframe(args.csv_path)
    class_names = [c for c in df.columns if c != "path"]
    num_classes = len(class_names)

    # Build model and load weights
    # Use ALL_OPS to match checkpoints saved from evaluate_darts_architecture.py
    model = build_model_from_arch(
        arch=arch,
        input_channels=args.input_channels,
        input_length=args.input_length,
        num_classes=num_classes,
        init_channels=args.init_channels,
        num_cells=args.num_cells,
        num_nodes=args.num_nodes,
        device=device,
        search_space=ALL_OPS,  # Match the checkpoint's search space
    )
    
    # Load checkpoint - handle both dict and nested state_dict formats
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint else checkpoint.get("model_state_dict", checkpoint)
    
    # Filter out _arch_params from checkpoint since we set it from architecture
    # This avoids shape mismatch errors when search spaces differ
    filtered_state_dict = {k: v for k, v in state_dict.items() if k != "_arch_params"}
    
    # Load with strict=False to allow _arch_params mismatch
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    if missing_keys:
        print_warning(f"Missing keys in checkpoint: {missing_keys[:5]}...")
    if unexpected_keys:
        print_warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
    
    # Set architecture parameters from the architecture dict (overrides checkpoint)
    alpha = torch.zeros_like(model.arch_params)
    for cell_idx in range(args.num_cells):
        if cell_idx not in arch:
            continue
        cell_arch = arch[cell_idx]
        edge_idx = 0
        for node_idx in range(args.num_nodes):
            if node_idx < len(cell_arch):
                for prev_node, op_name in cell_arch[node_idx]:
                    if op_name in ALL_OPS:
                        op_idx = ALL_OPS.index(op_name)
                        alpha[cell_idx, edge_idx, op_idx] = 100.0
                    edge_idx += 1
            else:
                edge_idx += node_idx + 2
    model._arch_params.data = alpha
    
    model.eval()
    print_success("Loaded model checkpoint.")

    # DataLoader (full dataset)
    loader = make_dataloader(Path(args.csv_path), args.batch_size, args.input_channels, args.input_length)

    print_section("Evaluating")
    metrics = evaluate(model, loader, device, args.threshold, class_names)
    print_info(f"Macro F1: {metrics['macro_f1']:.4f} | Macro AUC: {metrics['macro_auc']:.4f} | Macro AP: {metrics['macro_ap']:.4f}")
    print_info(f"Exact Match: {metrics['exact_match']:.4f} | Hamming: {metrics['hamming']:.4f}")

    # Plots
    plot_confusions(metrics, output_dir)
    plot_roc_pr(metrics, output_dir)
    plot_prediction_heatmap(metrics, output_dir)

    # DARTS process plots
    plot_history_from_results(Path(args.results_path), output_dir)

    # Save metrics JSON
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({
            "macro_f1": metrics["macro_f1"],
            "macro_auc": metrics["macro_auc"],
            "macro_ap": metrics["macro_ap"],
            "micro_f1": metrics["micro_f1"],
            "micro_p": metrics["micro_p"],
            "micro_r": metrics["micro_r"],
            "exact_match": metrics["exact_match"],
            "hamming": metrics["hamming"],
        }, f, indent=2)
    print_success(f"Saved metrics and plots to {output_dir}")


if __name__ == "__main__":
    main()


