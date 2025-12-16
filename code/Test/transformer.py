import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# ----------------------------
# Dataset Loader
# ----------------------------
class SleepDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)  # (N, C, T)
        self.labels = np.load(label_path).astype(int)
        print(f"Loaded dataset: {self.data.shape}, labels: {self.labels.shape}")

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ----------------------------
# Transformer Model
# ----------------------------
class SleepTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, num_classes):
        super(SleepTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
    
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.attn_weights = None  # for explainability

    def forward(self, x):
        # x: (B, C, T) → we want (B, T, C)
        x = x.permute(0, 2, 1)

        x = self.embedding(x)  # (B, T, hidden)

        # save attention weights using hook
        def attn_hook(module, input, output):
            self.attn_weights = module.self_attn.attn_output_weights.detach().cpu()

        handle = self.transformer.layers[-1].self_attn.register_forward_hook(attn_hook)
        out = self.transformer(x)
        handle.remove()

        out = out.mean(dim=1)  # global average pooling
        return self.classifier(out)

# ----------------------------
# Training Loop
# ----------------------------
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(y).sum().item()
                val_total += y.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")

    return model

# ----------------------------
# Attention Heatmap
# ----------------------------
def plot_attention_heatmap(model, dataset, num_samples=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    for _ in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        X, y = dataset[idx]
        X = X.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(X)
            pred = outputs.argmax(dim=1).item()

        attn = model.attn_weights[0]  # (heads*B, T, T)
        attn_mean = attn.mean(0).numpy()  # average across heads

        plt.figure(figsize=(10, 6))
        plt.imshow(attn_mean, aspect="auto", cmap="viridis")
        plt.colorbar(label="Attention Weight")
        plt.title(f"Sample {idx} | True: {y.item()} | Pred: {pred}")
        plt.xlabel("Time Steps")
        plt.ylabel("Time Steps")
        plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    data_path = r"sleep-edf-database-expanded-1.0.0\sleep-cassette\processed"
    label_path = r"sleep-edf-database-expanded-1.0.0\sleep-cassette\processed"

    dataset = SleepDataset(data_path, label_path)

    n_classes = 8  # EDF labels 0–7
    input_dim = dataset.data.shape[1]

    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = SleepTransformer(
        input_dim=input_dim,
        num_heads=4,
        num_layers=2,
        hidden_dim=128,
        num_classes=n_classes
    )

    model = train_model(model, train_loader, val_loader, num_epochs=10)

    # Explainability
    plot_attention_heatmap(model, dataset, num_samples=2)
