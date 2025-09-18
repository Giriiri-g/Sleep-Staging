import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Dataset Loader
# ----------------------------
class SleepDataset(Dataset):
    def __init__(self, data_folder):
        self.epochs, self.labels = [], []
        for f in os.listdir(data_folder):
            if f.endswith("_epochs.npy"):
                base = f.replace("_epochs.npy", "")
                X = np.load(os.path.join(data_folder, f))
                y = np.load(os.path.join(data_folder, base + "_labels.npy"))
                self.epochs.append(X)
                self.labels.append(y)
        self.epochs = np.concatenate(self.epochs, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        print(f"Loaded dataset: {self.epochs.shape}, labels: {self.labels.shape}")

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        X = torch.tensor(self.epochs[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return X, y

# ----------------------------
# Transformer Model
# ----------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, d_ff=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=num_heads, dim_feedforward=d_ff, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        x = self.embedding(x)    # (batch, time, 64)
        x = x.permute(1, 0, 2)   # (time, batch, 64)
        out = self.transformer_encoder(x)
        out = out.mean(dim=0)    # (batch, 64)
        return self.fc(out)

# ----------------------------
# Training Loop
# ----------------------------
def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                pred_labels = preds.argmax(dim=1)
                correct += (pred_labels == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")
    return model

# ----------------------------
# Heatmap Explainability
# ----------------------------
def plot_attention_heatmap(model, sample, class_names=None):
    model.eval()
    with torch.no_grad():
        sample = sample.unsqueeze(0)  # (1, channels, time)
        emb = model.embedding(sample.permute(0, 2, 1))
        emb = emb.permute(1, 0, 2)  # (time, batch, d_model)

        # Pass through transformer encoder layer by layer and capture attention
        attn_weights_all = []
        for layer in model.transformer_encoder.layers:
            src = emb
            attn_output, attn_weights = layer.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
            emb = layer.linear2(layer.dropout(layer.activation(layer.linear1(attn_output)))) + src
            attn_weights_all.append(attn_weights[0].mean(0).cpu().numpy())

        # Plot the last layer attention heatmap
        attn_map = attn_weights_all[-1]
        plt.figure(figsize=(6, 5))
        plt.imshow(attn_map, cmap="viridis")
        plt.colorbar()
        plt.title("Attention Heatmap (last layer)")
        plt.xlabel("Time steps")
        plt.ylabel("Time steps")
        plt.show()

# ----------------------------
# Run Example
# ----------------------------
if __name__ == "__main__":
    data_folder = r"sleep-edf-database-expanded-1.0.0/sleep-cassette/processed"
    dataset = SleepDataset(data_folder)

    # Train/val split
    n = len(dataset)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=16, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=16, shuffle=False)

    # Model init
    input_dim = dataset[0][0].shape[0]  # channels (EEG + age + sex)
    num_classes = len(np.unique(dataset.labels))  # should be 7
    model = TransformerModel(input_dim=input_dim, num_classes=num_classes)

    # Train
    model = train_model(model, train_loader, val_loader, num_epochs=5)

    # Pick one sample for explainability
    sample, label = dataset[0]
    plot_attention_heatmap(model, sample)