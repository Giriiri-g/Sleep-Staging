import os
import numpy as np
from torch.utils.data import Dataset
import torch

class SleepDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # collect all pairs *_epochs.npy + *_labels.npy
        for fname in os.listdir(data_dir):
            if fname.endswith("_epochs.npy"):
                base = fname.replace("_epochs.npy", "")
                epoch_path = os.path.join(data_dir, f"{base}_epochs.npy")
                label_path = os.path.join(data_dir, f"{base}_labels.npy")

                if os.path.exists(label_path):
                    self.samples.append((epoch_path, label_path))

        print(f"Found {len(self.samples)} files with epoch/label pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        epoch_path, label_path = self.samples[idx]

        # load data
        X = np.load(epoch_path)  # shape (n_epochs, n_channels, n_times)
        y = np.load(label_path)  # shape (n_epochs,)

        # map labels into 5 classes
        y = self.map_labels(y)

        # drop any invalid labels (movement/unscored)
        valid_idx = np.isin(y, [0, 1, 2, 3, 4])
        X, y = X[valid_idx], y[valid_idx]

        # convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)  # (n_epochs, n_channels, n_times)
        y = torch.tensor(y, dtype=torch.long)     # (n_epochs,)

        if self.transform:
            X = self.transform(X)

        return X, y

    def map_labels(self, labels):
        mapped = []
        for l in labels:
            if l == 6:   # Wake
                mapped.append(0)
            elif l == 1: # N1
                mapped.append(1)
            elif l == 2: # N2
                mapped.append(2)
            elif l in [3, 5]: # N3 + N4
                mapped.append(3)
            elif l == 4: # REM
                mapped.append(4)
            else: # movement/unscored (7)
                mapped.append(-1)  # mark invalid
        return np.array(mapped)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# ===================== #
# MobileNet1D Backbone
# ===================== #
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class MobileNet1D(nn.Module):
    def __init__(self, num_classes=5, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 512, stride=2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ===================== #
# Training Setup
# ===================== #
def collate_fn(batch):
    X_list, y_list = [], []
    for X, y in batch:
        X_list.append(X)  # (n_epochs, 7, 3000)
        y_list.append(y)  # (n_epochs,)
    X = torch.cat(X_list, dim=0)  # all epochs
    y = torch.cat(y_list, dim=0)
    return X, y

def train_model(train_loader, test_loader, num_epochs=10, lr=1e-3, device="cuda"):
    model = MobileNet1D(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # reshape: (batch, channels=1, 7, 3000)
            X = X.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total

        # ---- Evaluation ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                X = X.unsqueeze(1)
                outputs = model(X)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        test_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    return model


from torch.utils.data import DataLoader

# Load datasets
train_dataset = SleepDataset(r"sleep-edf-database-expanded-1.0.0\sleep-cassette\processed")
test_dataset = SleepDataset(r"sleep-edf-database-expanded-1.0.0\sleep-telemetry\processed")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Train
device = "cuda" if torch.cuda.is_available() else "cpu"
model = train_model(train_loader, test_loader, num_epochs=20, lr=1e-3, device=device)

# Save model
torch.save(model.state_dict(), "mobilenet1d_sleep.pth")
