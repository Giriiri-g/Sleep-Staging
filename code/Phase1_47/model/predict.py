import torch
from model import CFSNet
import os

# -------------------------------
# DEVICE
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# PATHS
# -------------------------------
checkpoint_path = r"F:\model\checkpoints\cfsnet_best.pth"
pt_file_path = r"F:\model\800011_preprocessed.pt"   # <-- CHANGE THIS

# -------------------------------
# LOAD CHECKPOINT
# -------------------------------
checkpoint = torch.load(checkpoint_path, map_location=device)

model = CFSNet(
    in_channels=checkpoint["in_channels"],
    num_classes=checkpoint["num_classes"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -------------------------------
# LOAD ONE CFS FILE
# -------------------------------
x = torch.load(pt_file_path)     # shape: [C, T]
x = x.unsqueeze(0).to(device)    # shape: [1, C, T]

# -------------------------------
# PREDICTION
# -------------------------------
with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits)

# -------------------------------
# THRESHOLDING
# -------------------------------
threshold = 0.5
pred = (probs >= threshold).int()

# -------------------------------
# OUTPUT
# -------------------------------
print("Predicted probabilities:")
print(probs.squeeze(0).cpu().numpy())

print("\nPredicted disease labels (0/1):")
print(pred.squeeze(0).cpu().numpy())
