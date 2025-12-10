import os
import torch

DATA_DIR = r"D:\Sleep-Staging\cfs_preprocessed"

# Helper: try to find channel names in common keys
def extract_channel_names(obj):
    possible_keys = ["channels", "channel_names", "ch_names", "ch", "names"]
    for key in possible_keys:
        if key in obj and isinstance(obj[key], (list, tuple)):
            return list(obj[key])
    return None  # Not found


def inspect_pt_file(path):
    obj = torch.load(path, map_location="cpu")

    # -------------------------
    # Case 1: tensor only
    # -------------------------
    if isinstance(obj, torch.Tensor):
        num_channels = obj.shape[0] if obj.dim() >= 2 else 1
        if num_channels == 7:
            print(f"\n=== {os.path.basename(path)} ===")
            print("Special case: 7 channels detected, likely EEG+EOG+EMG")
        return
        print("Type: Tensor")
        print("Shape:", obj.shape)

        print("Num Channels:", num_channels)

        # No names exist → print generic names
        print("Channel Names:", [f"Channel_{i}" for i in range(num_channels)])
        return

    # -------------------------
    # Case 2: dict format
    # -------------------------
    if isinstance(obj, dict):
        print("Type: Dict")

        # Try reading channel names
        channel_names = extract_channel_names(obj)
        if channel_names:
            print("Channel Names:", channel_names)

        # Print tensor shapes
        for key, item in obj.items():
            print(f" - Key '{key}': type={type(item)}")

            if isinstance(item, torch.Tensor):
                print(f"   Shape={item.shape}")

                if item.dim() >= 2:
                    n_ch = item.shape[0]
                    print("   Num Channels:", n_ch)

                    if channel_names is None:
                        print("   Channel Names:", 
                              [f"Channel_{i}" for i in range(n_ch)])
        return

    # -------------------------
    # Case 3: tuple or list
    # -------------------------
    if isinstance(obj, (tuple, list)):
        print(f"Type: {type(obj)} with {len(obj)} elements")
        for i, item in enumerate(obj):
            print(f" - Element {i}: type={type(item)}")

            if isinstance(item, torch.Tensor):
                print("   Shape:", item.shape)

                if item.dim() >= 2:
                    n_ch = item.shape[0]
                    print("   Num Channels:", n_ch)
                    print("   Channel Names:", 
                          [f"Channel_{k}" for k in range(n_ch)])
        return

    print("Unknown format:", type(obj))


# -------------------------
# Iterate through all .pt files
# -------------------------
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pt")])

print("Found", len(files), ".pt files")

for f in files:
    inspect_pt_file(os.path.join(DATA_DIR, f))