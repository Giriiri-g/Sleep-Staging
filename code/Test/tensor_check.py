import torch
import sys
from pathlib import Path

def print_structure(obj, prefix=""):
    """
    Recursively print shapes and types of PyTorch-loaded objects.
    Supports tensors, dicts, lists, and nested structures.
    """

    if torch.is_tensor(obj):
        print(f"{prefix}Tensor: shape={tuple(obj.shape)}, dtype={obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{prefix}Dict with {len(obj)} keys:")
        for key, value in obj.items():
            print(f"{prefix}  Key: {key}")
            print_structure(value, prefix + "    ")
    elif isinstance(obj, list):
        print(f"{prefix}List with {len(obj)} elements:")
        for idx, item in enumerate(obj):
            print(f"{prefix}  Index {idx}:")
            print_structure(item, prefix + "    ")
    else:
        print(f"{prefix}Unsupported type: {type(obj)}")


def main():
    file_path = r"D:\Sleep-Staging\cfs_preprocessed\800002_preprocessed.pt"
    print(f"Loading {file_path}...\n")
    data = torch.load(file_path, map_location="cpu")

    print("----- ANALYSIS RESULT -----")
    print_structure(data)
    print("---------------------------")


if __name__ == "__main__":
    main()
