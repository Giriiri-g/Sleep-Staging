"""
Memory management utilities for GPU training
"""
import torch
import gc


def clear_gpu_cache():
    """Clear GPU cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - reserved
    }


def print_gpu_memory():
    """Print current GPU memory usage"""
    info = get_gpu_memory_info()
    if info:
        print(f"GPU Memory - Allocated: {info['allocated_gb']:.2f} GB, "
              f"Reserved: {info['reserved_gb']:.2f} GB, "
              f"Free: {info['free_gb']:.2f} GB")


def set_memory_fraction(fraction=0.9):
    """Set memory fraction for PyTorch CUDA"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)

