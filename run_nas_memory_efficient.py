"""
Memory-efficient NAS run script
This script runs NAS with reduced memory footprint settings
"""
import subprocess
import sys

# Memory-efficient settings
cmd = [
    sys.executable,
    "code/main_nas.py",
    "--data_path", r"F:\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette",
    "--strategy", "both",
    "--batch_size", "2",  # Very small batch size
    "--num_cells", "4",   # Fewer cells
    "--num_nodes", "3",   # Fewer nodes
    "--init_channels", "16",  # Smaller initial channels
    "--num_epochs", "20",  # Fewer epochs for faster testing
    "--num_iterations", "30",  # Fewer RL iterations
    "--eval_epochs", "2",  # Fewer eval epochs
    "--final_train_epochs", "10"
]

print("Running NAS with memory-efficient settings...")
print("Command:", " ".join(cmd))
print("\nNote: If you still get out-of-memory errors, try:")
print("1. Using --strategy darts or --strategy rl (instead of both)")
print("2. Further reducing batch_size to 1")
print("3. Removing attention operations from the search space")
print("4. Using CPU instead of GPU (modify device selection in code)\n")

subprocess.run(cmd)







