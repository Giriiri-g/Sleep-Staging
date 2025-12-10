"""
Quick run script for evaluating DARTS architecture
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    
    # Default arguments - modify as needed
    results_path = project_root / "efficient_darts_results" / "efficient_darts_results.json"
    csv_path = project_root / "csv-docs" / "cfs_visit5_selected.csv"
    output_dir = project_root / "darts_evaluation_results"
    checkpoint_path = None  # Set to checkpoint path if you have a trained model
    train_epochs = 50
    skip_training = False  # Set to True if you have a checkpoint
    
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "evaluate_darts_architecture.py"),
        "--results_path", str(results_path),
        "--csv_path", str(csv_path),
        "--output_dir", str(output_dir),
        "--train_epochs", str(train_epochs),
    ]
    
    if checkpoint_path:
        cmd.extend(["--checkpoint_path", str(checkpoint_path)])
    
    if skip_training:
        cmd.append("--skip_training")
    
    print("Running evaluation with command:")
    print(" ".join(cmd))
    print()
    
    # Import and run
    from evaluate_darts_architecture import main
    import sys as sys_module
    sys_module.argv = cmd
    main()

