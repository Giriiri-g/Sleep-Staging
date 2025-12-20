"""
Quick script to generate NAS reward curve plot
Run this from the project root directory
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

from plot_nas_reward_curve import plot_reward_curve, load_nas_results

if __name__ == "__main__":
    # Default paths
    input_json = "efficient_darts_results/efficient_darts_results.json"
    output_path = "plots/nas_reward_curve.png"
    
    # Load data
    print(f"Loading NAS results from: {input_json}")
    data = load_nas_results(input_json)
    
    # Create output directory if it doesn't exist
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Plot reward curve
    print(f"Generating reward curve plot...")
    plot_reward_curve(data, str(output_path), show_efficiency=False)
    print(f"Plot saved to: {output_path}")
    
    # Also create version with efficiency metrics
    output_path_efficiency = "plots/nas_reward_curve_with_efficiency.png"
    print(f"\nGenerating reward curve with efficiency metrics...")
    plot_reward_curve(data, str(output_path_efficiency), show_efficiency=True)
    print(f"Plot saved to: {output_path_efficiency}")
    
    print("\nDone!")

