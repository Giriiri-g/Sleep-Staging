# Attention Heatmap Visualization for MESA Transformer

This module provides comprehensive attention visualization tools for the MESA Transformer model, enabling explainability analysis of the model's decision-making process.

## Features

The visualization module generates three types of attention heatmaps:

1. **Temporal Attention**: Shows attention patterns within each channel over time (within each epoch)
2. **Channel Attention**: Shows how the model attends to different channels when making predictions
3. **Inter-Epoch Attention**: Shows attention patterns between different epochs in the sequence

## Usage

### Quick Start

```bash
# Simple usage with default settings
python visualize_attention.py
```

### Advanced Usage

```bash
# Using the main script with custom options
python code/visualize_attention_heatmaps.py \
    --checkpoint_path checkpoints_mesa/best_model.pth \
    --preprocessed_dir C:\mesa \
    --csv_path mesa_final.csv \
    --output_dir attention_visualizations \
    --num_samples 5 \
    --channel_names EEG1 EEG2 EEG3 \
    --device cuda
```

### Programmatic Usage

```python
from code.visualize_attention_heatmaps import load_model, visualize_sample_attention
from code.mesa_dataloader import create_mesa_dataloader
import torch

# Load model
model = load_model(
    checkpoint_path="checkpoints_mesa/best_model.pth",
    num_channels=3,
    time_steps=3840,
    seq_len=20,
    num_classes=6,
    device="cuda"
)

# Create dataloader
dataloader = create_mesa_dataloader(
    preprocessed_dir=r"C:\mesa",
    csv_path="mesa_final.csv",
    seq_len=20,
    batch_size=1,
    shuffle=False
)

# Visualize attention for a sample
for features, labels in dataloader:
    visualize_sample_attention(
        model, features, labels,
        channel_names=["EEG1", "EEG2", "EEG3"],
        output_dir="attention_visualizations",
        sample_idx=0,
        seq_len=20
    )
    break
```

## Output Files

For each sample, the following visualizations are generated:

1. **`attention_dashboard_sample_{idx}.png`**: Comprehensive dashboard showing:
   - Inter-epoch attention (full sequence view)
   - Channel attention for selected epochs
   - Temporal attention for selected epochs and channels

2. **`epoch_attention_sample_{idx}.png`**: Detailed inter-epoch attention heatmap with predictions

3. **`channel_attention_sample_{idx}_epoch_{epoch}.png`**: Channel attention for specific epochs

4. **`temporal_attention_sample_{idx}_epoch_{epoch}_ch_{channel}.png`**: Temporal attention for specific epochs and channels

## Understanding the Heatmaps

### Temporal Attention
- **X-axis**: Key positions (time steps within epoch)
- **Y-axis**: Query positions (time steps within epoch)
- **Intensity**: How much each time step attends to other time steps
- **Interpretation**: Shows which parts of the signal are important for prediction

### Channel Attention
- **X-axis**: Key channels
- **Y-axis**: Query channels
- **Intensity**: How much each channel attends to other channels
- **Interpretation**: Shows channel interactions and which channels are most important

### Inter-Epoch Attention
- **X-axis**: Key epochs (in sequence)
- **Y-axis**: Query epochs (in sequence)
- **Intensity**: How much each epoch attends to other epochs
- **Interpretation**: Shows temporal context and dependencies between epochs
- **Labels**: Include true→predicted class labels for each epoch

## Example Output

The comprehensive dashboard provides a quick overview of all attention mechanisms:
- Top row: Inter-epoch attention showing how epochs attend to each other
- Subsequent rows: For each selected epoch, shows channel attention and temporal attention for different channels

## Customization

You can customize the visualization by:
- Adjusting the number of samples to visualize (`--num_samples`)
- Selecting specific epochs to visualize in detail
- Customizing channel names
- Changing colormaps (modify in the code: `viridis`, `plasma`, `YlOrRd`)

## Requirements

- PyTorch
- Matplotlib
- NumPy
- Trained MESA Transformer model checkpoint

