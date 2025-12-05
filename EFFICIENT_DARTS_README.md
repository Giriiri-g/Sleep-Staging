# Efficient DARTS Search for CFS Health Risk Prediction

This implementation provides an efficient Neural Architecture Search (NAS) using DARTS for predicting health risks from the CFS dataset, with constraints on FLOPs, MACs, and model parameters.

## Features

- **Efficiency Constraints**: Tracks and constrains FLOPs, MACs, and parameter counts
- **Attention Layers**: Includes efficient attention operations in the search space
- **Smaller Models**: Optimized for smaller, more efficient architectures
- **Quick Results**: Configured for fast iteration (20 epochs by default)
- **Comprehensive Logging**: Logs dumped to files periodically for monitoring

## Quick Start

### Run the Efficient DARTS Search

```bash
cd code
python run_efficient_darts.py
```

Or with custom arguments:

```bash
python efficient_darts_search.py \
    --data_path ../csv-docs/cfs_visit5_selected.csv \
    --num_epochs 20 \
    --max_flops 50000000 \
    --max_macs 25000000 \
    --max_params 500000 \
    --save_dir efficient_darts_results
```

## Architecture Configuration

The default configuration uses a smaller architecture for efficiency:

- **Cells**: 4 (reduced from 6)
- **Nodes per cell**: 3
- **Initial channels**: 32 (reduced from 64)

## Efficiency Constraints

You can set constraints on:
- **FLOPs**: Maximum floating point operations
- **MACs**: Maximum multiply-accumulate operations
- **Parameters**: Maximum model parameters

Set to `None` to disable a constraint. The system tracks these metrics and logs them, but doesn't enforce hard constraints during gradient-based search (since efficiency requires discretization).

## Search Space

The search space includes efficient operations:

- **Convolutions**: 1x1, 3x1, 5x1, 7x1
- **Separable Convolutions**: Depthwise separable (3x1, 5x1)
- **Dilated Convolutions**: 3x1, 5x1 with dilation
- **Disjoint CNN**: Temporal + spatial convolutions
- **Attention**: Standard (4 heads) and lightweight (2 heads)
- **LSTM**: Bidirectional LSTM
- **Pooling**: Max and average pooling
- **Identity**: Skip connections

## Output Files

The search creates the following in the save directory:

- `darts_search.log`: Continuous logging file
- `logs_epoch_N.json`: Periodic log dumps (every N epochs)
- `checkpoints/`: Model checkpoints
  - `checkpoint_best.pth`: Best model based on validation accuracy
  - `checkpoint_latest.pth`: Latest checkpoint
- `efficient_darts_results.json`: Final results with architecture and metrics

## Monitoring

Logs are written to:
1. **Console**: Real-time progress updates
2. **Log file**: `darts_search.log` with detailed information
3. **JSON dumps**: Periodic snapshots in `logs_epoch_N.json`

Each log entry includes:
- Training and validation metrics (loss, accuracy)
- Efficiency metrics (FLOPs, MACs, parameters)
- Architecture information

## Customization

### Adjust Efficiency Constraints

Edit `run_efficient_darts.py` or pass arguments:

```python
max_flops = 30_000_000   # 30M FLOPs
max_macs = 15_000_000    # 15M MACs
max_params = 300_000      # 300K parameters
```

### Modify Architecture Size

```python
num_cells = 6        # More cells (slower, potentially better)
num_nodes = 4        # More nodes per cell
init_channels = 48   # More initial channels
```

### Change Training Duration

```python
num_epochs = 30      # More epochs for better results
print_freq = 5       # Print every 5 epochs
log_freq = 3         # Log every 3 epochs
```

## Notes

- The efficiency constraints are tracked and logged but not directly enforced in the gradient-based search (since efficiency calculation requires architecture discretization)
- For hard constraints, you can post-process results and select architectures that meet your requirements
- The search prioritizes efficient operations (separable convs, lightweight attention) in the search space
- Smaller architectures (fewer cells, nodes, channels) naturally lead to lower FLOPs/MACs/parameters

## Requirements

- PyTorch
- MNE (for EDF file reading)
- NumPy, Pandas
- Standard Python libraries

See `requirements.txt` for full list.

