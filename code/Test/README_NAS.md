# Neural Architecture Search for Sleep Stage Classification

This implementation provides a comprehensive Neural Architecture Search (NAS) framework for time series classification, specifically designed for sleep stage classification with 7 input channels at 100Hz and 30-second windows (3000 time steps).

## Features

- **DARTS (Differentiable Architecture Search)**: Efficient gradient-based architecture search
- **RL-based Search**: Reinforcement learning approach using REINFORCE algorithm
- **Comprehensive Search Space**: Includes convolutions, LSTM, attention mechanisms, and pooling operations
- **Evaluation Tools**: Performance estimation and architecture evaluation utilities

## Architecture

### Files

1. **`nas_search_space.py`**: Defines the search space with various operations:
   - Convolution operations (1D conv with different kernel sizes)
   - Dilated convolutions
   - Depthwise separable convolutions
   - LSTM operations
   - Multi-head self-attention
   - Temporal pooling (max/avg)
   - Identity and zero operations

2. **`darts.py`**: Implements DARTS algorithm:
   - Differentiable architecture search
   - First-order and second-order (unrolled) optimization
   - Architecture parameter optimization

3. **`rl_search.py`**: Implements RL-based search:
   - Policy network (LSTM-based controller)
   - REINFORCE algorithm
   - Architecture sampling and evaluation

4. **`nas_evaluator.py`**: Evaluation utilities:
   - Performance estimation (parameters, FLOPs)
   - Architecture evaluation with full/partial training
   - Zero-cost proxy scores

5. **`main_nas.py`**: Main script to run NAS experiments

## Usage

### Basic Usage

```bash
# Run DARTS search
python code/main_nas.py \
    --data_path /path/to/sleep/data \
    --strategy darts \
    --num_epochs 50 \
    --batch_size 32

# Run RL-based search
python code/main_nas.py \
    --data_path /path/to/sleep/data \
    --strategy rl \
    --num_iterations 100 \
    --eval_epochs 5 \
    --batch_size 32

# Run both strategies
python code/main_nas.py \
    --data_path /path/to/sleep/data \
    --strategy both
```

### Key Arguments

#### Data Arguments
- `--data_path`: Path to sleep data directory
- `--input_channels`: Number of input channels (default: 7)
- `--input_length`: Input sequence length (default: 3000 for 30s at 100Hz)
- `--num_classes`: Number of classes (default: 7)
- `--batch_size`: Batch size (default: 32)

#### Architecture Arguments
- `--num_cells`: Number of cells in network (default: 8)
- `--num_nodes`: Number of nodes per cell (default: 4)
- `--init_channels`: Initial number of channels (default: 64)

#### DARTS Arguments
- `--w_lr`: Learning rate for model weights (default: 0.025)
- `--alpha_lr`: Learning rate for architecture parameters (default: 3e-4)
- `--unrolled`: Use unrolled optimization (second-order DARTS)

#### RL Arguments
- `--num_iterations`: Number of search iterations (default: 100)
- `--eval_epochs`: Number of epochs to train each architecture (default: 5)
- `--reward_type`: Reward type - 'accuracy' or 'loss' (default: 'accuracy')
- `--temperature`: Temperature for policy sampling (default: 1.0)
- `--entropy_coeff`: Entropy coefficient for exploration (default: 0.0001)

#### Training Arguments
- `--num_epochs`: Number of epochs for DARTS (default: 50)
- `--final_train_epochs`: Number of epochs for final architecture training (default: 20)
- `--print_freq`: Print frequency (default: 10)

#### Save Arguments
- `--save_dir`: Directory to save results (default: auto-generated with timestamp)

## Search Space

The search space includes the following operations:

1. **Convolution Operations**:
   - `conv_1x1`: 1x1 convolution
   - `conv_3x1`: 3x1 convolution
   - `conv_5x1`: 5x1 convolution
   - `conv_7x1`: 7x1 convolution

2. **Dilated Convolutions**:
   - `dil_conv_3x1`: Dilated 3x1 convolution
   - `dil_conv_5x1`: Dilated 5x1 convolution

3. **Depthwise Separable Convolutions**:
   - `sep_conv_3x1`: Depthwise separable 3x1 convolution
   - `sep_conv_5x1`: Depthwise separable 5x1 convolution

4. **Pooling Operations**:
   - `max_pool_3x1`: Max pooling with kernel size 3
   - `avg_pool_3x1`: Average pooling with kernel size 3

5. **Sequence Modeling**:
   - `lstm`: Bidirectional LSTM
   - `attention`: Multi-head self-attention

6. **Special Operations**:
   - `identity`: Identity operation (skip connection)
   - `zero`: Zero operation (no connection)

## Network Structure

The network consists of:
1. **Stem**: Initial convolution layer to process input
2. **Cells**: Stack of searchable cells (normal and reduction cells)
3. **Classifier**: Global average pooling and linear classifier

Each cell has:
- Two input nodes (from previous cells)
- Multiple intermediate nodes
- Each node connects to previous nodes via selected operations

## Evaluation

The framework provides several evaluation methods:

1. **Full Evaluation**: Train architecture fully and evaluate
2. **Quick Evaluation**: Train for a few epochs and evaluate (for RL search)
3. **Performance Estimation**: Estimate parameters and FLOPs
4. **Zero-cost Proxies**: SNIP, grad norm, etc. (for fast ranking)

## Output

The script saves:
- Architecture JSON file
- Training results JSON file
- Model weights (for DARTS) or policy weights (for RL)
- Evaluation metrics

## Example Output

```
Final Architecture:
============================================================
Architecture:
============================================================

Cell 0:
  Node 0:
    Edge from node 0 -> conv_3x1
    Edge from node 1 -> sep_conv_5x1
  Node 1:
    Edge from node 0 -> attention
    Edge from node 1 -> lstm
    Edge from node 2 -> identity
...

Final Evaluation Results:
Best Val Accuracy: 85.23%
Final Val Accuracy: 84.56%

Architecture Metrics:
Parameters: 2,345,678
Estimated FLOPs: 1,234,567,890
```

## Notes

- DARTS is typically faster but requires more memory
- RL-based search is more flexible but slower (requires training each architecture)
- For large datasets, consider using smaller `eval_epochs` for RL search
- The framework supports both first-order and second-order DARTS (unrolled)
- Architecture parameters are optimized separately from model weights in DARTS

## Requirements

- PyTorch >= 1.8.0
- NumPy
- Other dependencies from `requirements.txt`

## References

- DARTS: Liu et al., "DARTS: Differentiable Architecture Search", ICLR 2019
- RL-based NAS: Zoph & Le, "Neural Architecture Search with Reinforcement Learning", ICLR 2017

