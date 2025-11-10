# Neural Architecture Search (NAS) Scripts - Main Concepts Explained

This document explains the main concepts used in the NAS scripts for sleep stage classification.

## Overview

The NAS framework implements two main search strategies:
1. **DARTS (Differentiable Architecture Search)** - Gradient-based search
2. **RL-based Search** - Reinforcement learning using REINFORCE algorithm

Both methods search for optimal neural network architectures for sleep stage classification using 7-channel PSG data (3000 time steps = 30 seconds at 100Hz).

---

## 1. Search Space (`nas_search_space.py`)

### Concept: Define Operations and Network Structure

The search space defines what operations can be used in the architecture and how they're organized.

### Key Components:

#### **Operations (Primitives)**
- **Convolution Operations**: `conv_1x1`, `conv_3x1`, `conv_5x1`, `conv_7x1`
  - 1D convolutions with different kernel sizes for temporal pattern extraction
  
- **Dilated Convolutions**: `dil_conv_3x1`, `dil_conv_5x1`
  - Convolutions with dilation to capture longer-range dependencies
  
- **Depthwise Separable Convolutions**: `sep_conv_3x1`, `sep_conv_5x1`
  - Efficient convolutions that reduce parameters and computation
  
- **Pooling Operations**: `max_pool_3x1`, `avg_pool_3x1`
  - Temporal pooling for dimension reduction
  
- **Sequence Modeling**: 
  - `lstm`: Bidirectional LSTM for sequential dependencies
  - `attention`: Multi-head self-attention mechanism
  
- **Special Operations**:
  - `identity`: Skip connection (residual connection)
  - `zero`: No connection (for pruning)

#### **Cell Structure**
- **Cell**: Basic building block of the network
  - Contains multiple nodes (default: 4)
  - Each node connects to previous nodes via selected operations
  - Two input nodes from previous cells
  - Output is concatenation of all node outputs
  
- **Network Structure**:
  - **Stem**: Initial convolution layer (7 channels → init_channels)
  - **Cells**: Stack of searchable cells (normal and reduction cells)
  - **Reduction Cells**: Double channels at specific positions
  - **Classifier**: Global average pooling + linear layer

#### **Supernet**
- Contains ALL possible operations simultaneously
- Uses weighted combinations during search (DARTS) or discrete selections (RL)
- Architecture parameters (alpha) control which operations are used

---

## 2. DARTS (`darts.py`)

### Concept: Differentiable Architecture Search

DARTS makes architecture search differentiable by:
- Treating architecture selection as continuous optimization
- Using softmax over operation weights (alpha parameters)
- Optimizing architecture parameters and model weights alternately

### Key Concepts:

#### **Bilevel Optimization**
- **Inner Loop**: Update model weights (w) on training set
- **Outer Loop**: Update architecture parameters (alpha) on validation set
- This prevents overfitting to training data

#### **Architecture Parameters (Alpha)**
- Shape: `[num_cells, num_edges, num_ops]`
- Controls the weight of each operation in the weighted sum
- Softmax applied to get probabilities over operations

#### **First-Order vs Second-Order Approximation**
- **First-Order (Default)**: Faster, approximates gradient
  - Updates alpha directly using validation loss gradient
- **Second-Order (Unrolled)**: More accurate, computationally expensive
  - Computes Hessian-vector products
  - Requires unrolling the computation graph

#### **Discretization**
- After search, convert continuous alpha to discrete architecture
- Select operation with highest weight for each edge
- This gives the final architecture

### Training Process:
1. Split data into train and validation sets
2. For each batch:
   - Update model weights (w) on training batch
   - Update architecture parameters (alpha) on validation batch
3. After training, discretize alpha to get final architecture

---

## 3. RL-based Search (`rl_search.py`)

### Concept: Reinforcement Learning for Architecture Search

Uses a policy network to generate architectures and REINFORCE algorithm to learn from rewards.

### Key Components:

#### **Policy Network**
- **LSTM-based Controller**: Generates sequences of operation choices
- Takes previous operation as input (embedded)
- Outputs probability distribution over operations for each edge
- Sequential decision-making process

#### **REINFORCE Algorithm**
- **Policy Gradient Method**: Directly optimizes policy parameters
- **Reward**: Validation accuracy or negative loss after training architecture
- **Baseline**: Moving average of rewards (reduces variance)
- **Advantage**: Reward - Baseline (measures how good the architecture is)

#### **Training Process**:
1. Policy network samples an architecture (sequence of operations)
2. Architecture is converted to a model
3. Model is trained for a few epochs
4. Validation accuracy is computed as reward
5. Policy is updated using REINFORCE:
   - Policy loss = -log_prob(actions) × advantage
   - Entropy regularization encourages exploration
6. Repeat for multiple iterations

#### **Exploration vs Exploitation**
- **Temperature**: Controls randomness in sampling (higher = more exploration)
- **Entropy Regularization**: Encourages policy to explore different architectures
- **Baseline**: Reduces variance in gradient estimates

---

## 4. Architecture Evaluation (`nas_evaluator.py`)

### Concept: Evaluate and Estimate Architecture Performance

### Key Methods:

#### **Full Evaluation**
- Train architecture fully on training set
- Evaluate on validation set
- Returns accuracy, loss, and training history
- Used for final architecture evaluation

#### **Quick Evaluation**
- Train architecture for a few epochs only
- Used during RL search for speed
- Gives approximate performance estimate

#### **Performance Estimation**
- **Parameter Count**: Estimate number of parameters
- **FLOPs Estimation**: Estimate computational cost
- Useful for comparing architectures

#### **Zero-Cost Proxies**
- **Gradient Norm**: Measure of how well architecture can learn
- **SNIP Score**: Connection sensitivity
- Fast methods to rank architectures without training

---

## 5. Main Script (`main_nas.py`)

### Concept: Orchestrate the NAS Process

### Workflow:

1. **Data Loading**:
   - Load sleep staging dataset
   - Split into train/val/test sets
   - Create data loaders

2. **Search Strategy Selection**:
   - **DARTS**: Create supernet, run DARTS trainer
   - **RL**: Create policy network, run RL trainer
   - **Both**: Run both and compare results

3. **Search Execution**:
   - Run search for specified epochs/iterations
   - Track best architecture
   - Save intermediate results

4. **Final Evaluation**:
   - Train discovered architecture fully
   - Evaluate on validation set
   - Estimate performance metrics

5. **Results Saving**:
   - Save architecture (JSON)
   - Save training results
   - Save model weights

---

## Key Design Decisions

### 1. **Cell-Based Architecture**
- Reusable building blocks
- Allows for hierarchical search
- Reduces search space size

### 2. **Weight Sharing (DARTS)**
- All operations share weights in supernet
- More efficient than training each architecture separately
- Enables gradient-based optimization

### 3. **Bilevel Optimization (DARTS)**
- Separates architecture search from weight optimization
- Prevents overfitting to training data
- Uses validation set for architecture selection

### 4. **REINFORCE with Baseline (RL)**
- Reduces variance in gradient estimates
- Moves baseline reduces bias
- Entropy regularization encourages exploration

### 5. **Quick Evaluation (RL)**
- Trains architectures for only a few epochs
- Speeds up search process
- Trade-off between speed and accuracy

---

## Mathematical Foundations

### DARTS Loss:
```
L_train(w, alpha): Loss on training set
L_val(w*, alpha): Loss on validation set

Optimize: min_alpha L_val(w*, alpha)
Subject to: w* = argmin_w L_train(w, alpha)
```

### REINFORCE Update:
```
∇θ J(θ) = E[∇θ log π(a|s) × (R - b)]

Where:
- θ: Policy parameters
- π(a|s): Policy probability of action a given state s
- R: Reward
- b: Baseline
```

### Architecture Parameter Update (DARTS):
```
α ← α - η_α ∇_α L_val(w*, α)

Where w* is updated on training set:
w* ← w* - η_w ∇_w L_train(w, α)
```

---

## Advantages and Limitations

### DARTS:
**Advantages**:
- Fast convergence (gradient-based)
- Efficient (weight sharing)
- Continuous optimization

**Limitations**:
- Memory intensive (stores all operations)
- May not find optimal discrete architecture
- Sensitive to hyperparameters

### RL-based Search:
**Advantages**:
- Flexible (can use any reward function)
- Can incorporate constraints
- Direct optimization of discrete architectures

**Limitations**:
- Slow (requires training each architecture)
- High variance (policy gradients)
- Requires many iterations

---

## Application to Sleep Staging

### Task-Specific Adaptations:

1. **Time Series Input**: 
   - 7 channels (EEG, EOG, EMG, etc.)
   - 3000 time steps (30 seconds at 100Hz)
   - 1D convolutions for temporal patterns

2. **Multi-class Classification**:
   - 7 sleep stages (W, R, 1, 2, 3, 4, ?)
   - Cross-entropy loss
   - Accuracy as evaluation metric

3. **Sequence Modeling**:
   - LSTM and attention operations
   - Capture temporal dependencies
   - Important for sleep stage transitions

4. **Efficiency Considerations**:
   - Depthwise separable convolutions
   - Pooling operations
   - Balance accuracy and computational cost

---

## Usage Summary

### Running DARTS:
```python
python main_nas.py --strategy darts --data_path <path> --num_epochs 50
```

### Running RL Search:
```python
python main_nas.py --strategy rl --data_path <path> --num_iterations 100
```

### Key Parameters:
- `num_cells`: Number of cells in network (default: 8)
- `num_nodes`: Number of nodes per cell (default: 4)
- `init_channels`: Initial number of channels (default: 64)
- `w_lr`: Learning rate for weights (DARTS)
- `alpha_lr`: Learning rate for architecture parameters (DARTS)
- `eval_epochs`: Epochs to train each architecture (RL)

---

## Conclusion

The NAS framework provides a comprehensive solution for automatic architecture search in sleep stage classification. It combines:
- **Flexible search space** with diverse operations
- **Efficient search strategies** (DARTS and RL)
- **Comprehensive evaluation** tools
- **Task-specific adaptations** for time series classification

The implementation follows state-of-the-art NAS methods while being tailored for the specific requirements of sleep staging with multi-channel PSG data.

