# Sleep Staging Transformer

A hierarchical transformer model for sleep stage classification using PSG data from EDF files.

## Features

- **Hierarchical Transformer Architecture**: Two-tier transformer with local and global encoders
- **Automatic Data Loading**: Reads EDF files and hypnograms from Sleep-EDF database
- **Spectrogram Feature Extraction**: Converts EEG/PSG signals to spectrograms
- **Checkpoint Management**: Automatic saving/loading of model checkpoints
- **Colored Debug Output**: Easy-to-read training progress with colored terminal output
- **Comprehensive Metrics**: Per-class precision, recall, and F1 scores
- **Warning Suppression**: Clean output without MNE/PyTorch warnings

## Installation

Make sure you have the required dependencies:

```bash
pip install torch numpy scipy scikit-learn mne
```

## Usage

### Basic Training

```bash
python code/sleep_staging_transformer.py \
    --data_folder "F:\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette" \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### Resume from Checkpoint

```bash
python code/sleep_staging_transformer.py \
    --data_folder "F:\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette" \
    --resume_from "checkpoints/latest_checkpoint.pt"
```

### Custom Model Architecture

```bash
python code/sleep_staging_transformer.py \
    --data_folder "F:\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette" \
    --hidden_dim 512 \
    --num_heads 16 \
    --num_encoder_layers_local 6 \
    --num_encoder_layers_global 3 \
    --segment_size 15
```

## Arguments

- `--data_folder`: Path to sleep-cassette folder containing EDF files
- `--num_epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--segment_size`: Number of consecutive epochs per sample (default: 10)
- `--hidden_dim`: Transformer hidden dimension (default: 256)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_encoder_layers_local`: Local encoder layers (default: 4)
- `--num_encoder_layers_global`: Global encoder layers (default: 2)
- `--dropout`: Dropout rate (default: 0.1)
- `--checkpoint_dir`: Checkpoint directory (default: "checkpoints")
- `--resume_from`: Path to checkpoint to resume from (default: None)

## Model Architecture

The model consists of three main components:

1. **Local Transformer Encoder**: Processes individual 30-second epochs
2. **Global Transformer Encoder**: Processes sequences of epoch embeddings
3. **Prediction Head**: Classifies each epoch into sleep stages

### Sleep Stages

- W: Wake
- R: REM
- 1: Stage 1
- 2: Stage 2
- 3: Stage 3
- 4: Stage 4

## Checkpoints

Checkpoints are saved in the `checkpoints` directory:
- `checkpoint_epoch_{N}.pt`: Checkpoint for epoch N
- `latest_checkpoint.pt`: Latest checkpoint
- `best_model.pt`: Best model based on validation accuracy

## Output

The training script provides colored debug output:
- **Cyan**: Information messages
- **Blue**: Training progress
- **Magenta**: Validation metrics
- **Green**: Success messages
- **Yellow**: Warnings
- **Red**: Errors

## Data Format

The script expects:
- PSG files: `{subject_id}-PSG.edf`
- Hypnogram files: `{subject_id}*-Hypnogram.edf`

Each epoch is 30 seconds long, and the model processes sequences of consecutive epochs.

## Notes

- The model uses spectrograms with frequency range 0-30 Hz
- Signals are resampled to 100 Hz for consistency
- Unscored epochs are filtered out by default
- The dataset is split 80/20 for training/validation

