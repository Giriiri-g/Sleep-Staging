# NAS Training Features

## ✅ Implemented Features

### 1. Colored Key Information Printers
- **Header printing**: Large colored headers for major sections
- **Section printing**: Colored section separators
- **Info/Success/Warning/Error messages**: Color-coded status messages
- **Key-value pairs**: Formatted key-value information display
- **Metrics**: Colored metric displays with units

### 2. MNE Warnings Suppression
- All MNE warnings are suppressed at module import
- RuntimeWarning and UserWarning from MNE are filtered
- MNE logging level set to ERROR
- Warnings suppressed in dataloader during file reading

### 3. Checkpointing and Resume
- **Automatic checkpointing**: Saves checkpoints every N epochs (configurable)
- **Best model saving**: Automatically saves best validation accuracy model
- **Error recovery**: Saves checkpoint on error for recovery
- **Resume support**: Can resume from checkpoint using `--resume_from` argument
- **Auto-detection**: Automatically detects and resumes from latest checkpoint if available

### 4. Checkpoint Files
- `checkpoint_latest.pth`: Latest checkpoint (saved periodically)
- `checkpoint_best.pth`: Best validation accuracy checkpoint
- `checkpoint_final.pth`: Final checkpoint after training completes
- `checkpoint_error.pth`: Checkpoint saved on error

## Usage

### Basic Training
```bash
python code/main_nas.py --data_path "F:\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette" --strategy darts --batch_size 2 --num_cells 4 --num_nodes 3 --init_channels 16
```

### Resume from Checkpoint
```bash
python code/main_nas.py --data_path "F:\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette" --strategy darts --resume_from "nas_results_darts_20231109_120000/checkpoints/checkpoint_latest.pth"
```

### Custom Checkpoint Frequency
```bash
python code/main_nas.py --data_path "F:\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette" --strategy darts --checkpoint_freq 10
```

## Checkpoint Contents
Each checkpoint contains:
- Model state dictionary
- Optimizer states (both weight and architecture optimizers)
- Scheduler states
- Architecture parameters (alpha)
- Training history
- Best validation accuracy
- Best architecture
- Current epoch number

## Output Features
- **Colored output**: Key information is highlighted with colors
- **Clean output**: No MNE warnings or verbose messages
- **Progress tracking**: Clear progress indicators
- **Error handling**: Graceful error handling with checkpoint saving

## File Structure
```
nas_results_darts_TIMESTAMP/
├── checkpoints/
│   ├── checkpoint_latest.pth
│   ├── checkpoint_best.pth
│   ├── checkpoint_final.pth
│   └── checkpoint_error.pth (if error occurred)
├── darts_architecture.json
├── darts_results.json
└── darts_model.pth
```





