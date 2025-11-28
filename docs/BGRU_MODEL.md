# BGRU Model Architecture Documentation

This document provides detailed documentation of the Bidirectional GRU (BGRU) model architecture used for BankNifty directional prediction.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Model Components](#model-components)
4. [Data Pipeline](#data-pipeline)
5. [Training Configuration](#training-configuration)
6. [Usage Guide](#usage-guide)
7. [Performance Considerations](#performance-considerations)

---

## Overview

The BGRU model is designed to predict the directional movement (UP/DOWN) of the BankNifty index using historical OHLCV (Open, High, Low, Close, Volume) data. It uses a Bidirectional GRU architecture that processes time series data in both forward and backward directions to capture temporal patterns.

### Key Features

- **Binary Classification**: Predicts UP (1) or DOWN (0) price movement
- **Sequence-based Input**: Uses sliding windows of 60 time steps (15 hours of 15-minute data)
- **Bidirectional Processing**: Captures patterns from both past and future context within each sequence
- **Regularization**: Multiple dropout layers to prevent overfitting

---

## Architecture Design

### Visual Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│                   [batch, 60, 5] (OHLCV)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BGRU LAYER 1                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Forward GRU (128 units) ──────────────────────────────▶│   │
│   │  Backward GRU (128 units) ◀────────────────────────────│   │
│   └─────────────────────────────────────────────────────────┘   │
│                    Output: [batch, 60, 256]                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DROPOUT (0.3)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BGRU LAYER 2                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Forward GRU (64 units) ───────────────────────────────▶│   │
│   │  Backward GRU (64 units) ◀─────────────────────────────│   │
│   └─────────────────────────────────────────────────────────┘   │
│                    Output: [batch, 60, 128]                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DROPOUT (0.3)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAST TIME STEP SELECTION                        │
│                    Output: [batch, 128]                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DENSE LAYER                                 │
│                  32 units, ReLU activation                       │
│                    Output: [batch, 32]                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DROPOUT (0.2)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                │
│                  1 unit, Sigmoid activation                      │
│                    Output: [batch, 1]                            │
│                  (Probability of UP move)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Summary

| Layer | Type | Units | Output Shape | Parameters |
|-------|------|-------|--------------|------------|
| Input | - | 5 | [batch, 60, 5] | 0 |
| BGRU 1 | Bidirectional GRU | 128×2 | [batch, 60, 256] | ~135K |
| Dropout 1 | Dropout | - | [batch, 60, 256] | 0 |
| BGRU 2 | Bidirectional GRU | 64×2 | [batch, 60, 128] | ~99K |
| Dropout 2 | Dropout | - | [batch, 60, 128] | 0 |
| Dense | Linear + ReLU | 32 | [batch, 32] | 4,128 |
| Dropout 3 | Dropout | - | [batch, 32] | 0 |
| Output | Linear + Sigmoid | 1 | [batch, 1] | 33 |
| **Total** | | | | **~231K** |

---

## Model Components

### 1. BGRUModel Class

The core neural network architecture implemented in PyTorch.

```python
class BGRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,      # OHLCV features
        hidden_dim: int = 128,   # First layer hidden size
        num_layers: int = 2,     # Number of GRU layers
        dropout: float = 0.3     # Dropout rate
    )
```

**Key Design Decisions:**

1. **Bidirectional GRUs**: Process sequences in both directions to capture patterns that may depend on future context within the sequence window.

2. **Decreasing Hidden Dimensions**: The second BGRU layer uses 64 units (half of the first layer), creating a funnel-like architecture that compresses information.

3. **Last Time Step Output**: Only the final time step's output is used for classification, as it contains the most recent pattern information.

4. **Multiple Dropout Layers**: Prevents overfitting by randomly zeroing elements during training.

### 2. BGRUPredictor Class

The high-level wrapper for training, prediction, and model management.

```python
class BGRUPredictor:
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: Optional[str] = None  # Auto-detects GPU
    )
```

**Methods:**

| Method | Description |
|--------|-------------|
| `build_model()` | Constructs and initializes the BGRUModel |
| `normalize_data(df, method, window)` | Applies rolling z-score normalization |
| `prepare_sequences(df, sequence_length)` | Creates sliding window sequences |
| `train(train_df, val_df, epochs, ...)` | Training loop with early stopping |
| `predict(test_df)` | Generates predictions on test data |
| `save_model(path)` | Saves model checkpoint |
| `load_model(path)` | Loads model checkpoint |

---

## Data Pipeline

### 1. Input Data Format

The model expects OHLCV data with the following columns:

| Column | Description | Type |
|--------|-------------|------|
| `open` | Opening price | float |
| `high` | Highest price | float |
| `low` | Lowest price | float |
| `close` | Closing price | float |
| `volume` | Trading volume | float |
| `target` | Binary label (1=UP, 0=DOWN) | int |

### 2. Normalization (Rolling Z-Score)

Each feature is normalized using a rolling z-score with a window of 100:

```
z_score = (x - rolling_mean) / rolling_std
```

Where:
- `rolling_mean`: Mean of the last 100 values
- `rolling_std`: Standard deviation of the last 100 values

**Why Rolling Z-Score?**

- Adapts to changing market regimes
- Handles non-stationary financial data
- Preserves relative relationships between features

### 3. Sequence Preparation

Sliding windows are created from the normalized data:

```
For each position i:
    X[i] = data[i : i + sequence_length]  # 60 time steps
    y[i] = target[i + sequence_length - 1]  # Label at end of window
```

**Output Shapes:**
- X: `[num_samples, 60, 5]` - Sequences of OHLCV data
- y: `[num_samples]` - Binary targets

---

## Training Configuration

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 64 | Number of samples per gradient update |
| Learning Rate | 0.001 | Initial learning rate for Adam optimizer |
| Weight Decay | 1e-5 | L2 regularization |
| Epochs | 50 | Maximum training epochs |
| Patience | 10 | Early stopping patience |
| Gradient Clipping | 1.0 | Max gradient norm |

### Optimizer: Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)
```

### Learning Rate Scheduler: ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # Reduce LR by half
    patience=5       # Wait 5 epochs before reducing
)
```

### Loss Function: Binary Cross-Entropy

```python
criterion = nn.BCELoss()
```

For imbalanced datasets, weighted BCE can be used:
```python
def weighted_bce_loss(outputs, targets):
    weights = torch.ones_like(targets)
    weights[targets == 1] = pos_weight
    bce = F.binary_cross_entropy(outputs, targets, reduction='none')
    return (bce * weights).mean()
```

### Early Stopping

Training stops if validation accuracy doesn't improve for `patience` epochs:
- Monitors: Validation accuracy
- Patience: 10 epochs
- Saves: Best model based on validation accuracy

---

## Usage Guide

### Command Line Interface

```bash
# Training
python models/bgru_base.py --train \
    --data_dir data/processed/ \
    --sequence_length 60 \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001

# Prediction
python models/bgru_base.py --predict \
    --data_dir data/processed/ \
    --model_path models/checkpoints/bgru_baseline.pt
```

### Python API

```python
from models.bgru_base import BGRUPredictor
import pandas as pd

# Initialize predictor
predictor = BGRUPredictor(
    input_dim=5,
    hidden_dim=128,
    dropout=0.3
)

# Load data
train_df = pd.read_csv('data/processed/train.csv', index_col=0, parse_dates=True)
val_df = pd.read_csv('data/processed/val.csv', index_col=0, parse_dates=True)

# Train model
history = predictor.train(
    train_df=train_df,
    val_df=val_df,
    epochs=50,
    batch_size=64,
    sequence_length=60
)

# Generate predictions
test_df = pd.read_csv('data/processed/test.csv', index_col=0, parse_dates=True)
predictions, probabilities = predictor.predict(test_df)

# Save model
predictor.save_model('models/checkpoints/bgru_baseline.pt')
```

---

## Performance Considerations

### GPU Acceleration

The model automatically detects and uses GPU if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Memory Optimization

For large datasets:
1. Use smaller batch sizes if GPU memory is limited
2. Consider gradient accumulation for effective larger batch sizes
3. Use mixed precision training (FP16) for faster training

### Inference Speed

- CPU: ~100-500 samples/second
- GPU: ~10,000+ samples/second

### Model Checkpointing

Checkpoints include:
- Model state dictionary
- Model configuration
- Training history
- Timestamp

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'input_dim': 5,
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'training_history': history,
    'saved_at': datetime.now().isoformat()
}
```

---

## Files and Outputs

### Directory Structure

```
models/
├── bgru_base.py              # Main model implementation
├── checkpoints/
│   ├── bgru_baseline.pt      # Model checkpoint
│   ├── training_history.json # Training metrics
│   └── training_curves.png   # Loss/accuracy plots
└── logs/
    └── training.log          # Detailed training logs
```

### Training Outputs

1. **Model Checkpoint** (`bgru_baseline.pt`)
   - PyTorch state dictionary
   - Can be loaded for inference or continued training

2. **Training History** (`training_history.json`)
   - Epoch-wise loss and accuracy
   - Learning rate schedule

3. **Training Curves** (`training_curves.png`)
   - Loss curves (train/val)
   - Accuracy curves (train/val)

---

## References

- [GRU Paper](https://arxiv.org/abs/1406.1078): Learning Phrase Representations using RNN Encoder-Decoder
- [Bidirectional RNNs](https://www.cs.toronto.edu/~hinton/absps/bidirectional.pdf): Bidirectional Recurrent Neural Networks
- [PyTorch GRU Documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
