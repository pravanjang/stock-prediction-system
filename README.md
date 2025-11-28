# BankNifty Directional Prediction System

A comprehensive machine learning system for predicting the directional movement of BankNifty index using technical indicators, price action patterns, and hybrid modeling approaches.

## 📂 Project Structure

```
├── data/
│   ├── raw/                 # Downloaded OHLCV data
│   ├── processed/           # Cleaned & featured data
│   └── data_loader.py       # Data fetching, cleaning & preprocessing script
│
├── features/
│   ├── technical.py         # Technical indicators implementation
│   ├── temporal.py          # Time-based features (e.g., time of day, day of week)
│   └── price_action.py      # Candlestick patterns and price action logic
│
├── models/
│   ├── bgru_base.py         # Bidirectional GRU baseline model
│   ├── timesfm_base.py      # TimesFM model integration
│   ├── hybrid_model.py      # Feature fusion architecture
│   └── train.py             # Model training loop
│
├── evaluation/
│   ├── metrics.py           # Performance metrics (Accuracy, Precision, etc.)
│   └── backtest.py          # Trading simulation and backtesting engine
│
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_phase1.ipynb      # BGRU Baseline experiments
│   └── 03_phase2.ipynb      # Phase 2 experiments
│
├── docs/
│   └── BGRU_MODEL.md        # BGRU model architecture documentation
│
├── configs/
│   └── config.yaml          # Configuration for hyperparameters & paths
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stock-prediction-system
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Pipeline

The system includes a robust data loading and processing module located in `data/data_loader.py`.

### Features
- **Fetch Data**: Downloads historical data from Yahoo Finance (`^NSEBANK`) or loads from a local CSV.
- **Cleaning**:
  - Removes duplicate timestamps.
  - Filters for Indian Market trading hours (9:15 AM - 3:30 PM IST).
  - Handles missing values and invalid OHLC relationships.
  - Flags abnormal price gaps (>5%).
- **Splitting**: Splits data into Training, Validation, and Test sets chronologically.

### Usage

To fetch and process data via the command line:

```bash
python data/data_loader.py --start_date 2023-01-01 --end_date 2023-12-31 --interval 15m
```

**Arguments:**
- `--start_date`: Start date (YYYY-MM-DD)
- `--end_date`: End date (YYYY-MM-DD)
- `--interval`: Data timeframe (default: `15m`)
- `--source`: Data source, either `yfinance` or `csv` (default: `yfinance`)
- `--output_dir`: Directory to save processed files (default: `data/processed/`)
- `--train_ratio`: Ratio of data for training (default: `0.7`)
- `--val_ratio`: Ratio of data for validation (default: `0.15`)

## 🧠 BGRU Model

The system includes a Bidirectional GRU (BGRU) baseline model for directional prediction. See [docs/BGRU_MODEL.md](docs/BGRU_MODEL.md) for detailed architecture documentation.

### Architecture Overview

```
Input: [batch, 60, 5] (60 timesteps × 5 OHLCV features)
    → BGRU(128 units, bidirectional) → Dropout(0.3)
    → BGRU(64 units, bidirectional) → Dropout(0.3)
    → Dense(32, ReLU) → Dropout(0.2)
    → Dense(1, Sigmoid)
```

### Quick Start

**Training:**
```bash
python models/bgru_base.py --train --data_dir data/processed/ --sequence_length 60
```

**Python API:**
```python
from models.bgru_base import BGRUPredictor

predictor = BGRUPredictor()
predictor.train(train_df, val_df, epochs=50, batch_size=64)
predictions, probabilities = predictor.predict(test_df)
```

### Model Features

- **Input**: Rolling normalized OHLCV sequences (60 time steps)
- **Output**: Probability of upward price movement
- **Training**: Adam optimizer with early stopping and LR scheduler
- **Regularization**: Dropout layers and gradient clipping
- **Checkpointing**: Auto-saves best model and training history

## 🛠️ Modules Overview

### Features (`features/`)
- **technical.py**: Will contain implementations of indicators like RSI, MACD, Bollinger Bands, etc.
- **temporal.py**: Will extract time-based features useful for intraday patterns.
- **price_action.py**: Will identify candlestick patterns (Doji, Hammer, Engulfing, etc.).

### Models (`models/`)
- **bgru_base.py**: Bidirectional GRU baseline model for directional prediction.
- **timesfm_base.py**: Integration with Google's TimesFM or similar time-series foundation models.
- **hybrid_model.py**: Architecture to combine time-series embeddings with tabular technical features.
- **train.py**: Script to manage the training lifecycle.

### Evaluation (`evaluation/`)
- **metrics.py**: Custom metrics for financial ML (Directional Accuracy, Sharpe Ratio, etc.).
- **backtest.py**: Event-driven or vectorised backtesting engine to simulate trading strategies.

## 📓 Notebooks

- **01_eda.ipynb**: Step-by-step exploratory data analysis with visualizations.
- **02_phase1.ipynb**: BGRU baseline model building with debugging-friendly code.
- **03_phase2.ipynb**: Advanced modeling and feature engineering experiments.

## 📚 Documentation

- [BGRU Model Architecture](docs/BGRU_MODEL.md): Detailed documentation of the BGRU model design, training configuration, and usage.

## ⚙️ Configuration

The `configs/config.yaml` file is used to centralize configuration settings such as:
- Data paths
- Model hyperparameters (learning rate, batch size, layers)
- Feature lists
- Backtest parameters

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
