## 📋 Command Execution Order

---

### **Step 1: Data Preparation**

```powershell
# 1. Fetch and clean raw data (creates train.csv, val.csv, test.csv)
python data/data_loader.py --start_date 2007-10-01 --end_date 2025-11-27 --source csv --csv_path .\data\raw\Nifty_Bank_Historical_Data_2007-08-01_2025-11-28.csv --interval 1d
```

---

### **Step 2: Feature Engineering**

```powershell
# 2. Add technical indicators (RSI, MACD, Bollinger Bands, etc.)
python features/technical.py --input data/processed/train.csv --output data/processed/train_tech.csv
python features/technical.py --input data/processed/val.csv --output data/processed/val_tech.csv --load-scaler
python features/technical.py --input data/processed/test.csv --output data/processed/test_tech.csv --load-scaler

# 3. Add temporal features (hour, day of week, etc.)
python features/temporal.py --input data/processed/train_tech.csv --output data/processed/train_temp.csv
python features/temporal.py --input data/processed/val_tech.csv --output data/processed/val_temp.csv
python features/temporal.py --input data/processed/test_tech.csv --output data/processed/test_temp.csv

# 4. Add price action features (candlestick patterns, support/resistance)
python features/price_action.py --input data/processed/train_temp.csv --output data/processed/train_final.csv
python features/price_action.py --input data/processed/val_temp.csv --output data/processed/val_final.csv
python features/price_action.py --input data/processed/test_temp.csv --output data/processed/test_final.csv
```

---

### **Step 3: Model Training (Choose ONE option)**

**Option A: BGRU Baseline (OHLCV only)**
```powershell
python models/bgru_base.py --train --sequence_length 60 --epochs 100 --lr 0.005 --batch_size 64
```

**Option B: BGRU with All Features**
```powershell
python models/bgru_base.py --train --data_dir data/processed/ --all_features --sequence_length 60 --epochs 50
```

**Option C: Hybrid BGRU (Sequential + Static features)**
```powershell
python models/bgru_hybrid.py --train --data_dir data/processed/ --sequence_length 60 --epochs 50
```

---
### **Step 4: Hyperparameter Optimization (Optional)**

```powershell
python models/optimize_hyperparams.py --data data/processed/train_final.csv --val_data data/processed/val_final.csv --all_features --n_trials 30
```

---

### **Step 5: Feature Selection (Optional)**

```powershell
python features/feature_selection.py --model models/checkpoints/bgru_hybrid.pt --data data/processed/test_final.csv --output_dir features/output
```

---

### **Step 6: Ensemble Training (Optional)**

```powershell
python models/ensemble.py --train --bgru_model models/checkpoints/bgru_hybrid.pt --data_dir data/processed/ --selected_features_path features/output/selected_features.txt --hyperparams_path models/best_hyperparams.json --sequence_length 40
```

---

### **Step 7: Evaluation**

**Baseline evaluation:**
```powershell
python evaluation/evaluate_baseline.py --model models/checkpoints/bgru_baseline.pt --data data/processed/test.csv
```

**Final ensemble evaluation:**
```powershell
python evaluation/evaluate_final.py --model models/checkpoints/ensemble_model.pkl --data data/processed/test_final.csv --sequence_length 40 --lot_size 35
```

---

### **Step 8: Prediction**

```powershell
python models/bgru_base.py --predict --model_path models/checkpoints/bgru_baseline.pt --data_dir data/processed/
```

---

### 📊 **Summary Table**

| Step | Script | Purpose | Input | Output |
|------|--------|---------|-------|--------|
| 1 | data_loader.py | Clean & split data | `data/raw/*.csv` | `train.csv`, `val.csv`, `test.csv` |
| 2 | technical.py | Add indicators | `*.csv` | `*_tech.csv` |
| 3 | temporal.py | Add time features | `*_tech.csv` | `*_temp.csv` |
| 4 | price_action.py | Add patterns | `*_temp.csv` | `*_final.csv` |
| 5 | `models/bgru_*.py --train` | Train model | `*_final.csv` | `*.pt` checkpoint |
| 6 | optimize_hyperparams.py | Tune hyperparams | Training data | Best params |
| 7 | feature_selection.py | Select features | Model + data | Feature rankings |
| 8 | ensemble.py | Build ensemble | Multiple models | Ensemble model |
| 9 | `evaluation/evaluate_*.py` | Evaluate | Model + test data | Reports & plots |
