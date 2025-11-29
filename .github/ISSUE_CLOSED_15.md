# Issue #15 Resolution

## Problem
RuntimeError in `feature_selection.py` when running with `bgru_hybrid` model:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x65 and 0x64)
```

## Root Cause
The model checkpoint `bgru_hybrid.pt` was saved with incorrect metadata:
- `n_static_features` was set to 0 in the weights (static_fc1.weight shape was [64, 0])
- But `static_columns` had 65 entries in the checkpoint

This mismatch caused the model to try to process 65 static features through a layer expecting 0 inputs.

## Fix Applied
Modified `load_model()` in `models/bgru_hybrid.py` to:
1. Detect the actual `n_static_features` from the saved weight shapes
2. If weights have 0 input features, warn the user and clear `static_columns`
3. Allow the model to load and function using only OHLCV features

## Note
The current model checkpoint was trained without static features. To use static features for predictions, the model needs to be retrained with proper static feature configuration.
