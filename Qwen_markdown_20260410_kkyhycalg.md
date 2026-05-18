# CRITICAL IMPLEMENTATION REQUIREMENTS

## 1. DATA PREPROCESSING FIRST
- **Step 1:** Load raw ECG signals from `records500` folder
- **Step 2:** Apply complete preprocessing pipeline:
  - Baseline wander removal (EMD)
  - Noise filtering (High-pass, Notch, Wavelet denoising)
  - Z-score normalization
  - R-peak detection and segmentation
- **Step 3:** Save preprocessed signals to new directories:
  - `data/preprocessed/train/` (80% of data)
  - `data/preprocessed/test/` (20% of data)
  - Ensure patient-wise splitting (no data leakage)

## 2. SERIOUS MODEL TRAINING (NO SAMPLE/PLACEHOLDER MODELS)
- **Remove any existing sample/test models** from previous runs
- Build a **PRODUCTION-READY Hybrid CNN-BiLSTM model** with:
  - Proper architecture (see Model Specification below)
  - Correct input shapes for 12-lead, 500Hz ECG data
  - Optimized hyperparameters for >90% accuracy
  
## 3. TRAINING REQUIREMENTS
- Train the model **COMPLETELY** (not just 1-2 epochs for testing)
- Use early stopping with patience=15
- Target: **Minimum 90% validation accuracy**
- Save the best model to `models/best_ecg_model.pth`
- Generate training curves (loss/accuracy plots)

## 4. MODEL SPECIFICATION (REAL IMPLEMENTATION)
```python
class HybridCNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=5):
        # CNN Branch for spatial features
        self.cnn = Sequential(
            Conv1d(12, 64, kernel_size=7, padding=3), BatchNorm1d(64), ReLU(), MaxPool1d(2),
            Conv1d(64, 128, kernel_size=5, padding=2), BatchNorm1d(128), ReLU(), MaxPool1d(2),
            Conv1d(128, 256, kernel_size=3, padding=1), BatchNorm1d(256), ReLU(), GlobalAvgPool1d()
        )
        
        # BiLSTM Branch for temporal features
        self.lstm = BiLSTM(256, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        
        # Classification head
        self.classifier = Sequential(
            Linear(256 + 256, 128), ReLU(), Dropout(0.5),
            Linear(128, num_classes)
        )