"""
ECG Diagnostic System — Central Configuration
==============================================
Single source of truth for all paths, signal parameters,
class mappings, preprocessing options, model architecture,
and training hyperparameters.
"""

import os
import torch

# --- Project Paths -----------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
RECORDS_DIR = os.path.join(DATASET_DIR, "records500")
DATABASE_CSV = os.path.join(DATASET_DIR, "ptbxl_database.csv")
SCP_STATEMENTS_CSV = os.path.join(DATASET_DIR, "scp_statements.csv")

# Preprocessed data output
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "preprocessed")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Model checkpoints & metrics
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "acc59_cnn-bilstm_hybrid.pth")
METRICS_PATH = os.path.join(MODEL_DIR, "performance_metrics.json")

# --- Signal Parameters ------------------------------------------------------
SAMPLING_RATE = 500          # Hz  (records500)
SIGNAL_LENGTH = 5000         # 10 seconds × 500 Hz
NUM_LEADS = 12
LEAD_NAMES = [
    'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
]

# --- 5-Class Diagnostic Mapping ---------------------------------------------
CLASS_MAP = {
    'NORM': 0,   # Normal ECG
    'MI':   1,   # Myocardial Infarction
    'CD':   2,   # Conduction Disturbance
    'STTC': 3,   # ST/T Change
    'HYP':  4,   # Hypertrophy
}

CLASS_NAMES = {
    0: 'Normal ECG',
    1: 'Myocardial Infarction (MI)',
    2: 'Conduction Disturbance (CD)',
    3: 'ST/T Change (STTC)',
    4: 'Hypertrophy (HYP)',
}

NUM_CLASSES = 5

# --- Data Splitting ---------------------------------------------------------
TRAIN_RATIO = 0.80
TEST_RATIO  = 0.20

# Number of records to process (set None for FULL dataset ~21,800 records)
# Use a subset for faster local runs on limited hardware.
SAMPLE_SIZE = 2000

# --- Preprocessing Options --------------------------------------------------
APPLY_EMD         = False      # Set True for best quality (very slow ~10s/lead)
EMD_MAX_ITER      = 100        # Max EMD iterations
WAVELET_DENOISE   = 'sym4'    # Wavelet family for denoising
WAVELET_LEVEL     = 4          # Decomposition level for denoising
NOTCH_FREQ        = 50.0       # Power-line frequency (Hz)
HIGHPASS_CUTOFF   = 0.5        # High-pass filter cutoff (Hz)
DERIVATIVE_WEIGHT = 0.3        # Derivative enhancement scaling

# --- Feature Extraction -----------------------------------------------------
FFT_FEATURES_PER_LEAD = 20
DWT_FEATURES_PER_LEAD = 15
DWT_WAVELET           = 'db4'
DWT_LEVEL             = 5

# Derived feature dimensions
MORPH_FEATURE_DIM  = 14
FFT_FEATURE_DIM    = NUM_LEADS * FFT_FEATURES_PER_LEAD     # 240
DWT_FEATURE_DIM    = NUM_LEADS * DWT_FEATURES_PER_LEAD     # 180
FUSED_FEATURE_DIM  = MORPH_FEATURE_DIM + FFT_FEATURE_DIM + DWT_FEATURE_DIM  # 434

# --- Training Hyperparameters -----------------------------------------------
BATCH_SIZE          = 32
LEARNING_RATE       = 1e-3
NUM_EPOCHS          = 50
EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY        = 1e-4
GRAD_CLIP_NORM      = 1.0

# --- Model Architecture -----------------------------------------------------
CNN_CHANNELS        = [64, 128, 256]
CNN_KERNELS         = [7, 5, 3]
LSTM_HIDDEN         = 128
LSTM_LAYERS         = 2
DROPOUT             = 0.3
CLASSIFIER_DROPOUT  = 0.5

# --- Device ------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Create Directories -----------------------------------------------------
for _d in [TRAIN_DIR, TEST_DIR, MODEL_DIR]:
    os.makedirs(_d, exist_ok=True)
