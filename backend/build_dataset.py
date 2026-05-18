"""
ECG Diagnostic System — Offline Dataset Builder
================================================
Processes the full PTB-XL dataset (records500) through the complete pipeline:

  1. Load raw ECG records + map SCP codes → 5-class labels
  2. Preprocess each signal (EMD → HP → Notch → Wavelet → Z-score → Derivative)
  3. Extract features (Morphological + FFT + DWT)
  4. Fuse features into unified 434-dim vectors
  5. Patient-wise stratified split: 80% Train / 20% Test
  6. Save to .npy files for efficient training

Output structure:
  data/preprocessed/
    train/
      X_train.npy     — Preprocessed signals   (N_train, 12, 5000)
      F_train.npy     — Fused feature vectors   (N_train, 434)
      y_train.npy     — Labels                  (N_train,)
      metadata.csv    — Patient IDs, filenames, labels
    test/
      X_test.npy      — Preprocessed signals   (N_test, 12, 5000)
      F_test.npy      — Fused feature vectors   (N_test, 434)
      y_test.npy      — Labels                  (N_test,)
      metadata.csv    — Patient IDs, filenames, labels
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import shutil

from config import (
    DATASET_DIR, DATABASE_CSV, SCP_STATEMENTS_CSV,
    NUM_LEADS, SIGNAL_LENGTH, CLASS_MAP, CLASS_NAMES,
    TRAIN_DIR, TEST_DIR, SAMPLE_SIZE, APPLY_EMD,
    FUSED_FEATURE_DIM, TEST_RATIO
)
from preprocessing import ECGPreprocessor
from feature_extraction import FeatureExtractor
from feature_fusion import FeatureFuser


# --- Metadata Loading --------------------------------------------------------

def load_metadata():
    """Load PTB-XL database CSV and diagnostic SCP statements."""
    print("-" * 50)
    print("  STEP 1 : Loading Metadata")
    print("-" * 50)

    df = pd.read_csv(DATABASE_CSV, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    scp = pd.read_csv(SCP_STATEMENTS_CSV, index_col=0)
    scp_diag = scp[scp.diagnostic == 1.0]

    print(f"  Total ECG records   : {len(df)}")
    print(f"  Diagnostic SCP codes: {len(scp_diag)}")
    return df, scp_diag


def map_to_superclass(scp_codes, scp_diag):
    """Map a record's SCP codes to diagnostic superclass labels."""
    classes = set()
    for code, confidence in scp_codes.items():
        if code in scp_diag.index:
            diag_class = scp_diag.loc[code].diagnostic_class
            if isinstance(diag_class, str) and diag_class in CLASS_MAP:
                classes.add(diag_class)
    return list(classes)


def assign_primary_label(class_list):
    """
    Assign a single primary label using clinical priority.
    Priority: MI > CD > STTC > HYP > NORM
    """
    if not class_list:
        return None
    for p in ['MI', 'CD', 'STTC', 'HYP', 'NORM']:
        if p in class_list:
            return CLASS_MAP[p]
    return None


# --- Raw Signal Loading ------------------------------------------------------

def load_raw_signal(filename):
    """Load a single WFDB record (records500) from the dataset."""
    filepath = os.path.join(DATASET_DIR, filename)
    try:
        signal, _ = wfdb.rdsamp(filepath)
        return signal  # (5000, 12)
    except Exception:
        return None


# --- Main Builder -------------------------------------------------------------

def build_dataset():
    """
    Complete dataset building pipeline.
    Processes PTB-XL records → preprocessed train/test splits with features.
    """
    print("\n" + "=" * 50)
    print("  ECG DATASET BUILDER")
    print("=" * 50 + "\n")

    # -- Step 1: Load metadata & map labels ------------------------------
    df, scp_diag = load_metadata()

    print("\n" + "-" * 50)
    print("  STEP 2 : Mapping Diagnostic Labels")
    print("-" * 50)

    df['superclass_list'] = df.scp_codes.apply(
        lambda x: map_to_superclass(x, scp_diag)
    )
    df['label'] = df['superclass_list'].apply(assign_primary_label)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    print(f"  Records with valid labels: {len(df)}")
    print("  Class distribution:")
    for cls_name, cls_id in CLASS_MAP.items():
        count = (df['label'] == cls_id).sum()
        print(f"    {cls_name} (Class {cls_id}): {count}")

    # -- Optional: use subset for faster local processing ----------------
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
        print(f"\n  Using subset of {SAMPLE_SIZE} records ...")
        df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), int(SAMPLE_SIZE * len(x) / len(df))),
                random_state=42
            )
        ).reset_index(drop=True)

    print(f"  Total records to process: {len(df)}")

    # -- Step 3: Preprocess signals --------------------------------------
    print("\n" + "-" * 50)
    print("  STEP 3 : Preprocessing Signals")
    print("-" * 50)

    preprocessor = ECGPreprocessor()
    extractor = FeatureExtractor()
    fuser = FeatureFuser()

    signals = []
    fused_features = []
    valid_indices = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
        sig = load_raw_signal(row['filename_hr'])

        if sig is None or sig.shape != (SIGNAL_LENGTH, NUM_LEADS):
            skipped += 1
            continue

        # Transpose to (12, 5000) for processing
        sig = sig.T.astype(np.float32)

        # Preprocess (EMD → HP → Notch → Wavelet → Z-score → Derivative)
        sig_processed = preprocessor.preprocess(sig, apply_emd=APPLY_EMD)

        # Extract features (Morphological + FFT + DWT)
        feat_dict = extractor.extract_all(sig_processed)

        # Fuse features (concatenation → 434-dim vector)
        fused = fuser.fuse(feat_dict)

        signals.append(sig_processed)
        fused_features.append(fused)
        valid_indices.append(idx)

    if skipped > 0:
        print(f"  Skipped {skipped} invalid records")

    signals = np.array(signals, dtype=np.float32)
    fused_features = np.array(fused_features, dtype=np.float32)
    df = df.loc[valid_indices].reset_index(drop=True)
    labels = df['label'].values
    patient_ids = df['patient_id'].values

    print(f"  Processed signals shape  : {signals.shape}")
    print(f"  Fused features shape     : {fused_features.shape}")

    # -- Step 4: Patient-wise stratified split (80/20) -------------------
    print("\n" + "-" * 50)
    print("  STEP 4 : Patient-Wise Split (80% Train / 20% Test)")
    print("-" * 50)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=42)
    train_idx, test_idx = next(gss.split(signals, labels, groups=patient_ids))

    X_train = signals[train_idx]
    F_train = fused_features[train_idx]
    y_train = labels[train_idx]

    X_test = signals[test_idx]
    F_test = fused_features[test_idx]
    y_test = labels[test_idx]

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples : {len(X_test)}")

    # Train class distribution
    print("  Train class distribution:")
    for cls_name, cls_id in CLASS_MAP.items():
        print(f"    {cls_name}: {np.sum(y_train == cls_id)}")

    # -- Step 5: Save to disk --------------------------------------------
    print("\n" + "-" * 50)
    print("  STEP 5 : Saving Preprocessed Data")
    print("-" * 50)

    # Train
    np.save(os.path.join(TRAIN_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(TRAIN_DIR, "F_train.npy"), F_train)
    np.save(os.path.join(TRAIN_DIR, "y_train.npy"), y_train)

    train_meta = df.iloc[train_idx][['patient_id', 'filename_hr', 'label']].copy()
    train_meta.to_csv(os.path.join(TRAIN_DIR, "metadata.csv"), index=False)

    # Test
    np.save(os.path.join(TEST_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(TEST_DIR, "F_test.npy"), F_test)
    np.save(os.path.join(TEST_DIR, "y_test.npy"), y_test)

    test_meta = df.iloc[test_idx][['patient_id', 'filename_hr', 'label']].copy()
    test_meta.to_csv(os.path.join(TEST_DIR, "metadata.csv"), index=False)

    print(f"  Saved to: {TRAIN_DIR}")
    print(f"  Saved to: {TEST_DIR}")
    
    # Copy raw test files for frontend testing
    print("\n  Copying raw test files to 'frontend_test_samples' for easy UI testing...")
    test_samples_dir = os.path.join(os.path.dirname(TRAIN_DIR), "frontend_test_samples")
    os.makedirs(test_samples_dir, exist_ok=True)
    
    for _, row in tqdm(test_meta.iterrows(), total=len(test_meta), desc="  Copying"):
        base_filename = row['filename_hr']
        full_path = os.path.join(DATASET_DIR, base_filename)
        # Copy .hea and .dat
        for ext in ['.hea', '.dat']:
            src = full_path + ext
            dst = os.path.join(test_samples_dir, os.path.basename(full_path) + ext)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
    print(f"  Raw test files saved to: {test_samples_dir}")

    print("\n" + "=" * 50)
    print("  [OK] DATASET BUILD COMPLETE")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    build_dataset()
