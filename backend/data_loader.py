"""
ECG Diagnostic System — Data Loader
====================================
PyTorch Dataset and DataLoader for training the Hybrid CNN-BiLSTM model.
Loads preprocessed .npy files (signals + fused features + labels) via
memory-mapped access for RAM efficiency.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    TRAIN_DIR, TEST_DIR, BATCH_SIZE,
    NUM_CLASSES, DEVICE
)


class ECGDataset(Dataset):
    """
    PyTorch Dataset that loads preprocessed ECG data from .npy files.

    Each sample returns:
      - signal:   (12, 5000) float32   — preprocessed 12-lead ECG
      - features: (434,)    float32   — fused feature vector
      - label:    int64               — class label (0–4)
    """

    def __init__(self, data_dir, split='train'):
        prefix = split  # 'train' or 'test'

        sig_path  = os.path.join(data_dir, f"X_{prefix}.npy")
        feat_path = os.path.join(data_dir, f"F_{prefix}.npy")
        lbl_path  = os.path.join(data_dir, f"y_{prefix}.npy")

        if not os.path.exists(sig_path):
            raise FileNotFoundError(
                f"Preprocessed data not found at {data_dir}.\n"
                "Run 'python build_dataset.py' first."
            )

        # Memory-mapped for large datasets
        self.signals  = np.load(sig_path, mmap_mode='r')
        self.labels   = np.load(lbl_path, mmap_mode='r')

        # Features file may not exist yet (backward compat)
        if os.path.exists(feat_path):
            self.features = np.load(feat_path, mmap_mode='r')
        else:
            self.features = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = torch.FloatTensor(np.array(self.signals[idx]))
        label  = torch.LongTensor([self.labels[idx]])[0]

        if self.features is not None:
            features = torch.FloatTensor(np.array(self.features[idx]))
        else:
            features = torch.zeros(434)  # fallback

        return signal, features, label


def compute_class_weights(labels):
    """Compute inverse-frequency class weights for imbalanced data."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.FloatTensor(weights)


def get_data_loaders():
    """
    Create train and test DataLoaders from preprocessed data.

    Returns:
        train_loader, test_loader, class_weights
    """
    print("Loading preprocessed datasets ...")

    train_ds = ECGDataset(TRAIN_DIR, split='train')
    test_ds  = ECGDataset(TEST_DIR,  split='test')

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(DEVICE.type == 'cuda')
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == 'cuda')
    )

    class_weights = compute_class_weights(train_ds.labels)

    print(f"  Train : {len(train_ds)} samples")
    print(f"  Test  : {len(test_ds)} samples")
    print(f"  Class weights: {class_weights.numpy().round(3)}")

    return train_loader, test_loader, class_weights


if __name__ == "__main__":
    print("=" * 50)
    print("  Testing Data Loader")
    print("=" * 50)

    train_loader, test_loader, weights = get_data_loaders()

    for sig, feat, lbl in train_loader:
        print(f"\n  Batch signal shape   : {sig.shape}")    # (32, 12, 5000)
        print(f"  Batch features shape : {feat.shape}")     # (32, 434)
        print(f"  Batch labels shape   : {lbl.shape}")      # (32,)
        print(f"  Labels in batch      : {lbl.unique().tolist()}")
        break

    print("  [OK] Data loader test passed!")
