"""
ECG Diagnostic System — Feature Fusion Module
==============================================
Concatenates morphological, FFT, and DWT feature vectors
into a single unified feature vector for model input.

Feature Dimensions:
  Morphological :  14
  FFT           : 240  (12 leads × 20)
  DWT           : 180  (12 leads × 15)
  ---------------------
  Total (Fused) : 434
"""

import numpy as np

from config import (
    MORPH_FEATURE_DIM, FFT_FEATURE_DIM,
    DWT_FEATURE_DIM, FUSED_FEATURE_DIM
)


class FeatureFuser:
    """
    Fuses morphological + FFT + DWT feature vectors via concatenation
    into a single unified feature vector of dimension 434.
    """

    def __init__(self):
        self.morph_dim = MORPH_FEATURE_DIM    # 14
        self.fft_dim   = FFT_FEATURE_DIM      # 240
        self.dwt_dim   = DWT_FEATURE_DIM      # 180
        self.total_dim = FUSED_FEATURE_DIM    # 434

    def fuse(self, feature_dict):
        """
        Concatenate all feature vectors.

        Args:
            feature_dict: dict with keys 'morphological', 'fft', 'dwt'

        Returns:
            Fused feature vector of shape (434,)
        """
        morph = feature_dict['morphological']
        fft   = feature_dict['fft']
        dwt   = feature_dict['dwt']

        fused = np.concatenate([morph, fft, dwt])
        return fused.astype(np.float32)

    def fuse_batch(self, morph_batch, fft_batch, dwt_batch):
        """
        Fuse feature batches.

        Args:
            morph_batch: (N, 14)
            fft_batch:   (N, 240)
            dwt_batch:   (N, 180)

        Returns:
            (N, 434) fused feature array
        """
        return np.concatenate(
            [morph_batch, fft_batch, dwt_batch], axis=1
        ).astype(np.float32)

    def get_feature_names(self):
        """Return human-readable feature names for interpretability."""
        names = []

        # Morphological (14)
        names.extend([
            'heart_rate', 'rr_mean', 'rr_std', 'rr_min', 'rr_max',
            'pr_mean', 'pr_std', 'pr_min', 'pr_max',
            'qrs_mean', 'qrs_std', 'qrs_min', 'qrs_max',
            'num_beats'
        ])

        # FFT per lead (240)
        for lead in range(12):
            for feat in range(20):
                names.append(f'fft_lead{lead}_f{feat}')

        # DWT per lead (180)
        for lead in range(12):
            for feat in range(15):
                names.append(f'dwt_lead{lead}_f{feat}')

        return names


if __name__ == "__main__":
    print("=" * 50)
    print("  Testing FeatureFuser")
    print("=" * 50)

    fuser = FeatureFuser()

    mock = {
        'morphological': np.random.randn(14).astype(np.float32),
        'fft':           np.random.randn(240).astype(np.float32),
        'dwt':           np.random.randn(180).astype(np.float32),
    }

    fused = fuser.fuse(mock)
    print(f"  Fused vector shape : {fused.shape}")
    print(f"  Expected dimension : {fuser.total_dim}")
    assert fused.shape[0] == fuser.total_dim
    print("  [OK] Feature fusion test passed!")
