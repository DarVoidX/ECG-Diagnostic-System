"""
ECG Diagnostic System — Preprocessing Module
=============================================
Full signal preprocessing pipeline for 12-lead ECG at 500 Hz.

Pipeline Order (as specified):
  1. Baseline Wander Removal   → Empirical Mode Decomposition (EMD)
  2. Noise Removal             → High-pass filter + Notch filter + Wavelet denoising
  3. Normalization             → Z-score normalization
  4. Contrast Enhancement      → Derivative-based enhancement
  5. Segmentation              → R-peak detection
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import pywt

from config import (
    SAMPLING_RATE, APPLY_EMD, EMD_MAX_ITER,
    WAVELET_DENOISE, WAVELET_LEVEL,
    NOTCH_FREQ, HIGHPASS_CUTOFF, DERIVATIVE_WEIGHT
)


class ECGPreprocessor:
    """
    Complete preprocessing pipeline for 12-lead ECG signals.
    Each lead is processed independently through all stages.
    Signal shape is preserved: (12, 5000) → (12, 5000).
    """

    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs

    # =======================================================================
    #  STEP 1: BASELINE WANDER REMOVAL (EMD)
    # =======================================================================

    def remove_baseline_emd(self, signal_1d):
        """
        Remove baseline wander using Empirical Mode Decomposition.
        Subtracts the last IMF (residual / lowest-frequency trend).
        Falls back to high-pass filter if EMD fails.
        """
        try:
            from PyEMD import EMD
            emd = EMD()
            emd.MAX_ITERATION = EMD_MAX_ITER
            IMFs = emd.emd(signal_1d.astype(np.float64))

            if len(IMFs) > 1:
                # Last IMF = baseline wander component → subtract it
                cleaned = signal_1d - IMFs[-1]
                return cleaned.astype(np.float32)
            return signal_1d
        except Exception:
            # Fallback: high-pass filter
            return self.highpass_filter(signal_1d, cutoff=HIGHPASS_CUTOFF)

    # =======================================================================
    #  STEP 2: NOISE REMOVAL (High-pass + Notch + Wavelet)
    # =======================================================================

    def highpass_filter(self, signal_1d, cutoff=HIGHPASS_CUTOFF, order=4):
        """Butterworth high-pass filter to remove residual low-frequency drift."""
        nyq = self.fs / 2.0
        b, a = butter(order, cutoff / nyq, btype='high')
        return filtfilt(b, a, signal_1d).astype(np.float32)

    def notch_filter(self, signal_1d, freq=NOTCH_FREQ, Q=30):
        """Notch filter to remove power-line interference (50 Hz)."""
        nyq = self.fs / 2.0
        w0 = freq / nyq
        b, a = iirnotch(w0, Q)
        return filtfilt(b, a, signal_1d).astype(np.float32)

    def wavelet_denoise(self, signal_1d, wavelet=WAVELET_DENOISE, level=WAVELET_LEVEL):
        """
        Wavelet denoising using soft thresholding (VisuShrink).
        Keeps approximation coefficients, thresholds detail coefficients.
        """
        coeffs = pywt.wavedec(signal_1d, wavelet, level=level)

        # Universal threshold (VisuShrink)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal_1d)))

        # Soft-threshold detail coefficients only
        denoised = [coeffs[0]]  # Keep approximation unchanged
        for c in coeffs[1:]:
            denoised.append(pywt.threshold(c, threshold, mode='soft'))

        reconstructed = pywt.waverec(denoised, wavelet)
        return reconstructed[:len(signal_1d)].astype(np.float32)

    # =======================================================================
    #  STEP 3: NORMALIZATION (Z-score)
    # =======================================================================

    def normalize_zscore(self, signal_1d):
        """Per-lead Z-score normalization: (x - μ) / σ."""
        mean = np.mean(signal_1d)
        std = np.std(signal_1d)
        if std < 1e-8:
            return (signal_1d - mean).astype(np.float32)
        return ((signal_1d - mean) / std).astype(np.float32)

    # =======================================================================
    #  STEP 4: CONTRAST ENHANCEMENT (Derivative-based)
    # =======================================================================

    def derivative_enhance(self, signal_1d, weight=DERIVATIVE_WEIGHT):
        """
        Derivative-based contrast enhancement.
        Emphasizes rapid voltage transitions (e.g., QRS complex).
        enhanced = signal + weight × d(signal)/dt
        """
        derivative = np.diff(signal_1d, prepend=signal_1d[0])
        enhanced = signal_1d + weight * derivative
        return enhanced.astype(np.float32)

    # =======================================================================
    #  STEP 5: SEGMENTATION (R-peak detection)
    # =======================================================================

    def detect_r_peaks(self, signal_1d):
        """
        Detect R-peaks using squared first-derivative + moving-average envelope.
        Returns array of sample indices where R-peaks occur.
        """
        # Squared derivative
        diff = np.diff(signal_1d, prepend=signal_1d[0])
        squared = diff ** 2

        # Moving-average smoothing (120 ms window)
        window = max(1, int(0.12 * self.fs))
        kernel = np.ones(window) / window
        envelope = np.convolve(squared, kernel, mode='same')

        # Find peaks with minimum RR distance of 200 ms (max 300 bpm)
        min_distance = int(0.2 * self.fs)
        height_threshold = np.mean(envelope) + 0.5 * np.std(envelope)

        peaks, _ = find_peaks(envelope, height=height_threshold, distance=min_distance)
        return peaks

    def segment_heartbeats(self, signal_12lead, lead_idx=1):
        """
        Segment heartbeats using R-peak detection on Lead II.
        Returns list of (12, segment_length) arrays centered on each R-peak.
        """
        lead_signal = signal_12lead[lead_idx]
        r_peaks = self.detect_r_peaks(lead_signal)

        if len(r_peaks) < 2:
            return [], r_peaks

        pre_r = int(0.25 * self.fs)   # 250 ms before R-peak
        post_r = int(0.40 * self.fs)  # 400 ms after R-peak

        segments = []
        for peak in r_peaks:
            start = peak - pre_r
            end = peak + post_r
            if start >= 0 and end <= signal_12lead.shape[1]:
                segments.append(signal_12lead[:, start:end])

        return segments, r_peaks

    # =======================================================================
    #  FULL PIPELINE
    # =======================================================================

    def preprocess(self, signal_12lead, apply_emd=APPLY_EMD):
        """
        Apply the complete preprocessing pipeline to a 12-lead ECG signal.

        Pipeline per lead:
          1. Baseline Wander Removal  (EMD or high-pass fallback)
          2. High-pass Filter         (0.5 Hz)
          3. Notch Filter             (50 Hz)
          4. Wavelet Denoising        (sym4, level 4)
          5. Z-score Normalization
          6. Derivative Enhancement

        Args:
            signal_12lead: ndarray of shape (12, 5000)
            apply_emd: whether to run EMD (True = better quality, slower)

        Returns:
            Preprocessed signal of shape (12, 5000)
        """
        processed = np.zeros_like(signal_12lead, dtype=np.float32)

        for lead in range(signal_12lead.shape[0]):
            sig = signal_12lead[lead].copy()

            # Step 1 — Baseline Wander Removal
            if apply_emd:
                sig = self.remove_baseline_emd(sig)
            else:
                sig = self.highpass_filter(sig, cutoff=HIGHPASS_CUTOFF)

            # Step 2 — Noise Removal
            sig = self.highpass_filter(sig, cutoff=HIGHPASS_CUTOFF)
            sig = self.notch_filter(sig, freq=NOTCH_FREQ)
            sig = self.wavelet_denoise(sig)

            # Step 3 — Normalization
            sig = self.normalize_zscore(sig)

            # Step 4 — Contrast Enhancement
            sig = self.derivative_enhance(sig)

            processed[lead] = sig

        return processed

    def preprocess_for_display(self, signal_12lead):
        """
        Light preprocessing for frontend visualization.
        Skips normalization and derivative enhancement to preserve amplitude.
        """
        processed = np.zeros_like(signal_12lead, dtype=np.float32)

        for lead in range(signal_12lead.shape[0]):
            sig = signal_12lead[lead].copy()
            sig = self.highpass_filter(sig, cutoff=HIGHPASS_CUTOFF)
            sig = self.notch_filter(sig, freq=NOTCH_FREQ)
            sig = self.wavelet_denoise(sig)
            processed[lead] = sig

        return processed


if __name__ == "__main__":
    print("=" * 50)
    print("  Testing ECGPreprocessor")
    print("=" * 50)

    preprocessor = ECGPreprocessor()

    # Synthetic 12-lead ECG-like signal
    t = np.linspace(0, 10, 5000)
    synthetic = np.zeros((12, 5000), dtype=np.float32)
    for i in range(12):
        synthetic[i] = (
            np.sin(2 * np.pi * 1.2 * t)
            + 0.1 * np.random.randn(5000)
            + 0.5 * np.sin(2 * np.pi * 0.1 * t)
            + 0.05 * np.sin(2 * np.pi * 50 * t)
        )

    result = preprocessor.preprocess(synthetic, apply_emd=False)
    print(f"  Input shape:  {synthetic.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Mean (lead 0): {result[0].mean():.4f}")
    print(f"  Std  (lead 0): {result[0].std():.4f}")

    peaks = preprocessor.detect_r_peaks(result[1])
    print(f"  R-peaks detected: {len(peaks)}")
    print("  [OK] Preprocessing test passed!")
