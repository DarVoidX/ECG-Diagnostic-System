"""
ECG Diagnostic System — Feature Extraction Module
==================================================
Extracts three categories of features from preprocessed ECG signals:

1. Morphological Features:
   - P-wave, QRS complex, T-wave detection
   - RR interval (time between R-peaks)
   - PR interval and QRS duration

2. Transform-Based — FFT Features:
   - Frequency-band power distribution
   - Dominant frequencies and magnitudes

3. Transform-Based — DWT Features:
   - Wavelet coefficient statistics per decomposition level
   - Energy distribution across levels
"""

import numpy as np
from scipy.signal import find_peaks
import pywt

from config import (
    SAMPLING_RATE, NUM_LEADS,
    FFT_FEATURES_PER_LEAD, DWT_FEATURES_PER_LEAD,
    DWT_WAVELET, DWT_LEVEL
)


class FeatureExtractor:
    """
    Extract morphological, FFT, and DWT features from a preprocessed
    12-lead ECG signal of shape (12, 5000).
    """

    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs

    # =======================================================================
    #  1. MORPHOLOGICAL FEATURES
    # =======================================================================

    def detect_qrs_complex(self, lead_signal):
        """
        Detect QRS complexes using derivative + squared + integrated approach.
        Returns list of (onset, peak, offset) tuples.
        """
        # First derivative → square → moving-average integration
        diff1 = np.diff(lead_signal, prepend=lead_signal[0])
        squared = diff1 ** 2
        win = max(1, int(0.15 * self.fs))  # 150 ms window
        kernel = np.ones(win) / win
        integrated = np.convolve(squared, kernel, mode='same')

        # Detect R-peaks
        min_dist = int(0.25 * self.fs)
        threshold = np.mean(integrated) + 0.6 * np.std(integrated)
        peaks, _ = find_peaks(integrated, height=threshold, distance=min_dist)

        qrs_complexes = []
        qrs_half = int(0.05 * self.fs)  # ~100 ms total QRS width

        for peak in peaks:
            temp_onset = max(0, peak - qrs_half)
            temp_offset = min(len(lead_signal), peak + qrs_half)
            
            if temp_onset < temp_offset:
                window = lead_signal[temp_onset:temp_offset]
                baseline = np.mean(window)
                true_idx = np.argmax(np.abs(window - baseline))
                true_r_peak = temp_onset + true_idx
            else:
                true_r_peak = peak
                
            onset = max(0, true_r_peak - qrs_half)
            offset = min(len(lead_signal) - 1, true_r_peak + qrs_half)
            
            qrs_complexes.append((onset, true_r_peak, offset))

        return qrs_complexes

    def detect_p_wave(self, lead_signal, qrs_onset):
        """Detect P-wave in the 200 ms region before QRS onset."""
        start = max(0, qrs_onset - int(0.20 * self.fs))
        end = qrs_onset - int(0.02 * self.fs)

        if start >= end:
            return None

        region = lead_signal[start:end]
        if len(region) < 5:
            return None

        peaks, props = find_peaks(region, height=0)
        if len(peaks) > 0:
            best = peaks[np.argmax(props['peak_heights'])]
            return start + best
        return None

    def detect_t_wave(self, lead_signal, qrs_offset):
        """Detect T-wave in the 80–400 ms region after QRS offset."""
        start = qrs_offset + int(0.08 * self.fs)
        end = min(len(lead_signal), qrs_offset + int(0.40 * self.fs))

        if start >= end:
            return None

        region = lead_signal[start:end]
        if len(region) < 5:
            return None

        # Check positive T-wave
        peaks, props = find_peaks(region, height=0)
        if len(peaks) > 0:
            best = peaks[np.argmax(props['peak_heights'])]
            return start + best

        # Check inverted T-wave
        peaks_inv, props_inv = find_peaks(-region, height=0)
        if len(peaks_inv) > 0:
            best = peaks_inv[np.argmax(props_inv['peak_heights'])]
            return start + best

        return None

    def compute_morphological_features(self, signal_12lead):
        """
        Compute morphological features from all 12 leads.

        Returns dict with:
          - r_peaks, rr_intervals, heart_rate
          - pr_intervals, qrs_durations
          - p_wave_positions, qrs_positions, t_wave_positions
        """
        # Use Lead II for primary morphological detection
        lead_ii = signal_12lead[1]
        qrs_list = self.detect_qrs_complex(lead_ii)

        features = {
            'r_peaks': [],
            'rr_intervals': [],
            'heart_rate': 0.0,
            'pr_intervals': [],
            'qrs_durations': [],
            'p_wave_positions': [],
            'qrs_positions': qrs_list,
            't_wave_positions': [],
        }

        if len(qrs_list) < 2:
            return features

        # R-peaks and RR intervals
        r_peaks = [q[1] for q in qrs_list]
        features['r_peaks'] = r_peaks

        rr_intervals = np.diff(r_peaks) / self.fs  # seconds
        features['rr_intervals'] = rr_intervals.tolist()
        features['heart_rate'] = (
            60.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.0
        )

        # QRS durations
        features['qrs_durations'] = [
            (q[2] - q[0]) / self.fs for q in qrs_list
        ]

        # P-waves, PR intervals, T-waves
        for onset, peak, offset in qrs_list:
            p_pos = self.detect_p_wave(lead_ii, onset)
            if p_pos is not None:
                features['p_wave_positions'].append(p_pos)
                features['pr_intervals'].append((onset - p_pos) / self.fs)

            t_pos = self.detect_t_wave(lead_ii, offset)
            if t_pos is not None:
                features['t_wave_positions'].append(t_pos)

        return features

    def morphological_to_vector(self, morph_features):
        """
        Convert morphological feature dict → fixed-length vector (14-dim).

        Layout:
          [0]    heart_rate
          [1-4]  rr_mean, rr_std, rr_min, rr_max
          [5-8]  pr_mean, pr_std, pr_min, pr_max
          [9-12] qrs_mean, qrs_std, qrs_min, qrs_max
          [13]   num_beats
        """
        vec = []

        # Heart rate (1)
        vec.append(morph_features.get('heart_rate', 0.0))

        # RR interval stats (4)
        rr = morph_features.get('rr_intervals', [])
        if len(rr) > 0:
            vec.extend([np.mean(rr), np.std(rr), np.min(rr), np.max(rr)])
        else:
            vec.extend([0, 0, 0, 0])

        # PR interval stats (4)
        pr = morph_features.get('pr_intervals', [])
        if len(pr) > 0:
            vec.extend([np.mean(pr), np.std(pr), np.min(pr), np.max(pr)])
        else:
            vec.extend([0, 0, 0, 0])

        # QRS duration stats (4)
        qrs = morph_features.get('qrs_durations', [])
        if len(qrs) > 0:
            vec.extend([np.mean(qrs), np.std(qrs), np.min(qrs), np.max(qrs)])
        else:
            vec.extend([0, 0, 0, 0])

        # Number of beats (1)
        vec.append(len(morph_features.get('r_peaks', [])))

        return np.array(vec, dtype=np.float32)  # 14 features

    # =======================================================================
    #  2. FFT FEATURES (Fast Fourier Transform)
    # =======================================================================

    def compute_fft_features(self, signal_12lead, n_features=FFT_FEATURES_PER_LEAD):
        """
        Compute FFT-based spectral features for each lead.

        Per-lead features (20):
          [0-4]   Band powers: Delta, Theta, Alpha, Beta, HF
          [5-9]   Top-5 dominant frequencies (Hz)
          [10-14] Top-5 normalized magnitudes
          [15]    Mean magnitude
          [16]    Std magnitude
          [17]    Max magnitude
          [18]    Dominant frequency (Hz)
          [19]    Total spectral power

        Returns: vector of shape (12 × 20 = 240,)
        """
        all_features = []

        for lead in range(signal_12lead.shape[0]):
            sig = signal_12lead[lead]

            # FFT
            fft_vals = np.fft.rfft(sig)
            fft_mag = np.abs(fft_vals)
            freqs = np.fft.rfftfreq(len(sig), d=1.0 / self.fs)

            # Band powers (normalized)
            bands = [
                (0.5, 4),     # Delta
                (4, 8),       # Theta
                (8, 15),      # Alpha / low ECG
                (15, 40),     # Beta / QRS content
                (40, 100),    # High-frequency
            ]
            total_power = np.sum(fft_mag ** 2) + 1e-10
            band_powers = []
            for low, high in bands:
                mask = (freqs >= low) & (freqs < high)
                band_powers.append(np.sum(fft_mag[mask] ** 2) / total_power)

            # Top-5 dominant frequencies
            top_indices = np.argsort(fft_mag)[-5:]
            top_freqs = freqs[top_indices]
            top_mags = fft_mag[top_indices] / (np.max(fft_mag) + 1e-10)

            # Aggregate stats
            lead_features = np.concatenate([
                band_powers,                                          # 5
                top_freqs,                                            # 5
                top_mags,                                             # 5
                [np.mean(fft_mag),                                    # 1
                 np.std(fft_mag),                                     # 1
                 np.max(fft_mag),                                     # 1
                 np.argmax(fft_mag) * (self.fs / len(sig)),           # 1 dominant freq
                 np.sum(fft_mag ** 2)],                               # 1 total power
            ])

            all_features.append(lead_features[:n_features])

        return np.concatenate(all_features).astype(np.float32)  # 240

    # =======================================================================
    #  3. DWT FEATURES (Discrete Wavelet Transform)
    # =======================================================================

    def compute_dwt_features(self, signal_12lead,
                              wavelet=DWT_WAVELET, level=DWT_LEVEL,
                              n_features=DWT_FEATURES_PER_LEAD):
        """
        Compute DWT features for each lead.

        Per-lead features (15):
          For each decomposition level: mean, std, max_abs  (3 × (level+1))
          Then energy of each level.
          Padded/truncated to n_features.

        Returns: vector of shape (12 × 15 = 180,)
        """
        all_features = []

        for lead in range(signal_12lead.shape[0]):
            sig = signal_12lead[lead]

            # Wavelet decomposition
            coeffs = pywt.wavedec(sig, wavelet, level=level)

            lead_features = []
            # Statistical features per level
            for c in coeffs:
                lead_features.extend([
                    np.mean(c),
                    np.std(c),
                    np.max(np.abs(c)),
                ])

            # Energy per level
            for c in coeffs:
                lead_features.append(np.sum(c ** 2))

            # Pad / truncate to fixed size
            lead_features = np.array(lead_features[:n_features])
            if len(lead_features) < n_features:
                lead_features = np.pad(
                    lead_features, (0, n_features - len(lead_features))
                )

            all_features.append(lead_features)

        return np.concatenate(all_features).astype(np.float32)  # 180

    # =======================================================================
    #  EXTRACT ALL FEATURES
    # =======================================================================

    def extract_all(self, signal_12lead):
        """
        Extract all feature categories from a preprocessed 12-lead signal.

        Args:
            signal_12lead: shape (12, 5000)

        Returns:
            dict with keys: 'morphological' (14,), 'fft' (240,),
            'dwt' (180,), 'morph_raw' (raw detection results)
        """
        morph_raw = self.compute_morphological_features(signal_12lead)
        morph_vec = self.morphological_to_vector(morph_raw)
        fft_vec = self.compute_fft_features(signal_12lead)
        dwt_vec = self.compute_dwt_features(signal_12lead)

        return {
            'morphological': morph_vec,     # 14 features
            'fft': fft_vec,                 # 240 features
            'dwt': dwt_vec,                 # 180 features
            'morph_raw': morph_raw,         # Raw detection results
        }


if __name__ == "__main__":
    print("=" * 50)
    print("  Testing FeatureExtractor")
    print("=" * 50)

    extractor = FeatureExtractor()

    t = np.linspace(0, 10, 5000)
    synthetic = np.zeros((12, 5000), dtype=np.float32)
    for i in range(12):
        synthetic[i] = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(5000)

    features = extractor.extract_all(synthetic)
    print(f"  Morphological vector : {features['morphological'].shape}")
    print(f"  FFT vector           : {features['fft'].shape}")
    print(f"  DWT vector           : {features['dwt'].shape}")
    print(f"  Heart rate           : {features['morph_raw']['heart_rate']:.1f} bpm")
    print("  [OK] Feature extraction test passed!")
