"""
ECG Diagnostic System — Inference Module
=========================================
Single-record inference pipeline:
  Load ECG → Preprocess → Extract Features → Fuse → Predict → Return Results

Used by the Flask API for real-time analysis of uploaded ECG records.
"""

import os
import numpy as np
import torch
import wfdb

from config import (
    DEVICE, DATASET_DIR, MODEL_PATH,
    SAMPLING_RATE, SIGNAL_LENGTH, NUM_LEADS,
    CLASS_NAMES, LEAD_NAMES, FUSED_FEATURE_DIM
)
from model import HybridCNN_BiLSTM
from preprocessing import ECGPreprocessor
from feature_extraction import FeatureExtractor
from feature_fusion import FeatureFuser


class ECGInference:
    """
    End-to-end inference pipeline for a single ECG record.

    Steps:
      1. Load WFDB record (.hea + .dat)
      2. Preprocess (HP → Notch → Wavelet → Z-score → Derivative)
      3. Extract features (Morphological + FFT + DWT)
      4. Fuse features (concatenation → 434-dim)
      5. Run model prediction (CNN + BiLSTM + Features → 5-class)
      6. Return results with visualization data
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH

        self.model = HybridCNN_BiLSTM().to(DEVICE)
        self.model_loaded = False

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_loaded = True
            print(f"[OK] Model loaded from {model_path}")
        else:
            print(f"[WARN] No model at {model_path} - using random weights")

        self.model.eval()

        self.preprocessor = ECGPreprocessor()
        self.extractor = FeatureExtractor()
        self.fuser = FeatureFuser()

    def load_record(self, record_path):
        """
        Load a WFDB record.

        Args:
            record_path: path to .hea/.dat file (without extension)

        Returns:
            signal: (12, 5000) numpy array
            metadata: dict with record info
        """
        signal, meta = wfdb.rdsamp(record_path)
        n_samples, n_leads = signal.shape

        if n_leads != NUM_LEADS:
            raise ValueError(
                f"Expected {NUM_LEADS}-lead ECG, got {n_leads} leads."
            )

        # Pad or trim to SIGNAL_LENGTH
        if n_samples < SIGNAL_LENGTH:
            pad = SIGNAL_LENGTH - n_samples
            signal = np.pad(signal, ((0, pad), (0, 0)), mode='constant')
        elif n_samples > SIGNAL_LENGTH:
            signal = signal[:SIGNAL_LENGTH, :]

        signal = signal.T.astype(np.float32)  # (12, 5000)

        metadata = {
            'fs': meta['fs'],
            'sig_name': meta['sig_name'],
            'units': meta['units'],
            'n_sig': meta['n_sig'],
            'actual_samples': n_samples,
        }
        return signal, metadata

    def predict(self, signal_12lead):
        """
        Run full prediction pipeline on a raw 12-lead signal.

        Args:
            signal_12lead: (12, 5000) numpy array

        Returns:
            dict with prediction, metrics, waveform, markers, features
        """
        print(f"\n[NEURAL ENGINE] ECG Upload Detected. Starting Inference Pipeline...", flush=True)
        print(f"  --> 1. Applying High-pass & Notch Filters...", flush=True)
        # Preprocess for model (full pipeline with normalization)
        processed = self.preprocessor.preprocess(signal_12lead, apply_emd=False)

        print(f"  --> 2. Preparing display signals...", flush=True)
        # Preprocess for display (light, no normalization)
        display_signal = self.preprocessor.preprocess_for_display(signal_12lead)

        print(f"  --> 3. Extracting Morphological, FFT, and DWT Features...", flush=True)
        # Extract features
        feat_dict = self.extractor.extract_all(processed)

        # Fuse features
        fused = self.fuser.fuse(feat_dict)
        print(f"  --> 4. Mapping inputs to GPU Tensors (Signal: {processed.shape}, Features: {fused.shape})", flush=True)

        print(f"  --> 5. Executing Hybrid CNN-BiLSTM Architecture...", flush=True)
        # Model prediction
        with torch.no_grad():
            sig_tensor  = torch.FloatTensor(processed).unsqueeze(0).to(DEVICE)
            feat_tensor = torch.FloatTensor(fused).unsqueeze(0).to(DEVICE)
            print(f"      |-- Passing spatial data through CNN blocks...", flush=True)
            print(f"      |-- Passing temporal sequences through BiLSTM...", flush=True)
            print(f"      |-- Softmax probability calculation...", flush=True)
            logits = self.model(sig_tensor, feat_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Always use the model's actual output (no random guessing)
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        print(f"\n[DIAGNOSIS COMPLETE]", flush=True)
        print(f"  >> Predicted Class: {CLASS_NAMES[predicted_class]}", flush=True)
        print(f"  >> Confidence:      {confidence*100:.2f}%", flush=True)
        print(f"  >> R-Peaks found:   {len(feat_dict['morph_raw']['r_peaks'])}", flush=True)
        print("-" * 50, flush=True)

        # -- Build results -----------------------------------------------
        morph = feat_dict['morph_raw']
        time_axis = np.linspace(0, SIGNAL_LENGTH / SAMPLING_RATE, SIGNAL_LENGTH).tolist()

        results = {
            'prediction': {
                'class_id': predicted_class,
                'class_name': CLASS_NAMES[predicted_class],
                'confidence': round(confidence * 100, 2),
                'probabilities': {
                    CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                    for i in range(len(probs))
                }
            },
            'metrics': {
                'heart_rate': round(morph.get('heart_rate', 0), 1),
                'pr_interval': round(
                    np.mean(morph.get('pr_intervals', [0])) * 1000, 1
                ) if morph.get('pr_intervals') else 0,
                'qrs_duration': round(
                    np.mean(morph.get('qrs_durations', [0])) * 1000, 1
                ) if morph.get('qrs_durations') else 0,
                'rr_interval': round(
                    np.mean(morph.get('rr_intervals', [0])) * 1000, 1
                ) if morph.get('rr_intervals') else 0,
            },
            'waveform': {
                'time': time_axis,
                'leads': {},
                'actual_samples': SIGNAL_LENGTH,
            },
            'markers': {
                'p_waves': [int(p) for p in morph.get('p_wave_positions', [])],
                'qrs_complexes': [
                    {'onset': int(q[0]), 'peak': int(q[1]), 'offset': int(q[2])}
                    for q in morph.get('qrs_positions', [])
                ],
                't_waves': [int(t) for t in morph.get('t_wave_positions', [])],
                'r_peaks': [int(r) for r in morph.get('r_peaks', [])],
            }
        }

        # Waveform data per lead
        for i, name in enumerate(LEAD_NAMES):
            results['waveform']['leads'][name] = display_signal[i].tolist()

        # -- Feature extraction summary for frontend ---------------------
        fft_vec = feat_dict['fft']
        dwt_vec = feat_dict['dwt']

        # FFT: dominant frequency per lead
        fft_dom_freqs = {}
        for i, name in enumerate(LEAD_NAMES):
            idx = i * 20 + 18
            fft_dom_freqs[name] = round(float(fft_vec[idx]), 2) if idx < len(fft_vec) else 0

        # FFT: band power distribution (Lead II)
        lead_ii_bands = {
            'Delta (0.5-4Hz)': round(float(fft_vec[1*20 + 0]) * 100, 2),
            'Theta (4-8Hz)':   round(float(fft_vec[1*20 + 1]) * 100, 2),
            'Alpha (8-15Hz)':  round(float(fft_vec[1*20 + 2]) * 100, 2),
            'Beta (15-40Hz)':  round(float(fft_vec[1*20 + 3]) * 100, 2),
            'HF (40-100Hz)':   round(float(fft_vec[1*20 + 4]) * 100, 2),
        }

        # DWT: energy per level (Lead II)
        dwt_energies = {}
        lead_ii_start = 1 * 15
        level_names = ['cA5', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1']
        for j, lname in enumerate(level_names):
            idx = lead_ii_start + 9 + j
            dwt_energies[lname] = round(float(dwt_vec[idx]), 4) if idx < len(dwt_vec) else 0

        results['feature_extraction'] = {
            'morphological': {
                'r_peaks_detected': len(morph.get('r_peaks', [])),
                'p_waves_detected': len(morph.get('p_wave_positions', [])),
                'qrs_complexes':    len(morph.get('qrs_positions', [])),
                't_waves_detected': len(morph.get('t_wave_positions', [])),
                'heart_rate':       round(morph.get('heart_rate', 0), 1),
                'mean_rr': round(np.mean(morph.get('rr_intervals', [0])) * 1000, 1),
                'mean_qrs': round(np.mean(morph.get('qrs_durations', [0])) * 1000, 1),
            },
            'fft': {
                'dominant_frequencies': fft_dom_freqs,
                'band_powers': lead_ii_bands,
            },
            'dwt': {
                'wavelet': 'db4',
                'levels': 5,
                'energies': dwt_energies,
            },
        }

        return results

    def predict_from_file(self, record_path):
        """Full pipeline: load file → preprocess → predict."""
        signal, metadata = self.load_record(record_path)
        results = self.predict(signal)
        results['metadata'] = metadata
        results['waveform']['actual_samples'] = metadata.get('actual_samples', SIGNAL_LENGTH)
        return results


if __name__ == "__main__":
    print("=" * 50)
    print("  Testing ECGInference")
    print("=" * 50)

    inferencer = ECGInference()

    test_record = os.path.join(DATASET_DIR, "records500", "00000", "00001_hr")

    if os.path.exists(test_record + ".hea"):
        results = inferencer.predict_from_file(test_record)
        print(f"\n  Diagnosis  : {results['prediction']['class_name']}")
        print(f"  Confidence : {results['prediction']['confidence']}%")
        print(f"  Heart Rate : {results['metrics']['heart_rate']} bpm")
        print(f"  R-peaks    : {len(results['markers']['r_peaks'])}")
        print("  [OK] Inference test passed!")
    else:
        print(f"  [WARN] Record not found: {test_record}")
