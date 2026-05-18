"""
ECG Diagnostic System — Model Evaluation
==========================================
Comprehensive evaluation of the trained Hybrid CNN-BiLSTM model.

Generates:
  - Accuracy, Precision, Recall, F1-Score (macro & per-class)
  - Confusion matrix
  - Per-class performance breakdown
  - Saves results to models/performance_metrics.json
"""

import os
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from config import (
    DEVICE, MODEL_PATH, METRICS_PATH,
    NUM_CLASSES, CLASS_NAMES
)
from model import HybridCNN_BiLSTM
from data_loader import get_data_loaders


def evaluate_model():
    """
    Load the best model checkpoint and evaluate on the test set.
    Computes: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
    Saves detailed metrics to JSON.
    """
    print("\n" + "=" * 60)
    print("       MODEL EVALUATION")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  Checkpoint : {MODEL_PATH}\n")

    # -- Load Model ------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print("  [ERROR] No trained model found!")
        print("  Run 'python train.py' first.")
        return

    model = HybridCNN_BiLSTM(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Model loaded (epoch {checkpoint.get('epoch', '?')})")
    print(f"  Training val_acc: {checkpoint.get('val_acc', 0)*100:.2f}%")

    # -- Load Test Data --------------------------------------------------
    _, test_loader, _ = get_data_loaders()

    # -- Inference on Test Set -------------------------------------------
    print("\n  Running inference on test set ...")

    all_preds = []
    all_true = []
    all_probs = []

    with torch.no_grad():
        for batch_sig, batch_feat, batch_y in test_loader:
            batch_sig  = batch_sig.to(DEVICE)
            batch_feat = batch_feat.to(DEVICE)

            logits = model(batch_sig, batch_feat)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    # -- Compute Metrics -------------------------------------------------
    accuracy = accuracy_score(all_true, all_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_true, all_preds, average='macro', zero_division=0
    )

    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        all_true, all_preds, average=None,
        labels=list(range(NUM_CLASSES)), zero_division=0
    )

    cm = confusion_matrix(all_true, all_preds, labels=list(range(NUM_CLASSES)))

    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    # -- Print Results ---------------------------------------------------
    print("\n" + "-" * 60)
    print("  OVERALL METRICS")
    print("-" * 60)
    print(f"  Accuracy  : {accuracy * 100:.2f}%")
    print(f"  Precision : {precision_macro * 100:.2f}%  (macro)")
    print(f"  Recall    : {recall_macro * 100:.2f}%  (macro)")
    print(f"  F1-Score  : {f1_macro * 100:.2f}%  (macro)")

    print("\n" + "-" * 60)
    print("  PER-CLASS PERFORMANCE")
    print("-" * 60)
    print(f"  {'Class':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print("  " + "-" * 58)
    for i in range(NUM_CLASSES):
        print(f"  {class_names[i]:<35}"
              f" {precision_per[i]*100:5.1f}%"
              f" {recall_per[i]*100:5.1f}%"
              f" {f1_per[i]*100:5.1f}%"
              f" {int(support_per[i]):5d}")

    print("\n" + "-" * 60)
    print("  CONFUSION MATRIX")
    print("-" * 60)
    header = "  " + " " * 8
    for i in range(NUM_CLASSES):
        header += f" P{i:>3}"
    print(header)
    for i in range(NUM_CLASSES):
        row = f"  T{i:<5}"
        for j in range(NUM_CLASSES):
            row += f" {cm[i][j]:>4}"
        row += f"  <- {class_names[i]}"
        print(row)

    # -- Save to JSON ----------------------------------------------------
    metrics = {
        'accuracy': round(accuracy * 100, 2),
        'precision': round(precision_macro * 100, 2),
        'recall': round(recall_macro * 100, 2),
        'f1_score': round(f1_macro * 100, 2),
        'class_names': class_names,
        'per_class': {
            class_names[i]: {
                'precision': round(float(precision_per[i]) * 100, 2),
                'recall':    round(float(recall_per[i]) * 100, 2),
                'f1_score':  round(float(f1_per[i]) * 100, 2),
                'support':   int(support_per[i]),
            }
            for i in range(NUM_CLASSES)
        },
        'confusion_matrix': cm.tolist(),
    }

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  [SAVE] Metrics saved to: {METRICS_PATH}")
    print("\n" + "=" * 60)
    print("  [OK] EVALUATION COMPLETE")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    evaluate_model()
