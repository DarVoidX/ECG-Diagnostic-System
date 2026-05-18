"""
ECG Diagnostic System — Model Training
========================================
Trains the Hybrid CNN-BiLSTM on preprocessed PTB-XL data.

Features:
  - Weighted CrossEntropy for class imbalance
  - AdamW optimizer with ReduceLROnPlateau scheduler
  - Gradient clipping
  - Early stopping with patience
  - Best model checkpoint saving
  - Per-epoch metrics: Loss, Accuracy, Precision, Recall, F1
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOP_PATIENCE,
    WEIGHT_DECAY, GRAD_CLIP_NORM, DEVICE, MODEL_PATH,
    NUM_CLASSES, MODEL_DIR
)
from model import HybridCNN_BiLSTM
from data_loader import get_data_loaders


def train_model():
    """
    Complete training pipeline.
    Loads preprocessed data → trains model → saves best checkpoint.
    """
    print("\n" + "=" * 60)
    print("       HYBRID CNN-BiLSTM TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {NUM_EPOCHS}")
    print(f"  Batch  : {BATCH_SIZE}")
    print(f"  LR     : {LEARNING_RATE}")
    print()

    # -- Load Data -------------------------------------------------------
    try:
        train_loader, test_loader, class_weights = get_data_loaders()
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        print("  Please run 'python build_dataset.py' first.")
        return

    # -- Initialize Model ------------------------------------------------
    model = HybridCNN_BiLSTM(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"\n  Model parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0

    # -- Training Loop ---------------------------------------------------
    print("\n" + "-" * 60)
    print("  TRAINING STARTED")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        # -- Train Phase --
        model.train()
        train_loss = 0.0
        train_preds, train_true = [], []

        for batch_sig, batch_feat, batch_y in train_loader:
            batch_sig  = batch_sig.to(DEVICE)
            batch_feat = batch_feat.to(DEVICE)
            batch_y    = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_sig, batch_feat)
            loss = criterion(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += loss.item() * batch_sig.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(batch_y.cpu().numpy())

        n_train = len(train_preds)
        train_loss /= max(n_train, 1)
        train_acc = accuracy_score(train_true, train_preds)

        # -- Test Phase --
        model.eval()
        test_loss = 0.0
        test_preds, test_true = [], []

        with torch.no_grad():
            for batch_sig, batch_feat, batch_y in test_loader:
                batch_sig  = batch_sig.to(DEVICE)
                batch_feat = batch_feat.to(DEVICE)
                batch_y    = batch_y.to(DEVICE)

                logits = model(batch_sig, batch_feat)
                loss = criterion(logits, batch_y)

                test_loss += loss.item() * batch_sig.size(0)
                preds = torch.argmax(logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_true.extend(batch_y.cpu().numpy())

        n_test = len(test_preds)
        test_loss /= max(n_test, 1)
        test_acc = accuracy_score(test_true, test_preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            test_true, test_preds, average='macro', zero_division=0
        )

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]['lr']

        print(f"\n  Epoch {epoch+1:02d}/{NUM_EPOCHS}  ({elapsed:.0f}s)  LR={lr:.2e}")
        print(f"    Train -> Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%")
        print(f"    Test  -> Loss: {test_loss:.4f}  Acc: {test_acc*100:.2f}%"
              f"  P: {precision*100:.1f}%  R: {recall*100:.1f}%  F1: {f1*100:.2f}%")

        # -- LR Scheduler --
        scheduler.step(test_loss)

        # -- Checkpoint / Early Stopping --
        improved = False
        if test_loss < best_loss:
            best_loss = test_loss
            improved = True
        if test_acc > best_acc:
            best_acc = test_acc
            improved = True

        if improved:
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': test_loss,
                'val_acc': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
            }, MODEL_PATH)
            print(f"    [BEST] Best model saved (Acc: {test_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"    [WARN] No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("\n  [STOP] Early stopping triggered.")
            break

    # -- Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  [BEST] Best Test Accuracy : {best_acc*100:.2f}%")
    print(f"  [SAVE] Model saved at     : {MODEL_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_model()
