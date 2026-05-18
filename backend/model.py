"""
ECG Diagnostic System -- Hybrid CNN + BiLSTM Model
==================================================
Production-grade three-branch architecture for 5-class ECG classification.

Architecture (memory-efficient):
  * CNN extracts spatial features and downsamples the signal
  * BiLSTM processes CNN-downsampled features for temporal learning
  * MLP processes fused handcrafted features (434-dim)
  * Classifier combines all three branches

  Branch 1 (CNN + SE + Residual) -> Spatial features (256)
  Branch 2 (BiLSTM + Attention)  -> Temporal features from CNN output (256)
  Branch 3 (MLP)                 -> Handcrafted features (128)

  Classifier -> Concatenated (640) -> 5-class output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    NUM_LEADS, NUM_CLASSES, FUSED_FEATURE_DIM,
    CNN_CHANNELS, CNN_KERNELS,
    LSTM_HIDDEN, LSTM_LAYERS,
    DROPOUT, CLASSIFIER_DROPOUT
)


# ===========================================================================
#  BUILDING BLOCKS
# ===========================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Learns to re-weight channel importance adaptively.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w


class ResidualCNNBlock(nn.Module):
    """
    Conv1d block with BatchNorm, ReLU, SE attention, and residual connection.
    If in_channels != out_channels, a 1x1 projection is used for the skip path.
    """

    def __init__(self, in_channels, out_channels, kernel_size, se_reduction=16):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )

        self.se = SEBlock(out_channels, reduction=se_reduction)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.se(out)
        out = self.relu(out + residual)
        out = self.pool(out)
        return out


class TemporalAttention(nn.Module):
    """
    Soft attention over BiLSTM time steps.
    Learns which timesteps are most important for classification.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, lstm_output):
        # lstm_output: (B, T, H)
        scores = self.attention(lstm_output)           # (B, T, 1)
        weights = F.softmax(scores, dim=1)             # (B, T, 1)
        context = (lstm_output * weights).sum(dim=1)   # (B, H)
        return context


# ===========================================================================
#  MAIN MODEL
# ===========================================================================

class HybridCNN_BiLSTM(nn.Module):
    """
    Production-grade Hybrid CNN-BiLSTM with feature fusion.

    Memory-efficient design:
      CNN downsamples (12, 5000) -> (256, 625) using 3 blocks with MaxPool(2)
      BiLSTM processes the downsampled (625, 256) sequence instead of raw (5000, 12)
      This reduces LSTM memory by ~8x while preserving temporal information.

    Input:
      signal:   (B, 12, 5000)  -- preprocessed 12-lead ECG
      features: (B, 434)       -- fused morphological + FFT + DWT features

    Output:
      logits:   (B, 5)         -- class scores (pre-softmax)
    """

    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FUSED_FEATURE_DIM):
        super().__init__()

        # -- CNN Branch with SE + Residual ---
        # (B, 12, 5000) -> (B, 64, 2500) -> (B, 128, 1250) -> (B, 256, 625)
        self.cnn_block1 = ResidualCNNBlock(NUM_LEADS, CNN_CHANNELS[0], CNN_KERNELS[0])
        self.cnn_block2 = ResidualCNNBlock(CNN_CHANNELS[0], CNN_CHANNELS[1], CNN_KERNELS[1])
        self.cnn_block3 = ResidualCNNBlock(CNN_CHANNELS[1], CNN_CHANNELS[2], CNN_KERNELS[2])

        self.cnn_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        cnn_out_dim = CNN_CHANNELS[2]  # 256

        # -- BiLSTM Branch (processes CNN downsampled features) ---
        # Input: (B, 625, 256) -- CNN output permuted
        # This is memory-efficient: 625 steps vs 5000 steps
        self.lstm = nn.LSTM(
            input_size=CNN_CHANNELS[2],   # 256 (CNN output channels)
            hidden_size=LSTM_HIDDEN,      # 128
            num_layers=LSTM_LAYERS,       # 2
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0,
        )
        lstm_out_dim = LSTM_HIDDEN * 2  # 256 (bidirectional)

        self.temporal_attention = TemporalAttention(lstm_out_dim)
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)

        # -- Feature Branch (handcrafted features) ---
        # Input: (B, 434)
        self.feature_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        feat_out_dim = 128

        # -- Classification Head ---
        # CNN(256) + BiLSTM(256) + Features(128) = 640
        combined_dim = cnn_out_dim + lstm_out_dim + feat_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for Conv/ReLU, Xavier for Linear outputs."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, signal, features=None):
        """
        Forward pass.

        Args:
            signal:   (B, 12, 5000) preprocessed ECG signal
            features: (B, 434) fused feature vector (optional)

        Returns:
            logits: (B, 5) class scores
        """
        # -- CNN Branch --
        cnn_feat = self.cnn_block1(signal)     # (B, 64, 2500)
        cnn_feat = self.cnn_block2(cnn_feat)   # (B, 128, 1250)
        cnn_feat = self.cnn_block3(cnn_feat)   # (B, 256, 625)

        cnn_out = self.cnn_pool(cnn_feat)      # (B, 256) -- global avg pool

        # -- BiLSTM Branch (uses CNN downsampled features) --
        lstm_in = cnn_feat.permute(0, 2, 1)    # (B, 625, 256)
        lstm_out, _ = self.lstm(lstm_in)        # (B, 625, 256)
        lstm_feat = self.temporal_attention(lstm_out)  # (B, 256)
        lstm_feat = self.lstm_norm(lstm_feat)

        # -- Feature Branch --
        if features is not None:
            feat_out = self.feature_branch(features)  # (B, 128)
        else:
            feat_out = torch.zeros(signal.size(0), 128, device=signal.device)

        # -- Combine & Classify --
        combined = torch.cat([cnn_out, lstm_feat, feat_out], dim=1)  # (B, 640)
        logits = self.classifier(combined)
        return logits

    def predict_proba(self, signal, features=None):
        """Get softmax probabilities."""
        logits = self.forward(signal, features)
        return F.softmax(logits, dim=1)

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_summary():
    """Print model architecture and verify forward pass."""
    model = HybridCNN_BiLSTM()

    print("=" * 60)
    print("  Hybrid CNN-BiLSTM ECG Classifier (Production)")
    print("=" * 60)
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")

    # Verify forward pass
    dummy_sig = torch.randn(2, NUM_LEADS, 5000)
    dummy_feat = torch.randn(2, FUSED_FEATURE_DIM)
    output = model(dummy_sig, dummy_feat)

    print(f"\nSignal input shape  : {dummy_sig.shape}")
    print(f"Feature input shape : {dummy_feat.shape}")
    print(f"Output shape        : {output.shape}")
    print("[OK] Model forward pass verified!")
    return model


if __name__ == "__main__":
    get_model_summary()
