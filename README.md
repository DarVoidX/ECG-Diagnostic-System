# 🫀 ECG Diagnostic System

An advanced, production-grade Electrocardiography (ECG) Diagnostic System built to classify heart conditions using deep learning. This system processes raw ECG data from the PTB-XL dataset and visualizes the AI's diagnostic reasoning through a stunning Liquid Glass interface.

## ✨ Key Features

- **Hybrid AI Architecture:** Utilizes a Hybrid CNN-BiLSTM (Convolutional Neural Network + Bidirectional LSTM) designed in PyTorch to capture both spatial and temporal features of ECG waves.
- **Advanced Preprocessing Pipeline:** Incorporates Empirical Mode Decomposition (EMD), noise filtering, and signal normalization for clean data inputs.
- **Complex Feature Extraction:** Extracts morphological, FFT (Fast Fourier Transform), and DWT (Discrete Wavelet Transform) features from 12-lead ECG signals.
- **Liquid Glass UI:** A responsive, premium frontend interface featuring a crimson theme, dark/light modes, and dynamic background handling for maximum GPU efficiency.
- **Real-Time Inference Logging:** The backend terminal provides a step-by-step visual of the "thought process" of the AI during diagnostics.

## 📁 Project Architecture

### `backend/` (Python, PyTorch, AI Pipeline)
- `app.py`: The main API server handling requests from the frontend.
- `preprocessing.py`: Signal cleaning, normalization, and EMD logic.
- `feature_extraction.py` & `feature_fusion.py`: Extracts and combines FFT, DWT, and morphological features.
- `model.py`: Defines the Hybrid CNN-BiLSTM neural network architecture.
- `train.py`, `evaluate.py`, `inference.py`: Scripts for training, testing, and running real-time diagnostics on the PTB-XL dataset.
- `build_dataset.py`, `data_loader.py`: Patient-wise stratified splitting and dataset management.

### `frontend/` (HTML, CSS, JS)
- `index.html`, `stitch_landing.html`, `stitch_results.html`: The core pages of the web app.
- `app.js`: Connects to the Python backend and manages the dynamic Liquid Glass UI state.
- `index.css`: Contains custom CSS styling for the sleek, premium crimson medical aesthetics.

### `models/`
- Contains trained model weights (`acc59_cnn-bilstm_hybrid.pth`) and performance logs.

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.8+
- Node.js (Optional, for Live Server)

### 2. Backend Setup
```bash
cd backend
pip install -r ../requirements.txt
python app.py
```

### 3. Frontend Setup
Launch `frontend/index.html` via a local HTTP server (like VS Code Live Server).

*(Note: The PTB-XL dataset and some background videos are excluded from version control due to GitHub size limits).*

## 👨‍💻 Developed By
Darshan Naidu
