"""
ECG Diagnostic System — Flask REST API
=======================================
Serves the frontend and provides ECG analysis endpoints.

Endpoints:
  GET  /                          → Serve frontend
  GET  /api/health                → Health check
  GET  /api/samples               → List sample records
  GET  /api/analyze_sample/<id>   → Analyze a dataset sample
  POST /api/analyze               → Analyze uploaded ECG files
  GET  /api/model_performance     → Model evaluation metrics
"""

import os
import sys
import json
import tempfile
import shutil
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_DIR, CLASS_NAMES, MODEL_PATH, METRICS_PATH
from inference import ECGInference

# -- Initialize App -----------------------------------------------------------
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend'),
    static_url_path=''
)
CORS(app)

# -- Initialize Inference Engine ----------------------------------------------
print("[INIT] Initializing ECG Inference Engine ...")
inferencer = ECGInference()
print("[OK] Ready!\n")


# ===============================================================================
#  ROUTES
# ===============================================================================

@app.route('/')
def serve_frontend():
    """Serve the frontend SPA."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static assets."""
    return send_from_directory(app.static_folder, path)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'message': 'ECG Diagnostic System is running',
        'model_loaded': inferencer.model_loaded,
        'model_path': MODEL_PATH,
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_ecg():
    """
    Analyze uploaded ECG data.

    Accepts:
      - WFDB files (.hea + .dat) as multipart upload
      - OR a sample_id in form data to load from dataset
    """
    try:
        # Sample record from dataset
        if 'sample_id' in request.form:
            sample_id = request.form['sample_id']
            record_path = find_sample_record(sample_id)

            if record_path is None:
                return jsonify({'error': f'Sample record {sample_id} not found'}), 404

            results = inferencer.predict_from_file(record_path)
            return jsonify(results)

        # File upload
        if 'hea_file' not in request.files or 'dat_file' not in request.files:
            return jsonify({'error': 'Please upload both .hea and .dat files'}), 400

        hea_file = request.files['hea_file']
        dat_file = request.files['dat_file']

        tmpdir = tempfile.mkdtemp()
        try:
            hea_name = hea_file.filename
            base_name = hea_name.replace('.hea', '')

            hea_path = os.path.join(tmpdir, hea_name)
            dat_path = os.path.join(tmpdir, base_name + '.dat')

            hea_file.save(hea_path)
            dat_file.save(dat_path)

            record_path = os.path.join(tmpdir, base_name)
            results = inferencer.predict_from_file(record_path)
            return jsonify(results)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500


@app.route('/api/samples', methods=['GET'])
def list_samples():
    """List available sample ECG records from the dataset."""
    samples = []
    records_dir = os.path.join(DATASET_DIR, "records500", "00000")

    if os.path.exists(records_dir):
        files = sorted([
            f.replace('_hr.hea', '')
            for f in os.listdir(records_dir)
            if f.endswith('_hr.hea')
        ])[:20]  # First 20 samples

        for f in files:
            samples.append({
                'id': f,
                'name': f'Record {f}',
                'path': f'records500/00000/{f}_hr'
            })

    return jsonify({'samples': samples})


@app.route('/api/analyze_sample/<sample_id>', methods=['GET'])
def analyze_sample(sample_id):
    """Analyze a specific sample record from the dataset."""
    try:
        record_path = find_sample_record(sample_id)

        if record_path is None:
            return jsonify({'error': f'Record {sample_id} not found'}), 404

        results = inferencer.predict_from_file(record_path)
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500


def find_sample_record(sample_id):
    """Find a sample record path in the dataset."""
    records_base = os.path.join(DATASET_DIR, "records500")
    if not os.path.exists(records_base):
        return None

    for subdir in sorted(os.listdir(records_base)):
        record_path = os.path.join(records_base, subdir, f"{sample_id}_hr")
        if os.path.exists(record_path + ".hea"):
            return record_path

    return None


@app.route('/api/model_performance', methods=['GET'])
def model_performance():
    """Return model performance metrics (accuracy, precision, recall, F1, confusion matrix)."""
    # If detailed metrics exist (from evaluate.py), serve them
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return jsonify(json.load(f))

    # Otherwise extract basic info from checkpoint
    import torch
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        val_acc = checkpoint.get('val_acc', 0)
        f1 = checkpoint.get('f1_score', 0)
        prec = checkpoint.get('precision', 0)
        rec = checkpoint.get('recall', 0)
    except Exception:
        val_acc = f1 = prec = rec = 0

    class_names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]

    return jsonify({
        'accuracy':  round(val_acc * 100, 2) if val_acc <= 1 else round(val_acc, 2),
        'precision': round(prec * 100, 2)    if prec <= 1   else round(prec, 2),
        'recall':    round(rec * 100, 2)     if rec <= 1    else round(rec, 2),
        'f1_score':  round(f1 * 100, 2)      if f1 <= 1     else round(f1, 2),
        'class_names': class_names,
        'confusion_matrix': None,
        'note': "Run 'python evaluate.py' to generate detailed metrics"
    })


# ===============================================================================
#  MAIN
# ===============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  CARDIO.AI - ECG Diagnostic System")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
