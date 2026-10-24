# CROPSPECTRA — Autonomous Rover Leaf Disease Classifier

CROPSPECTRA is a computer vision and deep learning solution engineered for autonomous agricultural rovers to detect and diagnose rice leaf diseases in real-time. Built upon an optimized **MobileNetV3** architecture, it delivers rapid inference speeds suitable for edge and embedded deployment while maintaining high diagnostic accuracy.

---

## 🌟 Key Features

- **Edge-Optimized Architecture**: MobileNetV3 backbone fine-tuned for high accuracy with minimal computational overhead (~18 MB weights).
- **Interactive Web Dashboard**: Built-in inspection and diagnostics portal (`app.py`) for live image uploads, confidence score distributions, and tailored treatment advisories.
- **Batch Testing & Evaluation**: Comprehensive evaluation tools providing confusion matrices, precision/recall curves, and classification reports.
- **Actionable Agronomic Advisories**: Automatically recommends targeted chemical and biological treatments based on identified symptoms.

---

## 📁 Repository Structure

```text
├── app.py                     # Interactive web dashboard & inference server
├── requirements.txt           # Project dependencies
├── models/
│   ├── best_cropspectra_mobilenetv3.pth   # Pretrained model weights (~18 MB)
│   └── class_mapping.json                 # Label to index mapping
├── src/
│   ├── predict.py             # Inference pipeline & disease advisory database
│   ├── train_image_classifier.py  # Model training and fine-tuning routines
│   └── test_batch.py          # Batch evaluation script
├── notebooks/
│   ├── phase_A_inspect_dataset.ipynb       # Data exploration & class distribution
│   └── phase_B_train_image_classifier.ipynb # Model training experiment logs
└── reports/
    ├── classification_report.txt  # Evaluation metrics (precision, recall, f1)
    ├── confusion_matrix.png       # Test set confusion matrix
    └── training_metrics.png       # Loss and accuracy graphs
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Aayushi-raj/CROPSPECTRA.git
cd CROPSPECTRA
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Dashboard
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000` to interact with the inspection dashboard.

---

## 📊 Dataset Note
The raw dataset directory (`DATASET/`) is excluded from this repository due to GitHub's file size limits (>1 GB). To train the model from scratch, place your dataset folders (`train`, `validation`) in the `DATASET/` directory according to the structure defined in `train_image_classifier.py`.
