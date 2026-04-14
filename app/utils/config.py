# =============================================================================
# config.py — Konfigurasi Global Proyek
# =============================================================================
# File ini adalah FONDASI KE-1 yang harus dibuat sebelum file lain.
# Semua file lain (preprocessing.py, cnn_models.py, notebook scripts)
# akan meng-import konstanta dari sini.
# =============================================================================

import os
import torch
from pathlib import Path

# =============================================================================
# PATH DIRECTORIES
# =============================================================================

# Root directory backend (otomatis mendeteksi dari lokasi file ini)
# Struktur: backend/app/utils/config.py → naik 3 level = backend/
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/

# Dataset paths — KEDUA dataset digunakan (digabungkan)
DATASET_DIR = BASE_DIR / "dataset" / "Indian Medicinal Leaves Image Datasets"
PLANT_DATASET_DIR = DATASET_DIR / "Medicinal plant dataset"   # 40 kelas
LEAF_DATASET_DIR = DATASET_DIR / "Medicinal Leaf dataset"     # 80 kelas

# Output directories
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
MODEL_MOBILENET_DIR = TRAINED_MODELS_DIR / "mobilenetv2"
MODEL_RESNET_DIR = TRAINED_MODELS_DIR / "resnet50"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_EDA_DIR = REPORTS_DIR / "eda"
REPORTS_PREP_DIR = REPORTS_DIR / "preprocessing"
REPORTS_EVAL_DIR = REPORTS_DIR / "evaluation"

# Pastikan folder output ada
for d in [TRAINED_MODELS_DIR, MODEL_MOBILENET_DIR, MODEL_RESNET_DIR, REPORTS_DIR, REPORTS_EDA_DIR, REPORTS_PREP_DIR, REPORTS_EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HYPERPARAMETER TRAINING
# =============================================================================

# Ukuran input gambar — KENAPA 224x224?
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 1. ResNet50 dan MobileNetV2 adalah model "pre-trained" yang       │
# │    sudah dilatih oleh Google/Facebook pada dataset ImageNet        │
# │    (14 juta gambar). Mereka dilatih dengan ukuran input 224x224.  │
# │                                                                    │
# │ 2. Layer-layer di dalam model ini (convolutional, pooling, dll)   │
# │    sudah di-desain secara matematis untuk memproses input          │
# │    berukuran tepat 224x224 pixel.                                  │
# │                                                                    │
# │ 3. Jika kita pakai ukuran lain (misal 256x256 atau 100x100),     │
# │    model akan error atau hasilnya jauh lebih buruk karena          │
# │    "otak bawaan"-nya tidak mengenali dimensi tersebut.             │
# │                                                                    │
# │ KENAPA HARUS DI-RESIZE?                                           │
# │ Gambar di dataset kita punya ukuran yang BERVARIASI (ada yang     │
# │ 100x100, ada yang 1000x800, dsb). Neural network butuh input      │
# │ yang SERAGAM/TETAP karena operasi matriks di GPU memerlukan       │
# │ dimensi yang konsisten dalam satu batch.                           │
# └─────────────────────────────────────────────────────────────────────┘
IMAGE_SIZE = 224

BATCH_SIZE = 32           # Jumlah gambar yang diproses sekaligus per iterasi
NUM_EPOCHS = 25           # Jumlah putaran training (bisa disesuaikan)
LEARNING_RATE = 0.001     # Kecepatan belajar model (Adam optimizer)

# Data split — 80% training, 20% testing
TRAIN_RATIO = 0.80
TEST_RATIO = 0.20

# =============================================================================
# NORMALISASI (ImageNet Standard)
# =============================================================================
# Nilai mean dan std dari dataset ImageNet (jutaan gambar).
# Karena kita menggunakan model pre-trained dari ImageNet,
# kita HARUS menggunakan normalisasi yang sama agar model bisa
# "mengenali" pola gambar dengan benar.

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Mean per channel (R, G, B)
IMAGENET_STD = [0.229, 0.224, 0.225]   # Std per channel (R, G, B)

# =============================================================================
# DEVICE (CPU / GPU)
# =============================================================================
# Otomatis mendeteksi apakah komputer punya GPU NVIDIA (CUDA).
# Jika ada GPU → training jauh lebih cepat (10-100x)
# Jika tidak ada → pakai CPU (lebih lambat tapi tetap jalan)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# DAFTAR KELAS (akan diisi otomatis saat preprocessing)
# =============================================================================
# Variabel ini akan di-populate oleh preprocessing.py
# berdasarkan folder-folder yang ditemukan di kedua dataset

CLASS_NAMES = []  # Diisi otomatis oleh get_class_names()

# =============================================================================
# PRINT KONFIGURASI (untuk debugging)
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("⚙️  KONFIGURASI PROYEK")
    print("=" * 60)
    print(f"📂 Base directory:     {BASE_DIR}")
    print(f"📂 Plant dataset:      {PLANT_DATASET_DIR} (exists: {PLANT_DATASET_DIR.exists()})")
    print(f"📂 Leaf dataset:       {LEAF_DATASET_DIR} (exists: {LEAF_DATASET_DIR.exists()})")
    print(f"📂 Trained models:     {TRAINED_MODELS_DIR}")
    print(f"📂 Reports:            {REPORTS_DIR}")
    print(f"")
    print(f"🖼️  Image size:         {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"📦 Batch size:         {BATCH_SIZE}")
    print(f"🔄 Epochs:             {NUM_EPOCHS}")
    print(f"📈 Learning rate:      {LEARNING_RATE}")
    print(f"📊 Split:              {TRAIN_RATIO*100:.0f}% train / {TEST_RATIO*100:.0f}% test")
    print(f"🖥️  Device:             {DEVICE}")
    print(f"")
    print(f"🎨 ImageNet Mean:      {IMAGENET_MEAN}")
    print(f"🎨 ImageNet Std:       {IMAGENET_STD}")
    print("=" * 60)
