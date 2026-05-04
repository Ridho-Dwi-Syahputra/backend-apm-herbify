# =============================================================================
# preprocessing.py - Data Pipeline
# =============================================================================
# File ini berisi semua fungsi untuk:
#   1. Load dataset bersih (dataset_bersih/) — 80 kelas, sudah 224x224
#   2. Normalisasi menggunakan mean & std ImageNet (Normalisasi Piksel - 76.2% SLR)
#   3. Augmentasi data (sesuai preprocessing_metadata.json) [85.7% SLR]
#   4. Split data 80% train, 20% test
#   5. Membuat PyTorch DataLoader untuk training & testing
#   6. Load class weights dari preprocessing_metadata.json [23.8% SLR]
#
# Teknik SLR yang diterapkan:
#   - Standarisasi Ukuran (Resizing) -> 224x224 [95.2%]
#   - Augmentasi Data (Rotasi, Flip, ColorJitter, GaussianBlur) [85.7%]
#   - Normalisasi Piksel (ImageNet mean/std) [76.2%]
#   - Class Weighting untuk Ketidakseimbangan Kelas [23.8%]
#   - GaussianBlur untuk Robustness terhadap Noise [52.4%]
#
# Dataset: Indian Medicinal Leaf dataset — 80 kelas, 6,900 gambar
# Sumber: dataset_bersih/ (hasil preprocessing notebook 02, gambar 224x224)
#
# Fungsi utama yang diekspor:
#   - get_data_loaders()     -> return train_loader, test_loader
#   - get_class_names()      -> return list nama kelas (sorted)
#   - get_class_weights()    -> return tensor class weights (dari metadata)
# =============================================================================

import json
import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from app.utils.config import (
    CLEAN_DATASET_DIR,
    PREPROCESSING_METADATA_PATH,
    IMAGE_SIZE, BATCH_SIZE, TRAIN_RATIO, TEST_RATIO,
    IMAGENET_MEAN, IMAGENET_STD
)


# =============================================================================
# TRANSFORMASI (sesuai preprocessing_metadata.json)
# =============================================================================
# Augmentasi training mengikuti konfigurasi dari notebook 02 (preprocessing):
#   RandomHorizontalFlip(p=0.5)
#   RandomVerticalFlip(p=0.3)
#   RandomRotation(15)
#   ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
#   RandomAffine(scale=(0.85, 1.15))
#   GaussianBlur(kernel_size=3)

# Transformasi untuk data TRAINING (dengan augmentasi)
train_transforms = transforms.Compose([
    # Resize — gambar sudah 224x224 di dataset_bersih, tapi tetap resize untuk konsistensi
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # [SLR: Augmentasi Data - 85.7%]
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),

    # [SLR: Variasi Pencahayaan - 42.8% tantangan]
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),

    # Zoom & skala
    transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),

    # [SLR: Noise Robustness - 52.4% tantangan]
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.5),

    # [SLR: Normalisasi Piksel - 76.2%]
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Transformasi untuk data TESTING (TANPA augmentasi — hanya resize & normalize)
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# =============================================================================
# FUNGSI UTAMA — get_data_loaders()
# =============================================================================

def get_data_loaders(batch_size=None):
    """
    Membuat DataLoader untuk training dan testing dari dataset_bersih/.

    Alur:
        1. Load gambar dari CLEAN_DATASET_DIR (dataset_bersih/)
        2. Split: 80% train, 20% test (seed=42 untuk reproducibility)
        3. Terapkan transformasi (augmentasi hanya untuk train)
        4. Bungkus dalam DataLoader

    Returns:
        train_loader: DataLoader untuk training
        test_loader: DataLoader untuk testing
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    if not CLEAN_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset bersih tidak ditemukan: {CLEAN_DATASET_DIR}\n"
            "Jalankan notebook 02_preprocessing.ipynb terlebih dahulu."
        )

    # Load semua gambar menggunakan ImageFolder
    full_dataset = datasets.ImageFolder(
        root=str(CLEAN_DATASET_DIR),
        transform=None  # Transform diterapkan setelah split
    )

    total_images = len(full_dataset)
    num_classes = len(full_dataset.classes)
    print(f"\n[INFO] Dataset Info:")
    print(f"   Path:         {CLEAN_DATASET_DIR}")
    print(f"   Total gambar: {total_images}")
    print(f"   Total kelas:  {num_classes} (Indian Medicinal Leaf dataset)")

    # Split 80% train, 20% test
    train_size = int(TRAIN_RATIO * total_images)
    test_size = total_images - train_size

    train_subset, test_subset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train: {train_size} gambar ({TRAIN_RATIO*100:.0f}%)")
    print(f"   Test:  {test_size} gambar ({TEST_RATIO*100:.0f}%)")

    # Wrapper untuk transform berbeda train vs test
    train_dataset = _TransformSubset(train_subset, transform=train_transforms)
    test_dataset = _TransformSubset(test_subset, transform=test_transforms)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches:  {len(test_loader)}")

    return train_loader, test_loader


class _TransformSubset(torch.utils.data.Dataset):
    """Wrapper untuk menerapkan transform berbeda pada train vs test subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================================================
# get_class_names()
# =============================================================================

def get_class_names():
    """
    Mengembalikan daftar nama kelas yang terurut (sorted) dari dataset_bersih/.
    80 kelas tanaman obat dari Indian Medicinal Leaf dataset.

    Returns:
        list[str]: Daftar nama kelas, contoh: ['Aloevera', 'Amla', ...]
    """
    if not CLEAN_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset bersih tidak ditemukan: {CLEAN_DATASET_DIR}\n"
            "Jalankan notebook 02_preprocessing.ipynb terlebih dahulu."
        )

    class_names = sorted([
        d for d in os.listdir(CLEAN_DATASET_DIR)
        if os.path.isdir(CLEAN_DATASET_DIR / d)
    ])

    return class_names


# =============================================================================
# get_class_weights()
# =============================================================================
# [SLR: Ketidakseimbangan Kelas - 23.8% tantangan]
# Weights dimuat dari preprocessing_metadata.json (pre-calculated di notebook 02)
# untuk konsistensi dengan hasil preprocessing.

def get_class_weights(device=None):
    """
    Memuat class weights dari preprocessing_metadata.json.
    Weights sudah dihitung di notebook 02 dan disimpan di metadata.

    Formula yang digunakan saat preprocessing:
        weight_i = total_samples / (num_classes * count_i)
        (dinormalisasi agar mean weight ~ 1.0)

    Args:
        device: torch device (cpu/cuda). Default: auto-detect

    Returns:
        torch.Tensor: Class weights tensor, shape [num_classes], urutan sesuai class_names sorted
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not PREPROCESSING_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing metadata tidak ditemukan: {PREPROCESSING_METADATA_PATH}\n"
            "Jalankan notebook 02_preprocessing.ipynb terlebih dahulu."
        )

    with open(PREPROCESSING_METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    class_weights_dict = metadata["class_weights"]
    class_names = get_class_names()

    # Susun weights sesuai urutan class_names (sorted)
    weights = []
    for cls in class_names:
        if cls not in class_weights_dict:
            raise KeyError(f"Kelas '{cls}' tidak ditemukan di metadata class_weights")
        weights.append(class_weights_dict[cls])

    weights_tensor = torch.FloatTensor(weights).to(device)

    print(f"\n[INFO] Class Weights ({len(class_names)} kelas, dari preprocessing_metadata.json):")
    print(f"   Min weight: {min(weights):.4f} (kelas terbesar - Tulsi/Tamarind)")
    print(f"   Max weight: {max(weights):.4f} (kelas terkecil - Lemongrass/Pepper)")
    print(f"   Mean weight: {np.mean(weights):.4f}")

    return weights_tensor


# =============================================================================
# TEST — jalankan langsung untuk verifikasi
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("[TEST] TESTING PREPROCESSING PIPELINE")
    print("=" * 60)

    class_names = get_class_names()
    print(f"\n[LIST] Kelas yang ditemukan ({len(class_names)}):")
    for i, name in enumerate(class_names, 1):
        print(f"   {i:3d}. {name}")

    train_loader, test_loader = get_data_loaders()

    images, labels = next(iter(train_loader))
    print(f"\n[BATCH] Contoh batch:")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Labels contoh: {labels[:5].tolist()}")

    weights = get_class_weights()
    print(f"   Weights shape: {weights.shape}")

    print(f"\n[OK] Preprocessing pipeline berhasil!")
    print(f"   Dataset: dataset_bersih/ | Kelas: {len(class_names)} | "
          f"Weights: dari preprocessing_metadata.json")
