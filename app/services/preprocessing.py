# =============================================================================
# preprocessing.py — Data Pipeline (Fondasi ke-2)
# =============================================================================
# File ini berisi semua fungsi untuk:
#   1. Menggabungkan kedua dataset (40 + 80 kelas) menjadi satu
#   2. Resize gambar ke 224x224
#   3. Normalisasi menggunakan mean & std ImageNet
#   4. Augmentasi data (flip, rotate, color jitter) — hanya untuk training
#   5. Split data menjadi 80% train, 20% test
#   6. Membuat PyTorch DataLoader untuk training & testing
#
# Fungsi utama yang diekspor:
#   - get_data_loaders()  → return train_loader, test_loader
#   - get_class_names()   → return list nama kelas (sorted)
# =============================================================================

import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from app.utils.config import (
    PLANT_DATASET_DIR, LEAF_DATASET_DIR, BASE_DIR,
    IMAGE_SIZE, BATCH_SIZE, TRAIN_RATIO, TEST_RATIO,
    IMAGENET_MEAN, IMAGENET_STD
)

# =============================================================================
# LANGKAH 1: Gabungkan kedua dataset ke folder sementara (combined)
# =============================================================================
# Kedua subdataset memiliki format folder yang sama:
#   dataset/NamaKelas/gambar.jpg
# Kita akan menggabungkan keduanya ke satu folder "combined"
# supaya PyTorch ImageFolder bisa membacanya sebagai satu dataset.
# Jika ada kelas yang sama di kedua dataset (misal: "Aloevera"),
# gambarnya akan digabungkan ke satu folder.

COMBINED_DATASET_DIR = BASE_DIR / "dataset" / "_combined_dataset"


def _create_combined_dataset():
    """
    Menggabungkan Medicinal Plant (40 kelas) dan Medicinal Leaf (80 kelas)
    ke satu folder. Menggunakan symlink/copy untuk menghindari duplikasi disk.
    
    Struktur output:
       _combined_dataset/
       ├── Aloevera/     ← gambar dari kedua dataset
       ├── Amla/
       ├── Bamboo/
       └── ... (semua kelas unik dari kedua dataset)
    """
    if COMBINED_DATASET_DIR.exists():
        # Jika sudah ada, skip (supaya tidak copy ulang setiap kali)
        class_count = len([d for d in os.listdir(COMBINED_DATASET_DIR) 
                          if os.path.isdir(COMBINED_DATASET_DIR / d)])
        if class_count > 0:
            print(f"✅ Combined dataset sudah ada: {class_count} kelas")
            return
    
    COMBINED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print("🔄 Menggabungkan kedua dataset...")
    
    total_copied = 0
    
    # Proses kedua dataset
    for dataset_name, dataset_dir in [("Plant", PLANT_DATASET_DIR), 
                                       ("Leaf", LEAF_DATASET_DIR)]:
        if not dataset_dir.exists():
            print(f"⚠️  {dataset_name} dataset tidak ditemukan: {dataset_dir}")
            continue
        
        classes = [d for d in os.listdir(dataset_dir) 
                   if os.path.isdir(dataset_dir / d)]
        
        for class_name in classes:
            src_dir = dataset_dir / class_name
            dst_dir = COMBINED_DATASET_DIR / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy semua gambar ke folder gabungan
            for img_file in os.listdir(src_dir):
                src_path = src_dir / img_file
                if src_path.is_file():
                    # Tambahkan prefix untuk menghindari nama file duplikat
                    # contoh: plant_gambar1.jpg, leaf_gambar1.jpg
                    prefix = dataset_name.lower()
                    dst_filename = f"{prefix}_{img_file}"
                    dst_path = dst_dir / dst_filename
                    
                    if not dst_path.exists():
                        shutil.copy2(str(src_path), str(dst_path))
                        total_copied += 1
        
        print(f"   ✅ {dataset_name} dataset: {len(classes)} kelas ditambahkan")
    
    total_classes = len([d for d in os.listdir(COMBINED_DATASET_DIR) 
                        if os.path.isdir(COMBINED_DATASET_DIR / d)])
    print(f"✅ Gabungan selesai: {total_classes} kelas unik, {total_copied} gambar")


# =============================================================================
# LANGKAH 2: Definisi Transformasi (Resize, Normalisasi, Augmentasi)
# =============================================================================

# Transformasi untuk data TRAINING (dengan augmentasi)
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),      # Resize ke 224x224
    transforms.RandomHorizontalFlip(p=0.5),           # Flip horizontal acak
    transforms.RandomVerticalFlip(p=0.3),             # Flip vertikal acak
    transforms.RandomRotation(15),                    # Rotasi acak ±15 derajat
    transforms.ColorJitter(                           # Variasi warna acak
        brightness=0.2, 
        contrast=0.2, 
        saturation=0.2, 
        hue=0.1
    ),
    transforms.ToTensor(),                            # Ubah gambar → Tensor PyTorch
    transforms.Normalize(                             # Normalisasi ImageNet
        mean=IMAGENET_MEAN, 
        std=IMAGENET_STD
    ),
])

# Transformasi untuk data TESTING (TANPA augmentasi — hanya resize & normalize)
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),      # Resize ke 224x224
    transforms.ToTensor(),                            # Ubah gambar → Tensor PyTorch
    transforms.Normalize(                             # Normalisasi ImageNet
        mean=IMAGENET_MEAN, 
        std=IMAGENET_STD
    ),
])


# =============================================================================
# LANGKAH 3: Fungsi Utama — get_data_loaders()
# =============================================================================

def get_data_loaders(batch_size=None):
    """
    Membuat DataLoader untuk training dan testing.
    
    Alur:
        1. Gabungkan kedua dataset (jika belum)
        2. Load semua gambar menggunakan ImageFolder
        3. Split: 80% train, 20% test
        4. Terapkan transformasi (augmentasi hanya untuk train)
        5. Bungkus dalam DataLoader (untuk batch processing)
    
    Returns:
        train_loader: DataLoader untuk training
        test_loader: DataLoader untuk testing
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    # Step 1: Gabungkan dataset
    _create_combined_dataset()
    
    # Step 2: Load semua gambar menggunakan ImageFolder
    # ImageFolder otomatis membaca struktur folder:
    #   _combined_dataset/NamaKelas/gambar.jpg → label = index kelas
    full_dataset = datasets.ImageFolder(
        root=str(COMBINED_DATASET_DIR),
        transform=None  # Transform diterapkan nanti setelah split
    )
    
    total_images = len(full_dataset)
    num_classes = len(full_dataset.classes)
    print(f"\n📊 Dataset Info:")
    print(f"   Total gambar: {total_images}")
    print(f"   Total kelas:  {num_classes}")
    
    # Step 3: Split 80% train, 20% test
    train_size = int(TRAIN_RATIO * total_images)
    test_size = total_images - train_size
    
    train_subset, test_subset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Seed untuk reproducibility
    )
    
    print(f"   Train: {train_size} gambar ({TRAIN_RATIO*100:.0f}%)")
    print(f"   Test:  {test_size} gambar ({TEST_RATIO*100:.0f}%)")
    
    # Step 4: Wrapper class untuk menerapkan transform yang berbeda
    # pada train vs test subset
    train_dataset = _TransformSubset(train_subset, transform=train_transforms)
    test_dataset = _TransformSubset(test_subset, transform=test_transforms)
    
    # Step 5: Bungkus dalam DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,       # Acak urutan setiap epoch (penting untuk training!)
        num_workers=0,      # 0 = main thread (aman di Windows)
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,      # Test tidak perlu diacak
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    return train_loader, test_loader


class _TransformSubset(torch.utils.data.Dataset):
    """
    Wrapper class untuk menerapkan transform pada Subset.
    Diperlukan karena random_split menghasilkan Subset yang
    tidak mendukung transform berbeda untuk train/test.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # image masih berupa PIL Image di sini (belum di-transform)
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================================================
# LANGKAH 4: Fungsi Helper — get_class_names()
# =============================================================================

def get_class_names():
    """
    Mengembalikan daftar nama kelas yang terurut (sorted).
    Harus dipanggil SETELAH get_data_loaders() atau setelah
    _create_combined_dataset() dijalankan.
    
    Returns:
        list[str]: Daftar nama kelas, contoh: ['Aloevera', 'Amla', ...]
    """
    _create_combined_dataset()  # Pastikan folder combined sudah ada
    
    class_names = sorted([
        d for d in os.listdir(COMBINED_DATASET_DIR) 
        if os.path.isdir(COMBINED_DATASET_DIR / d)
    ])
    
    return class_names


# =============================================================================
# TEST (jalankan file ini langsung untuk verifikasi)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔄 TESTING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Test get_class_names
    class_names = get_class_names()
    print(f"\n📋 Kelas yang ditemukan ({len(class_names)}):")
    for i, name in enumerate(class_names, 1):
        print(f"   {i:3d}. {name}")
    
    # Test get_data_loaders
    train_loader, test_loader = get_data_loaders()
    
    # Test ambil 1 batch
    images, labels = next(iter(train_loader))
    print(f"\n📦 Contoh batch:")
    print(f"   Images shape: {images.shape}")  # Harusnya [batch_size, 3, 224, 224]
    print(f"   Labels shape: {labels.shape}")   # Harusnya [batch_size]
    print(f"   Labels contoh: {labels[:5].tolist()}")
    print(f"\n✅ Preprocessing pipeline berhasil!")
