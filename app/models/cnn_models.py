# =============================================================================
# cnn_models.py — Definisi Arsitektur CNN (Fondasi ke-3)
# =============================================================================
# File ini berisi fungsi untuk membuat model:
#   1. MobileNetV2 (transfer learning dari ImageNet)
#   2. ResNet50 (transfer learning dari ImageNet)
#
# Kedua model menggunakan teknik "Transfer Learning":
#   - Mengambil model yang sudah dilatih di ImageNet (14 juta gambar)
#   - Mengganti layer terakhir (classifier) untuk jumlah kelas kita
#   - Fine-tune pada dataset tanaman obat kita
#
# Fungsi utama yang diekspor:
#   - create_mobilenetv2(num_classes, pretrained)
#   - create_resnet50(num_classes, pretrained)
# =============================================================================

import torch.nn as nn
from torchvision import models


def create_mobilenetv2(num_classes, pretrained=True):
    """
    Membuat model MobileNetV2 dengan transfer learning.
    
    MobileNetV2 — Arsitektur CNN yang ringan dan efisien:
    - Ukuran model: ~14 MB (sangat kecil!)
    - Inference time: 15-25 ms per gambar
    - Cocok untuk deployment di mobile/edge device
    - Menggunakan inverted residual blocks + linear bottlenecks
    
    Args:
        num_classes (int): Jumlah kelas output (= jumlah jenis tanaman)
        pretrained (bool): Jika True, gunakan bobot dari ImageNet.
                          Jika False, inisialisasi acak (untuk loading .pth)
    
    Returns:
        model: PyTorch model MobileNetV2 yang siap ditraining/digunakan
    """
    if pretrained:
        # Muat model dengan bobot pre-trained ImageNet
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
    else:
        # Tanpa pre-trained (untuk load bobot dari file .pth nanti)
        model = models.mobilenet_v2(weights=None)
    
    # ====================================================================
    # GANTI LAYER TERAKHIR (classifier)
    # ====================================================================
    # MobileNetV2 asli punya classifier untuk 1000 kelas (ImageNet).
    # Kita ganti dengan classifier baru untuk jumlah kelas tanaman kita.
    #
    # Struktur asli:
    #   model.classifier = Sequential(
    #       Dropout(0.2),
    #       Linear(1280, 1000)  ← 1000 kelas ImageNet
    #   )
    #
    # Kita ganti menjadi:
    #   model.classifier = Sequential(
    #       Dropout(0.2),
    #       Linear(1280, num_classes)  ← jumlah kelas tanaman kita
    #   )
    # ====================================================================
    
    in_features = model.classifier[1].in_features  # = 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def create_resnet50(num_classes, pretrained=True):
    """
    Membuat model ResNet50 dengan transfer learning.
    
    ResNet50 — Arsitektur CNN yang dalam dan powerful:
    - 50 layer deep (residual connections mencegah vanishing gradient)
    - Akurasi tinggi: 97-99% pada banyak benchmark
    - Ukuran model: ~98 MB (lebih besar dari MobileNetV2)
    - Inference time: lebih lambat, tapi akurasi lebih tinggi
    
    Args:
        num_classes (int): Jumlah kelas output
        pretrained (bool): Jika True, gunakan bobot dari ImageNet
    
    Returns:
        model: PyTorch model ResNet50 yang siap ditraining/digunakan
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)
    
    # ====================================================================
    # GANTI LAYER TERAKHIR (fc = fully connected)
    # ====================================================================
    # ResNet50 asli:
    #   model.fc = Linear(2048, 1000)  ← 1000 kelas ImageNet
    #
    # Kita ganti:
    #   model.fc = Linear(2048, num_classes)  ← jumlah kelas tanaman kita
    # ====================================================================
    
    in_features = model.fc.in_features  # = 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    return model


# =============================================================================
# TEST (jalankan file ini langsung untuk verifikasi)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 TESTING MODEL DEFINITIONS")
    print("=" * 60)
    
    num_classes = 100  # Contoh: gabungan kedua dataset
    
    # Test MobileNetV2
    print("\n📱 MobileNetV2:")
    mobilenet = create_mobilenetv2(num_classes=num_classes, pretrained=False)
    total_params = sum(p.numel() for p in mobilenet.parameters())
    trainable = sum(p.numel() for p in mobilenet.parameters() if p.requires_grad)
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Classifier: {mobilenet.classifier}")
    
    # Test ResNet50
    print("\n🏗️  ResNet50:")
    resnet = create_resnet50(num_classes=num_classes, pretrained=False)
    total_params = sum(p.numel() for p in resnet.parameters())
    trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Classifier (fc): {resnet.fc}")
    
    # Test forward pass
    import torch
    dummy_input = torch.randn(1, 3, 224, 224)  # 1 gambar, 3 channel, 224x224
    
    mobilenet.eval()
    with torch.no_grad():
        output = mobilenet(dummy_input)
    print(f"\n✅ MobileNetV2 output shape: {output.shape}")  # [1, num_classes]
    
    resnet.eval()
    with torch.no_grad():
        output = resnet(dummy_input)
    print(f"✅ ResNet50 output shape:    {output.shape}")  # [1, num_classes]
    
    print(f"\n✅ Kedua model berhasil dibuat dan diverifikasi!")
