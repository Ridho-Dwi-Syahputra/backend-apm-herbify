# =============================================================================
# prediction_service.py - Service prediksi gambar tanaman obat
# =============================================================================
# Menggunakan MobileNetV2 dengan deteksi non-tanaman via confidence threshold.
# Semua nama kelas menggunakan Bahasa Indonesia (prefix "Daun").
# =============================================================================

import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from app.models.cnn_models import create_mobilenetv2
from app.services.plant_database import get_plant_info_safe
from app.utils.config import (
    MODEL_MOBILENET_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CONFIDENCE_THRESHOLD
)

# =============================================================================
# CLASS NAMES - 75 kelas Bahasa Indonesia (prefix "Daun")
# Diambil dari sorted(os.listdir(_combined_dataset))
# Urutan ini HARUS sama persis dengan saat training (ImageFolder sorts A-Z)
# =============================================================================

# Load dari file yang sudah diverifikasi
_CLASS_NAMES_FILE = Path(__file__).parent.parent.parent / "class_names.json"

if _CLASS_NAMES_FILE.exists():
    with open(_CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
else:
    # Hardcoded fallback - 75 kelas Bahasa Indonesia (setelah merge & translasi)
    CLASS_NAMES = [
        "Daun Adhatoda (Malabar Nut)", "Daun Amla (Malaka India)",
        "Daun Asam Jawa", "Daun Ashoka", 
        "Daun Badipala (Serai Malabar)",
        "Daun Bambu", "Daun Bawang Merah",
        "Daun Bayam Malabar (Gendola)", "Daun Biduri", "Daun Biji Poppy India",
        "Daun Brahmi (Pegagan India)", "Daun Brotowali", "Daun Bunga Kenop",
        "Daun Bunga Marigold", "Daun Bunga Tecoma", "Daun Bunga Thumbe (Leucas)",
        "Daun Cabai", "Daun Cempaka Kuning",         "Daun Chakte (Teak India)", "Daun Delima", "Daun Ganigale",
"Daun Handeuleum", "Daun Honge (Pongamia)",
        "Daun Inggu (Common Rue)", "Daun Insulin", "Daun Jahe",
        "Daun Jamblang (Jamun)", "Daun Jambu Biji", "Daun Jarak",
        "Daun Jeruk Lemon", "Daun Jeruk Sitrun",
        "Daun Jintan India (Indian Borage)", "Daun Kacang Buncis",
        "Daun Kacang Polong", "Daun Kambajala (Hop Bush)", "Daun Kamboja Jepang",
        "Daun Kapur Barus", "Daun Kari", "Daun Kasambruga (Trengguli)",
        "Daun Kasturi (Musk Mallow)", "Daun Kayu Putih (Eucalyptus)",
         "Daun Kelor", "Daun Kemangi Suci (Tulsi)",
        "Daun Kembang Sepatu", "Daun Ketumbar", "Daun Kohlrabi (Kubis Rabi)",
        "Daun Kopi", "Daun Kunyit", "Daun Labu Kuning", "Daun Lada (Merica)",
        "Daun Lantana", "Daun Lidah Buaya", "Daun Lobak", "Daun Mangga",
        "Daun Mawar", "Daun Melati", "Daun Mengkudu", "Daun Mimba (Neem)",
        "Daun Mint", "Daun Nangka", "Daun Pacar Kuku (Henna)",
        "Daun Padri (Night Jasmine)", "Daun Paria Gunung", "Daun Parijata",
        "Daun Patikan Kebo", "Daun Pepaya", "Daun Sambiloto",
        "Daun Sawo", "Daun Serai",
        "Daun Sirih", "Daun Srikaya", "Daun Talas", "Daun Tapak Dara",
        "Daun Tomat", "Daun Urang-Aring",
    ]

NUM_CLASSES = len(CLASS_NAMES)  # Harus 75

# =============================================================================
# Transform - HARUS sama dengan test_transforms saat training
# =============================================================================

inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# =============================================================================
# PREDICTION SERVICE - MobileNetV2 + Non-Plant Detection
# =============================================================================

class PredictionService:
    """Service untuk memuat model MobileNetV2 dan melakukan prediksi."""

    def __init__(self):
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load model MobileNetV2 .pth ke memory. Dipanggil saat startup."""
        model_path = MODEL_MOBILENET_DIR / "mobilenetv2_best.pth"

        self._model = create_mobilenetv2(
            num_classes=NUM_CLASSES, pretrained=False
        )

        state_dict = torch.load(
            model_path, map_location=self._device, weights_only=True
        )
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

        print("[PredictionService] MobileNetV2 loaded on {}".format(self._device))
        print("  Path: {}".format(model_path))
        print("  Classes: {}".format(NUM_CLASSES))

    def predict(self, image_bytes, top_k=5):
        """
        Prediksi gambar tanaman dari bytes.

        Jika confidence tertinggi < CONFIDENCE_THRESHOLD, dianggap bukan
        tanaman obat dan mengembalikan response khusus.

        Returns dict:
            class_name, confidence, top_predictions, plant_info, is_plant
        """
        if self._model is None:
            raise RuntimeError("Model belum di-load. Panggil load_model() dulu.")

        # 1. Baca gambar
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Transform
        tensor = inference_transform(image).unsqueeze(0).to(self._device)

        # 3. Inference
        with torch.no_grad():
            outputs = self._model(tensor)
            probabilities = F.softmax(outputs, dim=1)

        # 4. Top-k predictions
        k = min(top_k, NUM_CLASSES)
        top_probs, top_indices = torch.topk(probabilities, k)
        top_probs = top_probs.squeeze().cpu().tolist()
        top_indices = top_indices.squeeze().cpu().tolist()

        if not isinstance(top_probs, list):
            top_probs = [top_probs]
            top_indices = [top_indices]

        # 5. Map ke nama kelas (Bahasa Indonesia, prefix "Daun")
        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            top_predictions.append({
                "class_name": CLASS_NAMES[idx],
                "confidence": round(prob, 4),
            })

        # 6. Cek confidence threshold - deteksi non-tanaman
        best_confidence = top_probs[0]

        if best_confidence < CONFIDENCE_THRESHOLD:
            # Confidence terlalu rendah - kemungkinan bukan tanaman obat
            return {
                "class_name": None,
                "confidence": round(best_confidence, 4),
                "model": "mobilenetv2",
                "is_plant": False,
                "message": (
                    "Objek tidak dikenali sebagai tanaman obat. "
                    "Pastikan Anda memfoto daun tanaman dengan jelas dan "
                    "pencahayaan yang cukup."
                ),
                "top_predictions": top_predictions,
                "plant_info": None,
            }

        # 7. Best prediction + plant info
        best_class = CLASS_NAMES[top_indices[0]]
        plant_info = get_plant_info_safe(best_class)

        return {
            "class_name": best_class,
            "confidence": round(best_confidence, 4),
            "model": "mobilenetv2",
            "is_plant": True,
            "top_predictions": top_predictions,
            "plant_info": plant_info,
        }

    @property
    def is_loaded(self):
        return self._model is not None


# Singleton instance
prediction_service = PredictionService()
