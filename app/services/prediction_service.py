# =============================================================================
# prediction_service.py — Service prediksi gambar tanaman obat
# =============================================================================
# Hanya menggunakan MobileNetV2 (akurasi ~96.88%)
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
from app.utils.config import MODEL_MOBILENET_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# =============================================================================
# CLASS NAMES — Diambil langsung dari sorted(os.listdir(_combined_dataset))
# Urutan ini HARUS sama persis dengan saat training (ImageFolder sorts A-Z)
# =============================================================================

# Load dari file yang sudah diverifikasi
_CLASS_NAMES_FILE = Path(__file__).parent.parent.parent / "class_names.json"

if _CLASS_NAMES_FILE.exists():
    with open(_CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
else:
    # Hardcoded fallback — harus cocok dengan sorted(os.listdir())
    CLASS_NAMES = [
        "Aloevera", "Amla", "Amruta_Balli", "Amruthaballi", "Arali",
        "Ashoka", "Ashwagandha", "Astma_weed", "Avacado", "Badipala",
        "Balloon_Vine", "Bamboo", "Basale", "Beans", "Betel",
        "Betel_Nut", "Bhrami", "Brahmi", "Bringaraja", "Caricature",
        "Castor", "Catharanthus", "Chakte", "Chilly",
        "Citron lime (herelikai)", "Coffee", "Common rue(naagdalli)",
        "Coriender", "Curry", "Curry_Leaf", "Doddapatre", "Doddpathre",
        "Drumstick", "Ekka", "Eucalyptus", "Ganigale", "Ganike",
        "Gasagase", "Gauva", "Geranium", "Ginger", "Globe Amarnath",
        "Guava", "Henna", "Hibiscus", "Honge", "Insulin", "Jackfruit",
        "Jasmine", "Kambajala", "Kasambruga", "Kohlrabi", "Lantana",
        "Lemon", "Lemon_grass", "Lemongrass", "Malabar_Nut",
        "Malabar_Spinach", "Mango", "Marigold", "Mint", "Nagadali",
        "Neem", "Nelavembu", "Nerale", "Nithyapushpa", "Nooni", "Onion",
        "Padri", "Palak(Spinach)", "Papaya", "Pappaya", "Parijatha",
        "Pea", "Pepper", "Pomegranate", "Pomoegranate", "Pumpkin",
        "Raddish", "Raktachandini", "Rose", "Sampige", "Sapota",
        "Seethaashoka", "Seethapala", "Spinach1", "Tamarind", "Taro",
        "Tecoma", "Thumbe", "Tomato", "Tulasi", "Tulsi", "Turmeric",
        "Wood_sorel", "camphor", "kamakasturi", "kepala",
    ]

NUM_CLASSES = len(CLASS_NAMES)  # Harus 98

# =============================================================================
# Transform — HARUS sama dengan test_transforms saat training
# =============================================================================

inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# =============================================================================
# PREDICTION SERVICE — Hanya MobileNetV2
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

        Returns dict:
            class_name, confidence, top_predictions, plant_info
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

        # 5. Map ke nama kelas
        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            top_predictions.append({
                "class_name": CLASS_NAMES[idx],
                "confidence": round(prob, 4),
            })

        # 6. Best prediction + plant info
        best_class = CLASS_NAMES[top_indices[0]]
        best_confidence = top_probs[0]
        plant_info = get_plant_info_safe(best_class)

        return {
            "class_name": best_class,
            "confidence": round(best_confidence, 4),
            "model": "mobilenetv2",
            "top_predictions": top_predictions,
            "plant_info": plant_info,
        }

    @property
    def is_loaded(self):
        return self._model is not None


# Singleton instance
prediction_service = PredictionService()
