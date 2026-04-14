# =============================================================================
# routes.py — FastAPI API Endpoints
# =============================================================================

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.prediction_service import prediction_service
from app.services.plant_database import (
    get_plant_info_safe, get_all_plant_names, get_plant_count,
)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": prediction_service.is_loaded,
        "model": "mobilenetv2",
        "total_plants": get_plant_count(),
    }


@router.post("/predict")
async def predict_plant(file: UploadFile = File(...)):
    """
    Prediksi tanaman obat dari gambar daun.
    Menerima file gambar (JPEG/PNG), return JSON lengkap.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPEG/PNG)")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Gagal membaca file: {}".format(e))

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File kosong")

    try:
        result = prediction_service.predict(image_bytes, top_k=5)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail="Model belum siap: {}".format(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error prediksi: {}".format(e))

    plant = result["plant_info"]
    return {
        "success": True,
        "data": {
            "class_name": result["class_name"],
            "nama_indonesia": plant.get("nama_indonesia", result["class_name"]),
            "nama_latin": plant.get("nama_latin", ""),
            "family": plant.get("family", ""),
            "deskripsi": plant.get("deskripsi", ""),
            "confidence": result["confidence"],
            "model": result["model"],
            "manfaat": plant.get("manfaat", []),
            "penyakit": plant.get("penyakit", []),
            "cara_penggunaan": plant.get("cara_penggunaan", []),
            "peringatan": plant.get("peringatan", []),
            "top_predictions": result["top_predictions"],
        },
    }


@router.get("/plants")
async def list_plants():
    """Daftar semua tanaman yang dikenali sistem."""
    names = get_all_plant_names()
    plants = []
    for name in names:
        info = get_plant_info_safe(name)
        plants.append({
            "class_name": name,
            "nama_indonesia": info.get("nama_indonesia", name),
            "nama_latin": info.get("nama_latin", ""),
            "family": info.get("family", ""),
        })
    return {"total": len(plants), "plants": plants}


@router.get("/plants/{class_name}")
async def get_plant(class_name: str):
    """Detail satu tanaman berdasarkan nama kelas."""
    info = get_plant_info_safe(class_name)
    return {"class_name": class_name, **info}
