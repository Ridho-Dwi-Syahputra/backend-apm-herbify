# =============================================================================
# plant_database.py — Service untuk mengakses data tanaman obat
# =============================================================================

import json
from pathlib import Path

_DATA_FILE = Path(__file__).parent.parent / "data" / "plant_data.json"
_plant_data = None


def _load_data():
    """Load plant_data.json ke memory (cached)."""
    global _plant_data
    if _plant_data is None:
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            _plant_data = json.load(f)
        print("[PlantDB] Loaded {} plant entries".format(len(_plant_data)))
    return _plant_data


def get_plant_info(class_name):
    """Cari info tanaman berdasarkan nama kelas. Return None jika tidak ada."""
    data = _load_data()
    return data.get(class_name)


def get_plant_info_safe(class_name):
    """
    Sama seperti get_plant_info, tapi return entry default jika tidak ditemukan.
    Agar frontend selalu mendapat data (tidak crash).
    """
    info = get_plant_info(class_name)
    if info:
        return info

    return {
        "nama_indonesia": class_name.replace("_", " ").title(),
        "nama_latin": "Tidak diketahui",
        "family": "Tidak diketahui",
        "deskripsi": "Tanaman {} teridentifikasi oleh model AI. Informasi detail belum tersedia.".format(
            class_name.replace("_", " ")
        ),
        "manfaat": [{"judul": "Data Belum Tersedia", "deskripsi": "Informasi manfaat sedang dalam proses pengumpulan."}],
        "penyakit": [],
        "cara_penggunaan": [{"langkah": "Konsultasi", "deskripsi": "Konsultasikan dengan ahli herbal untuk penggunaan yang tepat."}],
        "peringatan": [{"judul": "Perhatian", "deskripsi": "Konsultasikan dengan profesional kesehatan sebelum menggunakan tanaman obat."}],
    }


def get_all_plant_names():
    """Return daftar semua nama kelas di database."""
    data = _load_data()
    return sorted(data.keys())


def get_plant_count():
    """Return jumlah tanaman dalam database."""
    data = _load_data()
    return len(data)
