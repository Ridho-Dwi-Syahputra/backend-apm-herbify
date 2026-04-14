# Herbify Backend (FastAPI) 🌿

Backend Machine Learning untuk mendeteksi jenis tanaman obat menggunakan algoritma MobileNetV2.

## 🚀 Persyaratan Sistem
Sebelum memulai, pastikan kamu sudah menginstall:
- **Python 3.10+** (disarankan)
- **Git**

## 📥 Cara Clone dan Install

**1. Clone Repository Bebas di folder mana aja (Misal di Desktop)**
Buka Terminal/CMD, lalu jalankan:
```bash
git clone https://github.com/Ridho-Dwi-Syahputra/backend-apm-herbify.git
cd backend-apm-herbify
```

**2. Buat Virtual Environment (Sangat Disarankan)**
Ini supaya library python untuk project ini tidak tercampur dengan project lain di komputermu.
```bash
python -m venv .venv
```

**3. Aktifkan Virtual Environment**
- Windows (CMD/PowerShell):
  ```bash
  .venv\Scripts\activate
  ```
- MacOS/Linux:
  ```bash
  source .venv/bin/activate
  ```
*(Pastikan ada tulisan `(.venv)` di awal baris terminal kamu).*

**4. Install Library/Dependency**
Setelah environment aktif, install semua paket yang dibutuhkan (PyTorch, FastAPI, dll):
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Cara Menjalankan Backend

Jalankan perintah ini di dalam folder `backend-apm-herbify` tempat kamu berada:
```bash
python -m uvicorn main:app --reload --port 8000
```
Backend akan otomatis berjalan pada: **http://127.0.0.1:8000** atau **http://localhost:8000**

**Endpoints Penting:**
- Dokumentasi API (Swagger UI): `http://127.0.0.1:8000/docs`
- Endpoint Prediksi Gambar: `POST /api/predict/`

**Note Troubleshooting:**
- Jika gagal run dengan error _"module not found"_, pastikan kamu sudah mengaktifkan `.venv` sebelum merun `uvicorn`.
- Gunakan fitur port forwarding dari VS Code atau aplikasi **ngrok** (`ngrok http 8000`) jika ingin membagikan atau menyambungkan ke HP Android.
