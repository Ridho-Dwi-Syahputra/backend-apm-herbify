# 📓 Panduan Notebook & Script Referensi

## Struktur Folder

```
notebooks/
├── 📄 01_eda.ipynb                          ← Notebook Jupyter (dikerjakan di VS Code)
├── 📄 02_preprocessing.ipynb               ← Notebook Jupyter
├── 📄 03_training_mobilenetv2.ipynb         ← Notebook Jupyter
├── 📄 04_training_resnet50.ipynb            ← Notebook Jupyter
├── 📄 05_model_comparison.ipynb             ← Notebook Jupyter
├── 📄 NOTEBOOK_GUIDE.md                     ← File ini
│
└── 📂 scripts/                              ← Script Python referensi (acuan untuk notebook)
    ├── 📄 01_eda.py
    ├── 📄 02_preprocessing.py
    ├── 📄 03_training_mobilenetv2.py
    ├── 📄 04_training_resnet50.py
    └── 📄 05_model_comparison.py
```

---

## Apa itu Folder `scripts/`?

Folder `scripts/` berisi **kode Python referensi** yang menjadi **acuan/blueprint** untuk setiap notebook `.ipynb`. Script ini ditulis di Antigravity (AI coding assistant) karena Antigravity tidak bisa langsung mengedit file `.ipynb`.

### Cara Menggunakannya

1. **Buka notebook** `.ipynb` di **VS Code** (dengan extension Jupyter)
2. **Buka script** `.py` yang sesuai dari folder `scripts/`
3. **Salin cell-by-cell** dari script ke notebook
   - Setiap blok yang dimulai dengan `# %% [CELL X]` = satu cell di notebook
4. **Jalankan** cell di notebook satu per satu
5. **Modifikasi** sesuai kebutuhan (script hanya acuan, boleh diubah)

### Kenapa Pakai Script Referensi?

| Alasan | Penjelasan |
| :--- | :--- |
| **Antigravity limitation** | Antigravity tidak bisa membuat/mengedit file `.ipynb` |
| **Dokumentasi kode** | Script berisi komentar lengkap dan penjelasan setiap langkah |
| **Version control** | File `.py` lebih mudah di-review di Git dibanding `.ipynb` |
| **Konsistensi** | Memastikan semua anggota tim mengikuti alur yang sama |
| **Copy-paste ready** | Setiap `# %% [CELL]` langsung bisa jadi satu cell di notebook |

---

## Urutan Pengerjaan Notebook

```
┌─────────────────────────────┐
│ 1. 01_eda.ipynb              │  ← Pahami data dulu!
│    Script: 01_eda.py         │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│ 2. 02_preprocessing.ipynb    │  ← Bersihkan data & analisis imbalance
│    Script: 02_preprocessing.py│
│    Output: preprocessing_metadata.json
└──────────┬──────────────────┘
           ↓
┌───────────────────────────────────────┐
│ 3. 03_training_mobilenetv2.ipynb       │  ← Training model pertama
│    Script: 03_training_mobilenetv2.py  │
│    Output: mobilenetv2_best.pth        │
└──────────┬────────────────────────────┘
           ↓
┌───────────────────────────────────────┐
│ 4. 04_training_resnet50.ipynb          │  ← Training model kedua
│    Script: 04_training_resnet50.py     │
│    Output: resnet50_best.pth           │
└──────────┬────────────────────────────┘
           ↓
┌───────────────────────────────────────┐
│ 5. 05_model_comparison.ipynb           │  ← Bandingkan & pilih model terbaik
│    Script: 05_model_comparison.py      │
│    Output: reports/ (grafik, tabel)    │
└───────────────────────────────────────┘
```

---

## Detail Setiap Script

### `01_eda.py` — Exploratory Data Analysis

**Tujuan:** Memahami dataset sebelum preprocessing & training.

**Mencakup kedua dataset:**
- `Medicinal plant dataset` (40 kelas)
- `Medicinal Leaf dataset` (80 kelas)

**Yang dilakukan:**

| Cell | Isi | Output |
| :--- | :--- | :--- |
| 1-2 | Import library & definisi path | - |
| 3-4 | Scan dataset, hitung gambar per kelas | DataFrame info |
| 5-6 | Bar chart distribusi kelas | `reports/eda_distribusi_kelas_*.png` |
| 7 | Cek imbalance ratio | Print ratio |
| 8 | Histogram resolusi gambar | `reports/eda_resolusi_gambar.png` |
| 9-10 | Grid sampel gambar | `reports/eda_sampel_*.png` |
| 11 | Analisis channel RGB (mean & std) | `reports/eda_rgb_*.png` |
| 12 | Deteksi duplikat | Print hasil |
| 13 | Info format & mode gambar | Print hasil |
| 14 | Preview efek augmentasi | `reports/eda_preview_augmentasi.png` |
| 15 | Ringkasan EDA | Print summary |

---

### `02_preprocessing.py` — Data Cleaning & Preprocessing

**Tujuan:** Membersihkan dataset dari file corrupt, menganalisis class imbalance, dan menghitung class weights.

**PENTING:** Jalankan SETELAH EDA dan SEBELUM training!

**Yang dilakukan:**

| Cell | Isi | Output |
| :--- | :--- | :--- |
| 1 | Import libraries & definisi path | Setup, folder karantina dibuat |
| 2 | Scan seluruh file di kedua dataset | Inventarisasi file valid & corrupt |
| 3 | Detail file corrupt (extension, error) | Print detail setiap file corrupt |
| 4 | Pindahkan file corrupt ke `_quarantine/` | File corrupt dipindahkan (bukan dihapus) |
| 5 | Analisis distribusi kelas (setelah cleaning) | Print statistik & imbalance ratio |
| 6 | Visualisasi distribusi kelas bersih | `reports/preprocessing_distribusi_bersih.png` |
| 7 | Hitung class weights untuk loss function | Print tabel class weights |
| 8 | Simpan metadata preprocessing ke JSON | `reports/preprocessing_metadata.json` |
| 9 | Verifikasi dataset bersih & ringkasan | Print summary final |

> **Catatan:** Script ini TIDAK mengubah kode di `app/`. File corrupt hanya dipindahkan ke folder karantina, bukan dihapus permanen.

---

### `03_training_mobilenetv2.py` — Training MobileNetV2

**Tujuan:** Training model MobileNetV2 dengan transfer learning.

**Depends on:**
- `app/services/preprocessing.py` — Pipeline data (resize, normalisasi, augmentasi, split)
- `app/models/cnn_models.py` — Definisi arsitektur MobileNetV2
- `app/utils/config.py` — Hyperparameter (learning rate, epochs, batch size)

**Yang dilakukan:**

| Cell | Isi | Output |
| :--- | :--- | :--- |
| 1-2 | Import & load data via preprocessing pipeline | DataLoaders |
| 3 | Buat model MobileNetV2 (transfer learning) | Model di GPU/CPU |
| 4 | Setup loss, optimizer, scheduler | - |
| 5 | Training loop (epoch-by-epoch) | Print progress |
| 6 | Simpan model & history | `mobilenetv2_best.pth`, `mobilenetv2_history.json` |
| 7 | Grafik training (loss & accuracy) | `training_history_mobilenetv2.png` |
| 8 | Quick evaluation di test set | Print akurasi |

---

### `04_training_resnet50.py` — Training ResNet50

**Tujuan:** Training model ResNet50 dengan transfer learning.

**Depends on:** Sama seperti notebook 03—menggunakan kode dari `app/` yang sama.

**Struktur:** Identik dengan `03_training_mobilenetv2.py`, hanya model yang berbeda.

---

### `05_model_comparison.py` — Evaluasi & Perbandingan

**Tujuan:** Membandingkan kedua model dan menentukan model terbaik untuk production.

**Depends on:**
- `trained_models/mobilenetv2_best.pth` — Output dari notebook 03
- `trained_models/resnet50_best.pth` — Output dari notebook 04
- `trained_models/*_history.json` — History training kedua model

**Yang dilakukan:**

| Cell | Isi | Output |
| :--- | :--- | :--- |
| 1-2 | Import & load kedua model dari `.pth` | Models loaded |
| 3 | Evaluasi lengkap (accuracy, precision, recall, F1) | Print metrik |
| 4 | Confusion matrix kedua model | `confusion_matrix_*.png` |
| 5 | Bar chart F1-score per kelas (side-by-side) | `per_class_f1_comparison.png` |
| 6 | Overlay grafik val loss & val accuracy | `training_comparison.png` |
| 7 | Tabel perbandingan final | Print tabel |
| 8 | Simpan hasil evaluasi ke JSON | `evaluation_results.json` |
| 9 | Rekomendasi model untuk production | `model_comparison.md` |

---

## Dependensi Antar File

```
app/utils/config.py          ← Konfigurasi global (path, hyperparameter)
        ↓
app/services/preprocessing.py ← Pipeline data (dipakai notebook 03, 04, 05)
        ↓
app/models/cnn_models.py      ← Definisi model (dipakai notebook 03, 04, 05)
        ↓
notebooks/02                  ← Cleaning data (TIDAK import dari app/)
        ↓
notebooks/03 & 04             ← Training → menghasilkan file .pth
        ↓
notebooks/05                  ← Evaluasi → memuat file .pth dari 03 & 04
        ↓
trained_models/*.pth          ← Digunakan oleh backend API (FastAPI)
```

> **Penting:** Sebelum menjalankan notebook 03-05, pastikan kode di `app/` sudah diimplementasikan (preprocessing, model definition, config). Kode `app/` dikerjakan di Antigravity, notebook dikerjakan di VS Code.
