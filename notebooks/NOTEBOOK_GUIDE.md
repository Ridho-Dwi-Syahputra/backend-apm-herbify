# Panduan Notebook & Script Referensi

## Struktur Folder

```
notebooks/
|-- 01_eda.ipynb                          <- Notebook Jupyter (dikerjakan di VS Code)
|-- 02_preprocessing.ipynb                <- Notebook Jupyter
|-- 03_training_mobilenetv2.ipynb         <- Notebook Jupyter
|-- 04_training_resnet50.ipynb            <- Notebook Jupyter
|-- 05_model_comparison.ipynb             <- Notebook Jupyter
|-- NOTEBOOK_GUIDE.md                     <- File ini
|
+-- scripts/                              <- Script Python referensi (acuan untuk notebook)
    |-- 01_eda.py
    |-- 02_preprocessing.py
    |-- 03_training_mobilenetv2.py
    |-- 04_training_resnet50.py
    +-- 05_model_comparison.py
```

---

## PENTING: Perubahan Dataset (v2)

Dataset telah melalui proses **merge label ganda**. 9 pasangan kelas duplikat digabungkan:

| Target (Dipertahankan) | Duplikat (Dihapus) |
|:---|:---|
| Amruta_Balli | Amruthaballi |
| Brahmi | Bhrami |
| Curry_Leaf | Curry |
| Doddapatre | Doddpathre |
| Guava | Gauva |
| Lemon_grass | Lemongrass |
| Papaya | Pappaya |
| Pomegranate | Pomoegranate |
| Tulsi | Tulasi |

**Total kelas: 98 -> 89** (setelah merge)

**Class weights** digunakan saat training untuk menangani class imbalance.

---

## Cara Menggunakan Script Referensi

1. **Buka notebook** `.ipynb` di **VS Code** (dengan extension Jupyter)
2. **Buka script** `.py` yang sesuai dari folder `scripts/`
3. **Salin cell-by-cell** dari script ke notebook
   - Setiap blok yang dimulai dengan `# %% [CELL X]` = satu cell di notebook
4. **Jalankan** cell di notebook satu per satu
5. **Modifikasi** sesuai kebutuhan (script hanya acuan, boleh diubah)

---

## Urutan Pengerjaan

```
1. 01_eda.ipynb
   -> Pahami dataset: distribusi 89 kelas, ukuran, kualitas, RGB stats
   -> Lihat hasil merge label ganda
   |
2. 02_preprocessing.ipynb
   -> Bersihkan file corrupt, validasi merge, hitung class weights
   -> Output: preprocessing_metadata.json, class_names.json
   |
3. 03_training_mobilenetv2.ipynb
   -> Training MobileNetV2 DENGAN class weights
   -> Output: mobilenetv2_best.pth
   |
4. 04_training_resnet50.ipynb
   -> Training ResNet50 DENGAN class weights
   -> Output: resnet50_best.pth
   |
5. 05_model_comparison.ipynb
   -> Bandingkan & pilih model terbaik
   -> Output: reports/evaluation/
```

---

## Detail Setiap Script

### `01_eda.py` - Exploratory Data Analysis

| Cell | Isi | Output |
|:---|:---|:---|
| 1 | Import & definisi path | Setup |
| 2 | Scan dataset asli (Plant + Leaf) | DataFrame |
| 3 | Scan combined dataset (89 kelas) | DataFrame |
| 4 | Tabel label ganda yang sudah di-merge | Print |
| 5 | Mapping nama India -> Indonesia | Print tabel |
| 6 | Visualisasi distribusi kelas | eda_distribusi_combined.png |
| 7 | Imbalance ratio | Print analisis |
| 8 | Resolusi gambar | eda_resolusi_gambar.png |
| 9 | Sampel gambar per kelas | eda_sampel_combined.png |
| 10 | Statistik RGB | eda_rgb_combined.png |
| 11 | Deteksi duplikat | Print |
| 12 | Pengecekan format file | Print |
| 13 | Simulasi augmentasi | eda_preview_augmentasi.png |
| 14 | Kesimpulan EDA | Print summary |

### `02_preprocessing.py` - Data Cleaning

| Cell | Isi | Output |
|:---|:---|:---|
| 1 | Import & definisi path | Setup |
| 2 | Scan seluruh file (validasi) | Valid/corrupt counts |
| 3 | Detail file corrupt | Print |
| 4 | Karantina file corrupt | File dipindahkan |
| 5 | Distribusi kelas pasca-cleaning | Tabel |
| 6 | Visualisasi distribusi bersih | preprocessing_distribusi_bersih.png |
| 7 | Kalkulasi class weights | Tabel weights |
| 8 | Ekspor metadata JSON | preprocessing_metadata.json |
| 9 | Verifikasi akhir | Assertions |

### `03_training_mobilenetv2.py` - Training MobileNetV2

| Cell | Isi | Output |
|:---|:---|:---|
| 1-2 | Import & load data | DataLoaders |
| 3 | Buat model MobileNetV2 | Model info |
| 4 | Loss (CLASS WEIGHTS), optimizer, scheduler | Setup |
| 5 | Training loop | Progress per epoch |
| 6 | Simpan model & history | .pth + .json |
| 7 | Grafik training | training_history_mobilenetv2.png |
| 8 | Quick evaluation | Print akurasi |

### `04_training_resnet50.py` - Training ResNet50

Struktur identik dengan notebook 03, hanya model yang berbeda.

### `05_model_comparison.py` - Evaluasi & Perbandingan

| Cell | Isi | Output |
|:---|:---|:---|
| 0 | Catatan dataset (merge, 89 kelas) | - |
| 1 | Import | Setup |
| 2 | Load data & kedua model | Models loaded |
| 3 | Evaluasi lengkap | Metrik per model |
| 4 | Confusion matrix | confusion_matrix_*.png |
| 5 | F1-score per kelas | per_class_f1_comparison.png |
| 6 | Training history comparison | training_comparison.png |
| 7 | Tabel perbandingan | Print tabel |
| 8 | Simpan evaluasi JSON | evaluation_results.json |
| 9 | Rekomendasi model | model_comparison.md |

---

## Dependensi Antar File

```
app/utils/config.py          <- Konfigurasi (89 kelas, paths, CONFIDENCE_THRESHOLD)
        |
app/services/preprocessing.py <- Pipeline data + class weights + merge function
        |
app/models/cnn_models.py      <- Definisi model (MobileNetV2, ResNet50)
        |
notebooks/01 & 02              <- EDA & Cleaning (validasi merge)
        |
notebooks/03 & 04              <- Training -> .pth files
        |
notebooks/05                   <- Evaluasi -> reports/
        |
trained_models/*.pth           <- Digunakan oleh backend API (FastAPI)
        |
prediction_service.py          <- Inference + non-plant detection (threshold 0.5)
```
