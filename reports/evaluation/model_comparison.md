# Perbandingan Model: MobileNetV2 vs ResNet50

## Tabel Perbandingan

| Metrik             |   MobileNetV2 |       ResNet50 |
|:-------------------|--------------:|---------------:|
| Accuracy           |   0.9689      |    0.7544      |
| Precision          |   0.9711      |    0.7735      |
| Recall             |   0.9689      |    0.7544      |
| F1-Score           |   0.9689      |    0.7492      |
| Avg Inference (ms) | 626.1         | 1971.81        |
| Model Size (MB)    |   9.2         |   90.7         |
| Total Parameters   |   2.34941e+06 |    2.37088e+07 |

## Rekomendasi

**Model terbaik untuk production: MOBILENETV2**

Model ini akan di-load oleh FastAPI backend untuk endpoint `/predict`.
