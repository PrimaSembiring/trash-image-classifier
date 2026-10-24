# Studio 2 — Klasifikasi Sampah · Kelompok 05 Pria Batak

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

Model klasifikasi gambar untuk lima kategori sampah menggunakan EfficientNet-B0 dan PyTorch. Submission mengikuti kontrak evaluator resmi melalui `submission/model.py` dan bobot terlatih `submission/model.pth`.

## Anggota Kelompok

| No. | Nama Lengkap | NIM | Kontribusi utama |
|---:|---|---|---|
| 1 | Prima Agusta Sembiring | 123140119 | Experimen, Training dan Evaluasi model |
| 2 | Gohan Tua Jeremia Ambarita | 123140160 | EDA dan dokumentasi |
| 3 | Rian Rafael Sangap Tamba | 123140190 | Persiapan submission, dan pengujian |

## Ringkasan Pendekatan

### Eksperimen awal: EfficientNet-B0

EfficientNet-B0 digunakan sebagai model awal karena merupakan arsitektur CNN yang relatif efisien dan lazim dipakai sebagai baseline untuk klasifikasi gambar. Pada hasil validasi lokal yang tercatat, EfficientNet-B0 memperoleh **accuracy 0.9564** dan **Macro-F1 0.9563**. Hasil ini sudah baik sebagai baseline, tetapi masih ada ruang untuk mencoba pendekatan yang mempunyai representasi fitur visual pretrained lebih kuat.

### Beralih ke DINOv2 ViT-S/14

Eksperimen berikutnya menggunakan DINOv2 ViT-S/14, yaitu backbone Vision Transformer dengan bobot pretrained dari metode self-supervised learning. DINOv2 dipilih untuk menguji apakah fitur visual yang dipelajari dari kumpulan gambar besar dapat membantu membedakan kategori sampah pada dataset proyek ini. Skor validasi terbaru DINOv2 yang didapat dalam eksperimen adalah **0.9890**.

## Gambaran Umum DINOv2 ViT-S/14

DINOv2 merupakan keluarga model computer vision yang dikembangkan oleh Meta AI Research dan diumumkan pada **17 April 2023**. Tujuannya adalah mempelajari representasi visual yang berguna untuk berbagai tugas computer vision tanpa memerlukan label kelas pada tahap pretraining. Model pretrained ini dapat digunakan sebagai backbone, lalu disesuaikan dengan tugas tertentu melalui classifier atau fine-tuning.

Sumber resmi:
- [Pengumuman DINOv2 dari Meta AI (17 April 2023)](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)
- [Paper DINOv2 di arXiv](https://arxiv.org/abs/2304.07193)
- [Repository resmi DINOv2](https://github.com/facebookresearch/dinov2)

### Varian utama DINOv2

Repository resmi menyediakan beberapa ukuran backbone berikut. Jumlah parameter adalah perkiraan pada backbone, bukan ukuran dataset pretraining.

| Varian | Perkiraan parameter backbone | Gambaran umum |
|---|---:|---|
| ViT-S/14 | 21 juta | Small; lebih ringan dan sesuai untuk eksperimen dengan sumber daya terbatas |
| ViT-B/14 | 86 juta | Base; lebih besar dan membutuhkan komputasi lebih banyak |
| ViT-L/14 | 300 juta | Large; jauh lebih besar |
| ViT-g/14 | 1,1 miliar | Giant; kebutuhan komputasi dan memori paling tinggi di antara varian utama tersebut |

Tersedia juga varian **with registers**, yaitu model dengan empat register tokens tambahan. Proyek ini menggunakan `vit_small_patch14_dinov2.lvd142m` tanpa register tokens. Karena itu, angka dan deskripsi varian lain di tabel hanya menjadi gambaran umum, bukan model yang digunakan untuk submission.

Rujukan model dan jumlah parameter: [README resmi DINOv2](https://github.com/facebookresearch/dinov2) dan [Model Card DINOv2](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).

## Ringkasan Dataset

| Kelas | Train | Validasi |
|---|---:|---:|
| `botol_plastik` | 418 | 105 |
| `gelas_plastik` | 417 | 104 |
| `kertas_kardus` | 292 | 73 |
| `saset_kemasan` | 538 | 135 |
| `styrofoam` | 537 | 134 |
| **Total** | **2.202** | **551** |

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---|---|---:|---|
| Preview 1 (14 Okt) | 14 Oktober 2026 | **0.9563** | **0.9222** |
| Preview 2 (21 Okt) | — | — | — |
| Final (25 Okt) | 24 Oktober 2026 | **0.9890** |  | 


## Cara Menjalankan Evaluasi

Pastikan Python, PyTorch, Torchvision, dan Pillow sudah tersedia di environment yang digunakan. Dari direktori utama repository, jalankan evaluator resmi dengan folder data validasi yang disusun dalam subfolder sesuai nama kelas:

```bash
python evaluate.py --submission ./submission --data ./my_val
```

Evaluator akan membangun arsitektur melalui `submission/model.py`, memuat bobot dari `submission/model.pth`, memprediksi gambar pada folder data, lalu menampilkan Macro-F1, akurasi, F1 per kelas, dan confusion matrix.

Training dan eksperimen dilakukan melalui notebook `notebooks/02_efficientnet_b0.ipynb`. Evaluasi lokal di atas tidak menjalankan training ulang. Folder `my_val/` hanya untuk pengujian lokal dan tidak perlu diunggah ke repository publik.

## Struktur Repository

```text
trash-image-classifier/
├── .gitignore
├── README.md
├── evaluate.py
├── notebooks/
│   ├── 01_efficientnet_b0.ipynb  # Eksperimen awal / baseline
│   └── 02_dinov2_vits14.ipynb    # Eksperimen utama DINOv2
└── submission/
    ├── model.py                  # get_model() dan get_transform()
    └── model.pth                 # state_dict model final
```
