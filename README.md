# Studio 2 — Klasifikasi Sampah · Kelompok 05 Pria Batak

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

Model klasifikasi gambar untuk lima kategori sampah menggunakan EfficientNet-B0 dan PyTorch. Submission mengikuti kontrak evaluator resmi melalui `submission/model.py` dan bobot terlatih `submission/model.pth`.

## Anggota Kelompok

| No. | Nama Lengkap | NIM | Kontribusi utama |
|---:|---|---|---|
| 1 | Prima Agusta Sembiring | 123140119 | Training dan evaluasi model |
| 2 | Gohan Tua Jeremia Ambarita | 123140160 | EDA dan dokumentasi |
| 3 | Rian Rafael Sangap Tamba | 123140190 | Eksperimen, persiapan submission, dan pengujian |

## Ringkasan Pendekatan

- **Arsitektur / backbone:** EfficientNet-B0 dengan classifier ber-output 5 kelas.
- **Strategi pelatihan:** seed `42`, total 10 epoch, 2 epoch awal dengan backbone dibekukan, lalu pelatihan dilanjutkan. Learning rate classifier/head `0.001`, learning rate backbone `0.0001`, dan weight decay `0.0001`. Bobot terbaik dipilih berdasarkan Macro-F1 validasi.
- **Data tambahan:** Tidak menggunakan data tambahan di luar dataset kompetisi.
- **Ukuran input & preprocessing:** input `224 × 224`; gambar diubah ukurannya menjadi 256, lalu `CenterCrop(224)`, dikonversi ke tensor, dan dinormalisasi menggunakan mean serta standard deviation ImageNet.
- **Pembagian data:** 2.753 gambar secara keseluruhan, dengan split train/validasi 80:20 (`seed=42`): 2.202 gambar train dan 551 gambar validasi.
- **Lingkungan dan library:** Python, Jupyter Notebook, PyTorch, dan Torchvision.

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
| Preview 1 (14 Okt) | Belum dicatat | **0.9563** (epoch terbaik: 5) | Belum diumumkan |
| Preview 2 (21 Okt) | — | — | — |
| Final (25 Okt) | — | — | — |

Metrik tambahan pada split validasi lokal untuk epoch terbaik:

- **Validation accuracy:** 0.9564
- **Macro precision:** 0.9559
- **Macro recall:** 0.9572

> Skor di atas merupakan hasil validasi lokal dari notebook training dan belum tentu sama dengan skor hidden validation yang dihitung oleh asisten dosen. Isi kolom commit dengan hash commit yang diserahkan untuk putaran terkait, serta isi kolom Macro-F1 diumumkan asisten setelah hasil resmi tersedia.

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
├── README.md
├── evaluate.py                 # Evaluator resmi dari dosen
├── .gitignore
├── notebooks/
│   └── 01_EDA.ipynb
│   └── 02_efficientnet_b0.ipynb
└── submission/
    ├── model.py                # get_model() dan get_transform()
    └── model.pth               # state_dict bobot terbaik
```
