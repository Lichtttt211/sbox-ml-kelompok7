# Konstruksi S-box AES Berbasis Surrogate-Assisted Genetic Algorithm (SAGA) untuk Enkripsi Citra

**Proyek Akhir Mata Kuliah Kriptografi — Kelompok 7**  
Departemen Ilmu Komputer, Universitas Negeri Semarang (UNNES), 2026

## Anggota Kelompok 7

| No | Nama | NIM | Pembagian Tugas Utama |
|---|---|---|---|
| 1 | **Cahya Din Ramadhan** | `2404130018` | Eksplorasi 30 polinomial irreducible $\text{GF}(2^8)$, ekstraksi dataset matriks, & pemodelan/ablasi ML Surrogate (KNN, Random Forest, Ridge 1-baris vs Ridge Row+Pairwise-XOR) |
| 2 | **Muhammad Nabil Fikri** | `2404130167` | Optimasi multi-objektif dengan *Surrogate-Assisted Genetic Algorithm* (SAGA), optimasi konstanta 8-bit $C$, konstruksi `S-box K7`, & pengujian dasar (Balance, Bijective, NL, SAC) |
| 3 | **Fathan Nabil Rahman** | `2404130140` | Pengujian lanjutan (BIC-NL, BIC-SAC, LAP, DAP, Fixed Point, Opposite Fixed Point, Cycle Structure) & perbandingan komprehensif terhadap AES, Alamsyah (2023), dan Alamsyah (2025) `S-box₄₄` |
| 4 | **Rashya Farellyndo Revanza** | `2404130132` | Implementasi AES-128 T-table (tervalidasi FIPS-197), mode CBC dua arah dengan *domain-separated IV*, enkripsi/dekripsi citra, & analisis Histogram ($\chi^2$), Entropy, Korelasi H/V/D, NPCR, dan UACI |

---

## Ringkasan Metodologi & Keunggulan `S-box K7`

Menindaklanjuti saran *future research* Alamsyah dkk. (2025) yang mengusulkan penggabungan eksplorasi matriks affine dengan seluruh polinomial irreducible di $\text{GF}(2^8)$, Kelompok 7 merancang pipeline **SAGA (*Surrogate-Assisted Genetic Algorithm*)**:

1. **Skrining 30 Polinomial Irreducible $\text{GF}(2^8)$:** Mengevaluasi kualitas pool komponen non-nol dari seluruh 30 polinomial irreducible derajat 8, menghasilkan polinomial terbaik `0x17B` ($x^8 + x^6 + x^5 + x^4 + x^3 + x + 1$).
2. **Ablasi Model Machine Learning (4 Model):**
   - `KNN (64-bit mentah)` & `Random Forest (64-bit mentah)` → gagal mempelajari struktur aljabar S-box ($R^2 \approx 0$).
   - `Ridge 1-Baris (bag-of-rows)` → akurat memprediksi SAC, namun **gagal memprediksi BIC-SAC** ($R^2 < 0$) karena BIC-SAC bergantung pada XOR antar-pasangan baris.
   - `Ridge Komposit Usulan (Row + Pairwise-XOR)` → memprediksi **SAC dan BIC-SAC secara akurat** ($R^2 \approx 1$).
3. **Surrogate-Assisted Genetic Algorithm (SAGA):** Karena optimasi simultan SAC (8 baris $r_j$) dan BIC-SAC (28 pasangan XOR $r_j \oplus r_k$) memiliki keterkaitan kuadratik di ruang matriks affine non-sirkulan $GL(8,2)$, model ML digunakan sebagai *surrogate fitness* cepat di dalam Algoritma Genetik.
4. **Optimasi Konstanta 8-bit $C$:** Memilih $C = \texttt{0xEE}$ (`238`) yang menjaga NL, SAC, BIC, LAP, dan DAP invarian sekaligus menjamin **0 Fixed Point**, **0 Opposite Fixed Point**, dan **1 siklus tunggal penuh sepanjang 256 elemen**.

### Hasil Perbandingan Metrik S-box

| S-box | Polinomial | NL | SAC | Dev. SAC | BIC-NL | BIC-SAC | Dev. BIC-SAC | LAP (bias) | DAP | Skor $S$ (2025) | Skor $SV$ (2025) | FP / OFP | Siklus |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **S-box K7 (Kelompok 7)** | `0x17B` | **112** | **0.50000** | **0.01562** | **112** | **0.50000** | **0.00446** | **0.0625** | **0.015625** | **0.00000** | **16.00000** | **0 / 0** | **1 (256)** |
| Alamsyah 2025 (`S-box₄₄`) | `0x11B` | 112 | 0.50073 | 0.02515 | 112 | 0.50237 | 0.00949 | 0.0625 | 0.015625 | 0.00155 | 16.00310 | 1 / 2 | 7 (124) |
| AES Standar (FIPS-197) | `0x11B` | 112 | 0.50488 | 0.02637 | 112 | 0.50460 | 0.01060 | 0.0625 | 0.015625 | 0.00474 | 16.00949 | 0 / 0 | 5 (87) |
| Alamsyah 2023 (`A0`) | `0x1F3` | 112 | 0.50146 | 0.02490 | 112 | 0.49972 | 0.00921 | 0.0625 | 0.015625 | 0.00087 | 16.00174 | 1 / 1 | 5 (129) |

---

## Struktur Folder

- `app.py` — Aplikasi web interaktif Streamlit (4 halaman: Beranda, Konstruksi SAGA, Uji & Perbandingan S-box, Enkripsi & Uji Citra).
- `sbox_core.py` — Modul inti $\text{GF}(2^8)$, eksplorasi 30 polinomial, ML surrogate, SAGA, optimasi $C$, dan seluruh metrik uji S-box.
- `aes_core.py` — Modul AES-128 berbasis T-table (FIPS-197), mode CBC dua arah dengan *domain-separated IV*, ECB, dan metrik citra.
- `Kelompok_7_Proyek_Akhir_Kriptografi.ipynb` — Notebook Google Colab lengkap dan *Run-All safe* dari awal sampai akhir.
- `mandrill.png` — Citra uji standar $512 \times 512$ RGB.
- `requirements.txt` & `.streamlit/config.toml` — Konfigurasi dependensi dan tema Streamlit.


## Referensi

1. Daemen, J., & Rijmen, V. (2002). *The Design of Rijndael: AES — The Advanced Encryption Standard*. Springer. (FIPS-197).
2. Wu, Y., Noonan, J. P., & Agaian, S. (2011). NPCR and UACI randomness tests for image encryption. *Cyber Journals: JSAT*, 1(2), 31–38.
3. Alamsyah, Prasetiyo, B., & Muhammad, A. (2023). S-box Construction on AES Algorithm using Affine Matrix Modification to Improve Image Encryption Security. *Scientific Journal of Informatics*, 10(2).
4. Alamsyah, Setiawan, A., Putra, A. T., Budiman, K., Muslim, M. A., Salahudin, S. N., & Prasetiyo, B. (2025). AES S-box modification uses affine matrices exploration for increased S-box strength. *Nonlinear Dynamics*, 113, 3869–3890.
