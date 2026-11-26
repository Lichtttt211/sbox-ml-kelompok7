"""Aplikasi Web Streamlit — Konstruksi S-box AES Berbasis SAGA untuk Enkripsi Citra.

Proyek Akhir Kriptografi — Kelompok 7 — Ilmu Komputer UNNES 2026:
  - Cahya Din Ramadhan (2404130018)
  - Muhammad Nabil Fikri (2404130167)
  - Fathan Nabil Rahman (2404130140)
  - Rashya Farellyndo Revanza (2404130132)
"""
import hashlib
import io
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import aes_core as ac
import sbox_core as sc

st.set_page_config(
    page_title="S-box K7 (SAGA) · Enkripsi Citra AES-128",
    page_icon="🔐",
    layout="wide",
)

# CSS ringan untuk mencegah pemotongan angka metrik dan memperjelas kartu ringkasan
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        white-space: nowrap;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    div[data-testid="metric-container"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_KEY = "KRIPTOGRAFI-KLP7"
DEFAULT_IV = "000102030405060708090a0b0c0d0e0f"
ANGGOTA = [
    ("Cahya Din Ramadhan", "2404130018", "Eksplorasi 30 Polinomial & Pemodelan ML Surrogate"),
    ("Muhammad Nabil Fikri", "2404130167", "Algoritma Genetik (SAGA), Optimasi C & Uji Dasar S-box"),
    ("Fathan Nabil Rahman", "2404130140", "Uji Lanjutan S-box (BIC, LAP, DAP, Siklus) & Komparasi"),
    ("Rashya Farellyndo Revanza", "2404130132", "Implementasi AES-128, CBC Dua Arah & Evaluasi Citra"),
]


# ---------------- Utilitas & Cache ----------------
def digest_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()[:16]


@st.cache_data(show_spinner=False)
def load_default_image() -> np.ndarray:
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandrill.png")
    return np.array(Image.open(img_path).convert("RGB"))


@st.cache_data(show_spinner=False)
def poly_screening_cached() -> pd.DataFrame:
    return pd.DataFrame(sc.screen_all_30_polynomials())


@st.cache_data(show_spinner=False)
def metrics_cached(sbox_tuple: tuple[int, ...]) -> dict:
    return sc.all_metrics(np.array(sbox_tuple, dtype=np.int64))


@st.cache_data(show_spinner=False)
def detail_metrics_cached(sbox_tuple: tuple[int, ...]) -> dict:
    S = np.array(sbox_tuple, dtype=np.int64)
    return {
        "bal": sc.balance(S),
        "bij": sc.bijective(S),
        "nl": sc.nonlinearity(S),
        "sac": sc.sac(S),
        "bic": sc.bic(S),
        "lap": sc.lap_detail(S),
        "dap": sc.dap_detail(S),
        "fp": sc.fixed_and_opposite_points(S),
        "cycles": sc.cycle_lengths(S),
    }


@st.cache_data(show_spinner=False)
def comparison_table_cached(extra_sbox: tuple[int, ...] | None = None) -> pd.DataFrame:
    refs = sc.reference_sboxes()
    if extra_sbox is not None:
        refs["Hasil SAGA Sesi Ini"] = np.array(extra_sbox, dtype=np.int64)
    rows = []
    for nama, S in refs.items():
        rows.append({"S-box": nama, **metrics_cached(tuple(int(v) for v in S))})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=6)
def _run_saga_cached(_img: np.ndarray, img_sig: str, poly_choice: int | None, pop: int, gens: int, seed: int, baselines: bool):
    return sc.construct_sbox_saga(_img, poly=poly_choice, pop_size=pop, generations=gens, seed=seed, run_baselines=baselines)


@st.cache_data(show_spinner=False, max_entries=8)
def _encrypt_cached(_img: np.ndarray, img_sig: str, key: bytes, iv: bytes, sbox_tuple: tuple[int, ...], mode: str):
    enc = ac.encrypt_image(_img, key, iv, list(sbox_tuple), mode=mode)
    dec = ac.decrypt_image(enc["cipher_flat"], _img.shape, key, iv, list(sbox_tuple), mode=mode)
    return enc, dec


@st.cache_data(show_spinner=False)
def fips_ok_cached() -> bool:
    return ac.verify_fips197(sc.SBOX_AES)


def sbox_df(S) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(S, dtype=np.int64).reshape(16, 16))
    df.index = [f"0x{i:X}_" for i in range(16)]
    df.columns = [f"_0x{j:X}" for j in range(16)]
    return df


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def format_sbox_txt(S) -> str:
    S = [int(v) for v in S]
    inv_s = [int(v) for v in np.argsort(S)]
    lines = ["S_BOX = ["]
    for i in range(16):
        lines.append("    " + ", ".join(f"{v:3d}" for v in S[i * 16:(i + 1) * 16]) + ",")
    lines.append("]\n")
    lines.append("INV_S_BOX = [")
    for i in range(16):
        lines.append("    " + ", ".join(f"{v:3d}" for v in inv_s[i * 16:(i + 1) * 16]) + ",")
    lines.append("]\n")
    return "\n".join(lines)


def plot_heatmap(M: np.ndarray, title: str, vmin: float, vmax: float, fmt: str = "{:.3f}"):
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_histogram(img: np.ndarray, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7), sharey=True)
    for k, (ax, col, name) in enumerate(zip(axes, ["#dc2626", "#16a34a", "#2563eb"], "RGB")):
        ax.bar(range(256), np.bincount(img[..., k].ravel(), minlength=256), color=col, width=1.0)
        ax.axhline(img[..., k].size / 256.0, color="#0f172a", ls="--", lw=0.9)
        ax.set_title(f"Kanal {name}", fontsize=9)
        ax.set_xlim(0, 255)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_correlation_scatter(plain: np.ndarray, cipher: np.ndarray):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.2))
    dirs = [("horizontal", "Horizontal"), ("vertical", "Vertikal"), ("diagonal", "Diagonal")]
    for col_idx, (dkey, dlabel) in enumerate(dirs):
        a_p, b_p = ac.correlation_pairs(plain[..., 0], direction=dkey, n=2000, seed=7)
        a_c, b_c = ac.correlation_pairs(cipher[..., 0], direction=dkey, n=2000, seed=7)
        axes[0, col_idx].scatter(a_p, b_p, s=2, alpha=0.35, color="#0f766e")
        axes[0, col_idx].set_title(f"Plain (R) · {dlabel}", fontsize=9)
        axes[0, col_idx].set_xlim(0, 255); axes[0, col_idx].set_ylim(0, 255)
        axes[1, col_idx].scatter(a_c, b_c, s=2, alpha=0.35, color="#2563eb")
        axes[1, col_idx].set_title(f"Cipher (R) · {dlabel}", fontsize=9)
        axes[1, col_idx].set_xlim(0, 255); axes[1, col_idx].set_ylim(0, 255)
    axes[0, 0].set_ylabel("Piksel tetangga")
    axes[1, 0].set_ylabel("Piksel tetangga")
    for ax in axes[1]:
        ax.set_xlabel("Piksel acuan")
    fig.suptitle("Sebaran Korelasi Piksel Bertetangga (Plain vs Cipher)", fontsize=10.5)
    fig.tight_layout()
    return fig


# ---------------- Sidebar: Navigasi, Identitas & Pilih S-box ----------------
st.sidebar.title("🔐 S-box K7 · SAGA")
st.sidebar.caption("Proyek Akhir Kriptografi · **Kelompok 7** · Ilmu Komputer UNNES 2026")

halaman = st.sidebar.radio(
    "Navigasi Halaman",
    ["Beranda", "Konstruksi S-box (ML + GA)", "Uji & Perbandingan S-box", "Enkripsi & Uji Citra (AES-128)"],
)

st.sidebar.divider()
opsi_sbox = list(sc.reference_sboxes().keys())
if "saga_sbox" in st.session_state:
    opsi_sbox.insert(1, "Hasil SAGA Sesi Ini")
opsi_sbox.append("Unggah S-box sendiri")

pilihan = st.sidebar.selectbox("S-box aktif", opsi_sbox)

if pilihan == "Hasil SAGA Sesi Ini":
    S_AKTIF = np.array(st.session_state["saga_sbox"], dtype=np.int64)
elif pilihan == "Unggah S-box sendiri":
    up = st.sidebar.file_uploader("File S-box (.txt / .csv, 256 angka)", type=["txt", "csv"])
    teks = st.sidebar.text_area("…atau tempel 256 angka (desimal / 0x..)", height=95)
    S_AKTIF = None
    try:
        if up is not None and up.name.lower().endswith(".csv"):
            df_up = pd.read_csv(up, index_col=0)
            if df_up.shape == (16, 16):
                S_AKTIF = df_up.values.astype(np.int64).ravel()
            else:
                S_AKTIF = sc.parse_sbox(up.getvalue().decode())
        elif up is not None:
            S_AKTIF = sc.parse_sbox(up.getvalue().decode())
        elif teks.strip():
            S_AKTIF = sc.parse_sbox(teks)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(str(e))
    if S_AKTIF is None:
        st.sidebar.info("Belum ada S-box diunggah. Sementara menggunakan S-box K7 (Kelompok 7).")
        S_AKTIF = sc.SBOX_K7.copy()
else:
    S_AKTIF = sc.reference_sboxes()[pilihan]

S_TUPLE = tuple(int(v) for v in S_AKTIF)

st.sidebar.divider()
with st.sidebar.expander("Anggota Kelompok 7", expanded=False):
    for nama, nim, _ in ANGGOTA:
        st.markdown(f"- **{nama}** (`{nim}`)")


# ---------------- Halaman 1: Beranda ----------------
if halaman == "Beranda":
    st.title("Konstruksi S-box AES Berbasis Surrogate-Assisted Genetic Algorithm (SAGA) untuk Enkripsi Citra")
    st.caption("Proyek Akhir Mata Kuliah Kriptografi · **Kelompok 7** · Departemen Ilmu Komputer, Universitas Negeri Semarang (2026)")

    st.markdown(
        """
Aplikasi ini mendemonstrasikan perancangan, pengujian kriptografi, dan implementasi enkripsi citra **AES-128**
menggunakan S-box hasil optimasi **Machine Learning + Algoritma Genetik (SAGA)** milik **Kelompok 7**:

1. **Eksplorasi 30 Polinomial Irreducible $\\text{GF}(2^8)$** — menindaklanjuti saran *future research* Alamsyah dkk. (2025)
   untuk memperluas pencarian melintasi seluruh 30 polinomial irreducible derajat 8, menghasilkan polinomial terbaik `0x17B` ($x^8+x^6+x^5+x^4+x^3+x+1$).
2. **Ablasi 4 Model ML Surrogate** — membandingkan `KNN (64-bit)`, `Random Forest (64-bit)`, `Ridge 1-Baris (bag-of-rows)`,
   dan model usulan **`Ridge Komposit (Row + Pairwise-XOR)`** yang secara eksplisit menangkap interaksi antar-pasangan bit untuk **BIC-SAC**.
3. **Surrogate-Assisted Genetic Algorithm (SAGA) + Optimasi Konstanta $C$** — mencari matriks affine non-sirkulan $A \\in GL(8, 2)$
   untuk mengoptimalkan **SAC** dan **BIC-SAC** secara simultan, dilanjutkan pemilihan konstanta $C = \\texttt{0xEE}$ untuk menjamin
   **0 Fixed Point**, **0 Opposite Fixed Point**, dan **1 siklus tunggal penuh [256]**.
4. **Implementasi Enkripsi Citra AES-128** — mengganti S-box standar pada `SubBytes` & `KeyExpansion` (tervalidasi vektor uji resmi **FIPS-197**),
   mode **CBC dua arah dengan *domain-separated IV***, serta uji lengkap Histogram ($\\chi^2$), Entropy, Korelasi H/V/D, NPCR, dan UACI (Wu dkk., 2011).
        """
    )

    m = metrics_cached(S_TUPLE)
    st.subheader(f"Ringkasan Metrik: {pilihan}")
    r1 = st.columns(4)
    r1[0].metric("Nonlinearity (NL)", f"{m['NL']}", help="Ideal bijective 8-bit: 112")
    r1[1].metric("SAC Rata-rata", f"{m['SAC']:.5f}", help="Ideal: 0.50000")
    r1[2].metric("Deviasi SAC", f"{m['Deviasi SAC']:.5f}", help="Rata-rata |M_ij - 0.5|, makin kecil makin baik")
    r1[3].metric("BIC-SAC Rata-rata", f"{m['BIC-SAC']:.5f}", help="Ideal: 0.50000")

    r2 = st.columns(4)
    r2[0].metric("Deviasi BIC-SAC", f"{m['Deviasi BIC-SAC']:.5f}", help="Rata-rata |B_jk - 0.5|, makin kecil makin baik")
    r2[1].metric("LAP (Bias) / DAP", f"{m['LAP (bias)']:.4f} / {m['DAP']:.6f}", help="Makin kecil makin tahan serangan linear & diferensial")
    r2[2].metric("Skor S / SV (2025)", f"{m['Skor S (2025)']:.5f} / {m['Skor SV (2025)']:.5f}", help="Metrik gabungan Alamsyah dkk. (2025): ideal S=0, SV min=16")
    r2[3].metric("FP / OFP / Siklus", f"{m['FP']} / {m['OFP']} / {m['Siklus']}", help="Fixed Point, Opposite Fixed Point, Jumlah Siklus (Panjang Maks)")

    st.subheader("Perbandingan Singkat: S-box K7 vs Referensi Utama")
    df_cmp = comparison_table_cached(tuple(int(v) for v in st.session_state["saga_sbox"]) if "saga_sbox" in st.session_state else None)
    st.dataframe(df_cmp, hide_index=True, use_container_width=True)

    st.subheader("Tim Pengembang — Kelompok 7 (Ilmu Komputer UNNES)")
    st.dataframe(
        pd.DataFrame(ANGGOTA, columns=["Nama Lengkap", "NIM", "Kontribusi Utama"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Referensi: Daemen & Rijmen (2002) FIPS-197; Wu, Noonan & Agaian (2011); "
        "Alamsyah, Prasetiyo & Muhammad (2023), SJI 10(2); "
        "Alamsyah dkk. (2025), Nonlinear Dynamics 113:3869–3890."
    )


# ---------------- Halaman 2: Konstruksi S-box (ML + GA) ----------------
elif halaman == "Konstruksi S-box (ML + GA)":
    st.title("Konstruksi S-box dengan Surrogate-Assisted Genetic Algorithm (SAGA)")
    st.markdown(
        """
**Mengapa perlu ML Surrogate + Algoritma Genetik?**
- **SAC** bergantung pada 8 baris matriks affine $r_0, \\dots, r_7$, sehingga model 1-baris (`bag-of-rows`) dapat memprediksi SAC,
  **tetapi gagal memprediksi BIC-SAC** ($R^2 < 0$) karena BIC-SAC bergantung pada **28 pasangan XOR** $r_j \\oplus r_k$.
- Model **Ridge Komposit (`Row + Pairwise-XOR`)** mempelajari bobot kontribusi baris tunggal sekaligus pasangan XOR dari dataset citra.
- Karena 8 baris saling berkait secara kuadratik lewat 28 pasangan XOR, optimasi simultan **SAC + BIC-SAC** di ruang $GL(8,2)$
  ($\\approx 5{,}35 \\times 10^{18}$ matriks) diselesaikan oleh **Algoritma Genetik** yang dipandu ML Surrogate.
- Terakhir, seluruh 256 konstanta 8-bit $C$ dievaluasi untuk mencapai **0 Fixed Point, 0 Opposite Fixed Point, dan 1 siklus penuh [256]**
  tanpa mengubah nilai NL, SAC, BIC, LAP, maupun DAP.
        """
    )

    with st.expander("Langkah 1 · Tabel Skrining Seluruh 30 Polinomial Irreducible GF(2⁸)", expanded=False):
        df_poly = poly_screening_cached()
        st.dataframe(df_poly.drop(columns=["poly_int"]), hide_index=True, use_container_width=True, height=320)
        st.caption(
            "Polinomial 0x17B (x^8 + x^6 + x^5 + x^4 + x^3 + x + 1) menempati peringkat #1 berdasarkan skor gabungan "
            "Top-16 Deviasi SAC + Top-32 Deviasi BIC-SAC."
        )

    st.subheader("Langkah 2 · Konfigurasi Dataset & SAGA")
    col1, col2 = st.columns([1.0, 1.3])
    with col1:
        sumber = st.radio("Citra sumber dataset", ["Mandrill bawaan (512×512)", "Unggah citra lain"], horizontal=True)
        img_src = load_default_image()
        if sumber == "Unggah citra lain":
            f = st.file_uploader("Unggah citra (PNG/JPG/BMP/TIFF)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="saga_img")
            if f is not None:
                img_src = np.array(Image.open(f).convert("RGB"))
        st.image(img_src, caption=f"Citra sumber ({img_src.shape[1]}×{img_src.shape[0]} RGB)", width=220)

    with col2:
        df_poly = poly_screening_cached()
        poly_options = {"Otomatis — Terbaik #1 (0x17B)": None}
        for _, row in df_poly.iterrows():
            tag = ""
            if row["poly_int"] == 0x11B:
                tag = " [AES / Alamsyah 2025]"
            elif row["poly_int"] == 0x1F3:
                tag = " [Alamsyah 2023]"
            poly_options[f"{row['Polinomial (Hex)']} · {row['Bentuk Polinomial']}{tag}"] = int(row["poly_int"])
        poly_label = st.selectbox("Polinomial Irreducible GF(2⁸)", list(poly_options.keys()))
        poly_choice = poly_options[poly_label]

        p_col = st.columns(3)
        pop_size = int(p_col[0].number_input("Ukuran populasi GA", min_value=60, max_value=240, value=140, step=20))
        generations = int(p_col[1].number_input("Jumlah generasi GA", min_value=60, max_value=300, value=160, step=20))
        seed = int(p_col[2].number_input("Random seed", min_value=1, max_value=999, value=7, step=1))
        run_baselines = st.checkbox("Sertakan pelatihan pembanding KNN & Random Forest (lebih lambat ~3–5 dtk)", value=True)

    current_sig = (digest_array(img_src), poly_choice, pop_size, generations, seed, run_baselines)

    if st.button("🚀 Jalankan Konstruksi S-box (SAGA)", type="primary"):
        with st.spinner("Membangun dataset invertible, melatih 4 model ML, menjalankan SAGA, dan mengoptimasi konstanta C..."):
            t_wall = time.perf_counter()
            res = _run_saga_cached(img_src, current_sig[0], poly_choice, pop_size, generations, seed, run_baselines)
            wall_dt = time.perf_counter() - t_wall
            from_cache = wall_dt < 0.08 and res["elapsed"] > 0.25
            st.session_state["saga_res"] = res
            st.session_state["saga_sig"] = current_sig
            st.session_state["saga_cached"] = from_cache
            st.session_state["saga_sbox"] = res["sbox"].tolist()

    if "saga_res" in st.session_state:
        if st.session_state.get("saga_sig") != current_sig:
            st.warning("Parameter atau citra telah diubah sejak eksekusi terakhir. Klik **Jalankan Konstruksi S-box (SAGA)** untuk memperbarui hasil.")

        r = st.session_state["saga_res"]
        cache_note = " (dari cache)" if st.session_state.get("saga_cached") else ""

        c = st.columns(4)
        c[0].metric("Kandidat / Invertible", f"{r['n_kandidat']:,} / {r['n_dataset']:,}".replace(",", "."))
        c[1].metric("Polinomial Terpilih", r["poly_hex"], help=r["poly_str"])
        c[2].metric("Konstanta C Optimal", f"{r['c_opt']['c_hex']} ({r['c_opt']['c_int']})", help=f"Siklus: {r['c_opt']['cycles']}")
        c[3].metric(f"Waktu Komputasi{cache_note}", f"{r['elapsed']:.2f} dtk")

        st.subheader("Ablasi & Evaluasi Model Machine Learning (Data Uji 20%)")
        df_eval = pd.DataFrame(r["evaluasi"], columns=["Model ML", "Target Prediksi", "R² (Uji)", "MAE (Uji)"])
        st.dataframe(df_eval.round(6), hide_index=True, use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
        y_bic = r["y_bic_te"]
        lim = [float(y_bic.min()), float(y_bic.max())]
        for ax, key, judul in [
            (axes[0], "ridge_row_bic", "Ridge 1-Baris → Deviasi BIC-SAC (Gagal)"),
            (axes[1], "ridge_pair_bic", "Ridge Komposit Pairwise-XOR → Deviasi BIC-SAC (Akurat)"),
        ]:
            pred = r["pred_plot"][key]
            ax.scatter(y_bic, pred, s=8, alpha=0.45, color="#0f766e" if "pair" in key else "#dc2626")
            ax.plot(lim, lim, "k--", lw=1)
            ax.set_title(judul, fontsize=9.5)
            ax.set_xlabel("Deviasi BIC-SAC aktual")
            ax.set_ylabel("Prediksi model")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Konvergensi Surrogate-Assisted Genetic Algorithm (SAGA)")
        fig2, ax2 = plt.subplots(figsize=(9.5, 3.2))
        gens_x = np.arange(len(r["ga_best_history"]))
        ax2.plot(gens_x, r["ga_best_history"], label="Best Fitness (Elit)", color="#0f766e", lw=2)
        ax2.plot(gens_x, r["ga_mean_history"], label="Rata-rata Populasi", color="#64748b", ls="--", lw=1.3)
        ax2.axhline(r["best_dataset_joint"], color="#dc2626", ls=":", lw=1.4, label=f"Terbaik Dataset tanpa SAGA ({r['best_dataset_joint']:.5f})")
        ax2.set_xlabel("Generasi")
        ax2.set_ylabel("Skor Rugi Multi-Objektif (makin kecil makin baik)")
        ax2.set_title(f"Konvergensi SAGA pada Polinomial {r['poly_hex']} (Turun dari {r['ga_best_history'][0]:.5f} ke {r['ga_best_history'][-1]:.5f})")
        ax2.legend(fontsize=8.5)
        ax2.grid(alpha=0.25)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        m_new = sc.all_metrics(r["sbox"])
        m_ds = sc.all_metrics(r["best_dataset_sbox"])
        st.success(
            f"**S-box hasil SAGA ({r['poly_hex']}, C={r['c_opt']['c_hex']}):** "
            f"SAC = **{m_new['SAC']:.5f}** (dev {m_new['Deviasi SAC']:.5f}), "
            f"BIC-SAC = **{m_new['BIC-SAC']:.5f}** (dev {m_new['Deviasi BIC-SAC']:.5f}), "
            f"Skor S = **{m_new['Skor S (2025)']:.5f}**, SV = **{m_new['Skor SV (2025)']:.5f}**, "
            f"FP = **{m_new['FP']}**, OFP = **{m_new['OFP']}**, Siklus = **{m_new['Siklus']}** "
            f"(terbaik dataset tanpa SAGA: dev SAC {m_ds['Deviasi SAC']:.5f}, dev BIC-SAC {m_ds['Deviasi BIC-SAC']:.5f}). "
            + ("Identik 100% dengan S-box K7 Kelompok 7." if np.array_equal(r["sbox"], sc.SBOX_K7) else "")
        )

        col_m, col_s = st.columns([1.0, 1.8])
        with col_m:
            st.markdown(f"**Matriks Affine $A$ ($8 \\times 8$, baris = `{r['rows']}`)**")
            st.dataframe(pd.DataFrame(r["A"], columns=[f"b{j}" for j in range(8)]), height=320, use_container_width=True)
            st.caption(
                f"Optimasi C: ditemukan {r['c_opt']['n_zero_fp_ofp']} konstanta ber-0 FP & 0 OFP, "
                f"dan {r['c_opt']['n_single_cycle_zero_fp_ofp']} di antaranya bersiklus tunggal [256]."
            )
        with col_s:
            st.markdown("**Tabel S-box Hasil Konstruksi ($16 \\times 16$)**")
            st.dataframe(sbox_df(r["sbox"]), height=420, use_container_width=True)

        d1, d2, d3 = st.columns(3)
        d1.download_button("Unduh S-box (.txt)", format_sbox_txt(r["sbox"]), "sbox_k7.txt", "text/plain")
        d2.download_button("Unduh S-box (.csv)", sbox_df(r["sbox"]).to_csv(), "sbox_k7.csv", "text/csv")
        meta_json = json.dumps({
            "kelompok": "Kelompok 7 - Ilmu Komputer UNNES 2026",
            "polynomial_hex": r["poly_hex"],
            "polynomial": r["poly_str"],
            "affine_rows": r["rows"],
            "constant_c_hex": r["c_opt"]["c_hex"],
            "metrics": m_new,
            "sbox": [int(v) for v in r["sbox"]],
        }, indent=2)
        d3.download_button("Unduh Metadata Lengkap (.json)", meta_json, "sbox_k7_metadata.json", "application/json")


# ---------------- Halaman 3: Uji & Perbandingan S-box ----------------
elif halaman == "Uji & Perbandingan S-box":
    st.title("Pengujian Kriptografi & Perbandingan S-box")
    st.caption(f"S-box aktif: **{pilihan}**")

    S = np.array(S_TUPLE, dtype=np.int64)
    m = metrics_cached(S_TUPLE)
    det = detail_metrics_cached(S_TUPLE)

    with st.expander("Lihat Tabel S-box & Invers S-box (16×16)", expanded=False):
        c_t1, c_t2 = st.columns(2)
        c_t1.markdown("**S-box ($16 \\times 16$)**")
        c_t1.dataframe(sbox_df(S), height=420, use_container_width=True)
        c_t2.markdown("**Invers S-box $S^{-1}$ ($16 \\times 16$)**")
        if det["bij"]:
            c_t2.dataframe(sbox_df(np.argsort(S)), height=420, use_container_width=True)
        else:
            c_t2.error("S-box tidak bijektif sehingga tidak memiliki invers.")

    # 3 baris x 4 kolom -> angka desimal penuh tanpa pemotongan '...'
    g1 = st.columns(4)
    g1[0].metric("Balance", m["Balance"], help="Tiap bit keluaran wajib 128 bit 0 dan 128 bit 1")
    g1[1].metric("Bijective", m["Bijective"], help="Seluruh 256 nilai 0..255 muncul unik")
    g1[2].metric("NL (Min / 255 Komp.)", f"{det['nl']['min']} / {det['nl']['min_komponen']}", help="Ideal: 112")
    g1[3].metric("BIC-NL (Min)", f"{det['bic']['nl_min']}", help="Ideal: 112")

    g2 = st.columns(4)
    g2[0].metric("SAC Rata-rata", f"{det['sac']['avg']:.5f}", help="Ideal: 0.50000")
    g2[1].metric("Deviasi SAC", f"{det['sac']['deviasi']:.5f}", help="Rentang: " + f"{det['sac']['min']:.4f}–{det['sac']['max']:.4f}")
    g2[2].metric("BIC-SAC Rata-rata", f"{det['bic']['sac_avg']:.5f}", help="Ideal: 0.50000")
    g2[3].metric("Deviasi BIC-SAC", f"{det['bic']['sac_deviasi']:.5f}", help="Rentang: " + f"{det['bic']['sac_min']:.4f}–{det['bic']['sac_max']:.4f}")

    g3 = st.columns(4)
    g3[0].metric("LAP (Bias)", f"{det['lap']['bias']:.6f}", help=f"Probabilitas maks: {det['lap']['prob']:.4f} (korelasi maks {det['lap']['max_corr']})")
    g3[1].metric("DAP", f"{det['dap']['dap']:.6f}", help=f"Maks {det['dap']['max_count']}/256 di Δx={det['dap']['dx']}, Δy={det['dap']['dy']}")
    g3[2].metric("Skor S / SV (2025)", f"{m['Skor S (2025)']:.5f} / {m['Skor SV (2025)']:.5f}", help="Rumus Alamsyah dkk. (2025) Pers. 9 & 20")
    g3[3].metric("FP / OFP / Siklus", f"{det['fp']['fp']} / {det['fp']['ofp']} / {m['Siklus']}", help=f"Panjang siklus: {det['cycles']}")

    t1, t2, t3, t4, t5 = st.tabs(["Balance & NL", "Matriks SAC", "BIC-NL & BIC-SAC", "LAT (LAP) & DDT (DAP)", "Perbandingan"])
    with t1:
        df_bn = pd.DataFrame(
            {
                "Jumlah bit 0": det["bal"]["jumlah_0"],
                "Jumlah bit 1": det["bal"]["jumlah_1"],
                "Nonlinearity": det["nl"]["per_bit"],
            },
            index=[f"f{j}" for j in range(8)],
        ).T
        st.dataframe(df_bn, use_container_width=True)
        st.caption(
            f"NL minimum dari seluruh 255 fungsi komponen tak-nol = **{det['nl']['min_komponen']}** (ideal 112). "
            f"Fixed Point: `{det['fp']['fp_list']}` · Opposite Fixed Point: `{det['fp']['ofp_list']}` · "
            f"Struktur Siklus: `{det['cycles']}`."
        )

    with t2:
        c1, c2 = st.columns([1.05, 1.0])
        fig_sac = plot_heatmap(det["sac"]["matrix"], "Matriks SAC (baris: bit input i, kolom: bit output j)", 0.42, 0.58)
        c1.pyplot(fig_sac)
        plt.close(fig_sac)
        c2.dataframe(pd.DataFrame(det["sac"]["matrix"], index=[f"in b{i}" for i in range(8)], columns=[f"out b{j}" for j in range(8)]).round(4), use_container_width=True)
        c2.caption(f"Rata-rata: **{det['sac']['avg']:.5f}** · Deviasi: **{det['sac']['deviasi']:.5f}** · Rentang: {det['sac']['min']:.4f}–{det['sac']['max']:.4f}")

    with t3:
        c1, c2 = st.columns(2)
        fig_bnl = plot_heatmap(det["bic"]["nl_matrix"], "Matriks BIC-NL", 104, 112, "{:.0f}")
        fig_bsac = plot_heatmap(det["bic"]["sac_matrix"], "Matriks BIC-SAC", 0.46, 0.54)
        c1.pyplot(fig_bnl); plt.close(fig_bnl)
        c2.pyplot(fig_bsac); plt.close(fig_bsac)
        st.caption(
            f"BIC-NL: min **{det['bic']['nl_min']}**, rata-rata **{det['bic']['nl_avg']:.2f}** · "
            f"BIC-SAC: rata-rata **{det['bic']['sac_avg']:.5f}**, deviasi **{det['bic']['sac_deviasi']:.5f}**, "
            f"rentang {det['bic']['sac_min']:.4f}–{det['bic']['sac_max']:.4f}."
        )

    with t4:
        c1, c2 = st.columns(2)
        fig_lat, ax_lat = plt.subplots(figsize=(5.0, 4.1))
        im1 = ax_lat.imshow(det["lap"]["lat"][1:, 1:], aspect="auto", cmap="RdBu_r", vmin=-32, vmax=32)
        ax_lat.set_title(f"LAT (Bias maks = {det['lap']['bias']:.4f}, Prob = {det['lap']['prob']:.4f})", fontsize=9.5)
        ax_lat.set_xlabel("Mask output b"); ax_lat.set_ylabel("Mask input a")
        fig_lat.colorbar(im1, ax=ax_lat, fraction=0.046)
        fig_lat.tight_layout()
        c1.pyplot(fig_lat); plt.close(fig_lat)

        fig_ddt, ax_ddt = plt.subplots(figsize=(5.0, 4.1))
        im2 = ax_ddt.imshow(det["dap"]["ddt"][1:], aspect="auto", cmap="viridis")
        ax_ddt.set_title(f"DDT (Maks = {det['dap']['max_count']}/256, DAP = {det['dap']['dap']:.6f})", fontsize=9.5)
        ax_ddt.set_xlabel("Δy (output)"); ax_ddt.set_ylabel("Δx (input)")
        fig_ddt.colorbar(im2, ax=ax_ddt, fraction=0.046)
        fig_ddt.tight_layout()
        c2.pyplot(fig_ddt); plt.close(fig_ddt)
        st.caption(
            "Catatan definisi LAP: sesuai Alamsyah dkk. (2025) Pers. (7) dan arahan mata kuliah (nilai ideal mendekati 0), "
            "LAP dilaporkan sebagai bias maksimum |P - 0,5| = 0,0625 (setara probabilitas maksimum P = 0,5625)."
        )

    with t5:
        df_cmp = comparison_table_cached(tuple(int(v) for v in st.session_state["saga_sbox"]) if "saga_sbox" in st.session_state else None)
        st.dataframe(df_cmp, hide_index=True, use_container_width=True)

        fig_cmp, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
        labels = [s.split(" (")[0] + ("\n" + s.split(" (")[1][:-1] if " (" in s else "") for s in df_cmp["S-box"]]
        colors = ["#0f766e" if "K7" in s else "#94a3b8" for s in df_cmp["S-box"]]
        for ax, col_name, ttl in [
            (axes[0], "Deviasi SAC", "Deviasi SAC (makin kecil makin baik)"),
            (axes[1], "Deviasi BIC-SAC", "Deviasi BIC-SAC (makin kecil makin baik)"),
            (axes[2], "Skor S (2025)", "Skor S Alamsyah 2025 (ideal = 0)"),
        ]:
            bars = ax.bar(range(len(labels)), df_cmp[col_name], color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=7, rotation=18, ha="right")
            ax.set_title(ttl, fontsize=9)
            ax.bar_label(bars, fmt="%.4f", fontsize=7.5)
        fig_cmp.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)
        st.caption(
            "Semua S-box berbasis invers GF(2⁸) memiliki NL=112, BIC-NL=112, LAP=0,0625, dan DAP=0,015625 yang setara. "
            "S-box K7 unggul tajam pada Deviasi SAC, Deviasi BIC-SAC, Skor S=0,00000, SV=16,00000, 0 FP, 0 OFP, serta 1 siklus penuh [256]."
        )


# ---------------- Halaman 4: Enkripsi & Uji Citra (AES-128) ----------------
else:
    st.title("Enkripsi & Evaluasi Keamanan Citra dengan AES-128")
    fips_badge = "✔ Lolos Vektor Uji Resmi FIPS-197 Appendix C.1" if fips_ok_cached() else "✘ Gagal FIPS-197"
    st.caption(f"S-box aktif: **{pilihan}** · SubBytes & KeyExpansion memakai S-box aktif · MixColumns GF(2⁸) 0x11B · {fips_badge}")

    col_a, col_b = st.columns([1.0, 1.1])
    with col_a:
        f_img = st.file_uploader("Unggah citra (kosongkan untuk Mandrill 512×512)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        img = np.array(Image.open(f_img).convert("RGB")) if f_img is not None else load_default_image()
        if img.shape[0] * img.shape[1] > 1024 * 1024:
            st.warning("Citra > 1 megapiksel membutuhkan waktu enkripsi lebih lama di Python murni. Disarankan ≤ 512×512.")
    with col_b:
        key_txt = st.text_input("Kunci AES-128 (tepat 16 karakter ASCII)", DEFAULT_KEY, max_chars=16)
        iv_hex = st.text_input("IV (32 digit heksadesimal = 16 byte)", DEFAULT_IV)
        mode = st.selectbox(
            "Mode Operasi Blok",
            ["CBC Dua Arah", "CBC Satu Arah", "ECB"],
            help="CBC Dua Arah menjalankan lintasan maju (IV) lalu lintasan mundur (IV_rev = IV XOR 0xA5) sehingga perubahan 1 piksel merambat ke seluruh citra.",
        )

    try:
        key = key_txt.encode("utf-8")
        iv = bytes.fromhex(iv_hex.strip())
        assert len(key) == 16, "Kunci harus tepat 16 byte (16 karakter ASCII)."
        assert len(iv) == 16, "IV harus tepat 16 byte (32 digit heksadesimal)."
        ac.AES128(key, S_TUPLE)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()

    img_sig = digest_array(img)
    enc_sig = (pilihan, S_TUPLE, img_sig, key, iv, mode)

    if st.button("🔐 Enkripsi, Dekripsi & Evaluasi Citra", type="primary"):
        with st.spinner("Menjalankan enkripsi & dekripsi AES-128..."):
            t_wall = time.perf_counter()
            enc, dec = _encrypt_cached(img, img_sig, key, iv, S_TUPLE, mode)
            from_cache = (time.perf_counter() - t_wall) < 0.05 and enc["elapsed"] > 0.15
        st.session_state["enc_res"] = {
            "enc": enc,
            "dec": dec,
            "img": img,
            "sig": enc_sig,
            "sbox_name": pilihan,
            "mode": mode,
            "key": key,
            "iv": iv,
            "sbox": S_TUPLE,
            "from_cache": from_cache,
        }

    if "enc_res" in st.session_state:
        r = st.session_state["enc_res"]
        if r["sig"] != enc_sig:
            st.warning(
                f"Hasil di bawah ini berasal dari konfigurasi sebelumnya (**{r['sbox_name']}**, mode **{r['mode']}**). "
                "Klik **Enkripsi, Dekripsi & Evaluasi Citra** untuk memperbarui sesuai konfigurasi aktif."
            )

        im_plain, im_cipher, im_dec = r["img"], r["enc"]["cipher_img"], r["dec"]["dec_img"]
        c_vis = st.columns(3)
        c_vis[0].image(im_plain, caption=f"Citra Asli ({im_plain.shape[1]}×{im_plain.shape[0]})", use_container_width=True)
        c_vis[1].image(im_cipher, caption=f"Citra Terenkripsi ({r['sbox_name']} · {r['mode']})", use_container_width=True)
        c_vis[2].image(im_dec, caption="Citra Hasil Dekripsi", use_container_width=True)

        sama = np.array_equal(im_plain, im_dec)
        cache_lbl = " (dari cache)" if r["from_cache"] else ""
        (st.success if sama else st.error)(
            f"Dekripsi {'identik 100% lossless' if sama else 'TIDAK identik'} · "
            f"piksel berbeda: **{int((im_plain != im_dec).sum())}** · "
            f"waktu enkripsi **{r['enc']['elapsed']:.2f} dtk** · dekripsi **{r['dec']['elapsed']:.2f} dtk**{cache_lbl}"
        )

        if r["enc"]["pad"] > 0:
            st.info(
                f"Ukuran byte citra bukan kelipatan 16 sehingga ditambahkan {r['enc']['pad']} byte padding. "
                "Dekripsi lossless memakai seluruh blok terenkripsi (`cipher_flat`, dapat diunduh sebagai `.bin`)."
            )

        dl_cols = st.columns(3 if r["enc"]["pad"] > 0 else 2)
        dl_cols[0].download_button("Unduh Citra Terenkripsi (PNG)", to_png_bytes(im_cipher), "citra_terenkripsi_k7.png", "image/png")
        dl_cols[1].download_button("Unduh Citra Dekripsi (PNG)", to_png_bytes(im_dec), "citra_dekripsi_k7.png", "image/png")
        if r["enc"]["pad"] > 0:
            dl_cols[2].download_button("Unduh Ciphertext Penuh (.bin)", r["enc"]["cipher_flat"].tobytes(), "ciphertext_penuh_k7.bin", "application/octet-stream")

        tab1, tab2, tab3 = st.tabs(["Histogram & Uji χ²", "Entropy & Korelasi (H, V, D)", "NPCR & UACI (Wu dkk., 2011)"])

        with tab1:
            fig_h1 = plot_histogram(im_plain, "Histogram Citra Asli (Plain)")
            fig_h2 = plot_histogram(im_cipher, f"Histogram Citra Terenkripsi — {r['sbox_name']} (garis putus-putus = frekuensi ideal)")
            st.pyplot(fig_h1); plt.close(fig_h1)
            st.pyplot(fig_h2); plt.close(fig_h2)

            chi_rows = []
            for k, ch_name in enumerate("RGB"):
                x_p = ac.chi_square(im_plain[..., k])
                x_c = ac.chi_square(im_cipher[..., k])
                chi_rows.append({
                    "Kanal": ch_name,
                    "Intensitas Muncul (Cipher)": f"{int((np.bincount(im_cipher[..., k].ravel(), minlength=256) > 0).sum())}/256",
                    "χ² Asli": round(x_p, 2),
                    "χ² Terenkripsi": round(x_c, 2),
                    "Status (χ² < 293,25)": "Lolos (Uniform)" if x_c < ac.CHI2_CRIT_005 else "Di atas ambang α=0,05",
                })
            st.dataframe(pd.DataFrame(chi_rows), hide_index=True, use_container_width=True)
            st.caption(f"Nilai kritis Chi-square untuk α = 0,05 dan derajat bebas 255 adalah **{ac.CHI2_CRIT_005}**.")

        with tab2:
            ec_rows = []
            for k, ch_name in enumerate("RGB"):
                ec_rows.append({
                    "Kanal": ch_name,
                    "Entropy Asli": ac.entropy(im_plain[..., k]),
                    "Entropy Cipher": ac.entropy(im_cipher[..., k]),
                    "Korelasi H Asli": ac.correlation(im_plain[..., k], "horizontal"),
                    "Korelasi H Cipher": ac.correlation(im_cipher[..., k], "horizontal"),
                    "Korelasi V Asli": ac.correlation(im_plain[..., k], "vertical"),
                    "Korelasi V Cipher": ac.correlation(im_cipher[..., k], "vertical"),
                    "Korelasi D Asli": ac.correlation(im_plain[..., k], "diagonal"),
                    "Korelasi D Cipher": ac.correlation(im_cipher[..., k], "diagonal"),
                })
            df_ec = pd.DataFrame(ec_rows)
            st.dataframe(df_ec.round(5), hide_index=True, use_container_width=True)
            st.caption(
                f"Rata-rata entropy terenkripsi: **{df_ec['Entropy Cipher'].mean():.5f} bit** (ideal 8,00000). "
                "Korelasi piksel bertetangga jatuh dari ~0,8–0,9 menjadi mendekati 0 di ketiga arah (H, V, D)."
            )
            fig_sc = plot_correlation_scatter(im_plain, im_cipher)
            st.pyplot(fig_sc)
            plt.close(fig_sc)

        with tab3:
            h_im, w_im = im_plain.shape[:2]
            npcr_crit, uaci_lo, uaci_hi = ac.wu_thresholds(h_im, w_im)
            st.markdown(
                f"Ambang lolos uji statistik **Wu, Noonan & Agaian (2011)** untuk citra {w_im}×{h_im} ($\\alpha = 0{{,}}05$): "
                f"**NPCR > {npcr_crit:.4f}%** (ideal {ac.NPCR_IDEAL:.4f}%) dan **{uaci_lo:.4f}% < UACI < {uaci_hi:.4f}%** (ideal {ac.UACI_IDEAL:.4f}%)."
            )

            col_p = st.columns(3)
            py = int(col_p[0].number_input("Baris piksel", 0, h_im - 1, 0))
            px = int(col_p[1].number_input("Kolom piksel", 0, w_im - 1, 0))
            ch_sel = col_p[2].selectbox("Kanal yang diubah (+1 mod 256)", ["R", "G", "B"])

            b_col = st.columns(2)
            if b_col[0].button("Hitung NPCR & UACI (1 Posisi)"):
                im2 = im_plain.copy()
                k_ch = "RGB".index(ch_sel)
                im2[py, px, k_ch] = (int(im2[py, px, k_ch]) + 1) % 256
                with st.spinner("Mengenkripsi citra yang diubah 1 piksel..."):
                    enc2, _ = _encrypt_cached(im2, digest_array(im2), r["key"], r["iv"], r["sbox"], r["mode"])
                c2 = enc2["cipher_img"]
                rows_nu = []
                for k2, nm in enumerate("RGB"):
                    n_, u_ = ac.npcr_uaci(im_cipher[..., k2], c2[..., k2])
                    rows_nu.append({
                        "Kanal": nm,
                        "NPCR (%)": n_,
                        "Status NPCR": "Lolos" if n_ > npcr_crit else "Tidak lolos",
                        "UACI (%)": u_,
                        "Status UACI": "Lolos" if (uaci_lo < u_ < uaci_hi) else "Tidak lolos",
                    })
                df_nu = pd.DataFrame(rows_nu)
                df_nu.loc[len(df_nu)] = {
                    "Kanal": "Rata-rata",
                    "NPCR (%)": df_nu["NPCR (%)"].mean(),
                    "Status NPCR": "Lolos" if df_nu["NPCR (%)"].mean() > npcr_crit else "Tidak lolos",
                    "UACI (%)": df_nu["UACI (%)"].mean(),
                    "Status UACI": "Lolos" if (uaci_lo < df_nu["UACI (%)"].mean() < uaci_hi) else "Tidak lolos",
                }
                st.dataframe(df_nu.round(4), hide_index=True, use_container_width=True)

            if b_col[1].button("Jalankan Uji Lengkap (5 Posisi Wu + Kunci 1-Bit + ECB)"):
                positions = [
                    (0, 0, 0),
                    (h_im // 4, (3 * w_im) // 4, 1),
                    (h_im // 2, w_im // 2, 1),
                    ((3 * h_im) // 4, w_im // 4, 2),
                    (h_im - 1, w_im - 1, 2),
                ]
                with st.spinner("Menjalankan 5 uji diferensial, sensitivitas kunci 1-bit, dan pembanding ECB..."):
                    rows_full = []
                    n_pass, u_pass = 0, 0
                    for iy, ix, ich in positions:
                        im2 = im_plain.copy()
                        im2[iy, ix, ich] = (int(im2[iy, ix, ich]) + 1) % 256
                        e2, _ = _encrypt_cached(im2, digest_array(im2), r["key"], r["iv"], r["sbox"], r["mode"])
                        ns, us = [], []
                        for k2 in range(3):
                            n_, u_ = ac.npcr_uaci(im_cipher[..., k2], e2["cipher_img"][..., k2])
                            ns.append(n_); us.append(u_)
                            n_pass += int(n_ > npcr_crit)
                            u_pass += int(uaci_lo < u_ < uaci_hi)
                        rows_full.append({
                            "Skenario": f"Ubah 1 piksel ({iy}, {ix}) {'RGB'[ich]}",
                            "NPCR R": ns[0], "NPCR G": ns[1], "NPCR B": ns[2], "NPCR Rata2": float(np.mean(ns)),
                            "UACI R": us[0], "UACI G": us[1], "UACI B": us[2], "UACI Rata2": float(np.mean(us)),
                        })

                    # Sensitivitas kunci 1 bit
                    key2 = r["key"][:-1] + bytes([r["key"][-1] ^ 1])
                    ek2, _ = _encrypt_cached(im_plain, digest_array(im_plain), key2, r["iv"], r["sbox"], r["mode"])
                    kn, ku = zip(*[ac.npcr_uaci(im_cipher[..., k2], ek2["cipher_img"][..., k2]) for k2 in range(3)])
                    rows_full.append({
                        "Skenario": "Sensitivitas Kunci (beda 1 bit)",
                        "NPCR R": kn[0], "NPCR G": kn[1], "NPCR B": kn[2], "NPCR Rata2": float(np.mean(kn)),
                        "UACI R": ku[0], "UACI G": ku[1], "UACI B": ku[2], "UACI Rata2": float(np.mean(ku)),
                    })

                    # Pembanding ECB
                    e_ecb1, _ = _encrypt_cached(im_plain, digest_array(im_plain), r["key"], r["iv"], r["sbox"], "ECB")
                    im_mid = im_plain.copy()
                    im_mid[h_im // 2, w_im // 2, 1] = (int(im_mid[h_im // 2, w_im // 2, 1]) + 1) % 256
                    e_ecb2, _ = _encrypt_cached(im_mid, digest_array(im_mid), r["key"], r["iv"], r["sbox"], "ECB")
                    en, eu = zip(*[ac.npcr_uaci(e_ecb1["cipher_img"][..., k2], e_ecb2["cipher_img"][..., k2]) for k2 in range(3)])
                    rows_full.append({
                        "Skenario": "Pembanding ECB (ubah 1 piksel tengah)",
                        "NPCR R": en[0], "NPCR G": en[1], "NPCR B": en[2], "NPCR Rata2": float(np.mean(en)),
                        "UACI R": eu[0], "UACI G": eu[1], "UACI B": eu[2], "UACI Rata2": float(np.mean(eu)),
                    })

                df_full = pd.DataFrame(rows_full)
                st.dataframe(df_full.round(4), hide_index=True, use_container_width=True)
                st.success(
                    f"Ringkasan 5 Posisi ({r['mode']}): Rata-rata NPCR = **{df_full.iloc[:5]['NPCR Rata2'].mean():.4f}%**, "
                    f"UACI = **{df_full.iloc[:5]['UACI Rata2'].mean():.4f}%** · "
                    f"Lolos Wu dkk. per kanal: **NPCR {n_pass}/15**, **UACI {u_pass}/15**."
                )
