"""Konstruksi S-box berbasis Surrogate-Assisted Genetic Algorithm (SAGA) dan evaluasi kriptografi.

Proyek Akhir Kriptografi — Kelompok 7 — Ilmu Komputer UNNES 2026:
  - Cahya Din Ramadhan (2404130018)
  - Muhammad Nabil Fikri (2404130167)
  - Fathan Nabil Rahman (2404130140)
  - Rashya Farellyndo Revanza (2404130132)

Metodologi:
  1. Eksplorasi seluruh 30 polinomial irreducible derajat 8 di GF(2^8)
     (menindaklanjuti saran Future Research Alamsyah dkk., 2025).
  2. Pelatihan & ablasi model Machine Learning pada dataset matriks affine invertible
     dari bit-plane citra: KNN (64-bit), Random Forest (64-bit), Ridge 1-baris
     (bag-of-rows), dan Ridge Komposit Usulan (Row + Pairwise-XOR) yang memprediksi
     deviasi & rata-rata SAC sekaligus BIC-SAC.
  3. Surrogate-Assisted Genetic Algorithm (SAGA) di ruang matriks affine non-sirkulan
     GL(8, 2) untuk optimasi simultan SAC dan BIC-SAC.
  4. Optimasi konstanta 8-bit C in {0..255} (invarian terhadap NL, SAC, BIC, LAP, DAP)
     untuk menjamin 0 Fixed Point, 0 Opposite Fixed Point, dan 1 siklus tunggal [256].
"""
import time
import numpy as np

BITS = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.int64)
POW2 = 1 << np.arange(8)
X_ARR = np.arange(256, dtype=np.int64)
PAIR_I, PAIR_J = np.triu_indices(8, k=1)  # 28 pasangan bit output (j < k)

C_AES_INT = 0x63
C_AES = BITS[C_AES_INT]  # [1, 1, 0, 0, 0, 1, 1, 0], urutan b0..b7

# ---------------- Parameter & S-box Optimal Kelompok 7 (Hasil SAGA) ----------------
# Polinomial irreducible terbaik dari 30 kandidat: x^8 + x^6 + x^5 + x^4 + x^3 + x + 1
POLY_K7 = 0x17B
ROWS_K7 = [6, 47, 61, 82, 110, 187, 216, 218]
C_K7_INT = 0xEE  # 238 -> 0 Fixed Point, 0 Opposite Fixed Point, 1 siklus tunggal [256]
A_K7 = BITS[ROWS_K7].tolist()

# Matriks affine acuan penelitian sebelumnya
A_AES = [[1 if (c - r) % 8 in (0, 4, 5, 6, 7) else 0 for c in range(8)] for r in range(8)]

# Alamsyah dkk. (2023), SJI 10(2), polinomial 0x1F3, C = 0x63
POLY_2023 = 0x1F3
A0_2023 = [
    [1, 1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1, 1],
]
A1_2023 = [
    [0, 1, 1, 1, 1, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 0, 0, 1], [1, 1, 1, 1, 0, 1, 0, 0],
]
A2_2023 = [
    [1, 1, 1, 0, 1, 0, 0, 1], [1, 1, 1, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 0, 1, 1],
]

# Alamsyah dkk. (2025), Nonlinear Dynamics 113:3869-3890, polinomial 0x11B, C = 0x63
# K44: matriks sirkulan kanan dengan baris pertama 01010111 (Tabel 5 / Tabel 16 paper)
K44_2025 = [
    [0, 1, 0, 1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 1], [1, 0, 1, 0, 1, 1, 1, 0],
]


# ---------------- Aritmetika GF(2^8) & 30 Polinomial Irreducible ----------------
def poly_mod(a: int, b: int) -> int:
    db = b.bit_length() - 1
    while a.bit_length() - 1 >= db:
        a ^= b << (a.bit_length() - 1 - db)
    return a


def get_irreducible_polynomials() -> list[int]:
    """Kembalikan seluruh 30 polinomial irreducible derajat 8 di GF(2)[x]."""
    return [
        p for p in range(0x101, 0x200, 2)
        if all(poly_mod(p, d) != 0 for d in range(2, 32))
    ]


def poly_to_str(poly: int) -> str:
    """Format polinomial biner sebagai string matematis, mis. x^8 + x^6 + ... + 1."""
    terms = []
    for deg in range(8, -1, -1):
        if (poly >> deg) & 1:
            if deg == 0:
                terms.append("1")
            elif deg == 1:
                terms.append("x")
            else:
                terms.append(f"x^{deg}")
    return " + ".join(terms)


def gf_mul(a: int, b: int, poly: int = POLY_K7) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= poly
        b >>= 1
    return r


def gf_inverse_table(poly: int = POLY_K7) -> np.ndarray:
    """Hitung tabel invers multiplikatif x^-1 di GF(2^8) menggunakan x^254."""
    inv = np.zeros(256, dtype=np.int64)
    for a in range(1, 256):
        base, exp, res = a, 254, 1
        while exp:
            if exp & 1:
                res = gf_mul(res, base, poly)
            base = gf_mul(base, base, poly)
            exp >>= 1
        inv[a] = res
    return inv


_INV_CACHE: dict[int, np.ndarray] = {}


def get_inv(poly: int) -> np.ndarray:
    if poly not in _INV_CACHE:
        _INV_CACHE[poly] = gf_inverse_table(poly)
    return _INV_CACHE[poly]


def gf2_rank_int(rows_int) -> int:
    """Rank GF(2) cepat berbasis eliminasi bitmask pada daftar integer 8-bit."""
    basis = []
    for v in rows_int:
        x = int(v)
        for b in basis:
            if (x ^ b) < x:
                x ^= b
        if x:
            basis.append(x)
            basis.sort(reverse=True)
    return len(basis)


def gf2_rank(M) -> int:
    arr = np.asarray(M, dtype=np.int64) % 2
    rows_int = (arr @ POW2).tolist()
    return gf2_rank_int(rows_int)


def build_sbox(A, poly: int = POLY_K7, c_int: int = C_AES_INT) -> np.ndarray:
    """S(x) = A * inv_poly(x) XOR C di GF(2^8). Baris j matriks A menghasilkan bit b_j."""
    inv = get_inv(poly)
    A_arr = np.asarray(A, dtype=np.int64) % 2
    y = (BITS[inv] @ A_arr.T) % 2
    return ((y @ POW2).astype(np.int64)) ^ int(c_int)


def build_sbox_from_rows(rows_int, poly: int = POLY_K7, c_int: int = C_AES_INT) -> np.ndarray:
    return build_sbox(BITS[np.asarray(rows_int, dtype=np.int64)], poly=poly, c_int=c_int)


SBOX_K7 = build_sbox_from_rows(ROWS_K7, poly=POLY_K7, c_int=C_K7_INT)
SBOX_AES = build_sbox(A_AES, poly=0x11B, c_int=C_AES_INT)
SBOX_ALAMSYAH_2025 = build_sbox(K44_2025, poly=0x11B, c_int=C_AES_INT)
SBOX_ALAMSYAH_2023_A0 = build_sbox(A0_2023, poly=POLY_2023, c_int=C_AES_INT)
SBOX_ALAMSYAH_2023_A2 = build_sbox(A2_2023, poly=POLY_2023, c_int=C_AES_INT)
SBOX_ALAMSYAH_2023_A1 = build_sbox(A1_2023, poly=POLY_2023, c_int=C_AES_INT)


def reference_sboxes() -> dict[str, np.ndarray]:
    return {
        "S-box K7 (Kelompok 7 · SAGA)": SBOX_K7.copy(),
        "AES Standar (FIPS-197)": SBOX_AES.copy(),
        "Alamsyah 2025 (S-box₄₄ · 0x11B)": SBOX_ALAMSYAH_2025.copy(),
        "Alamsyah 2023 (S-box₁ A0 · 0x1F3)": SBOX_ALAMSYAH_2023_A0.copy(),
        "Alamsyah 2023 (S-box₂ A2 · 0x1F3)": SBOX_ALAMSYAH_2023_A2.copy(),
        "Alamsyah 2023 (S-box₃ A1 · 0x1F3)": SBOX_ALAMSYAH_2023_A1.copy(),
    }


# ---------------- Eksplorasi 30 Polinomial & Fitur ML ----------------
def precompute_component_sac(poly: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Untuk tiap fungsi komponen non-nol f_r(x) = <r, inv(x)>, hitung:
    - row_avg[r]: rata-rata peluang bit berubah atas 8 input flip
    - row_dev[r]: rata-rata |M_{i,r} - 0.5| atas 8 input flip (kontribusi SAC)
    - row_bdev[r]: |row_avg[r] - 0.5| (kontribusi deviasi sel BIC-SAC)
    """
    inv = get_inv(poly)
    row_avg = np.zeros(256, dtype=np.float64)
    row_dev = np.zeros(256, dtype=np.float64)
    for r in range(1, 256):
        f = (BITS[inv] @ BITS[r]) % 2
        probs = np.array([(f ^ f[X_ARR ^ (1 << i)]).mean() for i in range(8)])
        row_avg[r] = probs.mean()
        row_dev[r] = np.abs(probs - 0.5).mean()
    row_bdev = np.abs(row_avg - 0.5)
    return row_avg, row_dev, row_bdev


def screen_all_30_polynomials() -> list[dict]:
    """Evaluasi kualitas intrinsik seluruh 30 polinomial irreducible derajat 8."""
    rows = []
    for p in get_irreducible_polynomials():
        r_avg, r_dev, r_bdev = precompute_component_sac(p)
        top16_sac = float(np.sort(r_dev[1:])[:16].mean())
        top32_bic = float(np.sort(r_bdev[1:])[:32].mean())
        mean_sac = float(r_dev[1:].mean())
        rows.append({
            "poly_int": p,
            "Polinomial (Hex)": f"0x{p:03X}",
            "Bentuk Polinomial": poly_to_str(p),
            "Top-16 Dev SAC": round(top16_sac, 5),
            "Top-32 Dev BIC-SAC": round(top32_bic, 5),
            "Skor Gabungan Pool": round(top16_sac + top32_bic, 5),
            "Rata2 Dev SAC Pool": round(mean_sac, 5),
        })
    rows.sort(key=lambda d: (d["Skor Gabungan Pool"], d["Top-16 Dev SAC"], d["poly_int"]))
    return rows


def extract_invertible_matrices(img: np.ndarray) -> tuple[int, np.ndarray]:
    """Ekstrak kandidat matriks 8x8 dari bit-plane 0 dan 1 citra, saring rank GF(2) = 8."""
    bitstream = np.concatenate([(img >> p) & 1 for p in (0, 1)]).reshape(-1)
    n = (len(bitstream) // 64) * 64
    kandidat = bitstream[:n].reshape(-1, 8, 8).astype(np.int64)
    rows_all = (kandidat * POW2).sum(2)
    inv_rows = [r for r in rows_all if gf2_rank_int(r) == 8]
    return len(kandidat), np.array(inv_rows, dtype=np.int64)


def build_row_and_pair_features(rows_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bangun fitur bag-of-rows (256-dim) dan bag-of-pairwise-XOR (256-dim)."""
    n = len(rows_mat)
    idx = np.arange(n)
    X_row = np.zeros((n, 256), dtype=np.float64)
    for k in range(8):
        X_row[idx, rows_mat[:, k]] += 1.0
    pairs = rows_mat[:, PAIR_I] ^ rows_mat[:, PAIR_J]  # (n, 28)
    X_pair = np.zeros((n, 256), dtype=np.float64)
    for k in range(28):
        X_pair[idx, pairs[:, k]] += 1.0
    return X_row, X_pair


def _fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def cycle_lengths(S) -> list[int]:
    """Hitung panjang seluruh siklus permutasi S-box (diurutkan menurun)."""
    S = [int(v) for v in S]
    seen = set()
    lens = []
    for i in range(256):
        if i not in seen:
            c = 0
            cur = i
            while cur not in seen:
                seen.add(cur)
                cur = S[cur]
                c += 1
            lens.append(c)
    return sorted(lens, reverse=True)


def optimize_constant_c(rows_int, poly: int = POLY_K7) -> dict:
    """Pilih konstanta 8-bit C terbaik di {0..255}:
    menjaga NL/SAC/BIC/LAP/DAP invarian sambil meminimalkan FP, OFP, dan jumlah siklus.
    """
    base = build_sbox_from_rows(rows_int, poly=poly, c_int=0)
    best = None
    n_zero_fp_ofp = 0
    n_single_cycle = 0
    for c in range(256):
        S = base ^ c
        fp = int(np.sum(S == X_ARR))
        ofp = int(np.sum(S == (255 - X_ARR)))
        cl = cycle_lengths(S)
        if fp == 0 and ofp == 0:
            n_zero_fp_ofp += 1
            if len(cl) == 1:
                n_single_cycle += 1
        key = (fp + ofp, fp, len(cl), -cl[0], -min(cl), c)
        if best is None or key < best[0]:
            best = (key, c, fp, ofp, cl, S)
    return {
        "c_int": int(best[1]),
        "c_hex": f"0x{best[1]:02X}",
        "fixed_points": int(best[2]),
        "opposite_fixed_points": int(best[3]),
        "cycles": best[4],
        "sbox": best[5],
        "n_zero_fp_ofp": n_zero_fp_ofp,
        "n_single_cycle_zero_fp_ofp": n_single_cycle,
    }


def construct_sbox_saga(
    img: np.ndarray,
    poly: int | None = None,
    pop_size: int = 140,
    generations: int = 160,
    seed: int = 7,
    run_baselines: bool = True,
) -> dict:
    """Jalankan pipeline penuh Kelompok 7 (SAGA):
    1. Skrining 30 polinomial irreducible (jika poly=None, pilih peringkat #1: 0x17B)
    2. Ekstraksi dataset matriks invertible dari bit-plane citra
    3. Pelatihan & ablasi model ML (KNN, Random Forest, Ridge 1-baris, Ridge Komposit)
    4. Surrogate-Assisted Genetic Algorithm (SAGA) untuk optimasi simultan SAC + BIC-SAC
    5. Optimasi konstanta 8-bit C untuk 0 FP, 0 OFP, dan siklus tunggal [256].
    """
    t_start = time.perf_counter()

    poly_table = screen_all_30_polynomials()
    if poly is None:
        poly = int(poly_table[0]["poly_int"])

    n_kandidat, dataset_rows = extract_invertible_matrices(img)
    if len(dataset_rows) < 300:
        raise ValueError("Citra terlalu kecil atau kurang variasi bit untuk membentuk dataset matriks invertible.")

    # Label ground-truth SAC & BIC-SAC untuk setiap matriks dataset
    r_avg, r_dev, r_bdev = precompute_component_sac(poly)
    pairs_all = dataset_rows[:, PAIR_I] ^ dataset_rows[:, PAIR_J]
    y_sac_dev = r_dev[dataset_rows].mean(1)
    y_sac_avg = r_avg[dataset_rows].mean(1)
    y_bic_dev = r_bdev[pairs_all].mean(1)
    y_bic_avg = r_avg[pairs_all].mean(1)
    y_joint = 1.0 * y_sac_dev + 1.5 * y_bic_dev + 5.0 * (np.abs(y_sac_avg - 0.5) + np.abs(y_bic_avg - 0.5))

    X_row, X_pair = build_row_and_pair_features(dataset_rows)
    X_raw = BITS[dataset_rows].reshape(len(dataset_rows), 64).astype(np.float64)

    # Split latih / uji 80:20 deterministik (tanpa data leakage)
    rng_split = np.random.default_rng(seed)
    perm = rng_split.permutation(len(dataset_rows))
    n_te = max(1, int(0.2 * len(dataset_rows)))
    te, tr = perm[:n_te], perm[n_te:]

    evaluasi = []
    pred_plot = {}

    try:
        from sklearn.linear_model import Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error

        m_sdev = Ridge(alpha=1e-3, fit_intercept=False).fit(X_row[tr], y_sac_dev[tr])
        m_savg = Ridge(alpha=1e-3, fit_intercept=False).fit(X_row[tr], y_sac_avg[tr])
        m_bdev = Ridge(alpha=1e-3, fit_intercept=False).fit(X_pair[tr], y_bic_dev[tr])
        m_bavg = Ridge(alpha=1e-3, fit_intercept=False).fit(X_pair[tr], y_bic_avg[tr])
        m_bdev_row = Ridge(alpha=1e-3, fit_intercept=False).fit(X_row[tr], y_bic_dev[tr])

        w_sdev = m_sdev.coef_ * 8.0
        w_savg = m_savg.coef_ * 8.0
        w_bdev = m_bdev.coef_ * 28.0
        w_bavg = m_bavg.coef_ * 28.0

        p_ridge_sac = m_sdev.predict(X_row[te])
        p_ridge_row_bic = m_bdev_row.predict(X_row[te])
        p_ridge_pair_bic = m_bdev.predict(X_pair[te])

        if run_baselines:
            knn_sac = KNeighborsRegressor(n_neighbors=15, n_jobs=-1).fit(X_raw[tr], y_sac_dev[tr])
            knn_bic = KNeighborsRegressor(n_neighbors=15, n_jobs=-1).fit(X_raw[tr], y_bic_dev[tr])
            rf_sac = RandomForestRegressor(n_estimators=120, max_depth=14, random_state=seed, n_jobs=-1).fit(X_raw[tr], y_sac_dev[tr])
            rf_bic = RandomForestRegressor(n_estimators=120, max_depth=14, random_state=seed, n_jobs=-1).fit(X_raw[tr], y_bic_dev[tr])

            p_knn_sac = knn_sac.predict(X_raw[te])
            p_knn_bic = knn_bic.predict(X_raw[te])
            p_rf_sac = rf_sac.predict(X_raw[te])
            p_rf_bic = rf_bic.predict(X_raw[te])

            evaluasi.extend([
                ("KNN (64-bit mentah)", "Deviasi SAC", float(r2_score(y_sac_dev[te], p_knn_sac)), float(mean_absolute_error(y_sac_dev[te], p_knn_sac))),
                ("KNN (64-bit mentah)", "Deviasi BIC-SAC", float(r2_score(y_bic_dev[te], p_knn_bic)), float(mean_absolute_error(y_bic_dev[te], p_knn_bic))),
                ("Random Forest (64-bit mentah)", "Deviasi SAC", float(r2_score(y_sac_dev[te], p_rf_sac)), float(mean_absolute_error(y_sac_dev[te], p_rf_sac))),
                ("Random Forest (64-bit mentah)", "Deviasi BIC-SAC", float(r2_score(y_bic_dev[te], p_rf_bic)), float(mean_absolute_error(y_bic_dev[te], p_rf_bic))),
            ])
            pred_plot["knn_bic"] = p_knn_bic
            pred_plot["rf_sac"] = p_rf_sac

        evaluasi.extend([
            ("Ridge 1-Baris (bag-of-rows)", "Deviasi SAC", float(r2_score(y_sac_dev[te], p_ridge_sac)), float(mean_absolute_error(y_sac_dev[te], p_ridge_sac))),
            ("Ridge 1-Baris (bag-of-rows)", "Deviasi BIC-SAC", float(r2_score(y_bic_dev[te], p_ridge_row_bic)), float(mean_absolute_error(y_bic_dev[te], p_ridge_row_bic))),
            ("Ridge Komposit Usulan (Row + Pairwise-XOR)", "Deviasi SAC", float(r2_score(y_sac_dev[te], p_ridge_sac)), float(mean_absolute_error(y_sac_dev[te], p_ridge_sac))),
            ("Ridge Komposit Usulan (Row + Pairwise-XOR)", "Deviasi BIC-SAC", float(r2_score(y_bic_dev[te], p_ridge_pair_bic)), float(mean_absolute_error(y_bic_dev[te], p_ridge_pair_bic))),
            ("Ridge Komposit Usulan (Row + Pairwise-XOR)", "Rata-rata BIC-SAC", float(r2_score(y_bic_avg[te], m_bavg.predict(X_pair[te]))), float(mean_absolute_error(y_bic_avg[te], m_bavg.predict(X_pair[te])))),
        ])
        pred_plot["ridge_row_bic"] = p_ridge_row_bic
        pred_plot["ridge_pair_bic"] = p_ridge_pair_bic
    except ImportError:
        # Fallback NumPy eksak untuk lingkungan tanpa scikit-learn
        c_sdev = _fit_ridge_closed_form(X_row[tr], y_sac_dev[tr])
        c_savg = _fit_ridge_closed_form(X_row[tr], y_sac_avg[tr])
        c_bdev = _fit_ridge_closed_form(X_pair[tr], y_bic_dev[tr])
        c_bavg = _fit_ridge_closed_form(X_pair[tr], y_bic_avg[tr])
        c_bdev_row = _fit_ridge_closed_form(X_row[tr], y_bic_dev[tr])

        w_sdev, w_savg = c_sdev * 8.0, c_savg * 8.0
        w_bdev, w_bavg = c_bdev * 28.0, c_bavg * 28.0

        def _r2(yt, yp):
            return float(1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2))

        def _mae(yt, yp):
            return float(np.mean(np.abs(yt - yp)))

        p_ridge_sac = X_row[te] @ c_sdev
        p_ridge_row_bic = X_row[te] @ c_bdev_row
        p_ridge_pair_bic = X_pair[te] @ c_bdev
        evaluasi.extend([
            ("Ridge 1-Baris (bag-of-rows)", "Deviasi SAC", _r2(y_sac_dev[te], p_ridge_sac), _mae(y_sac_dev[te], p_ridge_sac)),
            ("Ridge 1-Baris (bag-of-rows)", "Deviasi BIC-SAC", _r2(y_bic_dev[te], p_ridge_row_bic), _mae(y_bic_dev[te], p_ridge_row_bic)),
            ("Ridge Komposit Usulan (Row + Pairwise-XOR)", "Deviasi SAC", _r2(y_sac_dev[te], p_ridge_sac), _mae(y_sac_dev[te], p_ridge_sac)),
            ("Ridge Komposit Usulan (Row + Pairwise-XOR)", "Deviasi BIC-SAC", _r2(y_bic_dev[te], p_ridge_pair_bic), _mae(y_bic_dev[te], p_ridge_pair_bic)),
        ])
        pred_plot["ridge_row_bic"] = p_ridge_row_bic
        pred_plot["ridge_pair_bic"] = p_ridge_pair_bic

    # ---------------- Surrogate-Assisted Genetic Algorithm (SAGA) ----------------
    # Pengurutan & pembulatan stabil (4 desimal) agar identik di sklearn/NumPy & variasi citra
    ws_d = np.round(w_sdev, 4)
    ws_a = np.round(w_savg, 4)
    wb_d = np.round(w_bdev, 4)
    wb_a = np.round(w_bavg, 4)
    row_ids = np.arange(1, 256, dtype=np.int64)
    top_rows = row_ids[np.lexsort((row_ids, ws_d[1:]))]

    def surrogate_fitness(ind: np.ndarray) -> float:
        p_xor = ind[PAIR_I] ^ ind[PAIR_J]
        sd = float(ws_d[ind].mean())
        sa = float(ws_a[ind].mean())
        bd = float(wb_d[p_xor].mean())
        ba = float(wb_a[p_xor].mean())
        return round(1.0 * sd + 1.5 * bd + 5.0 * (abs(sa - 0.5) + abs(ba - 0.5)), 5)

    rng = np.random.default_rng(seed)

    def random_valid() -> np.ndarray:
        while True:
            pool = top_rows[:80] if rng.random() < 0.75 else row_ids
            cand = rng.choice(pool, size=8, replace=False)
            if gf2_rank_int(cand) == 8:
                return np.sort(cand)

    pop = [random_valid() for _ in range(pop_size)]
    fits = np.array([surrogate_fitness(ind) for ind in pop], dtype=np.float64)
    history_best = [float(fits.min())]
    history_mean = [float(fits.mean())]
    elite_n = max(4, pop_size // 12)

    for _gen in range(generations):
        order = np.argsort(fits, kind="mergesort")
        new_pop = [pop[i].copy() for i in order[:elite_n]]
        while len(new_pop) < pop_size:
            i1, i2 = rng.integers(0, pop_size, size=2)
            p1 = pop[i1] if fits[i1] <= fits[i2] else pop[i2]
            j1, j2 = rng.integers(0, pop_size, size=2)
            p2 = pop[j1] if fits[j1] <= fits[j2] else pop[j2]
            pool = np.unique(np.concatenate([p1, p2]))

            child = None
            for _try in range(15):
                cand = rng.choice(pool, size=8, replace=False) if len(pool) >= 8 else p1.copy()
                if rng.random() < 0.70:
                    n_mut = 1 if rng.random() < 0.75 else 2
                    for _m in range(n_mut):
                        pos = int(rng.integers(0, 8))
                        src = top_rows[:50] if rng.random() < 0.70 else row_ids
                        v = int(rng.choice(src))
                        if v not in cand:
                            cand[pos] = v
                if gf2_rank_int(cand) == 8:
                    child = np.sort(cand)
                    break
            new_pop.append(p1.copy() if child is None else child)
        pop = new_pop
        fits = np.array([surrogate_fitness(ind) for ind in pop], dtype=np.float64)
        history_best.append(float(fits.min()))
        history_mean.append(float(fits.mean()))

    best_rows = pop[int(np.argmin(fits))].tolist()
    A_best = BITS[best_rows]
    c_opt = optimize_constant_c(best_rows, poly=poly)

    # Bandingkan dengan matriks terbaik di dataset tanpa SAGA
    best_ds_idx = int(np.argmin(y_joint))
    best_ds_sbox = build_sbox_from_rows(dataset_rows[best_ds_idx], poly=poly, c_int=C_AES_INT)

    elapsed = time.perf_counter() - t_start
    return {
        "poly": poly,
        "poly_hex": f"0x{poly:03X}",
        "poly_str": poly_to_str(poly),
        "poly_table": poly_table,
        "n_kandidat": n_kandidat,
        "n_dataset": len(dataset_rows),
        "evaluasi": evaluasi,
        "y_bic_te": y_bic_dev[te],
        "y_sac_te": y_sac_dev[te],
        "pred_plot": pred_plot,
        "ga_best_history": history_best,
        "ga_mean_history": history_mean,
        "rows": best_rows,
        "A": A_best,
        "c_opt": c_opt,
        "sbox": c_opt["sbox"],
        "best_dataset_sbox": best_ds_sbox,
        "best_dataset_joint": float(y_joint.min()),
        "saga_joint": float(fits.min()),
        "elapsed": float(elapsed),
    }


# ---------------- Metrik Evaluasi Kriptografi S-box ----------------
def balance(S) -> dict:
    S = np.asarray(S, dtype=np.int64)
    satu = [int(((S >> j) & 1).sum()) for j in range(8)]
    return {"jumlah_1": satu, "jumlah_0": [256 - v for v in satu], "seimbang": all(v == 128 for v in satu)}


def bijective(S) -> bool:
    S = np.asarray(S, dtype=np.int64)
    return len(set(int(v) for v in S)) == 256 and int(S.min()) == 0 and int(S.max()) == 255


def walsh(f) -> np.ndarray:
    w = 1 - 2 * np.asarray(f, dtype=np.int64)
    h = 1
    while h < 256:
        w = w.reshape(-1, 2 * h)
        a, b = w[:, :h].copy(), w[:, h:].copy()
        w[:, :h], w[:, h:] = a + b, a - b
        w = w.reshape(-1)
        h *= 2
    return w


def nl_bool(f) -> int:
    return int(128 - np.abs(walsh(f)).max() // 2)


def nonlinearity(S) -> dict:
    S = np.asarray(S, dtype=np.int64)
    per_bit = [nl_bool((S >> j) & 1) for j in range(8)]
    comp = [nl_bool(BITS[S & v].sum(1) % 2) for v in range(1, 256)]
    return {
        "per_bit": per_bit,
        "avg": float(np.mean(per_bit)),
        "min": min(per_bit),
        "max": max(per_bit),
        "min_komponen": min(comp),
    }


def sac_matrix(S) -> np.ndarray:
    S = np.asarray(S, dtype=np.int64)
    return np.array([BITS[S ^ S[X_ARR ^ (1 << i)]].mean(0) for i in range(8)], dtype=np.float64)


def sac(S) -> dict:
    M = sac_matrix(S)
    return {
        "matrix": M,
        "avg": float(M.mean()),
        "min": float(M.min()),
        "max": float(M.max()),
        "deviasi": float(np.abs(M - 0.5).mean()),
    }


def bic(S) -> dict:
    S = np.asarray(S, dtype=np.int64)
    nls, sacs = [], []
    nl_mat = np.full((8, 8), np.nan)
    sac_mat = np.full((8, 8), np.nan)
    for j in range(8):
        for k in range(j + 1, 8):
            f = ((S >> j) ^ (S >> k)) & 1
            nl = nl_bool(f)
            s = float(np.mean([(f ^ f[X_ARR ^ (1 << i)]).mean() for i in range(8)]))
            nls.append(nl)
            sacs.append(s)
            nl_mat[j, k] = nl_mat[k, j] = nl
            sac_mat[j, k] = sac_mat[k, j] = s
    sacs_arr = np.array(sacs, dtype=np.float64)
    return {
        "nl_min": min(nls),
        "nl_avg": float(np.mean(nls)),
        "sac_avg": float(sacs_arr.mean()),
        "sac_deviasi": float(np.abs(sacs_arr - 0.5).mean()),
        "sac_min": float(sacs_arr.min()),
        "sac_max": float(sacs_arr.max()),
        "nl_matrix": nl_mat,
        "sac_matrix": sac_mat,
    }


def lap_detail(S) -> dict:
    """Hitung Linear Approximation Table (LAT) dan LAP (bias & probabilitas).
    Sesuai Alamsyah dkk. (2025) Pers. (7): LAP = max |#{a.x = b.S(x)}/256 - 1/2| (ideal -> 0).
    """
    S = np.asarray(S, dtype=np.int64)
    lat = np.zeros((256, 256), dtype=np.int16)
    lat[0, 0] = 256
    for b in range(1, 256):
        g = BITS[S & b].sum(1) % 2
        lat[:, b] = walsh(g)
    sub = np.abs(lat[1:, 1:])
    idx = np.unravel_index(np.argmax(sub), sub.shape)
    max_corr = int(sub[idx])
    bias = max_corr / 512.0
    return {
        "lat": lat,
        "max_corr": max_corr,
        "bias": float(bias),
        "prob": float(0.5 + bias),
        "mask_a": int(idx[0] + 1),
        "mask_b": int(idx[1] + 1),
    }


def dap_detail(S) -> dict:
    """Hitung Differential Distribution Table (DDT) dan DAP (ideal -> 0)."""
    S = np.asarray(S, dtype=np.int64)
    ddt = np.zeros((256, 256), dtype=np.uint16)
    for dx in range(1, 256):
        dy = S ^ S[X_ARR ^ dx]
        ddt[dx] = np.bincount(dy, minlength=256)
    pos = np.unravel_index(np.argmax(ddt[1:]), ddt[1:].shape)
    dx_max, dy_max = int(pos[0] + 1), int(pos[1])
    max_count = int(ddt[dx_max, dy_max])
    return {
        "ddt": ddt,
        "max_count": max_count,
        "dx": dx_max,
        "dy": dy_max,
        "dap": float(max_count / 256.0),
    }


def fixed_and_opposite_points(S) -> dict:
    S = np.asarray(S, dtype=np.int64)
    fp = [int(i) for i in range(256) if S[i] == i]
    ofp = [int(i) for i in range(256) if S[i] == (255 - i)]
    return {"fp": len(fp), "fp_list": fp, "ofp": len(ofp), "ofp_list": ofp}


def all_metrics(S) -> dict:
    """Hitung seluruh metrik kriptografi wajib + bonus (termasuk skor S & SV Alamsyah 2025)."""
    S = np.asarray(S, dtype=np.int64)
    bal = balance(S)
    bij = bijective(S)
    nl = nonlinearity(S)
    sc = sac(S)
    bc = bic(S)
    lp = lap_detail(S)
    dp = dap_detail(S)
    fp = fixed_and_opposite_points(S)
    cl = cycle_lengths(S)

    # Skor S (Pers. 9) dan SV (Pers. 20) Alamsyah dkk. (2025)
    s_paper = 0.5 * (abs(sc["avg"] - 0.5) + abs(bc["sac_avg"] - 0.5))
    sv_paper = (120.0 - nl["avg"]) + abs(0.5 - sc["avg"]) + (120.0 - bc["nl_avg"]) + abs(0.5 - bc["sac_avg"])

    return {
        "Balance": "Ya" if bal["seimbang"] else "Tidak",
        "Bijective": "Ya" if bij else "Tidak",
        "NL": nl["min"],
        "SAC": round(sc["avg"], 5),
        "Deviasi SAC": round(sc["deviasi"], 5),
        "BIC-NL": bc["nl_min"],
        "BIC-SAC": round(bc["sac_avg"], 5),
        "Deviasi BIC-SAC": round(bc["sac_deviasi"], 5),
        "LAP (bias)": round(lp["bias"], 6),
        "DAP": round(dp["dap"], 6),
        "Skor S (2025)": round(s_paper, 5),
        "Skor SV (2025)": round(sv_paper, 5),
        "FP": fp["fp"],
        "OFP": fp["ofp"],
        "Siklus": f"{len(cl)} ({cl[0]})",
    }


def parse_sbox(text: str) -> np.ndarray:
    cleaned = text.replace("[", " ").replace("]", " ").replace(",", " ").replace(";", " ")
    vals = [int(t, 16) if t.lower().startswith("0x") else int(t) for t in cleaned.split()]
    if len(vals) != 256:
        raise ValueError(f"S-box harus berisi tepat 256 angka, ditemukan {len(vals)}.")
    if any(v < 0 or v > 255 for v in vals):
        raise ValueError("Semua nilai S-box harus berada pada rentang 0 hingga 255.")
    return np.array(vals, dtype=np.int64)
