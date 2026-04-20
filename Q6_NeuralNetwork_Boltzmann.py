#!/usr/bin/env python3
"""
Q6 Neural Network — tehnika: Quantum Boltzmann Machine (QBM) — inference
(čisto kvantno, bez klasičnog treniranja i bez hibrida).

Arhitektura:
  - Fiksni Hamiltonijan iz CELOG CSV-a (transverzalno-Ising forma):
        H = Σ J_i  Z_i Z_{i+1}  +  Σ h_i  X_i
    sa deterministički izvedenim koeficijentima (bez randoma).
  - Gibbs stanje:  ρ_β = exp(-β H) / Tr[exp(-β H)]
  - Born distribucija u komp. bazi: p(x) = ⟨x| ρ_β |x⟩ = diag(ρ_β)
  - Readout → bias_39 → NEXT rastuća sedmorka ∈ {1..39}.

Sve deterministički: seed=39; H i β iz CELOG CSV-a / grida.
Deterministička grid-optimizacija (nq, β) po meri cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit.quantum_info import SparsePauliOp

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (6, 7, 8)
GRID_BETA = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


# =========================
# Hamiltonijan iz CSV-a
# =========================
def qbm_hamiltonian(H: np.ndarray, nq: int) -> SparsePauliOp:
    """
    H = Σ J_i Z_i Z_{i+1}  +  Σ h_i X_i, J_i, h_i iz CSV freq-a (deterministički).
    """
    f = freq_vector(H)
    denom = max(float(f.max() - f.min()), 1e-12)
    fn = (f - f.min()) / denom
    fn = 2.0 * fn - 1.0  # [-1, 1]

    pauli_strings: List[str] = []
    coeffs: List[float] = []

    # ZZ (zatvoreni lanac)
    for i in range(nq):
        j = (i + 1) % nq
        s = ["I"] * nq
        s[i] = "Z"
        s[j] = "Z"
        pauli_strings.append("".join(reversed(s)))
        coeffs.append(float(fn[i % N_MAX]))

    # X
    for i in range(nq):
        s = ["I"] * nq
        s[i] = "X"
        pauli_strings.append("".join(reversed(s)))
        coeffs.append(float(fn[(i + 13) % N_MAX]))

    return SparsePauliOp.from_list(list(zip(pauli_strings, coeffs)))


# =========================
# Gibbs stanje
# =========================
def gibbs_probs(H: np.ndarray, nq: int, beta: float) -> np.ndarray:
    """
    ρ_β = exp(-β H) / Tr[exp(-β H)]; vraća diag(ρ_β) u komp. bazi.
    Deterministička matrica (eigendekompozicija hermitskog H).
    """
    op = qbm_hamiltonian(H, nq)
    Hm = np.array(op.to_matrix(), dtype=np.complex128)
    # Hermitski: koristi eigh — stabilno i deterministički.
    w, V = np.linalg.eigh(Hm)
    # Stabilni exp(-βw) sa pomeranjem min-a u 0 (ne menja distribuciju jer se deli trag).
    w_shift = w - float(w.min())
    e = np.exp(-float(beta) * w_shift)
    Z = float(e.sum())
    if Z < 1e-300:
        return np.ones(Hm.shape[0], dtype=np.float64) / float(Hm.shape[0])
    p_diag = np.zeros(Hm.shape[0], dtype=np.float64)
    # diag(V * diag(e/Z) * V†) = sum_k |V[x,k]|^2 * (e_k/Z)
    absV2 = (V.real ** 2 + V.imag ** 2)
    p_diag = absV2 @ (e / Z)
    s = float(p_diag.sum())
    return p_diag / s if s > 0 else p_diag


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, β) po meri cos(bias, freq_csv)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    f_csv_n = f_csv / float(f_csv.sum() or 1.0)
    best = None
    for nq in GRID_NQ:
        for beta in GRID_BETA:
            try:
                probs = gibbs_probs(H, nq, beta)
                b = bias_39(probs)
                score = cosine(b, f_csv_n)
            except Exception:
                continue
            key = (score, -nq, -int(round(beta * 1000)))
            if best is None or key > best[0]:
                best = (key, dict(nq=nq, beta=float(beta), score=score))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q6 NN (QBM — Quantum Boltzmann Machine): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| β:", best["beta"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    probs = gibbs_probs(H, best["nq"], best["beta"])
    pred = pick_next_combination(probs)
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




"""
Q6 NN (QBM — Quantum Boltzmann Machine): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 8 | β: 0.25 | cos(bias, freq_csv): 0.992665
predikcija NEXT: (1, 3, x, y, z, 18, 22)
"""




"""
Q6_NeuralNetwork_Boltzmann.py — tehnika: QBM (Quantum Boltzmann Machine) — inference

Učita CEO CSV. Napravi fiksni Hamiltonijan H = Σ Jᵢ ZᵢZᵢ₊₁ + Σ hᵢ Xᵢ sa koeficijentima iz normalizovanog histograma (determinističke vrednosti u [-1, 1]).
Egzaktno izračuna Gibbs stanje: ρ_β = exp(−β H) / Tr[exp(−β H)] preko eigh hermitske dekompozicije (stabilan shift w - min(w)).
p(x) = diag(ρ_β) u komp. bazi → bias_39 → NEXT.
Deterministička grid-optimizacija (nq, β) po meri cos(bias, freq_csv).

Tehnike:
Gibbs-state priprema kao generativni kvantni NN (QBM u inference modu — bez treniranja težina Hamiltonijana).
Spektralna eigendekompozicija H = V Λ V† → diag(ρ_β) = |V|² · (e^{−βλ}/Z).
Transverzalno-Ising forma sa X i ZZ članovima iz CSV-a.
Čisto deterministički, bez uzorkovanja (nema Metropolis/thermofield-a).

Prednosti:
Termalni kvantni model — suštinski drugačiji od Grover/QNN porodice.
Inverzna temperatura β direktno kontroliše „oštrinu“ distribucije (mali β → uniformno, veliki β → koncentrisano oko osnovnog stanja).
Egzaktno, stabilno, brzo za mali nq.
Iako je implementacija klasična algebra, model je definisan kao kvantni Gibbs-state (egzaktan simulator).

Nedostaci:
Iako je formalno kvantni model, praktično se radi o expm(-βH) tj. klasičnoj matričnoj operaciji — nema pravog kvantnog sampling-a (što bi na hardveru tražilo amplitude amplification ili thermofield double pristup).
Hamiltonijan je 1D-lanac — ograničena izraznost (slično kao Q5).
Bez učenja: J, h, β su iz CSV-a direktno; ne uče se.
Mera optimizacije i dalje cos prema frekvenciji → tautološki bias.
Eksponencijalni memorijski trošak (puna 2^nq x 2^nq matrica za eigh), nq ≤ 8 je praktičan plafon.
"""
