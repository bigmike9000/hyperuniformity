"""Fast normalization check."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from substitution_tilings import generate_substitution_sequence, sequence_to_points
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

def compute_I(pts, G_vals):
    """Compute I_G = |F_G|^2 = |sum exp(iGx)|^2 / N^2"""
    N = len(pts)
    I = np.zeros(len(G_vals))
    for i, G in enumerate(G_vals):
        rh = np.sum(np.exp(1j * G * pts))
        I[i] = abs(rh)**2 / N**2
    return I

# --- Integer lattice calibration ---
N_lat = 500
pts_lat = np.arange(N_lat, dtype=float)
G_lat = np.array([2*np.pi*m for m in range(1,30)])
I_lat = compute_I(pts_lat, G_lat)
lb_lat = 4.0 * np.sum(I_lat / G_lat**2)
print(f"Integer lattice Lambda_bar: {lb_lat:.8f}  (expected 1/6={1/6:.8f})")
print(f"I_G for first 3 peaks: {I_lat[:3]}  (expected all 1.0)")

# --- Fibonacci (small, fast) ---
tau = (1+np.sqrt(5))/2
seq_fib = generate_substitution_sequence('fibonacci', 20)
pts_fib, L_fib = sequence_to_points(seq_fib, 'fibonacci')
N_fib = len(pts_fib)
rho_fib = N_fib / L_fib
ell_bar_fib = 1/rho_fib
print(f"\nFibonacci N={N_fib:,}")

G_fib = []
for p in range(-50,51):
    for q in range(-50,51):
        if p==0 and q==0: continue
        t = p + q*tau
        if 0 < t and 2*np.pi*t/ell_bar_fib < 50:
            G_fib.append(round(2*np.pi*t/ell_bar_fib, 8))
G_fib = np.array(sorted(set(G_fib)))
print(f"Candidate peaks in [0,50]: {len(G_fib)}")

I_fib = compute_I(pts_fib, G_fib)
lb_fib = 4*rho_fib * np.sum(I_fib/G_fib**2)
print(f"Lambda_bar (Bragg sum): {lb_fib:.8f}  (expected 0.20110)")

# --- Silver (small) ---
lam1 = 1+np.sqrt(2)
seq_sil = generate_substitution_sequence('silver', 12)
pts_sil, L_sil = sequence_to_points(seq_sil, 'silver')
N_sil = len(pts_sil)
rho_sil = N_sil / L_sil
ell_bar_sil = 1/rho_sil
print(f"\nSilver N={N_sil:,}")

G_sil = []
for p in range(-50,51):
    for q in range(-50,51):
        if p==0 and q==0: continue
        t = p + q*lam1
        if 0 < t and 2*np.pi*t/ell_bar_sil < 50:
            G_sil.append(round(2*np.pi*t/ell_bar_sil, 8))
G_sil = np.array(sorted(set(G_sil)))
print(f"Candidate peaks in [0,50]: {len(G_sil)}")

I_sil = compute_I(pts_sil, G_sil)
lb_sil = 4*rho_sil * np.sum(I_sil/G_sil**2)
print(f"Lambda_bar (Bragg sum): {lb_sil:.8f}  (expected ~0.25)")

# --- Monte Carlo for Silver ---
rng = np.random.default_rng(42)
R_arr = np.linspace(2.0, 200*2.0, 1000)
var_sil, _ = compute_number_variance_1d(pts_sil, L_sil, R_arr, num_windows=15000, rng=rng)
lb_mc = compute_lambda_bar(R_arr, var_sil)
print(f"Lambda_bar (MC):        {lb_mc:.8f}")
print(f"\nComparison: Bragg={lb_sil:.5f}, MC={lb_mc:.5f}, 1/4={0.25:.5f}")
print(f"Is Bragg converging to MC? diff = {lb_sil - lb_mc:.6f}")
