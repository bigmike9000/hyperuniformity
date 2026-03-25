"""
FINAL normalization fix.

Key observation from debug_bragg2.py:
  S(G)/N ~ N * (constant)  -- scales LINEARLY with N!

For a TRUE Bragg peak in a quasiperiodic tiling:
  S(G) = (1/N) * |sum_j exp(i G x_j)|^2 ~ N * |F_G|^2

where |F_G|^2 is the SQUARED AMPLITUDE of the structure factor per particle in the
thermodynamic limit (specific Bragg intensity, dimensionless, O(1)).

In the normalization S(k) = (1/N)|sum_j exp(ikx_j)|^2:
  - For the INTEGER LATTICE: S(G_m) = N (each of N terms contributes 1 to the sum)
  - For a QUASIPERIODIC TILING: S(G_m) ~ N * |F_G|^2 where |F_G|^2 ~ sinc^2(...)

So the correct SPECIFIC INTENSITY is:
  I_G = lim_{N->inf} S(G)/N^2   ... NO wait:

For integer lattice S(G)/N = N -> infinity? NO:
  (1/N)|sum_j exp(iGj)|^2 for G=2*pi*m:
  sum_j exp(iGj) = sum_{j=0}^{N-1} 1 = N
  S(G) = (1/N) * N^2 = N  ← Yes, S(G) = N for integer lattice.
  So S(G)/N = 1. The SPECIFIC INTENSITY per particle is I_G = S(G)/N = 1.

But in debug_bragg2, integer lattice gives S(G)/N = 1000 (= N!) -- this is a bug.
The formula compute_F_exact returns np.abs(rho_hat)^2 / N, but for integer lattice:
  rho_hat = sum_j exp(iGx_j) = N  => |rho_hat|^2 = N^2 => S(G) = N^2/N = N.
So compute_F_exact returns S(G) = N, not S(G)/N !!!

So the compute_F_exact function returns S(G), not S(G)/N.
And for the quasiperiodic tiling: S(G) ~ N * |F_G|^2 means S(G)/N -> |F_G|^2.

But wait: for Fibonacci (N=75025), S(G)/N (as computed) = 36972 for the (1,0) peak.
This would mean S(G)/N^2 = 36972/75025 = 0.493...

Actually: compute_F_exact returns |rho_hat|^2/N where rho_hat = sum exp(ikx).
So compute_F_exact(pts, L, [G])[0] = |sum_j exp(iGx_j)|^2 / N.
For integer lattice, G=2*pi, sum = N, so F_exact = N^2/N = N = 1000. ✓

For Fibonacci (1,0) peak: |sum_j exp(i G x_j)|^2 / N = 36972 for N=75025.
So |sum_j exp(iGx_j)|^2 = 36972 * 75025 ~ 2.77e9.
The NORMALIZED amplitude: |F_G|^2 = |sum exp(iGx_j)|^2 / N^2 = 36972/75025 = 0.4928.

The CORRECT formula (consistent with integer lattice calibration):
  Lambda_bar = (4*rho) * sum_{G>0} I_G / G^2
  where I_G = |F_G|^2 = lim S_computed(G) / N
            [where S_computed(G) = (1/N)|sum exp(iGx)|^2]
  So I_G = S_computed(G) / N  (dividing again by N)

Let me verify with integer lattice:
  I_G = S_computed(G)/N = N/N = 1 ✓
  Lambda_bar = 4*1 * sum_{m=1}^inf 1/(2*pi*m)^2 = 4/(4*pi^2) * pi^2/6 = 1/6 ✓
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from substitution_tilings import CHAINS, generate_substitution_sequence, sequence_to_points
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

def compute_Sk_normalized(pts, G_vals, batch=100):
    """
    Compute I_G = |F_G|^2 = lim_{N->inf} S(G)/N
    where S(G) = (1/N)|sum_j exp(iGx_j)|^2.

    So this returns S(G)/N = |sum exp(iGx)|^2 / N^2.
    """
    N = len(pts)
    I_G = np.zeros(len(G_vals))
    for i in range(0, len(G_vals), batch):
        Gb = G_vals[i:i+batch]
        phase = np.outer(Gb, pts)  # (batch, N)
        rho_hat = np.sum(np.exp(1j * phase), axis=1)  # (batch,)
        I_G[i:i+batch] = np.abs(rho_hat)**2 / N**2   # = S(G)/N
    return I_G

print("=" * 65)
print("  CALIBRATION: Integer lattice, Lambda_bar = 1/6")
print("=" * 65)
N_latt = 1000
pts_latt = np.arange(N_latt, dtype=float)
rho_latt = 1.0
G_latt = np.array([2*np.pi*m for m in range(1, 51)])
I_latt = compute_Sk_normalized(pts_latt, G_latt)
lb_latt = 4*rho_latt * np.sum(I_latt / G_latt**2)
print(f"I_G for first 3 peaks (should be 1.0): {I_latt[:3]}")
print(f"Lambda_bar (50 peaks): {lb_latt:.8f}  (expected 1/6={1/6:.8f})")
print()

# ================================================================
# FIBONACCI CHAIN
# ================================================================
print("=" * 65)
print("  Fibonacci chain")
print("=" * 65)
tau = (1+np.sqrt(5))/2
lam2_fib = 1 - tau  # = -0.6180...

for n_iters, label in [(23,'N~75k'), (25,'N~196k')]:
    seq_fib = generate_substitution_sequence('fibonacci', n_iters)
    pts_fib, L_fib = sequence_to_points(seq_fib, 'fibonacci')
    N_fib = len(pts_fib)
    rho_fib = N_fib/L_fib
    ell_bar_fib = 1/rho_fib

    # All candidate Bragg peaks
    G_all = []
    for p in range(-100, 101):
        for q in range(-100, 101):
            if p==0 and q==0: continue
            t = p + q*tau
            if t <= 0: continue
            G = 2*np.pi*t/ell_bar_fib
            if G < 200:
                G_all.append(round(G, 8))
    G_all = np.array(sorted(set(G_all)))

    print(f"N={N_fib:,} (iters={n_iters}): {len(G_all)} candidate peaks in [0,200]")

    # Compute I_G = S(G)/N (specific intensity per particle)
    I_all = compute_Sk_normalized(pts_fib, G_all, batch=50)

    # Lambda_bar
    lb = 4*rho_fib * np.sum(I_all/G_all**2)
    print(f"  Lambda_bar (Bragg sum): {lb:.8f}  (expected ~0.20110)")

print()

# ================================================================
# SILVER CHAIN (smaller N to avoid memory issues)
# ================================================================
print("=" * 65)
print("  Silver chain (N~50k)")
print("=" * 65)
lam1_sil = 1+np.sqrt(2)

for n_iters in [12, 14, 16, 17]:
    seq_sil = generate_substitution_sequence('silver', n_iters)
    pts_sil, L_sil = sequence_to_points(seq_sil, 'silver')
    N_sil = len(pts_sil)
    rho_sil = N_sil/L_sil
    ell_bar_sil = 1/rho_sil

    G_all_sil = []
    for p in range(-100, 101):
        for q in range(-100, 101):
            if p==0 and q==0: continue
            t = p + q*lam1_sil
            if t <= 0: continue
            G = 2*np.pi*t/ell_bar_sil
            if G < 200:
                G_all_sil.append(round(G, 8))
    G_all_sil = np.array(sorted(set(G_all_sil)))

    print(f"N={N_sil:,} (iters={n_iters}): {len(G_all_sil)} candidate peaks in [0,200]")

    I_all_sil = compute_Sk_normalized(pts_sil, G_all_sil, batch=50)
    lb_sil = 4*rho_sil * np.sum(I_all_sil/G_all_sil**2)
    print(f"  Lambda_bar (Bragg sum): {lb_sil:.8f}")
    print(f"  diff from 1/4:         {lb_sil-0.25:+.4e}")

# ================================================================
# MONTE CARLO comparison for Silver
# ================================================================
print()
print("=" * 65)
print("  Silver chain Monte Carlo (N~665k)")
print("=" * 65)
seq_sil15 = generate_substitution_sequence('silver', 15)
pts_sil15, L_sil15 = sequence_to_points(seq_sil15, 'silver')
N_sil15 = len(pts_sil15)
rho_sil15 = N_sil15/L_sil15
print(f"N={N_sil15:,}, rho={rho_sil15:.8f}")

rng = np.random.default_rng(42)
mean_sp = 1/rho_sil15
R_arr = np.linspace(mean_sp, 300*mean_sp, 2000)
var_sil15, _ = compute_number_variance_1d(pts_sil15, L_sil15, R_arr,
                                          num_windows=30000, rng=rng)
lb_mc = compute_lambda_bar(R_arr, var_sil15)
print(f"Lambda_bar (MC, N={N_sil15:,}): {lb_mc:.8f}")
print(f"diff from 1/4: {lb_mc-0.25:+.4e}")
