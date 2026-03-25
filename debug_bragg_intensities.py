"""
Careful derivation of the correct Bragg intensity formula for metallic-mean chains.
We compute S(k) = (1/N)|sum_j exp(ikx_j)|^2 numerically for the Silver chain
and extract the peak intensities I_G = S(G)/N.
Then verify: Lambda_bar = (4*rho) * sum_{G>0} I_G / G^2

This is a semi-analytical (numerical) approach.  Once we verify it matches
the Monte Carlo Lambda_bar = 0.25 for Silver, we can investigate whether 0.25 = 1/4 exactly.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from substitution_tilings import CHAINS, generate_substitution_sequence, sequence_to_points
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

# ================================================================
# STEP 1: First verify normalization on FIBONACCI chain
#         (known Lambda_bar = 0.20110)
# ================================================================
print("=" * 65)
print("  Verifying Bragg sum formula on Fibonacci chain")
print("=" * 65)

# Generate Fibonacci tiling
seq_fib = generate_substitution_sequence('fibonacci', 25)
pts_fib, L_fib = sequence_to_points(seq_fib, 'fibonacci')
N_fib = len(pts_fib)
rho_fib = N_fib / L_fib
print(f"Fibonacci: N={N_fib:,}, L={L_fib:.1f}, rho={rho_fib:.8f}")

# Compute S(k) at many k values to find peaks
# Use known Bragg peak positions for Fibonacci:
# G_{p,q} = 2*pi*(p + q*tau)/ell_bar  with tau = golden ratio
tau = (1 + np.sqrt(5)) / 2
ell_bar_fib = 1.0/rho_fib

# Generate candidate peak positions
P_max = 100
G_candidates = []
labels = []
for p in range(-P_max, P_max+1):
    for q in range(-P_max, P_max+1):
        if p == 0 and q == 0:
            continue
        G = 2 * np.pi * (p + q * tau) / ell_bar_fib
        if G > 0.001:
            G_candidates.append(G)
            labels.append((p, q))

G_candidates = np.array(sorted(set(round(g, 6) for g in G_candidates)))
G_candidates = G_candidates[G_candidates < 100.0]

print(f"Candidate Bragg peaks in [0,100]: {len(G_candidates)}")

# Compute S(G) at each candidate G
# S(k) = (1/N) * |sum_j exp(i k x_j)|^2
def compute_Sk(pts, k_vals, batch=200):
    N = len(pts)
    Sk = np.zeros(len(k_vals))
    for i in range(0, len(k_vals), batch):
        kbatch = k_vals[i:i+batch]
        phase = np.outer(kbatch, pts)  # (batch, N)
        rho_hat = np.sum(np.exp(1j * phase), axis=1)
        Sk[i:i+batch] = np.abs(rho_hat)**2 / N
    return Sk

print("Computing S(G) at candidate positions...")
Sk_candidates = compute_Sk(pts_fib, G_candidates)

# I_G = S(G)/N  (peak intensity per particle)
# Filter: keep only peaks with S(G)/N > threshold
# For Bragg peaks in a substitution tiling, S(G)/N scales as |F_G|^2 which can be large.
threshold = 0.001
mask_peaks = Sk_candidates > threshold * N_fib
G_peaks_fib = G_candidates[mask_peaks]
I_peaks_fib = Sk_candidates[mask_peaks] / N_fib  # normalize per particle

print(f"Bragg peaks with S(G)/N > {threshold}: {np.sum(mask_peaks)}")
print(f"Strongest peaks (top 10):")
top10 = np.argsort(I_peaks_fib)[::-1][:10]
for idx in top10:
    print(f"  G={G_peaks_fib[idx]:.6f}, S(G)/N={I_peaks_fib[idx]:.8f}")

# Lambda_bar = 4*rho * sum I_G/G^2
lb_bragg_fib = 4 * rho_fib * np.sum(I_peaks_fib / G_peaks_fib**2)
print(f"\nFibonacci Lambda_bar (Bragg sum): {lb_bragg_fib:.8f}")
print(f"Expected (Z&T 2009):              0.20110")
print(f"Ratio: {lb_bragg_fib/0.20110:.6f}")

# Also compute Monte Carlo variance for comparison
print("\nComputing Fibonacci Monte Carlo variance...")
rng = np.random.default_rng(42)
mean_sp = 1.0/rho_fib
R_arr = np.linspace(mean_sp, 300*mean_sp, 1500)
var_fib, _ = compute_number_variance_1d(pts_fib, L_fib, R_arr, num_windows=20000, rng=rng)
lb_mc_fib = compute_lambda_bar(R_arr, var_fib)
print(f"Fibonacci Lambda_bar (MC):        {lb_mc_fib:.8f}")

print()

# ================================================================
# STEP 2: Silver chain
# ================================================================
print("=" * 65)
print("  Silver chain: Bragg sum + Monte Carlo")
print("=" * 65)

# Generate Silver tiling
seq_sil = generate_substitution_sequence('silver', 18)
pts_sil, L_sil = sequence_to_points(seq_sil, 'silver')
N_sil = len(pts_sil)
rho_sil = N_sil / L_sil
print(f"Silver: N={N_sil:,}, L={L_sil:.1f}, rho={rho_sil:.8f}")

lam1_sil = 1 + np.sqrt(2)
ell_bar_sil = 1.0/rho_sil
print(f"ell_bar = {ell_bar_sil:.8f}  (expected 2.0)")

# Generate candidate peak positions for Silver
G_cands_sil = []
for p in range(-P_max, P_max+1):
    for q in range(-P_max, P_max+1):
        if p == 0 and q == 0:
            continue
        G = 2 * np.pi * (p + q * lam1_sil) / ell_bar_sil
        if G > 0.001:
            G_cands_sil.append(round(G, 8))

G_cands_sil = np.array(sorted(set(G_cands_sil)))
G_cands_sil = G_cands_sil[G_cands_sil < 100.0]
print(f"Candidate Bragg peaks in [0,100]: {len(G_cands_sil)}")

print("Computing S(G) for Silver...")
Sk_sil_cands = compute_Sk(pts_sil, G_cands_sil)

mask_sil = Sk_sil_cands > threshold * N_sil
G_peaks_sil = G_cands_sil[mask_sil]
I_peaks_sil = Sk_sil_cands[mask_sil] / N_sil

print(f"Bragg peaks with S(G)/N > {threshold}: {np.sum(mask_sil)}")
print(f"Strongest peaks (top 10):")
top10_sil = np.argsort(I_peaks_sil)[::-1][:10]
for idx in top10_sil:
    print(f"  G={G_peaks_sil[idx]:.6f}, S(G)/N={I_peaks_sil[idx]:.8f}")

lb_bragg_sil = 4 * rho_sil * np.sum(I_peaks_sil / G_peaks_sil**2)
print(f"\nSilver Lambda_bar (Bragg sum): {lb_bragg_sil:.8f}")
print(f"diff from 1/4: {lb_bragg_sil - 0.25:.2e}")

# Monte Carlo
print("\nComputing Silver Monte Carlo variance...")
mean_sp_sil = 1.0/rho_sil
R_arr_sil = np.linspace(mean_sp_sil, 300*mean_sp_sil, 1500)
var_sil, _ = compute_number_variance_1d(pts_sil, L_sil, R_arr_sil, num_windows=20000, rng=rng)
lb_mc_sil = compute_lambda_bar(R_arr_sil, var_sil)
print(f"Silver Lambda_bar (MC):        {lb_mc_sil:.8f}")

# ================================================================
# STEP 3: Investigate the missing contribution
# ================================================================
print()
print("=" * 65)
print("  Investigating missing contribution")
print("=" * 65)
print()
print(f"Fibonacci: Bragg sum = {lb_bragg_fib:.6f}, MC = {lb_mc_fib:.6f}")
print(f"Silver:    Bragg sum = {lb_bragg_sil:.6f}, MC = {lb_mc_sil:.6f}")
print()
print(f"Fibonacci ratio Bragg/MC: {lb_bragg_fib/lb_mc_fib:.6f}")
print(f"Silver    ratio Bragg/MC: {lb_bragg_sil/lb_mc_sil:.6f}")
print()

# The ratio should be 1 if the formula is correct.
# If the sum is missing peaks (cutoff too small), adding more should help.
# Let's try P_max = 200 for Silver:
print("Extending to P_max=200 for Silver...")
G_cands_sil2 = []
P_max2 = 200
for p in range(-P_max2, P_max2+1):
    for q in range(-P_max2, P_max2+1):
        if p == 0 and q == 0:
            continue
        G = 2 * np.pi * (p + q * lam1_sil) / ell_bar_sil
        if G > 0.001 and G < 200.0:
            G_cands_sil2.append(round(G, 8))

G_cands_sil2 = np.array(sorted(set(G_cands_sil2)))
print(f"Candidate Bragg peaks in [0,200]: {len(G_cands_sil2)}")

Sk_sil2 = compute_Sk(pts_sil, G_cands_sil2)
mask_sil2 = Sk_sil2 > threshold * N_sil
G_peaks_sil2 = G_cands_sil2[mask_sil2]
I_peaks_sil2 = Sk_sil2[mask_sil2] / N_sil

lb_bragg_sil2 = 4 * rho_sil * np.sum(I_peaks_sil2 / G_peaks_sil2**2)
print(f"Silver Lambda_bar (Bragg sum, P_max=200, k<200): {lb_bragg_sil2:.8f}")
print(f"diff from 1/4: {lb_bragg_sil2 - 0.25:.2e}")

# ================================================================
# STEP 4: Check convergence as k_max increases
# ================================================================
print()
print("Convergence as function of k_max:")
for k_max_val in [20, 50, 100, 150, 200, 300, 500]:
    mask_kmax = G_peaks_sil2 < k_max_val
    if np.sum(mask_kmax) == 0:
        continue
    lb_kmax = 4 * rho_sil * np.sum(I_peaks_sil2[mask_kmax] / G_peaks_sil2[mask_kmax]**2)
    print(f"  k_max={k_max_val:5.0f}: N_peaks={np.sum(mask_kmax):6d}, Lambda_bar={lb_kmax:.8f}, diff={lb_kmax-0.25:+.4f}")

print()
print(f"Final MC value: Lambda_bar = {lb_mc_sil:.8f}")
print(f"1/4            = 0.25000000")
