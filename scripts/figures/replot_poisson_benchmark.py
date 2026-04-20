"""Replot Poisson benchmark without internal 'Figure 1' title."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
from quasicrystal_variance import compute_number_variance_1d

N = 50_000
rho = 1.0
L = N / rho
NUM_WINDOWS = 30_000
NUM_R = 50
N_REAL = 20  # average over multiple realizations

R_arr = np.linspace(1.0, 50.0, NUM_R)

print("Computing Poisson variance (averaging over realizations)...")
all_sig2 = []
for i in range(N_REAL):
    rng = np.random.default_rng(42 + i)
    points = rng.uniform(0, L, size=N)
    points.sort()
    s2, _ = compute_number_variance_1d(points, L, R_arr,
                                        num_windows=NUM_WINDOWS, rng=rng)
    all_sig2.append(s2)
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{N_REAL} done")

sig2 = np.mean(all_sig2, axis=0)
sig2_err = np.std(all_sig2, axis=0) / np.sqrt(N_REAL)
theory = 2 * rho * R_arr
rel_err = np.abs(sig2 - theory) / theory * 100
mean_err = np.mean(rel_err)
print(f"Mean relative error: {mean_err:.1f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: variance vs R
ax1.plot(R_arr, theory, 'r--', lw=2, label=r'Theory: $\sigma^2 = 2\rho R$')
ax1.errorbar(R_arr, sig2, yerr=2*sig2_err, fmt='k.', ms=5, capsize=2,
             label=r'Simulated $\sigma^2(R)$ ($\pm 2\sigma$)')
ax1.set_xlabel(r'Window half-width $R$', fontsize=12)
ax1.set_ylabel(r'Number variance $\sigma^2(R)$', fontsize=12)
ax1.set_title('(a) Poisson variance vs. window size', fontsize=12)
ax1.legend(fontsize=10)
ax1.text(0.03, 0.92, f'$N = {N:,}$, $\\rho = {rho:.0f}$',
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))

# Right: relative error
ax2.bar(R_arr, rel_err, width=R_arr[1]-R_arr[0]-0.1, color='steelblue', alpha=0.7)
ax2.axhline(5, color='red', ls='--', lw=1.5, label='5% threshold')
ax2.text(0.97, 0.92, f'Mean error: {mean_err:.1f}%',
         transform=ax2.transAxes, fontsize=10, ha='right',
         bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.8))
ax2.set_xlabel(r'Window half-width $R$', fontsize=12)
ax2.set_ylabel('Relative Error (%)', fontsize=10)
ax2.set_title('(b) Accuracy vs. exact theory', fontsize=12)
ax2.legend(fontsize=10)

plt.tight_layout()
out = os.path.join(BASE, 'results', 'figures', 'fig1_poisson_benchmark.png')
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved {out}")
