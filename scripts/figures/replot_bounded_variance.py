"""Replot bounded variance (4-panel) without verbose suptitle."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
from substitution_tilings import generate_substitution_sequence, sequence_to_points
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

NUM_WINDOWS = 15_000
NUM_R = 2000

chains = [
    ('Integer Lattice', None, None),
    ('Fibonacci', 'fibonacci', 1),
    ('Silver', 'silver', 2),
    ('Bronze', 'bronze', 3),
]
colors = ['#4477AA', '#228833', '#AA3377', '#EE6677']

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for idx, (name, chain_type, n) in enumerate(chains):
    ax = axes[idx]
    rng = np.random.default_rng(42)

    if chain_type is None:
        # Integer lattice
        N = 100_000
        points = np.arange(N, dtype=float)
        L = float(N)
        rho = 1.0
        mu = None
    else:
        for n_iter in range(5, 50):
            seq = generate_substitution_sequence(chain_type, n_iter)
            if len(seq) > 1_000_000:
                break
        points, L = sequence_to_points(seq, chain_type)
        N = len(points)
        rho = N / L
        mu_vals = {1: 1.618, 2: 2.414, 3: 3.303}
        mu = mu_vals.get(n)

    R_max = min(L / 4, 1000 if chain_type is None else L / 4)
    if chain_type is None:
        R_arr = np.linspace(0.5, 100, NUM_R)
    else:
        R_arr = np.linspace(1.0 / rho, R_max, NUM_R)

    print(f"Computing {name}: N={N:,}...")
    sig2, _ = compute_number_variance_1d(points, L, R_arr,
                                          num_windows=NUM_WINDOWS, rng=rng)
    lb = compute_lambda_bar(R_arr, sig2)

    ax.plot(R_arr, sig2, color=colors[idx], lw=0.5, alpha=0.8)
    ax.axhline(lb, color='crimson', ls='--', lw=1.5,
               label=rf'$\bar\Lambda = {lb:.3f}$')

    title = f'{name}'
    if mu:
        title += rf' ($\mu_{n} = {mu:.3f}$)'
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'Window half-width $R$', fontsize=11)
    ax.set_ylabel(r'$\sigma^2(R)$', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(bottom=0)
    if chain_type is not None:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))

    info = f'$N = {N:,}$'
    ax.text(0.97, 0.95, info, transform=ax.transAxes, fontsize=9,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='goldenrod', alpha=0.8))

plt.tight_layout()
out = os.path.join(BASE, 'results', 'figures', 'fig2_bounded_variance.png')
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved {out}")
