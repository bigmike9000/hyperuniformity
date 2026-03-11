"""
Stealthy Hyperuniform Configuration Analysis
=============================================
Analyzes the ~4000-configuration ensembles (N=2000, rho=1) provided by
a collaborating researcher for chi = 0.1, 0.2, 0.3.

Analysis steps:
  1. Parse configuration files for each chi
  2. Compute ensemble-averaged number variance -> Lambda_bar
  3. Analyze the provided structure factor files (sf_bin.txt, sf_res.txt):
       - Verify stealthy exclusion zone K = 2*pi*chi
       - Characterize S(k) envelope near k = K (edge behavior)
       - Compare to our optimization-based results
  4. Compute excess spreadability E(t) from ensemble-averaged S(k)
  5. Produce comparison figures
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar
from two_phase_media import (
    compute_spectral_density, compute_excess_spreadability, extract_alpha_fit
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
STEALTHY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'stealthy_configurations')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 2026
PHI2 = 0.35
CHI_VALUES = [0.1, 0.2, 0.3]

# How many configurations to load per chi for variance computation
# (all ~4316 is best; use all since variance computation is fast for N=2000)
MAX_CONFIGS = None   # None = use all

rng = np.random.default_rng(SEED)


# ============================================================
# File parsing
# ============================================================

def read_cco(ifile: str):
    """Parse a CCO-format configuration file (1D)."""
    with open(ifile, 'r') as f:
        lines = f.readlines()
    d = int(lines[0].strip())
    # Lattice vectors: lines 1..d
    lv = []
    for i in range(1, d + 1):
        vals = lines[i].split()
        lv.append([float(v) for v in vals[:d]])
    lattice_vectors = np.array(lv)
    # Coordinates: lines d+1 ..
    coords = []
    for j in range(d + 1, len(lines)):
        vals = lines[j].split()
        if len(vals) >= d:
            coords.append([float(v) for v in vals[:d]])
    return d, lattice_vectors, np.array(coords)


def load_all_configs(chi_val, max_n=None):
    """
    Load all configuration files for a given chi.
    Returns list of (points_1d, L) tuples.
    """
    folder = os.path.join(STEALTHY_DIR, f'{chi_val}_patterns')
    files = sorted(glob(os.path.join(folder, '*.txt')))
    files = [f for f in files if not f.endswith(('sf_bin.txt', 'sf_res.txt'))]
    if max_n is not None:
        files = files[:max_n]

    configs = []
    for ifile in files:
        try:
            d, lv, coords = read_cco(ifile)
            L = float(lv[0, 0])   # 1D domain length
            pts = np.sort(coords[:, 0])
            configs.append((pts, L))
        except Exception as e:
            print(f"  Warning: could not parse {os.path.basename(ifile)}: {e}")
    return configs


def load_sf(chi_val, binned=True):
    """Load the pre-computed structure factor for a given chi."""
    folder = os.path.join(STEALTHY_DIR, f'{chi_val}_patterns')
    fname = 'sf_bin.txt' if binned else 'sf_res.txt'
    fpath = os.path.join(folder, fname)
    data = np.loadtxt(fpath, delimiter=',', comments='#')
    if binned:
        # columns: |k|, S(k)
        k = data[:, 0]
        S = data[:, 1]
    else:
        # columns: k (signed), |k|, S(k)
        k = data[:, 1]   # use |k|
        S = data[:, 2]
    return k, S


# ============================================================
# Core analysis per chi value
# ============================================================

def analyze_chi(chi_val, configs):
    """
    Analyze a set of stealthy configurations for a given chi.
    Returns a results dict.
    """
    N = len(configs[0][0])
    L = configs[0][1]
    rho = N / L
    n_configs = len(configs)

    print(f"\n  --- chi={chi_val}, N={N}, L={L:.1f}, rho={rho:.4f}, "
          f"{n_configs} configs ---")

    # 1. Ensemble-averaged number variance
    # Compute for each config, then average
    print(f"  Computing variance for {n_configs} configurations...")
    mean_sp = 1.0 / rho
    R_max = min(200 * mean_sp, L / 4)
    R_arr = np.linspace(mean_sp * 0.5, R_max, 600)
    num_windows_per = max(500, min(2000, 50000 // n_configs))

    t0 = time.perf_counter()
    all_vars = np.zeros((n_configs, len(R_arr)))
    for i, (pts, Li) in enumerate(configs):
        v, _ = compute_number_variance_1d(pts, Li, R_arr,
                                          num_windows=num_windows_per, rng=rng)
        all_vars[i] = v
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_configs} done...")

    var_mean = np.mean(all_vars, axis=0)
    var_std  = np.std(all_vars, axis=0) / np.sqrt(n_configs)
    t_var = time.perf_counter() - t0
    print(f"  Variance done in {t_var:.1f}s")

    lb = compute_lambda_bar(R_arr, var_mean)
    # Error: propagate SEM through the mean
    lb_err_sem = np.mean(var_std[len(R_arr)//3:])

    # Bootstrap error on Lambda_bar (split configs into 10 groups)
    n_boot = 10
    group_size = n_configs // n_boot
    lb_boots = []
    for g in range(n_boot):
        group_var = np.mean(all_vars[g*group_size:(g+1)*group_size], axis=0)
        lb_boots.append(compute_lambda_bar(R_arr, group_var))
    lb_err = np.std(lb_boots)

    print(f"  Lambda_bar = {lb:.6f} ± {lb_err:.6f} (bootstrap)")

    # 2. Load provided structure factor
    k_bin, S_bin = load_sf(chi_val, binned=True)

    # 3. Analyze stealthy exclusion zone
    K_theory = 2 * np.pi * chi_val   # k < K should have S(k) = 0
    # Find the empirical edge: last k where S(k) < 0.05
    stealthy_mask = S_bin < 0.05
    if np.any(~stealthy_mask):
        K_empirical = k_bin[np.argmax(~stealthy_mask)]
    else:
        K_empirical = k_bin[-1]

    # Mean S(k) in the exclusion zone
    excl_mask = k_bin < K_theory
    S_excl_mean = np.mean(S_bin[excl_mask]) if np.any(excl_mask) else np.nan
    S_excl_max  = np.max(S_bin[excl_mask])  if np.any(excl_mask) else np.nan

    # Mean S(k) in the bulk (k >> K)
    bulk_mask = k_bin > 2 * K_theory
    S_bulk_mean = np.mean(S_bin[bulk_mask]) if np.any(bulk_mask) else np.nan

    print(f"  Stealthy exclusion: K_theory={K_theory:.4f}, K_empirical≈{K_empirical:.4f}")
    print(f"  S(k) in exclusion zone: mean={S_excl_mean:.3e}, max={S_excl_max:.3e}")
    print(f"  S(k) bulk (k>2K): mean={S_bulk_mean:.4f}")

    # 4. Diffusion spreadability from binned S(k)
    # Use the structure factor directly (it's already ensemble-averaged)
    # We need chi_V(k) = rho * |m(k)|^2 * S(k)
    a = PHI2 / (2 * rho)
    # Only use k > 0 for chi_V
    k_pos_mask = k_bin > 0
    k_pos = k_bin[k_pos_mask]
    S_pos = S_bin[k_pos_mask]
    chi_V = compute_spectral_density(k_pos, S_pos, rho, a)
    t_arr = np.logspace(-2, 8, 200)
    E_t = compute_excess_spreadability(k_pos, chi_V, PHI2, t_arr)
    alpha_fit, r2 = extract_alpha_fit(t_arr, E_t, t_min=1e1, t_max=1e4)
    print(f"  Alpha (spreadability fit): {alpha_fit:.4f}, R^2={r2:.4f}")

    # 5. S(k) near the edge: does it show a step or power-law onset?
    # Fit S(k) for k just above K: S(k) ~ (k - K)^gamma
    edge_mask = (k_bin > K_theory) & (k_bin < K_theory * 1.5)
    gamma_fit = np.nan
    if np.sum(edge_mask) >= 5:
        k_edge = k_bin[edge_mask]
        S_edge = S_bin[edge_mask]
        try:
            from scipy.optimize import curve_fit
            def edge_model(k, gamma, A):
                return A * np.maximum(k - K_theory, 0) ** gamma
            popt_e, _ = curve_fit(edge_model, k_edge, S_edge, p0=[1.0, 1.0],
                                   maxfev=3000)
            gamma_fit = popt_e[0]
            print(f"  S(k) near edge: ~ (k-K)^{gamma_fit:.3f}")
        except Exception:
            # Try log-log slope on S(k) in a small window above K
            log_k = np.log(k_edge - K_theory + 1e-6)
            log_S = np.log(np.maximum(S_edge, 1e-10))
            try:
                slope = np.polyfit(log_k, log_S, 1)[0]
                gamma_fit = slope
                print(f"  S(k) near edge (log slope): ~ (k-K)^{gamma_fit:.3f}")
            except Exception:
                pass

    return {
        'chi': chi_val,
        'N': N, 'L': float(L), 'rho': float(rho), 'n_configs': n_configs,
        'R_array': R_arr.tolist(),
        'var_mean': var_mean.tolist(),
        'var_std': var_std.tolist(),
        'lambda_bar': float(lb),
        'lambda_bar_err': float(lb_err),
        'K_theory': float(K_theory),
        'K_empirical': float(K_empirical),
        'S_excl_mean': float(S_excl_mean) if not np.isnan(S_excl_mean) else None,
        'S_excl_max': float(S_excl_max) if not np.isnan(S_excl_max) else None,
        'S_bulk_mean': float(S_bulk_mean) if not np.isnan(S_bulk_mean) else None,
        'alpha_fit': float(alpha_fit) if not np.isnan(alpha_fit) else None,
        'alpha_r2': float(r2),
        'gamma_edge': float(gamma_fit) if not np.isnan(gamma_fit) else None,
        't_array': t_arr.tolist(),
        'E_t': E_t.tolist(),
        'k_bin': k_bin.tolist(),
        'S_bin': S_bin.tolist(),
    }


# ============================================================
# Run analysis for all three chi values
# ============================================================

all_results = {}

for chi_val in CHI_VALUES:
    print(f"\n{'='*68}")
    print(f"  Loading chi={chi_val} configurations...")
    t0 = time.perf_counter()
    configs = load_all_configs(chi_val, max_n=MAX_CONFIGS)
    print(f"  Loaded {len(configs)} configs in {time.perf_counter()-t0:.1f}s")
    result = analyze_chi(chi_val, configs)
    all_results[chi_val] = result


# ============================================================
# Save results
# ============================================================

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)

json_path = os.path.join(RESULTS_DIR, 'stealthy_collab_results.json')
with open(json_path, 'w') as f:
    json.dump(make_serializable(all_results), f, indent=2)
print(f"\n  Saved numerical results to {json_path}")


# ============================================================
# Figures
# ============================================================

chi_colors = {0.1: '#f7b6d2', 0.2: '#e377c2', 0.3: '#9467bd'}

# ---- Figure G: Structure factor for all three chi ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(r'Ensemble-Averaged Structure Factor $S(k)$ for Stealthy 1D Patterns '
             r'($N=2000$, $\rho=1$, $\sim$4000 configs)', fontsize=13)

for i, chi_val in enumerate(CHI_VALUES):
    r = all_results[chi_val]
    ax = axes[i]
    k = np.array(r['k_bin'])
    S = np.array(r['S_bin'])
    K = r['K_theory']

    # Full S(k)
    ax.plot(k, S, '-', color=chi_colors[chi_val], lw=1.2, alpha=0.9)

    # Shade the exclusion zone
    ax.axvspan(0, K, alpha=0.15, color='red', label=rf'Exclusion zone $k<K$')
    ax.axvline(K, color='red', ls='--', lw=1.5,
               label=rf'$K = 2\pi\chi = {K:.3f}$')
    ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)

    ax.set_xlabel(r'$k$', fontsize=12)
    ax.set_ylabel(r'$S(k)$', fontsize=12)
    ax.set_title(rf'$\chi = {chi_val}$, $\bar{{\Lambda}} = {r["lambda_bar"]:.4f}$',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.05, 1.5)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, ls=':', alpha=0.4)

    # Inset: exclusion zone (log scale)
    ax_in = ax.inset_axes([0.05, 0.55, 0.45, 0.40])
    mask = k < K * 1.8
    ax_in.semilogy(k[mask], np.maximum(S[mask], 1e-20),
                   color=chi_colors[chi_val], lw=1.2)
    ax_in.axvline(K, color='red', ls='--', lw=1, alpha=0.7)
    ax_in.set_xlim(0, K * 1.8)
    ax_in.set_title('Near $K$', fontsize=7)
    ax_in.tick_params(labelsize=7)
    ax_in.grid(True, ls=':', alpha=0.3)

plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figG_stealthy_structure_factor.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ---- Figure H: Number variance for all three chi ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax2 = axes[1]

for chi_val in CHI_VALUES:
    r = all_results[chi_val]
    R = np.array(r['R_array'])
    var = np.array(r['var_mean'])
    var_std = np.array(r['var_std'])
    lb = r['lambda_bar']
    lb_err = r['lambda_bar_err']
    color = chi_colors[chi_val]

    mask = R > 0.5
    ax1.fill_between(R[mask],
                     (var - var_std)[mask], (var + var_std)[mask],
                     alpha=0.2, color=color)
    ax1.plot(R[mask], var[mask], '-', color=color, lw=1.5,
             label=rf'$\chi={chi_val}$, $\bar{{\Lambda}}={lb:.4f}\pm{lb_err:.4f}$')
    ax1.axhline(lb, color=color, ls='--', lw=0.8, alpha=0.7)

ax1.set_xlabel(r'$R$', fontsize=12)
ax1.set_ylabel(r'$\sigma^2(R)$', fontsize=12)
ax1.set_title('Ensemble-Averaged Number Variance', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, ls=':', alpha=0.4)

# Lambda_bar vs chi
chis_plot = np.array([r['chi'] for r in all_results.values()])
lbs_plot  = np.array([r['lambda_bar'] for r in all_results.values()])
errs_plot = np.array([r['lambda_bar_err'] for r in all_results.values()])

ax2.errorbar(chis_plot, lbs_plot, yerr=errs_plot, fmt='o-',
             color='#9467bd', ms=8, capsize=5, lw=1.5, label='Collab. ensemble')
ax2.axhline(1/6, color='black', ls='--', lw=1.5, alpha=0.7,
            label=r'Lattice: $1/6 \approx 0.1667$')

ax2.set_xlabel(r'Stealthiness $\chi$', fontsize=12)
ax2.set_ylabel(r'$\bar{\Lambda}$', fontsize=12)
ax2.set_title(r'$\bar{\Lambda}(\chi)$ from Collab. Ensemble', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, ls=':', alpha=0.4)
ax2.set_xlim(0.05, 0.35)
ax2.set_ylim(0)

plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figH_stealthy_variance.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ---- Figure I: Spreadability E(t) for three chi ----
fig, ax = plt.subplots(figsize=(10, 6))
t_ref = np.logspace(0, 6, 100)

for chi_val in CHI_VALUES:
    r = all_results[chi_val]
    t_arr = np.array(r['t_array'])
    E_t = np.array(r['E_t'])
    mask = E_t > 1e-20
    alpha_fit = r['alpha_fit']
    label = (rf'$\chi={chi_val}$' +
             (f', $\\hat{{\\alpha}}={alpha_fit:.2f}$' if alpha_fit else ''))
    ax.loglog(t_arr[mask], E_t[mask], color=chi_colors[chi_val],
              lw=1.8, label=label)

ax.axvspan(1e1, 1e4, alpha=0.08, color='gold', label='Fit window')
ax.set_xlabel(r'Diffusion time $t$', fontsize=13)
ax.set_ylabel(r'Excess spreadability $E(t)$', fontsize=13)
ax.set_title('Stealthy: Excess Spreadability from Ensemble-Averaged $S(k)$',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, ls=':', alpha=0.4)

plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figI_stealthy_spreadability.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ---- Figure J: Exclusion zone verification ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]   # S(k) near K for all three chi (zoomed in)
for chi_val in CHI_VALUES:
    r = all_results[chi_val]
    k = np.array(r['k_bin'])
    S = np.array(r['S_bin'])
    K = r['K_theory']
    # Normalize to k/K axis
    kK = k / K
    mask = (kK > 0.5) & (kK < 2.5) & (S > 0)
    ax1.semilogy(kK[mask], S[mask], color=chi_colors[chi_val],
                 lw=1.5, label=rf'$\chi={chi_val}$, $K={K:.3f}$')
ax1.axvline(1.0, color='black', ls='--', lw=1.5, alpha=0.7,
            label='$k = K$')
ax1.axhline(1.0, color='gray', ls=':', lw=1)
ax1.set_xlabel(r'$k / K$', fontsize=12)
ax1.set_ylabel(r'$S(k)$', fontsize=12)
ax1.set_title(r'Scaled $S(k)$ Near Exclusion Boundary', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, ls=':', alpha=0.4)

ax2 = axes[1]   # K_empirical vs K_theory
K_theories = [r['K_theory'] for r in all_results.values()]
K_empiricals = [r['K_empirical'] for r in all_results.values()]
ax2.scatter(K_theories, K_empiricals, s=100, color='#9467bd', zorder=5)
for chi_val, r in all_results.items():
    ax2.annotate(rf'$\chi={chi_val}$', (r['K_theory'], r['K_empirical']),
                 xytext=(5, 5), textcoords='offset points', fontsize=10)
lim_min = min(min(K_theories), min(K_empiricals)) * 0.9
lim_max = max(max(K_theories), max(K_empiricals)) * 1.1
ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1, alpha=0.5,
         label='$K_{\\rm empirical} = K_{\\rm theory}$')
ax2.set_xlabel(r'$K_{\rm theory} = 2\pi\chi$', fontsize=12)
ax2.set_ylabel(r'$K_{\rm empirical}$', fontsize=12)
ax2.set_title('Exclusion Zone: Theory vs Measured', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, ls=':', alpha=0.4)

plt.tight_layout()
path = os.path.join(RESULTS_DIR, 'figJ_exclusion_zone.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 72)
print(f"  {'STEALTHY HYPERUNIFORM ANALYSIS SUMMARY':^68s}")
print("=" * 72)
print(f"  {'chi':>6s}  {'N_cfg':>6s}  {'Lambda_bar':>12s}  {'Err':>8s}  "
      f"{'K_theory':>9s}  {'K_emp':>9s}  {'S_max_excl':>12s}  {'alpha_fit':>10s}")
print("  " + "-" * 72)
for chi_val in CHI_VALUES:
    r = all_results[chi_val]
    print(f"  {chi_val:6.1f}  {r['n_configs']:6d}  {r['lambda_bar']:12.6f}  "
          f"{r['lambda_bar_err']:8.6f}  {r['K_theory']:9.4f}  "
          f"{r['K_empirical']:9.4f}  {r['S_excl_max']:12.3e}  "
          f"{r['alpha_fit'] if r['alpha_fit'] else 'N/A':>10}")
print("=" * 72)
print("\n  Done.")
