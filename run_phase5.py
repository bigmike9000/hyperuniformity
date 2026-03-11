"""
Phase 5: Expanding the 1D Hyperuniform Catalog

Analyzes stealthy hyperuniform patterns and perturbed lattices to build
a comprehensive (alpha, Lambda_bar) ranking table across all 1D hyperuniform
systems studied in this project.

New patterns:
  - Stealthy hyperuniform (chi = 0.05 to 0.45): alpha -> inf, Class I
  - Perturbed lattice, uniform (URL, a=0.5, 1.0): alpha = 2, Class I
  - Perturbed lattice, Gaussian (sigma=0.3): alpha = 2, Class I
  - Perturbed lattice, Cauchy (gamma=0.1): alpha = 1, Class II

Figures produced:
  9.  Stealthy S(k) verification — shows exclusion zone
  10. Perturbed lattice variance comparison (URL, Gaussian, Cauchy)
  11. Comprehensive (alpha, Lambda_bar) ranking chart

All outputs saved to results/ directory.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from perturbed_lattices import (
    generate_perturbed_uniform, generate_perturbed_gaussian,
    generate_perturbed_cauchy, generate_perturbed_stable,
    lambda_bar_url_exact, structure_factor_analytical,
)
from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points,
    predict_chain_length, verify_eigenvalue_prediction,
)
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar
from two_phase_media import (
    compute_structure_factor as compute_sk_fft,
    compute_spectral_density, compute_excess_spreadability,
    extract_alpha_fit,
)

# ============================================================
# Configuration
# ============================================================
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
NUM_WINDOWS = 20_000
NUM_R_POINTS = 500
PHI2 = 0.35


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Figure 9: Stealthy S(k) verification (grad student data)
# ============================================================
STEALTHY_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'stealthy_data')


def read_cco(filepath):
    """Read a CCO configuration file. Returns (points_1d, L)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    d = int(lines[0].strip())
    L = float(lines[1].strip().split()[0])
    points = np.array([float(line.strip().split()[0]) for line in lines[2:]])
    return points, L


def run_stealthy_analysis(rng):
    """
    Analyze stealthy hyperuniform data from grad student configurations.
    N=2000 particles, density=1, chi=0.1, 0.2, 0.3.
    """
    print("\n" + "=" * 70)
    print("  Figure 9: Stealthy Hyperuniform Analysis")
    print("=" * 70)

    chi_values = [0.1, 0.2, 0.3]
    stealthy_results = {}
    N_CONFIGS = 500  # number of configurations to average over

    for chi in chi_values:
        print(f"\n  --- chi = {chi:.1f} ---")
        folder = os.path.join(STEALTHY_DATA_DIR, f'{chi}_patterns')

        # Read pre-computed S(k)
        sf_file = os.path.join(folder, 'sf_bin.txt')
        sf_data = np.loadtxt(sf_file, delimiter=',', skiprows=1)
        k_sf = sf_data[:, 0]
        S_sf = sf_data[:, 1]

        # Find exclusion zone boundary
        above_thresh = np.where(S_sf > 0.01)[0]
        K_boundary = k_sf[above_thresh[0]] if len(above_thresh) > 0 else k_sf[-1]
        max_S_excl = np.max(S_sf[k_sf < K_boundary * 0.9]) if K_boundary > 0 else 0
        print(f"  Exclusion boundary K ~ {K_boundary:.3f}")
        print(f"  max S(k) in exclusion zone: {max_S_excl:.2e}")

        # Get list of configuration files
        config_files = sorted([
            f for f in os.listdir(folder)
            if f.endswith('.txt') and not f.startswith('sf_')
        ])[:N_CONFIGS]

        # Read first file for parameters
        pts0, L0 = read_cco(os.path.join(folder, config_files[0]))
        N = len(pts0)
        rho = N / L0
        print(f"  N={N}, L={L0:.1f}, rho={rho:.4f}, "
              f"analyzing {len(config_files)} configurations...")

        # Compute Lambda_bar from ensemble
        t0 = time.perf_counter()
        R_max = min(300, L0 / 4)
        R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)
        lambda_bars = []

        for i, fname in enumerate(config_files):
            pts, L = read_cco(os.path.join(folder, fname))
            pts = np.sort(pts % L)
            var, _ = compute_number_variance_1d(
                pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
            lb = compute_lambda_bar(R_arr, var)
            lambda_bars.append(lb)

            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    [{i+1:4d}/{len(config_files)}] "
                      f"running mean={np.mean(lambda_bars):.6f} ({elapsed:.1f}s)")

        lb_arr = np.array(lambda_bars)
        lb_mean = np.mean(lb_arr)
        lb_std = np.std(lb_arr)
        lb_sem = lb_std / np.sqrt(len(lb_arr))
        elapsed = time.perf_counter() - t0

        print(f"  Lambda_bar = {lb_mean:.6f} +/- {lb_sem:.6f} "
              f"(std={lb_std:.6f}, {len(config_files)} configs, {elapsed:.1f}s)")

        stealthy_results[chi] = {
            'N': N, 'rho': rho, 'L': L0,
            'lambda_bar': lb_mean, 'lambda_bar_std': lb_std,
            'lambda_bar_sem': lb_sem,
            'n_configs': len(config_files),
            'k_sf': k_sf, 'S_sf': S_sf,
            'K_boundary': K_boundary,
        }

    # Plot: S(k) for chi=0.1 and chi=0.3
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, chi in enumerate([0.1, 0.3]):
        ax = axes[idx]
        res = stealthy_results[chi]
        k_plot = res['k_sf']
        S_plot = res['S_sf']
        K = res['K_boundary']

        ax.semilogy(k_plot, np.maximum(S_plot, 1e-20), 'b-', lw=0.8)
        ax.axvline(K, color='red', ls='--', lw=1.5,
                   label=rf'$K = {K:.2f}$ (exclusion boundary)')
        ax.axvspan(0, K, alpha=0.1, color='red')
        ax.set_xlabel(r'$k$', fontsize=13)
        ax.set_ylabel(r'$S(k)$', fontsize=13)
        ax.set_title(rf'Stealthy $\chi = {chi}$ (N={res["N"]})', fontsize=13)
        ax.set_ylim(1e-18, 10)
        ax.set_xlim(0, 8)
        ax.legend(fontsize=10)
        ax.text(0.25, 0.5, r'$S(k) = 0$', transform=ax.transAxes,
                fontsize=16, ha='center', va='center', color='red', alpha=0.4)

    fig.suptitle(
        'Figure 9: Stealthy Hyperuniform — Structure Factor Exclusion Zone\n'
        '(averaged over ~4000 configurations, N=2000)',
        fontsize=12, y=0.01, va='bottom', style='italic', color='0.3')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig9_stealthy_sk.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    return stealthy_results


# ============================================================
# Quasicrystal analysis (Copper & Nickel chains)
# ============================================================
TARGET_N_QC = 10_000_000  # target chain length for production runs


def run_quasicrystal_analysis(rng):
    """
    Generate Copper and Nickel metallic-mean quasicrystals at large N,
    compute Lambda_bar and alpha via spreadability.
    Returns dict of results keyed by chain name.
    """
    print("\n" + "=" * 70)
    print("  Quasicrystal Analysis: Copper & Nickel Chains")
    print("=" * 70)

    qc_results = {}
    chain_names = ['copper', 'nickel']

    for chain_name in chain_names:
        chain = CHAINS[chain_name]
        print(f"\n  --- {chain['name']} ---")

        # Verify eigenvalue prediction
        alpha_pred, lam1, lam2 = verify_eigenvalue_prediction(chain_name)
        print(f"  Eigenvalue prediction: alpha = {alpha_pred:.4f} "
              f"(lam1={lam1:.4f}, |lam2|={lam2:.4f})")

        # Find iteration count to reach TARGET_N_QC
        for iters in range(5, 60):
            n_pred = predict_chain_length(chain_name, iters)
            if n_pred >= TARGET_N_QC:
                break
        print(f"  Using {iters} iterations -> ~{n_pred:,} tiles")

        # Generate chain
        t0 = time.perf_counter()
        seq = generate_substitution_sequence(chain_name, iters)
        points, L_domain = sequence_to_points(seq, chain_name)
        N = len(points)
        rho = N / L_domain
        elapsed_gen = time.perf_counter() - t0
        print(f"  Generated: N={N:,}, L={L_domain:.1f}, rho={rho:.6f} ({elapsed_gen:.1f}s)")

        # Compute number variance and Lambda_bar
        t0 = time.perf_counter()
        R_max = min(500, L_domain / 4)
        R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)
        var, _ = compute_number_variance_1d(
            points, L_domain, R_arr, num_windows=NUM_WINDOWS, rng=rng)
        lb = compute_lambda_bar(R_arr, var)
        elapsed_var = time.perf_counter() - t0
        print(f"  Lambda_bar = {lb:.6f} ({elapsed_var:.1f}s)")

        # Compute alpha via spreadability
        # For large chains, generate a smaller chain for FFT (avoids enormous histogram)
        t0 = time.perf_counter()
        N_FFT_TARGET = 500_000
        if N > N_FFT_TARGET * 2:
            # Find iteration count for a smaller chain
            for iters_fft in range(5, 60):
                n_fft_pred = predict_chain_length(chain_name, iters_fft)
                if n_fft_pred >= N_FFT_TARGET:
                    break
            seq_fft = generate_substitution_sequence(chain_name, iters_fft)
            pts_fft, L_fft = sequence_to_points(seq_fft, chain_name)
            print(f"  (using {len(pts_fft):,}-tile chain for FFT-based alpha)")
            k_arr, S_k = compute_sk_fft(pts_fft, L_fft)
            rho_fft = len(pts_fft) / L_fft
        else:
            k_arr, S_k = compute_sk_fft(points, L_domain)
            rho_fft = rho
        a_rod = PHI2 / (2 * rho_fft)
        chi_V = compute_spectral_density(k_arr, S_k, rho_fft, a_rod)
        t_spread = np.logspace(-2, 8, 200)
        E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_spread)
        alpha_val, r2 = extract_alpha_fit(t_spread, E_t)
        elapsed_alpha = time.perf_counter() - t0
        print(f"  alpha (fit) = {alpha_val:.3f}, R^2 = {r2:.6f} ({elapsed_alpha:.1f}s)")

        qc_results[chain_name] = {
            'name': chain['name'].replace(' Ratio', ''),
            'N': N, 'rho': rho, 'L': L_domain,
            'lambda_bar': lb,
            'alpha': alpha_val, 'r2': r2,
            'alpha_predicted': alpha_pred,
        }

    return qc_results


# ============================================================
# Figure 10: Perturbed lattice variance comparison
# ============================================================
def run_perturbed_analysis(rng):
    print("\n" + "=" * 70)
    print("  Figure 10: Perturbed Lattice Analysis")
    print("=" * 70)

    N_pts = 100_000
    perturbed_results = {}

    # --- URL a=0.5 (Class I, alpha=2) ---
    print("\n  --- URL a=0.5 ---")
    t0 = time.perf_counter()
    pts, L = generate_perturbed_uniform(N_pts, a=0.5, rng=rng)
    rho = N_pts / L
    R_max = min(300, L / 4)
    R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)
    var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
    lb = compute_lambda_bar(R_arr, var)
    lb_exact = lambda_bar_url_exact(0.5)
    elapsed = time.perf_counter() - t0
    print(f"  N={N_pts}, Lambda_bar={lb:.6f} (exact={lb_exact:.6f}), time={elapsed:.1f}s")

    # Spreadability
    k_arr, S_k = compute_sk_fft(pts, L)
    a_rod = PHI2 / (2 * rho)
    chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
    t_spread = np.logspace(-2, 8, 200)
    E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_spread)
    alpha_val, r2 = extract_alpha_fit(t_spread, E_t)
    print(f"  alpha (fit) = {alpha_val:.3f}, R^2 = {r2:.6f}")

    perturbed_results['url_0.5'] = {
        'name': 'URL a=0.5', 'class': 'I', 'alpha_expected': 2,
        'N': N_pts, 'rho': rho, 'lambda_bar': lb, 'lambda_bar_exact': lb_exact,
        'alpha': alpha_val, 'r2': r2,
        'R_array': R_arr, 'variances': var,
    }

    # --- URL a=1.0 (Class I, alpha=2, cloaked) ---
    print("\n  --- URL a=1.0 (cloaked) ---")
    t0 = time.perf_counter()
    pts, L = generate_perturbed_uniform(N_pts, a=1.0, rng=rng)
    rho = N_pts / L
    R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)
    var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
    lb = compute_lambda_bar(R_arr, var)
    lb_exact = lambda_bar_url_exact(1.0)
    elapsed = time.perf_counter() - t0
    print(f"  N={N_pts}, Lambda_bar={lb:.6f} (exact={lb_exact:.6f}), time={elapsed:.1f}s")

    k_arr, S_k = compute_sk_fft(pts, L)
    a_rod = PHI2 / (2 * rho)
    chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
    E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_spread)
    alpha_val, r2 = extract_alpha_fit(t_spread, E_t)
    print(f"  alpha (fit) = {alpha_val:.3f}, R^2 = {r2:.6f}")

    perturbed_results['url_1.0'] = {
        'name': 'URL a=1.0 (cloaked)', 'class': 'I', 'alpha_expected': 2,
        'N': N_pts, 'rho': rho, 'lambda_bar': lb, 'lambda_bar_exact': lb_exact,
        'alpha': alpha_val, 'r2': r2,
        'R_array': R_arr, 'variances': var,
    }

    # --- URL sweep: a=0.1, 0.3, 0.8 (Class I, alpha=2) ---
    for a_val in [0.1, 0.3, 0.8]:
        label = f'URL a={a_val}'
        print(f"\n  --- {label} ---")
        t0 = time.perf_counter()
        pts, L = generate_perturbed_uniform(N_pts, a=a_val, rng=rng)
        rho = N_pts / L
        R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)
        var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
        lb = compute_lambda_bar(R_arr, var)
        lb_exact = lambda_bar_url_exact(a_val)
        elapsed = time.perf_counter() - t0
        print(f"  N={N_pts}, Lambda_bar={lb:.6f} (exact={lb_exact:.6f}), time={elapsed:.1f}s")

        k_arr, S_k = compute_sk_fft(pts, L)
        a_rod = PHI2 / (2 * rho)
        chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
        E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_spread)
        alpha_val, r2 = extract_alpha_fit(t_spread, E_t)
        print(f"  alpha (fit) = {alpha_val:.3f}, R^2 = {r2:.6f}")

        perturbed_results[f'url_{a_val}'] = {
            'name': label, 'class': 'I', 'alpha_expected': 2,
            'N': N_pts, 'rho': rho, 'lambda_bar': lb, 'lambda_bar_exact': lb_exact,
            'alpha': alpha_val, 'r2': r2,
            'R_array': R_arr, 'variances': var,
        }

    # --- Gaussian sigma=0.3 (Class I, alpha=2) ---
    print("\n  --- Gaussian sigma=0.3 ---")
    t0 = time.perf_counter()
    pts, L = generate_perturbed_gaussian(N_pts, sigma=0.3, rng=rng)
    rho = N_pts / L
    R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)
    var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
    lb = compute_lambda_bar(R_arr, var)
    elapsed = time.perf_counter() - t0
    print(f"  N={N_pts}, Lambda_bar={lb:.6f}, time={elapsed:.1f}s")

    k_arr, S_k = compute_sk_fft(pts, L)
    a_rod = PHI2 / (2 * rho)
    chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
    E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_spread)
    alpha_val, r2 = extract_alpha_fit(t_spread, E_t)
    print(f"  alpha (fit) = {alpha_val:.3f}, R^2 = {r2:.6f}")

    perturbed_results['gaussian_0.3'] = {
        'name': 'Gaussian sigma=0.3', 'class': 'I', 'alpha_expected': 2,
        'N': N_pts, 'rho': rho, 'lambda_bar': lb, 'lambda_bar_exact': None,
        'alpha': alpha_val, 'r2': r2,
        'R_array': R_arr, 'variances': var,
    }

    # --- Gaussian sweep: sigma=0.1, 0.2, 0.5 (Class I, alpha=2) ---
    for sigma_val in [0.1, 0.2, 0.5]:
        label = f'Gaussian sigma={sigma_val}'
        print(f"\n  --- {label} ---")
        t0 = time.perf_counter()
        pts, L = generate_perturbed_gaussian(N_pts, sigma=sigma_val, rng=rng)
        rho = N_pts / L
        R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)
        var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
        lb = compute_lambda_bar(R_arr, var)
        elapsed = time.perf_counter() - t0
        print(f"  N={N_pts}, Lambda_bar={lb:.6f}, time={elapsed:.1f}s")

        k_arr, S_k = compute_sk_fft(pts, L)
        a_rod = PHI2 / (2 * rho)
        chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
        E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_spread)
        alpha_val, r2 = extract_alpha_fit(t_spread, E_t)
        print(f"  alpha (fit) = {alpha_val:.3f}, R^2 = {r2:.6f}")

        perturbed_results[f'gaussian_{sigma_val}'] = {
            'name': label, 'class': 'I', 'alpha_expected': 2,
            'N': N_pts, 'rho': rho, 'lambda_bar': lb, 'lambda_bar_exact': None,
            'alpha': alpha_val, 'r2': r2,
            'R_array': R_arr, 'variances': var,
        }

    # --- Cauchy gamma=0.1 (Class II, alpha=1) ---
    print("\n  --- Cauchy gamma=0.1 ---")
    t0 = time.perf_counter()
    pts, L = generate_perturbed_cauchy(N_pts, gamma=0.1, rng=rng)
    rho = N_pts / L
    R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)
    var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
    lb = compute_lambda_bar(R_arr, var)
    elapsed = time.perf_counter() - t0
    print(f"  N={N_pts}, Lambda_bar={lb:.6f} (grows with R for Class II), time={elapsed:.1f}s")

    k_arr, S_k = compute_sk_fft(pts, L)
    a_rod = PHI2 / (2 * rho)
    chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
    E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_spread)
    alpha_val, r2 = extract_alpha_fit(t_spread, E_t)
    print(f"  alpha (fit) = {alpha_val:.3f}, R^2 = {r2:.6f}")

    perturbed_results['cauchy_0.1'] = {
        'name': 'Cauchy gamma=0.1', 'class': 'II', 'alpha_expected': 1,
        'N': N_pts, 'rho': rho, 'lambda_bar': lb, 'lambda_bar_exact': None,
        'alpha': alpha_val, 'r2': r2,
        'R_array': R_arr, 'variances': var,
    }

    # --- Stable distribution perturbations (Class III, alpha = s) ---
    for s_idx, s_val in [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]:
        label = f'Stable s={s_val}'
        print(f"\n  --- {label} (Class III, alpha={s_val}) ---")
        t0 = time.perf_counter()
        pts, L = generate_perturbed_stable(N_pts, stability=s_val, scale=0.1, rng=rng)
        rho = N_pts / L
        R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)
        var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
        lb = compute_lambda_bar(R_arr, var)
        elapsed = time.perf_counter() - t0
        print(f"  N={N_pts}, Lambda_bar={lb:.6f} (grows with R for Class III), time={elapsed:.1f}s")

        k_arr, S_k = compute_sk_fft(pts, L)
        a_rod = PHI2 / (2 * rho)
        chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
        E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_spread)
        alpha_val, r2 = extract_alpha_fit(t_spread, E_t)
        print(f"  alpha (fit) = {alpha_val:.3f}, R^2 = {r2:.6f}")

        perturbed_results[f'stable_{s_val}'] = {
            'name': label, 'class': 'III', 'alpha_expected': s_val,
            'N': N_pts, 'rho': rho, 'lambda_bar': lb, 'lambda_bar_exact': None,
            'alpha': alpha_val, 'r2': r2,
            'R_array': R_arr, 'variances': var,
        }

    # --- Stable distribution perturbations with s>1 (Class I, 1 < alpha < 2) ---
    for s_val in [1.3, 1.5, 1.7]:
        label = f'Stable s={s_val}'
        print(f"\n  --- {label} (Class I, alpha={s_val}) ---")
        t0 = time.perf_counter()
        pts, L = generate_perturbed_stable(N_pts, stability=s_val, scale=0.1, rng=rng)
        rho = N_pts / L
        R_arr = np.linspace(1.0, R_max, NUM_R_POINTS)
        var, _ = compute_number_variance_1d(pts, L, R_arr, num_windows=NUM_WINDOWS, rng=rng)
        lb = compute_lambda_bar(R_arr, var)
        elapsed = time.perf_counter() - t0
        print(f"  N={N_pts}, Lambda_bar={lb:.6f}, time={elapsed:.1f}s")

        k_arr, S_k = compute_sk_fft(pts, L)
        a_rod = PHI2 / (2 * rho)
        chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
        E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_spread)
        alpha_val, r2 = extract_alpha_fit(t_spread, E_t)
        print(f"  alpha (fit) = {alpha_val:.3f}, R^2 = {r2:.6f}")

        perturbed_results[f'stable_{s_val}'] = {
            'name': label, 'class': 'I', 'alpha_expected': s_val,
            'N': N_pts, 'rho': rho, 'lambda_bar': lb, 'lambda_bar_exact': None,
            'alpha': alpha_val, 'r2': r2,
            'R_array': R_arr, 'variances': var,
        }

    # Plot: variance comparison (Class I and II)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    configs = [
        ('url_0.5', '#1f77b4'),
        ('url_1.0', '#ff7f0e'),
        ('gaussian_0.3', '#2ca02c'),
        ('cauchy_0.1', '#d62728'),
    ]
    for idx, (key, color) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        r = perturbed_results[key]
        ax.plot(r['R_array'], r['variances'], '-', color=color, lw=0.5, alpha=0.8)
        if r['class'] == 'I':
            ax.axhline(r['lambda_bar'], color='red', ls='--', lw=1.5,
                       label=rf"$\bar{{\Lambda}} = {r['lambda_bar']:.4f}$")
        ax.set_xlabel(r'Window half-width $R$', fontsize=11)
        ax.set_ylabel(r'$\sigma^2(R)$', fontsize=11)
        title = f"{r['name']} (Class {r['class']}, α={r['alpha_expected']})"
        ax.set_title(title, fontsize=12)
        if r['class'] == 'I':
            ax.legend(fontsize=10)
        ax.grid(True, ls=':', alpha=0.5)

    fig.suptitle(r'Figure 10: Perturbed Lattice Number Variance — $\sigma^2(R)$',
                 fontsize=13, y=0.01, va='bottom', style='italic', color='0.3')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig10_perturbed_variance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # Figure 12: Class III variance growth (stable s<1)
    stable_keys_iii = [k for k in perturbed_results
                       if k.startswith('stable_') and perturbed_results[k]['class'] == 'III']
    if stable_keys_iii:
        fig, axes = plt.subplots(1, len(stable_keys_iii),
                                 figsize=(6 * len(stable_keys_iii), 5))
        if len(stable_keys_iii) == 1:
            axes = [axes]
        colors_iii = ['#e377c2', '#7f7f7f', '#bcbd22']
        for idx, key in enumerate(stable_keys_iii):
            ax = axes[idx]
            r = perturbed_results[key]
            ax.plot(r['R_array'], r['variances'], '-', color=colors_iii[idx],
                    lw=0.5, alpha=0.8)
            s_exp = r['alpha_expected']
            R_fit = r['R_array']
            coeff = r['variances'][-1] / R_fit[-1] ** (1 - s_exp) if s_exp < 1 else 1
            ax.plot(R_fit, coeff * R_fit ** (1 - s_exp), 'k--', lw=1.5, alpha=0.6,
                    label=rf'$\sim R^{{{1 - s_exp:.1f}}}$')
            ax.set_xlabel(r'Window half-width $R$', fontsize=11)
            ax.set_ylabel(r'$\sigma^2(R)$', fontsize=11)
            ax.set_title(
                rf"{r['name']} (Class III, $\alpha$={s_exp})", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, ls=':', alpha=0.5)

        fig.suptitle(
            r'Figure 12: Stable Distribution Perturbations — Class III $\sigma^2(R)$',
            fontsize=13, y=0.01, va='bottom', style='italic', color='0.3')
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, 'fig12_stable_variance.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {path}")

    # Figure 13: Class I stable (s>1) variance — bounded but with 1<alpha<2
    stable_keys_i = [k for k in perturbed_results
                     if k.startswith('stable_') and perturbed_results[k]['class'] == 'I']
    if stable_keys_i:
        fig, axes = plt.subplots(1, len(stable_keys_i),
                                 figsize=(6 * len(stable_keys_i), 5))
        if len(stable_keys_i) == 1:
            axes = [axes]
        colors_i = ['#17becf', '#8c564b', '#9467bd']
        for idx, key in enumerate(stable_keys_i):
            ax = axes[idx]
            r = perturbed_results[key]
            ax.plot(r['R_array'], r['variances'], '-', color=colors_i[idx],
                    lw=0.5, alpha=0.8)
            ax.axhline(r['lambda_bar'], color='red', ls='--', lw=1.5,
                       label=rf"$\bar{{\Lambda}} = {r['lambda_bar']:.4f}$")
            ax.set_xlabel(r'Window half-width $R$', fontsize=11)
            ax.set_ylabel(r'$\sigma^2(R)$', fontsize=11)
            s_exp = r['alpha_expected']
            ax.set_title(
                rf"{r['name']} (Class I, $\alpha$={s_exp})", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, ls=':', alpha=0.5)

        fig.suptitle(
            r'Figure 13: Stable Perturbations ($s > 1$) — Class I $\sigma^2(R)$',
            fontsize=13, y=0.01, va='bottom', style='italic', color='0.3')
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, 'fig13_stable_classI_variance.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {path}")

    # Figure 14: URL Lambda_bar(a) exact curve with numerical points
    print("\n  --- Figure 14: URL Lambda_bar(a) curve ---")
    a_curve = np.linspace(0.001, 2.0, 500)
    lb_curve = np.array([lambda_bar_url_exact(a) for a in a_curve])

    # Collect numerical URL points
    url_keys = sorted([k for k in perturbed_results if k.startswith('url_')])
    url_a_vals = []
    url_lb_num = []
    url_lb_exact = []
    for key in url_keys:
        r = perturbed_results[key]
        a_str = key.replace('url_', '')
        a = float(a_str)
        url_a_vals.append(a)
        url_lb_num.append(r['lambda_bar'])
        url_lb_exact.append(lambda_bar_url_exact(a))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(a_curve, lb_curve, 'b-', lw=2, label=r'Exact $\bar{\Lambda}(a)$ (Klatt et al. 2020)')
    ax.scatter(url_a_vals, url_lb_num, c='red', s=80, zorder=5, edgecolors='black',
               label=f'Numerical (N={N_pts:,})')
    # Mark special points
    ax.axhline(1/6, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(0.05, 1/6 + 0.005, r'$\bar{\Lambda} = 1/6$ (lattice)', fontsize=9, color='gray')
    ax.axhline(1/3, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(0.05, 1/3 + 0.005, r'$\bar{\Lambda} = 1/3$ (cloaked, $a=1$)', fontsize=9, color='gray')

    # Annotate numerical points with errors
    for a, lb_n, lb_e in zip(url_a_vals, url_lb_num, url_lb_exact):
        rel_err = abs(lb_n - lb_e) / lb_e * 100
        ax.annotate(f'{rel_err:.1f}%', (a, lb_n), textcoords='offset points',
                    xytext=(8, 8), fontsize=8, color='red')

    ax.set_xlabel(r'Displacement width $a$', fontsize=13)
    ax.set_ylabel(r'$\bar{\Lambda}(a)$', fontsize=13)
    ax.set_title(r'URL Model: $\bar{\Lambda}$ vs Displacement Width', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, ls=':', alpha=0.5)
    ax.set_xlim(0, 2.05)

    fig.suptitle(
        r'Figure 14: Exact $\bar{\Lambda}(a)$ for the Uniformly Randomized Lattice',
        fontsize=12, y=0.01, va='bottom', style='italic', color='0.3')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig14_url_lambda_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return perturbed_results


# ============================================================
# Figure 11: Comprehensive ranking chart
# ============================================================
def run_ranking_chart(stealthy_results, perturbed_results, qc_results=None):
    print("\n" + "=" * 70)
    print("  Figure 11: Comprehensive (alpha, Lambda_bar) Ranking")
    print("=" * 70)

    # Collect all results into ranking table
    ranking = []

    # Integer lattice
    ranking.append({
        'name': 'Integer Lattice', 'alpha': np.inf,
        'lambda_bar': 1/6, 'class': 'I', 'category': 'crystal',
    })

    # Quasicrystals (from Phase 4)
    qc_data = [
        ('Fibonacci', 3.049, 0.200),
        ('Silver', 2.992, 0.250),
        ('Bronze', 2.987, 0.282),
    ]
    for name, alpha, lb in qc_data:
        ranking.append({
            'name': name, 'alpha': alpha,
            'lambda_bar': lb, 'class': 'I', 'category': 'quasicrystal',
        })

    # Copper & Nickel (from run_quasicrystal_analysis)
    if qc_results:
        for chain_name in ['copper', 'nickel']:
            if chain_name in qc_results:
                r = qc_results[chain_name]
                ranking.append({
                    'name': r['name'], 'alpha': r['alpha'],
                    'lambda_bar': r['lambda_bar'], 'class': 'I',
                    'category': 'quasicrystal',
                })

    # Stealthy — alpha is effectively infinite (exponential decay, not power-law)
    for chi, res in sorted(stealthy_results.items()):
        ranking.append({
            'name': f'Stealthy chi={chi:.2f}', 'alpha': np.inf,
            'lambda_bar': res['lambda_bar'], 'class': 'I',
            'category': 'stealthy',
        })

    # Perturbed lattices
    for key, res in perturbed_results.items():
        ranking.append({
            'name': res['name'], 'alpha': res['alpha'],
            'lambda_bar': res['lambda_bar'], 'class': res['class'],
            'category': 'perturbed',
        })

    # Sort by Lambda_bar
    ranking.sort(key=lambda x: x['lambda_bar'])

    # Print table
    print(f"\n  {'#':>3s}  {'Pattern':30s}  {'Class':>6s}  {'alpha':>8s}  "
          f"{'Lambda_bar':>12s}")
    print(f"  {'-'*3}  {'-'*30}  {'-'*6}  {'-'*8}  {'-'*12}")
    for i, r in enumerate(ranking):
        alpha_str = 'inf' if np.isinf(r['alpha']) else f"{r['alpha']:.3f}"
        if np.isnan(r['lambda_bar']):
            lb_str = 'N/A'
        else:
            lb_str = f"{r['lambda_bar']:.4f}"
        print(f"  {i+1:3d}  {r['name']:30s}  {r['class']:>6s}  "
              f"{alpha_str:>8s}  {lb_str:>12s}")

    # Plot: Lambda_bar bar chart (log scale to show all classes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8),
                                    gridspec_kw={'width_ratios': [2, 1]})

    names = [r['name'] for r in ranking if not np.isnan(r['lambda_bar'])]
    lbs = [r['lambda_bar'] for r in ranking if not np.isnan(r['lambda_bar'])]
    cats = [r['category'] for r in ranking if not np.isnan(r['lambda_bar'])]
    classes = [r['class'] for r in ranking if not np.isnan(r['lambda_bar'])]
    alphas_plot = [r['alpha'] for r in ranking if not np.isnan(r['lambda_bar'])]

    cat_colors = {
        'crystal': '#1f77b4',
        'quasicrystal': '#2ca02c',
        'stealthy': '#9467bd',
        'perturbed': '#d62728',
    }
    colors = [cat_colors[c] for c in cats]

    # Left panel: full ranking with log scale
    bars = ax1.barh(range(len(names)), lbs, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel(r'$\bar{\Lambda}$ (surface-area coefficient, log scale)',
                   fontsize=13)
    ax1.set_title('All Hyperuniformity Classes', fontsize=13)
    ax1.invert_yaxis()

    for i, (lb_val, alpha_val, cls) in enumerate(zip(lbs, alphas_plot, classes)):
        alpha_str = r'$\alpha\to\infty$' if np.isinf(alpha_val) \
            else rf'$\alpha$={alpha_val:.1f}'
        ax1.text(lb_val * 1.15, i, f'{alpha_str}  [Class {cls}]',
                 va='center', fontsize=8, color='0.3')

    ax1.axvline(1/6, color='gray', ls=':', lw=1, alpha=0.5)

    # Right panel: Class I only (linear scale, zoomed in)
    class1 = [(n, lb, c, a) for n, lb, c, a, cls
              in zip(names, lbs, colors, alphas_plot, classes) if cls == 'I']
    if class1:
        c1_names, c1_lbs, c1_colors, c1_alphas = zip(*class1)
        ax2.barh(range(len(c1_names)), c1_lbs, color=c1_colors, alpha=0.8,
                 edgecolor='black', linewidth=0.5)
        ax2.set_yticks(range(len(c1_names)))
        ax2.set_yticklabels(c1_names, fontsize=9)
        ax2.set_xlabel(r'$\bar{\Lambda}$', fontsize=13)
        ax2.set_title('Class I Only (linear scale)', fontsize=13)
        ax2.invert_yaxis()

        for i, (lb_val, alpha_val) in enumerate(zip(c1_lbs, c1_alphas)):
            alpha_str = r'$\alpha\to\infty$' if np.isinf(alpha_val) \
                else rf'$\alpha$={alpha_val:.1f}'
            ax2.text(lb_val + 0.01, i, alpha_str, va='center', fontsize=8,
                     color='0.3')

        ax2.axvline(1/6, color='gray', ls=':', lw=1, alpha=0.5)
        ax2.text(1/6 + 0.005, len(c1_names) - 0.5,
                 r'Lattice $\bar{\Lambda}=1/6$', fontsize=8, color='gray')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cat_colors['crystal'], label='Crystal'),
        Patch(facecolor=cat_colors['quasicrystal'], label='Quasicrystal'),
        Patch(facecolor=cat_colors['stealthy'], label='Stealthy'),
        Patch(facecolor=cat_colors['perturbed'], label='Perturbed Lattice'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

    fig.suptitle(r'Figure 11: $(\alpha, \bar{\Lambda})$ ranking — '
                 'lower = more ordered',
                 fontsize=14, y=1.01, va='bottom', style='italic', color='0.3')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig11_ranking.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    return ranking


# ============================================================
# Literature validation
# ============================================================
def print_literature_comparison(qc_results, perturbed_results, stealthy_results):
    """
    Print a comprehensive comparison table of our results vs published values.
    """
    print("\n" + "=" * 70)
    print("  Literature Validation")
    print("=" * 70)

    print(f"\n  {'Pattern':30s}  {'Our value':>12s}  {'Literature':>12s}  "
          f"{'Source':30s}  {'Agreement':>10s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*30}  {'-'*10}")

    # --- Lambda_bar comparisons ---
    print(f"\n  Lambda_bar:")

    # Integer lattice
    lb_lattice = 1/6
    print(f"  {'Integer Lattice':30s}  {lb_lattice:12.4f}  {'0.1667':>12s}  "
          f"{'Torquato & Stillinger 2003':30s}  {'exact':>10s}")

    # Fibonacci (from Phase 4 hardcoded)
    print(f"  {'Fibonacci':30s}  {'0.200':>12s}  {'0.201':>12s}  "
          f"{'Zachary & Torquato 2009':30s}  {'0.5%':>10s}")

    # Silver, Bronze (novel)
    print(f"  {'Silver':30s}  {'0.250':>12s}  {'---':>12s}  "
          f"{'NOVEL':30s}  {'---':>10s}")
    print(f"  {'Bronze':30s}  {'0.282':>12s}  {'---':>12s}  "
          f"{'NOVEL':30s}  {'---':>10s}")

    # Copper, Nickel from qc_results
    for chain_name in ['copper', 'nickel']:
        if chain_name in qc_results:
            r = qc_results[chain_name]
            print(f"  {r['name']:30s}  {r['lambda_bar']:12.4f}  {'---':>12s}  "
                  f"{'NOVEL':30s}  {'---':>10s}")

    # URL exact comparisons
    url_keys = sorted([k for k in perturbed_results if k.startswith('url_')])
    for key in url_keys:
        r = perturbed_results[key]
        if r.get('lambda_bar_exact') is not None:
            rel_err = abs(r['lambda_bar'] - r['lambda_bar_exact']) / r['lambda_bar_exact'] * 100
            print(f"  {r['name']:30s}  {r['lambda_bar']:12.4f}  "
                  f"{r['lambda_bar_exact']:12.4f}  "
                  f"{'Klatt et al. 2020, Eq. B7':30s}  {rel_err:9.1f}%")

    # Stealthy
    for chi in sorted(stealthy_results.keys()):
        res = stealthy_results[chi]
        name = f'Stealthy chi={chi:.1f}'
        print(f"  {name:30s}  {res['lambda_bar']:12.4f}  {'---':>12s}  "
              f"{'no published value':30s}  {'---':>10s}")

    # --- Alpha comparisons ---
    print(f"\n  Alpha (hyperuniformity exponent):")

    # Metallic means: alpha should be 3
    qc_names_alpha = [
        ('Fibonacci', 3.049),
        ('Silver', 2.992),
        ('Bronze', 2.987),
    ]
    for name, alpha in qc_names_alpha:
        err = abs(alpha - 3.0) / 3.0 * 100
        print(f"  {name:30s}  {alpha:12.3f}  {'3.000':>12s}  "
              f"{'Oguz et al. 2019 eigenvalue':30s}  {err:9.1f}%")

    for chain_name in ['copper', 'nickel']:
        if chain_name in qc_results:
            r = qc_results[chain_name]
            err = abs(r['alpha'] - 3.0) / 3.0 * 100
            print(f"  {r['name']:30s}  {r['alpha']:12.3f}  {'3.000':>12s}  "
                  f"{'Oguz et al. 2019 eigenvalue':30s}  {err:9.1f}%")

    # Perturbed lattice alpha
    for key in sorted(perturbed_results.keys()):
        r = perturbed_results[key]
        if r['alpha_expected'] is not None:
            err = abs(r['alpha'] - r['alpha_expected']) / max(r['alpha_expected'], 0.01) * 100
            print(f"  {r['name']:30s}  {r['alpha']:12.3f}  "
                  f"{r['alpha_expected']:12.3f}  "
                  f"{'Theory':30s}  {err:9.1f}%")

    # Note on 2<alpha<3 gap
    print(f"\n  NOTE: The range 2 < alpha < 3 is an open gap in the 1D landscape.")
    print(f"  All metallic-mean chains give alpha=3 exactly (det(M)=-1).")
    print(f"  All finite-variance perturbations give alpha=2 exactly.")
    print(f"  No known 1D construction achieves 2 < alpha < 3.")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    total_t0 = time.perf_counter()
    rng = np.random.default_rng(seed=SEED)

    print("=" * 70)
    print("  Phase 5: Expanding the 1D Hyperuniform Catalog")
    print("=" * 70)

    ensure_results_dir()

    stealthy_results = run_stealthy_analysis(rng)
    qc_results = run_quasicrystal_analysis(rng)
    perturbed_results = run_perturbed_analysis(rng)
    ranking = run_ranking_chart(stealthy_results, perturbed_results, qc_results)
    print_literature_comparison(qc_results, perturbed_results, stealthy_results)

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n  Total Phase 5 runtime: {total_elapsed:.1f}s")
    print(f"  All figures saved to: {RESULTS_DIR}")
    print("\nDone.")
