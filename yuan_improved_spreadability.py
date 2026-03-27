"""
Yuan-Torquato (2026) improved spreadability fitting — Week 4, Task 7.

Implements the three fitting function types from arXiv:2602.17873:
  Type I:  ln[s^ex(t)] = -(d+alpha)/2 * ln(t) + ln(sum C_{i/2} t^{-i/2})
  Type II: adds ln(t) correction terms for integer alpha
  Type III: ln[s^ex(t)] = -(d+alpha)/2 * ln(t) + ln(sum C_i t^{-i})
                          (assumes analytic spectral density)

Tests on Bombieri-Taylor chain first, then Fibonacci.

NOTE: Yuan & Torquato (2026) Section VI explicitly states that quasicrystalline
media with dense Bragg peaks are NOT covered by their regular expansion theory.
For quasicrystals, s^ex(t) oscillates between C_+ t^{-gamma} and C_- t^{-gamma}.
We test Type I as a general fitting improvement regardless.

Output: results/yuan_spreadability_results.json
"""

import os
import sys
import json
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from substitution_tilings import (generate_substitution_sequence,
                                  sequence_to_points, sequence_to_points_general,
                                  compute_tile_lengths)
from two_phase_media import (compute_structure_factor, compute_spectral_density,
                              compute_excess_spreadability, extract_alpha)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIG_PATH = os.path.join(OUTPUT_DIR, 'fig_yuan_spreadability.png')
JSON_PATH = os.path.join(SCRIPT_DIR, 'results', 'yuan_spreadability_results.json')

PHI2 = 0.35
D_COEFF = 1.0
d_DIM = 1  # spatial dimension


# ── Fitting functions ──────────────────────────────────────────────────

def type_I_fit(t, alpha_hat, C_coeffs):
    """
    Type I fitting: ln[s^ex] = -(d+alpha)/2 * ln(t) + ln(C0 + C_{1/2}*t^{-1/2} + C1*t^{-1} + ...)

    C_coeffs = [C0, C_{1/2}, C1, C_{3/2}, ...] for fitting order n.
    """
    gamma = (d_DIM + alpha_hat) / 2.0
    correction = np.zeros_like(t)
    for i, c in enumerate(C_coeffs):
        correction += c * t ** (-i / 2.0)
    # Avoid log of negative numbers
    correction = np.maximum(correction, 1e-30)
    return -gamma * np.log(t) + np.log(correction)


def type_III_fit(t, alpha_hat, C_coeffs):
    """
    Type III fitting (analytic spectral density):
    ln[s^ex] = -(d+alpha)/2 * ln(t) + ln(C0 + C1*t^{-1} + C2*t^{-2} + ...)

    C_coeffs = [C0, C1, C2, ...] for fitting order n.
    """
    gamma = (d_DIM + alpha_hat) / 2.0
    correction = np.zeros_like(t)
    for i, c in enumerate(C_coeffs):
        correction += c * t ** (-i)
    correction = np.maximum(correction, 1e-30)
    return -gamma * np.log(t) + np.log(correction)


def fit_type_I(t, ln_sex, n_order):
    """
    Fit Type I: ln[s^ex] = -(d+alpha_hat)/2 * ln(t) + ln(sum_{i=0}^{n} C_{i/2} t^{-i/2})

    Parameters: alpha_hat, C0, C_{1/2}, ..., C_{n/2}  (n+2 total parameters)
    """
    def residuals(params):
        alpha_hat = params[0]
        C_coeffs = params[1:]
        model = type_I_fit(t, alpha_hat, C_coeffs)
        return model - ln_sex

    # Initial guess
    # Estimate alpha from simple slope
    mid = len(t) // 2
    slope = (ln_sex[-1] - ln_sex[mid]) / (np.log(t[-1]) - np.log(t[mid]))
    alpha_init = -2 * slope - d_DIM
    alpha_init = np.clip(alpha_init, -2, 10)

    x0 = np.zeros(n_order + 2)
    x0[0] = alpha_init
    x0[1] = 1.0  # C0

    result = least_squares(residuals, x0, method='lm', max_nfev=10000)

    alpha_hat = result.x[0]
    C_coeffs = result.x[1:]
    residual_norm = np.sqrt(np.mean(result.fun**2))

    return alpha_hat, C_coeffs, residual_norm


def fit_type_III(t, ln_sex, n_order, alpha_fixed=None):
    """
    Fit Type III: ln[s^ex] = -(d+alpha)/2 * ln(t) + ln(sum_{i=0}^{n} C_i t^{-i})
    """
    def residuals(params):
        if alpha_fixed is not None:
            alpha_hat = alpha_fixed
            C_coeffs = params
        else:
            alpha_hat = params[0]
            C_coeffs = params[1:]
        model = type_III_fit(t, alpha_hat, C_coeffs)
        return model - ln_sex

    mid = len(t) // 2
    slope = (ln_sex[-1] - ln_sex[mid]) / (np.log(t[-1]) - np.log(t[mid]))
    alpha_init = -2 * slope - d_DIM
    alpha_init = np.clip(alpha_init, -2, 10)

    if alpha_fixed is not None:
        x0 = np.zeros(n_order + 1)
        x0[0] = 1.0
    else:
        x0 = np.zeros(n_order + 2)
        x0[0] = alpha_init
        x0[1] = 1.0

    result = least_squares(residuals, x0, method='lm', max_nfev=10000)

    if alpha_fixed is not None:
        alpha_hat = alpha_fixed
        C_coeffs = result.x
    else:
        alpha_hat = result.x[0]
        C_coeffs = result.x[1:]

    residual_norm = np.sqrt(np.mean(result.fun**2))
    return alpha_hat, C_coeffs, residual_norm


# ── Chain computation ──────────────────────────────────────────────────

def compute_spreadability_for_chain(chain_name, N_target=500_000):
    """Compute s^ex(t) for a substitution chain."""
    print(f"\n  Generating {chain_name} chain...", end='', flush=True)

    # Find how many iterations needed
    _, lam1 = compute_tile_lengths(chain_name)
    n_iter = int(np.ceil(np.log(N_target) / np.log(lam1))) + 2
    seq = generate_substitution_sequence(chain_name, n_iter)
    try:
        points, L = sequence_to_points(seq, chain_name)
    except KeyError:
        points, L = sequence_to_points_general(seq, chain_name)
    N = len(points)
    rho = N / L
    print(f" N={N:,}, rho={rho:.4f}")

    a_rod = PHI2 / (2 * rho)

    print(f"  Computing S(k)...", end='', flush=True)
    k_arr, S_k = compute_structure_factor(points, L)
    chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)
    print(" done")

    # Time array: log-spaced from 1 to 10^6
    t_arr = np.logspace(0, 6, 500)

    print(f"  Computing E(t)...", end='', flush=True)
    E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_arr, D=D_COEFF)
    print(" done")

    # Normalize to s^ex(t)
    E_t_max = E_t[0]
    s_ex = E_t / E_t_max  # normalized

    # Actually, s^ex = E(t) directly (already normalized by definition)
    # Use E(t) as the excess spreadability

    return t_arr, E_t, rho


def run_fitting(chain_name, alpha_true, t_arr, E_t):
    """Run Type I and III fits at various orders."""
    # Filter to positive E(t) and large enough t
    mask = (E_t > 0) & (t_arr > 10)
    t = t_arr[mask]
    ln_sex = np.log(E_t[mask])

    print(f"\n  === Fitting {chain_name} (alpha_true={alpha_true}) ===")
    print(f"  Using {len(t)} points with t in [{t[0]:.1f}, {t[-1]:.0f}]")

    results = {'chain': chain_name, 'alpha_true': alpha_true,
               'type_I': [], 'type_III': []}

    # Type I fits at orders 0 through 6
    print(f"\n  Type I fits:")
    print(f"  {'order':>5s}  {'alpha_hat':>10s}  {'|delta_alpha|':>14s}  {'residual':>10s}")
    for n in range(7):
        try:
            alpha_hat, C_coeffs, res = fit_type_I(t, ln_sex, n)
            delta = abs(alpha_hat - alpha_true)
            print(f"  {n:5d}  {alpha_hat:10.4f}  {delta:14.6f}  {res:10.6f}")
            results['type_I'].append({
                'order': n, 'alpha_hat': float(alpha_hat),
                'delta_alpha': float(delta), 'residual': float(res),
                'C_coeffs': [float(c) for c in C_coeffs],
            })
        except Exception as e:
            print(f"  {n:5d}  FAILED: {e}")

    # Type III fits at orders 0 through 6
    print(f"\n  Type III fits:")
    print(f"  {'order':>5s}  {'alpha_hat':>10s}  {'|delta_alpha|':>14s}  {'residual':>10s}")
    for n in range(7):
        try:
            alpha_hat, C_coeffs, res = fit_type_III(t, ln_sex, n)
            delta = abs(alpha_hat - alpha_true)
            print(f"  {n:5d}  {alpha_hat:10.4f}  {delta:14.6f}  {res:10.6f}")
            results['type_III'].append({
                'order': n, 'alpha_hat': float(alpha_hat),
                'delta_alpha': float(delta), 'residual': float(res),
                'C_coeffs': [float(c) for c in C_coeffs],
            })
        except Exception as e:
            print(f"  {n:5d}  FAILED: {e}")

    # Also run old method (simple log-slope)
    alpha_old = extract_alpha(t_arr, E_t, window=20)
    # Use the last stable value
    mask_stable = (t_arr > 100) & (t_arr < t_arr[-1] / 2)
    if np.any(mask_stable):
        alpha_simple = float(np.median(alpha_old[mask_stable]))
    else:
        alpha_simple = float(np.median(alpha_old[len(alpha_old)//2:]))
    results['alpha_simple_slope'] = alpha_simple
    print(f"\n  Simple log-slope method: alpha = {alpha_simple:.4f}")

    return results


def make_figure(all_results):
    """Plot delta_alpha vs fitting order for each chain and method."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5),
                              squeeze=False)

    for idx, res in enumerate(all_results):
        ax = axes[0, idx]
        chain = res['chain']
        alpha_true = res['alpha_true']

        # Type I
        orders_I = [r['order'] for r in res['type_I']]
        deltas_I = [r['delta_alpha'] for r in res['type_I']]
        ax.semilogy(orders_I, deltas_I, 'bo-', ms=6, label='Type I')

        # Type III
        orders_III = [r['order'] for r in res['type_III']]
        deltas_III = [r['delta_alpha'] for r in res['type_III']]
        ax.semilogy(orders_III, deltas_III, 'rs-', ms=6, label='Type III')

        # Simple slope
        ax.axhline(y=abs(res['alpha_simple_slope'] - alpha_true),
                    color='green', ls='--', lw=1.5, label='Simple slope')

        ax.set_xlabel('Fitting order $n$')
        ax.set_ylabel(r'$|\hat\alpha - \alpha|$')
        ax.set_title(f'{chain}\n($\\alpha = {alpha_true}$)', fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylim(1e-5, 10)

    fig.suptitle('Yuan-Torquato Improved Spreadability Fitting',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {FIG_PATH}")


def main():
    print("=" * 70)
    print("  Yuan-Torquato (2026) Improved Spreadability Fitting")
    print("=" * 70)

    chains = [
        ('fibonacci', 3.0),
        ('bombieri_taylor', 1.545),
    ]

    all_results = []
    for chain_name, alpha_true in chains:
        t_arr, E_t, rho = compute_spreadability_for_chain(chain_name, N_target=500_000)
        results = run_fitting(chain_name, alpha_true, t_arr, E_t)
        all_results.append(results)

    make_figure(all_results)

    with open(JSON_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {JSON_PATH}")


if __name__ == '__main__':
    main()
