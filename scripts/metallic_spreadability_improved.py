"""
Run Yuan-Torquato improved spreadability for all 5 metallic-mean chains
to get the best possible alpha estimates for the JP table.

Uses N_target = 1,000,000 (matching research_catalog.py) and
Type I fitting at orders 0-3.
"""
import os, sys, json
import numpy as np
from scipy.optimize import least_squares

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from substitution_tilings import (generate_substitution_sequence,
                                  sequence_to_points, compute_tile_lengths,
                                  predict_chain_length)
from two_phase_media import (compute_structure_factor, compute_spectral_density,
                              compute_excess_spreadability, extract_alpha_fit)

PHI2 = 0.35
D_COEFF = 1.0
N_TARGET = 1_000_000


def type_I_fit_func(t, alpha_hat, C_coeffs):
    gamma = (1 + alpha_hat) / 2.0
    correction = np.zeros_like(t)
    for i, c in enumerate(C_coeffs):
        correction += c * t ** (-i / 2.0)
    correction = np.maximum(correction, 1e-30)
    return -gamma * np.log(t) + np.log(correction)


def fit_type_I(t, ln_sex, n_order):
    def residuals(params):
        alpha_hat = params[0]
        C_coeffs = params[1:]
        return type_I_fit_func(t, alpha_hat, C_coeffs) - ln_sex

    mid = len(t) // 2
    slope = (ln_sex[-1] - ln_sex[mid]) / (np.log(t[-1]) - np.log(t[mid]))
    alpha_init = np.clip(-2 * slope - 1, -2, 10)

    x0 = np.zeros(n_order + 2)
    x0[0] = alpha_init
    x0[1] = 1.0
    result = least_squares(residuals, x0, method='lm', max_nfev=10000)

    return float(result.x[0]), result.x[1:], float(np.sqrt(np.mean(result.fun**2)))


def run_chain(chain_name):
    print(f"\n{'='*60}")
    print(f"  {chain_name}")
    print(f"{'='*60}")

    lengths, lam1 = compute_tile_lengths(chain_name)
    for n_iter in range(5, 70):
        if predict_chain_length(chain_name, n_iter) > N_TARGET:
            break
    seq = generate_substitution_sequence(chain_name, n_iter)
    points, L = sequence_to_points(seq, chain_name)
    N = len(points)
    rho = N / L
    print(f"  N = {N:,}, rho = {rho:.5f}, L/S = {lam1:.4f}")

    a_rod = PHI2 / (2 * rho)
    k_arr, S_k = compute_structure_factor(points, L)
    chi_V = compute_spectral_density(k_arr, S_k, rho, a_rod)

    t_arr = np.logspace(0, 6, 500)
    E_t = compute_excess_spreadability(k_arr, chi_V, PHI2, t_arr, D=D_COEFF)

    # Simple log-slope (same method as research_catalog.py)
    alpha_simple, r2 = extract_alpha_fit(t_arr, E_t, t_min=1e2, t_max=1e5)
    print(f"  Simple log-slope: alpha = {alpha_simple:.4f} (R^2 = {r2:.6f})")

    # Yuan-Torquato Type I fits
    mask = (E_t > 0) & (t_arr > 10)
    t = t_arr[mask]
    ln_sex = np.log(E_t[mask])

    print(f"  Type I fits ({len(t)} points, t in [{t[0]:.1f}, {t[-1]:.0f}]):")
    print(f"  {'order':>5s}  {'alpha_hat':>10s}  {'error':>10s}  {'residual':>10s}")

    best_alpha = alpha_simple
    best_order = -1
    for n in range(4):
        try:
            alpha_hat, _, res = fit_type_I(t, ln_sex, n)
            err = alpha_hat - 3.0
            print(f"  {n:5d}  {alpha_hat:10.4f}  {err:+10.4f}  {res:10.6f}")
            if n == 1:  # Type I order 1 was best for Fibonacci
                best_alpha = alpha_hat
                best_order = n
        except Exception as e:
            print(f"  {n:5d}  FAILED: {e}")

    return {
        'chain': chain_name,
        'N': N,
        'rho': rho,
        'alpha_simple': float(alpha_simple),
        'alpha_typeI_order1': float(best_alpha),
    }


def main():
    chains = ['fibonacci', 'silver', 'bronze', 'copper', 'nickel']
    results = []

    for name in chains:
        r = run_chain(name)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Chain':<12s}  {'N':>10s}  {'Simple':>8s}  {'TypeI-1':>8s}  {'True':>6s}")
    for r in results:
        print(f"  {r['chain']:<12s}  {r['N']:>10,d}  {r['alpha_simple']:>8.3f}  "
              f"{r['alpha_typeI_order1']:>8.3f}  {'3.000':>6s}")

    out = os.path.join(SCRIPT_DIR, 'results', 'metallic_spreadability_improved.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
