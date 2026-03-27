"""
g2-invariant point processes from Torquato & Stillinger (2003) Section V.

1D examples:
  (a) Lattice-gas: tessellate line into intervals of length D,
      place one point uniformly in each. g2(r) = r/D for r <= D, 1 for r > D.
      rho = 1/D. Lambda_bar = 1/5 (Eq. 96).

  (b) Step-function g2: g2(r) = Theta(r-D). At terminal density phi_c = 1/2,
      this is hard rods at close packing = integer lattice. Lambda_bar = 1/6.

  (c) Step+delta g2: g2 = Theta(r-D) + Z/(rho*s1(D))*delta(r-D).
      For d=1: phi_c = 3/4. Lambda_bar from Eq. (125).

Output: results/g2_invariant_results.json
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_JSON = os.path.join(RESULTS_DIR, 'g2_invariant_results.json')

N_POINTS = 100_000
NUM_WINDOWS = 20_000
NUM_R = 500
SEED = 2029


def generate_lattice_gas(N, rng):
    """
    Lattice-gas model (Torquato & Stillinger 2003, Eq. 95).

    Tessellate [0, N) into N intervals of length D=1.
    Place one point uniformly at random in each interval.
    rho = 1/D = 1.

    g2(r) = r/D for r <= D, g2(r) = 1 for r > D.
    Lambda_bar = 1/5 (Eq. 96).
    """
    # Each point j is placed at j + U(0, 1)
    points = np.arange(N, dtype=np.float64) + rng.uniform(0, 1, N)
    L = float(N)
    return np.sort(points % L), L


def generate_shuffled_lattice(N, rng, sigma=0.5):
    """
    Shuffled lattice: each site of integer lattice displaced by
    a random variable with finite second moment.

    Using Gaussian displacement with std = sigma.
    Hyperuniform for any sigma > 0.
    """
    points = np.arange(N, dtype=np.float64) + rng.normal(0, sigma, N)
    L = float(N)
    return np.sort(points % L), L


def compute_lb(points, L, rng, label=""):
    """Compute Lambda_bar for a point pattern."""
    N = len(points)
    rho = N / L
    mean_spacing = 1.0 / rho
    R_max = min(250 * mean_spacing, L / 4)
    R_array = np.linspace(mean_spacing, R_max, NUM_R)

    variances, _ = compute_number_variance_1d(
        points, L, R_array, num_windows=NUM_WINDOWS, rng=rng, periodic=True)

    lb = compute_lambda_bar(R_array, variances)

    # Error estimate
    tail = variances[len(variances) // 3:]
    splits = np.array_split(tail, 4)
    boots = [np.mean(s) for s in splits if len(s) > 0]
    lb_err = np.std(boots) / np.sqrt(len(boots)) if len(boots) > 1 else 0.0

    print(f"  {label}: Lambda_bar = {lb:.5f} +/- {lb_err:.5f}  (N={N:,}, rho={rho:.4f})")
    return lb, lb_err, rho


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("  g2-invariant processes (Torquato & Stillinger 2003)")
    print("=" * 70)

    results = {}

    # (a) Lattice-gas: Lambda_bar should be 1/5 = 0.2000
    print("\n1. Lattice-gas (g2 = r/D for r <= D)")
    print(f"   Theory: Lambda_bar = 1/5 = {1/5:.5f}")
    points, L = generate_lattice_gas(N_POINTS, rng)
    lb, err, rho = compute_lb(points, L, rng, "Lattice-gas")
    results['lattice_gas'] = {
        'lambda_bar': float(lb), 'err': float(err),
        'lambda_bar_exact': 1/5,
        'description': 'g2(r) = r/D for r <= D, 1 for r > D',
    }

    # For comparison: URL with a=1 (different from lattice-gas!)
    print("\n2. URL a=1 (for comparison)")
    print(f"   Theory: Lambda_bar = 1/3 = {1/3:.5f}")
    from disordered_patterns import generate_url
    points_url, L_url = generate_url(N_POINTS, 1.0, rng=rng)
    lb_url, err_url, rho_url = compute_lb(points_url, L_url, rng, "URL a=1")
    results['url_a1'] = {
        'lambda_bar': float(lb_url), 'err': float(err_url),
        'lambda_bar_exact': 1/3,
    }

    # (b) Shuffled lattice with various sigma
    for sigma in [0.1, 0.2, 0.3]:
        print(f"\n3. Shuffled lattice sigma={sigma}")
        points_sh, L_sh = generate_shuffled_lattice(N_POINTS, rng, sigma=sigma)
        lb_sh, err_sh, rho_sh = compute_lb(points_sh, L_sh, rng,
                                            f"Shuffled sig={sigma}")
        results[f'shuffled_sigma_{sigma}'] = {
            'lambda_bar': float(lb_sh), 'err': float(err_sh),
            'sigma': sigma,
        }

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        exact = r.get('lambda_bar_exact', '?')
        print(f"  {name:25s}: Lb = {r['lambda_bar']:.5f} +/- {r['err']:.5f}"
              f"  (exact: {exact})")

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {OUT_JSON}")


if __name__ == '__main__':
    main()
