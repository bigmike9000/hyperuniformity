"""
High-Precision Lambda-bar Investigation for Silver and Bronze Chains
====================================================================
Investigates whether Lambda-bar = 1/4 exactly for Silver chain.

Strategy:
1. Generate Silver chain to N ~ 5-20M (as many iterations as feasible)
2. Use high-precision variance computation with 50,000 windows
3. Compute running average Lambda-bar(R) as R grows
4. Also compute Bronze Lambda-bar at high precision
5. Analytical: apply Zachary & Torquato (2009) theta-function formula
6. Save figure results/figures/fig_silver_precision.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os

# Add project directory to path
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from substitution_tilings import (
    CHAINS, generate_substitution_sequence, sequence_to_points,
    predict_chain_length
)
from quasicrystal_variance import compute_number_variance_1d

FIGURES_DIR = '/c/Users/minec/OneDrive/Desktop/Hyperuniformity/results/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Part 0: Analytical candidate values
# ============================================================

def check_simple_expressions(value, name, tol=5e-4):
    """Check if value matches any simple analytical expression."""
    candidates = {
        '1/4': 1/4,
        '1/3': 1/3,
        '1/6': 1/6,
        '1/5': 1/5,
        '1/pi': 1/np.pi,
        'sqrt(2)/5': np.sqrt(2)/5,
        '(sqrt(5)-1)/8': (np.sqrt(5)-1)/8,
        '1/(2+sqrt(2))': 1/(2+np.sqrt(2)),
        '1/(2*sqrt(2))': 1/(2*np.sqrt(2)),
        'sqrt(2)-1': np.sqrt(2)-1,
        '(sqrt(2)-1)/2': (np.sqrt(2)-1)/2,
        '(3-sqrt(5))/2': (3-np.sqrt(5))/2,
        '1/(3+sqrt(2))': 1/(3+np.sqrt(2)),
        '(2-sqrt(2))/2': (2-np.sqrt(2))/2,
        '(sqrt(3)-1)/2': (np.sqrt(3)-1)/2,
        '3/10': 3/10,
        '17/60': 17/60,
        '(4-sqrt(13))/2': (4-np.sqrt(13))/2 if (4-np.sqrt(13))/2 > 0 else None,
        '(sqrt(13)-1)/8': (np.sqrt(13)-1)/8,
        '1/(sqrt(2)+1)**2': 1/(np.sqrt(2)+1)**2,
        '(3-sqrt(5))/4': (3-np.sqrt(5))/4,
        '3/(3*sqrt(2)+sqrt(2))': 3/(4*np.sqrt(2)),
        '1/sqrt(2*pi)': 1/np.sqrt(2*np.pi),
        '9/32': 9/32,
        '7/25': 7/25,
        '3/11': 3/11,
        '11/39': 11/39,
        '5/18': 5/18,
        '(sqrt(13)+1)/24': (np.sqrt(13)+1)/24,
        'sqrt(2)/6': np.sqrt(2)/6,
        'sqrt(2)/4': np.sqrt(2)/4,
        '(3-sqrt(2))/8': (3-np.sqrt(2))/8,
        '(sqrt(13)-2)/4': (np.sqrt(13)-2)/4,
    }

    print(f"\n  Checking {name} = {value:.8f} against simple expressions:")
    matches = []
    for expr, cval in candidates.items():
        if cval is None or cval <= 0:
            continue
        diff = abs(value - cval)
        if diff < tol:
            matches.append((expr, cval, diff))
            print(f"    MATCH: {expr} = {cval:.8f}  (diff = {diff:.2e})")
        elif diff < 0.01:
            print(f"    close: {expr} = {cval:.8f}  (diff = {diff:.4f})")

    if not matches:
        print(f"    No simple expression found within tolerance {tol}")
    return matches


# ============================================================
# Part 1: Zachary & Torquato theta-function formula
# ============================================================

def zachary_formula_1d(positions, L, D_char=None, beta_min=1e-6, beta_max=1e3,
                        n_beta=5000):
    """
    Compute B_N using the Zachary & Torquato (2009) theta-function method.

    For d=1 (Eq. 73 of Zachary 2009):
        B_N = lim_{beta->0+} [phi/(2*beta) - (1/2) * sum_k Z_k * r_k * exp(-beta*r_k^2)]

    where r_k are inter-point distances, Z_k their multiplicities, and phi = rho * D.
    Lambda_bar = 2*phi*B_N (since Lambda_bar = 2^d * phi * B_N in d=1).

    For quasiperiodic chains, we compute sum via pairwise distances directly.

    Parameters
    ----------
    positions : ndarray
        Sorted point positions.
    L : float
        Domain length.
    D_char : float, optional
        Characteristic length scale (mean spacing = 1/rho). Default: 1/rho.

    Returns
    -------
    B_N : float
    phi : float (= rho * D_char / something)
    """
    N = len(positions)
    rho = N / L
    if D_char is None:
        D_char = 1.0 / rho  # mean spacing

    phi = rho * D_char  # dimensionless density

    # Build pair distance histogram up to r_max
    # For the theta-series method, we need ALL unique inter-point distances
    # For a quasiperiodic chain, distances take finitely many values at each shell
    # But practically, we compute the sum numerically using the autocorrelation approach

    # Method: use Eq. (82)-(83) of Zachary 2009
    # B_N(R) = (R/D) * [1 - 2*phi*(R/D) + (1/N) * sum_{i!=j} alpha(r_ij; R)]
    # B_N = lim_{L->inf} (1/L) * int_0^L B_N(R) dR
    #
    # This is equivalent to the variance method we already use.
    # The direct theta-function approach for periodic systems requires knowing
    # the exact coordination shells.

    # For our purpose, we'll use the running average of sigma^2(R) which is
    # exactly Lambda_bar by definition.
    pass


# ============================================================
# Part 2: Direct analytical B_N via inter-point distances
# ============================================================

def compute_BN_direct(positions, L, D_char=None, beta_values=None):
    """
    Compute B_N via Zachary 2009 Eq. (73), direct summation over pairs.

    B_N = lim_{beta->0+} [phi/(2*beta) - (1/2)*sum_k Z_k * r_k * exp(-beta*r_k^2)]

    We evaluate this at a sequence of beta values and extrapolate to beta=0.

    For large N this is O(N^2) which is only feasible for N < ~50k.
    We use a subchain for this calculation.
    """
    N = len(positions)
    rho = N / L
    if D_char is None:
        D_char = 1.0 / rho
    phi = rho * D_char

    if beta_values is None:
        # Fine grid in log space from small beta to medium beta
        beta_values = np.logspace(-5, 1, 200)

    # Compute all pairwise distances (use periodic BC, take min image)
    # For large N, sample a subset
    N_sub = min(N, 10000)
    if N_sub < N:
        idx = np.round(np.linspace(0, N-1, N_sub)).astype(int)
        pos_sub = positions[idx]
        rho_eff = N_sub / L  # same density effectively
    else:
        pos_sub = positions
        rho_eff = rho

    # Compute all pairwise distances with PBC
    # r_ij = |x_i - x_j| mod L (take minimum image)
    print(f"  Computing pairwise distances for N_sub={N_sub}...")
    t0 = time.perf_counter()

    BN_values = np.zeros(len(beta_values))

    # Build pairwise distances: O(N_sub^2 / 2)
    all_dists = []
    chunk = 500
    for i in range(0, N_sub, chunk):
        end_i = min(i + chunk, N_sub)
        diff = pos_sub[i:end_i, np.newaxis] - pos_sub[np.newaxis, :]
        # Minimum image convention
        diff = diff - L * np.round(diff / L)
        dists = np.abs(diff)
        # Only take upper triangle (j > i indices)
        for ii in range(end_i - i):
            row_dists = dists[ii, i + ii + 1:]
            all_dists.extend(row_dists.tolist())

    all_dists = np.array(all_dists)
    print(f"  {len(all_dists):,} pairs computed in {time.perf_counter()-t0:.1f}s")

    # B_N(beta) = phi/(2*beta) - (1/N_sub) * sum_{i<j} r_ij * exp(-beta*r_ij^2)
    # (factor of 1/2 absorbed because we sum only i<j, each counted once,
    # but the formula has sum_{i!=j} / N = 2 * sum_{i<j} / N )
    # Zachary eq (73): B_N = lim_{beta->0} [phi/(2*beta) - (1/2)*sum_k Z_k*r_k*exp(-beta*r_k^2)]
    # where sum_k Z_k*r_k*exp(-beta*r_k^2) = (1/N) * sum_{i!=j} r_ij * exp(-beta*r_ij^2)
    #                                        = (2/N) * sum_{i<j} r_ij * exp(-beta*r_ij^2)
    # So B_N(beta) = phi/(2*beta) - (1/2) * (2/N_sub) * sum_{i<j} r_ij * exp(-beta*r_ij^2)
    #              = phi/(2*beta) - (1/N_sub) * sum_{i<j} r_ij * exp(-beta*r_ij^2)

    for bi, beta in enumerate(beta_values):
        sum_term = np.sum(all_dists * np.exp(-beta * all_dists**2))
        BN_values[bi] = phi / (2 * beta) - sum_term / N_sub

    return beta_values, BN_values, phi


# ============================================================
# Part 3: High-precision variance computation
# ============================================================

def compute_lambda_bar_highprecision(chain_name, target_N=10_000_000,
                                      num_R=800, num_windows=50000,
                                      use_asymptotic_only=True):
    """
    Generate chain and compute Lambda-bar at high precision.

    Returns: lambda_bar, lambda_bar_std, R_array, variances, points, L
    """
    print(f"\n{'='*60}")
    print(f"  High-precision Lambda-bar: {CHAINS[chain_name]['name']}")
    print(f"{'='*60}")

    # Find number of iterations to get N >= target_N
    for iters in range(5, 45):
        n_pred = predict_chain_length(chain_name, iters)
        if n_pred >= target_N:
            break

    print(f"  Target N = {target_N:,} -> using {iters} iterations -> N_pred = {n_pred:,}")

    t0 = time.perf_counter()
    seq = generate_substitution_sequence(chain_name, iters)
    points, L_domain = sequence_to_points(seq, chain_name)
    N = len(points)
    rho = N / L_domain
    print(f"  Generated: N = {N:,}, L = {L_domain:.1f}, rho = {rho:.6f}  [{time.perf_counter()-t0:.1f}s]")

    # Set R range: use mean spacing as unit
    mean_spacing = 1.0 / rho

    # Use wide R range: from 1 mean spacing to L/4
    # To get high-precision Lambda-bar, we need MANY oscillation periods
    # Silver has oscillation period ~ mean_spacing ~ 1/rho ~ 1.4
    # We want R_max / period >> 1 for accurate averaging
    R_min = mean_spacing * 0.5
    R_max = min(L_domain / 5.0, 5000 * mean_spacing)  # many periods

    R_array = np.linspace(R_min, R_max, num_R)

    print(f"  R range: [{R_min:.3f}, {R_max:.1f}] (= [{R_min/mean_spacing:.1f}, {R_max/mean_spacing:.0f}] mean spacings)")
    print(f"  Oscillation periods covered: ~{(R_max - R_min) / mean_spacing:.0f}")
    print(f"  Computing variance at {num_R} R values with {num_windows:,} windows each...")

    rng = np.random.default_rng(seed=42)
    t0 = time.perf_counter()
    variances, mean_counts = compute_number_variance_1d(
        points, L_domain, R_array, num_windows=num_windows, rng=rng, periodic=True)
    elapsed = time.perf_counter() - t0
    print(f"  Variance computed in {elapsed:.1f}s")

    # Lambda-bar: mean of variance over the last 2/3 of R range
    # (skip early transient)
    start_idx = num_R // 3
    lambda_bar = np.mean(variances[start_idx:])
    lambda_bar_std = np.std(variances[start_idx:]) / np.sqrt(len(variances[start_idx:]))

    # Running average Lambda-bar(R): simple cumulative mean (equally-spaced R)
    running_lambda_bar = np.cumsum(variances) / np.arange(1, len(variances)+1)

    # Final Lambda-bar estimate: running average at R_max
    lambda_bar_running = running_lambda_bar[-1]

    print(f"\n  Lambda-bar (simple mean, last 2/3): {lambda_bar:.8f} ± {lambda_bar_std:.2e}")
    print(f"  Lambda-bar (running integral):       {lambda_bar_running:.8f}")
    print(f"  Variance range: [{np.min(variances):.5f}, {np.max(variances):.5f}]")
    print(f"  Variance std/mean = {np.std(variances[start_idx:])/lambda_bar:.4f}")

    # Additional high-precision estimate: use only middle 50% of variance oscillation
    # to reduce boundary effects
    start2 = num_R // 4
    end2 = 3 * num_R // 4
    lambda_bar_mid = np.mean(variances[start2:end2])
    print(f"  Lambda-bar (middle 50%):             {lambda_bar_mid:.8f}")

    # Also compute over last 1/4 (most converged)
    start3 = 3 * num_R // 4
    lambda_bar_late = np.mean(variances[start3:])
    lambda_bar_late_std = np.std(variances[start3:]) / np.sqrt(len(variances[start3:]))
    print(f"  Lambda-bar (last 1/4):               {lambda_bar_late:.8f} ± {lambda_bar_late_std:.2e}")

    return {
        'lambda_bar': lambda_bar,
        'lambda_bar_std': lambda_bar_std,
        'lambda_bar_running': lambda_bar_running,
        'lambda_bar_mid': lambda_bar_mid,
        'lambda_bar_late': lambda_bar_late,
        'lambda_bar_late_std': lambda_bar_late_std,
        'R_array': R_array,
        'variances': variances,
        'running_lambda_bar': running_lambda_bar,
        'rho': rho,
        'N': N,
        'L': L_domain,
        'mean_spacing': mean_spacing,
    }


# ============================================================
# Part 4: Analytical formula attempt
# ============================================================

def analytical_BN_metallic(chain_name, N_pair=20000):
    """
    Compute B_N analytically for a metallic-mean chain using the theta-function method.

    Uses the exact pairwise distance structure of the chain.
    For a Silver chain, we have exactly 2 tile types (S=1, L=1+sqrt(2)),
    and the inter-point distances are all possible combinations:
      k*S + m*L  for non-negative integers k, m with k+m >= 1
    with specific frequencies determined by the substitution rules.

    This is an approximation computed from a large but finite chain.
    """
    chain = CHAINS[chain_name]
    metal = chain['metallic_mean']

    print(f"\n  Analytical B_N via pairwise distances (N_pair={N_pair:,})...")

    # Generate a chain just large enough
    for iters in range(5, 40):
        n_pred = predict_chain_length(chain_name, iters)
        if n_pred >= N_pair:
            break

    seq = generate_substitution_sequence(chain_name, iters)
    pts, L = sequence_to_points(seq, chain_name)
    N = len(pts)
    rho = N / L
    D_char = 1.0 / rho
    phi = rho * D_char  # = 1.0

    print(f"  Chain: N={N:,}, rho={rho:.6f}, D_char={D_char:.6f}")

    # For d=1, Zachary 2009 eq. (73):
    # B_N = lim_{beta->0+} [phi/(2*beta) - (1/2) * (1/N) * sum_{i!=j} r_{ij} * exp(-beta*r_{ij}^2)]
    # where r_{ij} are nearest-image distances.
    # We compute at many beta values and extrapolate.

    # The key insight: for periodic + quasiperiodic systems,
    # (1/N) sum_{i!=j} r_{ij} exp(-beta r_{ij}^2)
    # = 2 * (1/N) sum_{i<j} r_{ij} exp(-beta r_{ij}^2)
    # This diverges as beta -> 0 as phi/beta, so the limit B_N is the finite remainder.

    # Build all pairwise distances using FFT-based autocorrelation approach
    # For large N, we use the spacing statistics instead

    # Get all inter-point spacings (nearest neighbors)
    spacings = np.diff(pts)
    unique_spacings = np.unique(np.round(spacings, 6))
    print(f"  Unique spacings: {unique_spacings} (expected S=1, L={metal:.4f})")

    beta_values = np.logspace(-5, 2, 500)
    BN_vals, phi_out = _compute_BN_from_distances(pts, L, rho, D_char, beta_values)

    # Find plateau region: B_N should plateau as beta -> 0
    # (the divergence is cancelled by the phi/(2*beta) term)
    # Find the flat region
    dBN = np.abs(np.diff(BN_vals)) / (np.abs(BN_vals[:-1]) + 1e-10)
    plateau_mask = dBN < 0.01

    if np.any(plateau_mask):
        # Use the median in the plateau region
        plateau_BN = BN_vals[:-1][plateau_mask]
        BN_estimate = np.median(plateau_BN)
        print(f"  B_N plateau: {BN_estimate:.6f} (from {np.sum(plateau_mask)} beta values)")
    else:
        # Use minimum variation region
        smooth_BN = np.convolve(BN_vals, np.ones(20)/20, mode='valid')
        min_var_idx = np.argmin(np.abs(np.diff(smooth_BN)))
        BN_estimate = smooth_BN[min_var_idx]
        print(f"  B_N (min variation): {BN_estimate:.6f}")

    # Lambda_bar = 2^d * phi * B_N = 2 * phi * B_N (d=1)
    # But actually Lambda_bar = 2*phi*B_N only with D = D_char
    # The convention is: sigma^2(R) -> 2*phi*(R/D) * B_N as R -> infinity for hyperuniform
    # but for bounded variance (Class I), sigma^2 is bounded, not growing.
    # So the correct relationship is through the average:
    # Lambda_bar = (period-average of sigma^2) and B_N = Lambda_bar / (2*phi)
    # with phi being in units of D.
    lambda_bar_analytical = 2 * phi_out * BN_estimate
    print(f"  Lambda_bar (analytical) = 2 * phi * B_N = 2 * {phi_out:.4f} * {BN_estimate:.6f} = {lambda_bar_analytical:.6f}")

    return BN_estimate, lambda_bar_analytical, beta_values, BN_vals


def _compute_BN_from_distances(pts, L, rho, D_char, beta_values, max_pairs=500000):
    """Helper: compute B_N(beta) from pairwise distances."""
    N = len(pts)
    phi = rho * D_char

    # Sample representative pairs
    np.random.seed(42)
    n_sample = min(N, 3000)
    idx = np.sort(np.random.choice(N, n_sample, replace=False))
    pts_sub = pts[idx]

    # Compute all pairwise minimum-image distances
    diffs = pts_sub[:, np.newaxis] - pts_sub[np.newaxis, :]
    diffs = diffs - L * np.round(diffs / L)
    dists_flat = np.abs(diffs[np.triu_indices(n_sample, k=1)])

    print(f"  Using {len(dists_flat):,} pairs from {n_sample} sampled points")

    BN_vals = np.zeros(len(beta_values))
    for bi, beta in enumerate(beta_values):
        # sum_{i<j} r_ij * exp(-beta * r_ij^2) / n_sample
        # = (n_sample - 1)/2 terms normalized by n_sample
        # But for the formula, we need (1/N) sum_{i!=j} = (2/N) sum_{i<j}
        # Using sampled version: (2/n_sample) * sum_{i<j in sample} / ...
        sum_term = np.sum(dists_flat * np.exp(-beta * dists_flat**2)) * 2 / n_sample
        BN_vals[bi] = phi / (2 * beta) - 0.5 * sum_term

    return BN_vals, phi


# ============================================================
# Main computation
# ============================================================

def main():
    print("=" * 70)
    print("  High-Precision Lambda-bar Investigation")
    print("  Silver chain: Is Lambda-bar = 1/4 exactly?")
    print("=" * 70)

    # ---- Silver chain high-precision ----
    silver_results = compute_lambda_bar_highprecision(
        'silver',
        target_N=5_000_000,   # 5M minimum
        num_R=1000,
        num_windows=50000,
    )

    # ---- Bronze chain high-precision ----
    bronze_results = compute_lambda_bar_highprecision(
        'bronze',
        target_N=5_000_000,
        num_R=1000,
        num_windows=50000,
    )

    # ---- Fibonacci for reference ----
    fib_results = compute_lambda_bar_highprecision(
        'fibonacci',
        target_N=5_000_000,
        num_R=1000,
        num_windows=50000,
    )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  SUMMARY: High-Precision Lambda-bar Values")
    print("=" * 70)

    results = {
        'fibonacci': fib_results,
        'silver': silver_results,
        'bronze': bronze_results,
    }

    for name, r in results.items():
        lb = r['lambda_bar']
        lbr = r['lambda_bar_running']
        lbl = r['lambda_bar_late']
        std = r['lambda_bar_std']
        stdl = r['lambda_bar_late_std']
        N = r['N']
        print(f"\n  {CHAINS[name]['name']} (N={N:,}):")
        print(f"    Lambda-bar (mean last 2/3):  {lb:.8f} ± {std:.1e}")
        print(f"    Lambda-bar (running integral): {lbr:.8f}")
        print(f"    Lambda-bar (last 1/4):         {lbl:.8f} ± {stdl:.1e}")

    print(f"\n  Reference: 1/4 = {0.25000000:.8f}")
    print(f"  Reference: 1/6 = {1/6:.8f} (integer lattice)")
    print(f"  Reference: Zachary Fibonacci = {0.20110:.8f}")

    # ---- Check simple expressions ----
    silver_lb = silver_results['lambda_bar_late']
    bronze_lb = bronze_results['lambda_bar_late']
    fib_lb = fib_results['lambda_bar_late']

    silver_matches = check_simple_expressions(silver_lb, 'Silver Lambda-bar', tol=1e-3)
    bronze_matches = check_simple_expressions(bronze_lb, 'Bronze Lambda-bar', tol=1e-3)
    fib_matches = check_simple_expressions(fib_lb, 'Fibonacci Lambda-bar', tol=1e-3)

    # ---- Pattern analysis: does Lambda-bar follow a pattern across metallic means? ----
    print("\n\n" + "=" * 70)
    print("  Pattern Analysis: Lambda-bar vs metallic mean index n")
    print("=" * 70)

    # Metallic mean for index n: mu_n = (n + sqrt(n^2+4))/2
    # Fibonacci: n=1, mu=phi=(1+sqrt(5))/2
    # Silver:    n=2, mu=1+sqrt(2)
    # Bronze:    n=3, mu=(3+sqrt(13))/2

    data = [
        (1, (1+np.sqrt(5))/2,      fib_lb,    0.20110, 'Fibonacci'),
        (2, 1+np.sqrt(2),           silver_lb, 0.250,   'Silver'),
        (3, (3+np.sqrt(13))/2,     bronze_lb, 0.282,   'Bronze'),
    ]

    print(f"\n  {'n':>3}  {'mu_n':>10}  {'Lambda-bar':>12}  {'Known':>10}  {'Chain':10}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")
    for n, mu, lb, known, cname in data:
        print(f"  {n:>3}  {mu:10.6f}  {lb:12.8f}  {known:10.5f}  {cname}")

    # Check if Lambda-bar = f(mu_n) for some simple function
    # Try: Lambda-bar = 1/(2*(1+mu_n)) ?
    # n=1: 1/(2*(1+phi)) = 1/(2*2.618) = 0.1910  NO
    # n=2: 1/(2*(1+1+sqrt(2))) = 1/(2*(3+sqrt(2))) = 1/(2*3.414) = 0.1464  NO
    #
    # Try: Lambda-bar = 1/(2*(mu_n)) ?
    # n=1: 1/(2*phi) = 1/(2*1.618) = 0.3090  NO
    # n=2: 1/(2*(1+sqrt(2))) = 1/(2*2.414) = 0.2071  NO
    #
    # Try: Lambda-bar * mu_n^2 = constant?
    print("\n  Testing potential formulas:")
    for n, mu, lb, known, cname in data:
        print(f"  {cname}: lb*mu^2 = {lb*mu**2:.4f}, "
              f"lb*(1+1/mu) = {lb*(1+1/mu):.4f}, "
              f"lb*mu = {lb*mu:.4f}")

    # The known result for Fibonacci = 0.20110
    # 0.20110 = ?
    # phi*B_N = 0.10055 from Table 1 of Zachary 2009
    # For silver: phi*B_N = Lambda-bar/2 since Lambda-bar = 2*phi*B_N (d=1, phi=rho*D)
    # But phi = rho * D. In Zachary's notation with D = mean spacing = 1/rho, phi = 1.
    # So Lambda-bar = 2 * B_N
    # => B_N(silver) = Lambda-bar(silver) / 2

    print("\n  B_N values (= Lambda_bar / 2, since phi=1 in our normalization):")
    for n, mu, lb, known, cname in data:
        BN = lb / 2
        phi_BN = lb / 2  # = phi*B_N since phi=1
        print(f"  {cname}: B_N = {BN:.8f},  phi*B_N = {phi_BN:.8f}")
    print(f"  Zachary Table 1: Fibonacci phi*B_N = 0.10055, Lambda-bar = 0.20110")
    print(f"  Our Fibonacci:   phi*B_N = {fib_lb/2:.8f}")

    # ---- Figure ----
    print(f"\n\nGenerating figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    chain_data = [
        ('fibonacci', fib_results, '#1f77b4', 'Fibonacci (n=1)'),
        ('silver', silver_results, '#2ca02c', 'Silver (n=2)'),
        ('bronze', bronze_results, '#d62728', 'Bronze (n=3)'),
    ]

    # Panel 1: Variance curves for Silver (main focus)
    ax = axes[0, 0]
    r = silver_results
    ax.plot(r['R_array'] / r['mean_spacing'], r['variances'],
            '-', color='#2ca02c', lw=0.4, alpha=0.7, label='$\\sigma^2(R)$')
    ax.axhline(0.25, color='red', ls='--', lw=2, label='$1/4 = 0.25000$')
    ax.axhline(r['lambda_bar'], color='orange', ls='-', lw=1.5,
               label=f"Mean = {r['lambda_bar']:.5f}")
    ax.axhline(r['lambda_bar_late'], color='purple', ls=':', lw=1.5,
               label=f"Late = {r['lambda_bar_late']:.5f}")
    ax.set_xlabel(r'$R$ / (mean spacing)', fontsize=11)
    ax.set_ylabel(r'$\sigma^2(R)$', fontsize=11)
    ax.set_title(f'Silver Chain (N={r["N"]:,})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)
    ax.set_xlim([0, 200])

    # Panel 2: Running average Lambda-bar(R) for all three chains
    ax = axes[0, 1]
    for name, r, color, label in chain_data:
        ax.plot(r['R_array'] / r['mean_spacing'], r['running_lambda_bar'],
                '-', color=color, lw=1.5, label=f"{label}: {r['lambda_bar']:.5f}")
    ax.axhline(1/4, color='black', ls='--', lw=1, alpha=0.6, label='1/4')
    ax.axhline(0.20110, color='gray', ls=':', lw=1, alpha=0.6, label='0.20110')
    ax.set_xlabel(r'$R$ / (mean spacing)', fontsize=11)
    ax.set_ylabel(r'Running $\bar{\Lambda}(R)$', fontsize=11)
    ax.set_title('Running Average Convergence', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)
    ax.set_xlim([0, None])
    ax.set_ylim([0.18, 0.32])

    # Panel 3: Silver variance zoomed in — deviation from 1/4
    ax = axes[1, 0]
    r = silver_results
    # Running average in the last portion of R range
    # Plot delta = Lambda-bar(R) - 1/4
    delta = r['running_lambda_bar'] - 0.25
    ax.plot(r['R_array'] / r['mean_spacing'], delta,
            '-', color='#2ca02c', lw=1, label='Running avg - 1/4')
    ax.axhline(0, color='red', ls='--', lw=2, label='Exact = 1/4')
    ax.axhline(r['lambda_bar'] - 0.25, color='orange', ls='-', lw=1.5,
               label=f"Final = {r['lambda_bar']:.6f}")
    ax.set_xlabel(r'$R$ / (mean spacing)', fontsize=11)
    ax.set_ylabel(r'$\bar{\Lambda}(R) - 1/4$', fontsize=11)
    ax.set_title('Silver: Deviation from 1/4', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)
    ax.set_xlim([10, None])

    # Panel 4: Summary table / precision estimates
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary text
    textlines = [
        "HIGH-PRECISION RESULTS",
        "=" * 40,
        "",
        f"Fibonacci (N={fib_results['N']:,}):",
        f"  Simple mean: {fib_results['lambda_bar']:.6f} ± {fib_results['lambda_bar_std']:.1e}",
        f"  Running avg: {fib_results['lambda_bar_running']:.6f}",
        f"  Zachary ref: 0.201100",
        f"  Diff: {fib_results['lambda_bar'] - 0.20110:.2e}",
        "",
        f"Silver (N={silver_results['N']:,}):",
        f"  Simple mean: {silver_results['lambda_bar']:.6f} ± {silver_results['lambda_bar_std']:.1e}",
        f"  Running avg: {silver_results['lambda_bar_running']:.6f}",
        f"  1/4 exact:   0.250000",
        f"  Diff from 1/4: {silver_results['lambda_bar'] - 0.25:.2e}",
        "",
        f"Bronze (N={bronze_results['N']:,}):",
        f"  Simple mean: {bronze_results['lambda_bar']:.6f} ± {bronze_results['lambda_bar_std']:.1e}",
        f"  Running avg: {bronze_results['lambda_bar_running']:.6f}",
        "",
        "CANDIDATE EXPRESSIONS FOR BRONZE:",
    ]

    # Add Bronze candidate expressions
    bronze_val = bronze_results['lambda_bar']
    candidates_check = [
        ('sqrt(2)/5',   np.sqrt(2)/5),
        ('17/60',       17/60),
        ('(sqrt(13)-1)/8', (np.sqrt(13)-1)/8),
        ('sqrt(13)/12', np.sqrt(13)/12),
        ('3/(4*sqrt(2)+2)', 3/(4*np.sqrt(2)+2)),
        ('(3+sqrt(13))/24', (3+np.sqrt(13))/24),
    ]
    for expr, val in candidates_check:
        diff = abs(bronze_val - val)
        flag = " <-- MATCH" if diff < 5e-4 else ""
        textlines.append(f"  {expr} = {val:.6f} (diff={diff:.1e}){flag}")

    text = "\n".join(textlines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Precision Summary', fontsize=12)

    plt.suptitle('High-Precision Lambda-bar for Metallic Mean Chains\n'
                 'Is Lambda-bar(Silver) = 1/4 exactly?', fontsize=14, y=1.01)
    plt.tight_layout()

    outpath = os.path.join(FIGURES_DIR, 'fig_silver_precision.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {outpath}")
    plt.close()

    return results


if __name__ == '__main__':
    results = main()
