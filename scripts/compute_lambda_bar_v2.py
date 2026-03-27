"""
Correct analytical computation of Lambda_bar using the Zachary-Torquato formula.

The correct approach uses:
  Lambda_bar = (2*rho/pi) * sum_{G>0} I_G / G^2

where I_G is the SPECIFIC BRAGG INTENSITY defined as:
  I_G = lim_{N->inf} (1/N) * S(G)
and S(G) = |rho_hat(G)|^2 is the UNSCALED structure factor:
  S(G) = |sum_{j=1}^N exp(i G x_j)|^2

This is DIFFERENT from the S(k)/N normalization. The specific intensity is:
  I_G = S_unscaled(G) / N = |rho_hat(G)|^2 / N

For the INTEGER LATTICE:
  rho_hat(G) = N at each G = 2*pi*m -> I_G = N -> lambda_bar = (2/pi)*N*sum 1/G^2 DIVERGES

This can't be right. Let me start from scratch with the CORRECT derivation.

---
THE CORRECT FORMULA comes from the pair-correlation function.

For a 1D point process with pair correlation g_2(r):
  sigma^2(R) = 2*rho*R + 2*rho^2 * integral_0^{2R} (2R-r)*h(r) dr
where h(r) = g_2(r) - 1.

For a quasiperiodic tiling, h(r) = g_2(r) - 1 has the form:
  h(r) = sum_{k=1}^inf Z_k * delta(r - r_k) / (rho * 2) - 1
where Z_k is the number of pairs at distance r_k (coordination number),
and the factor 1/(rho*2) comes from the 1D surface area = 2.

Actually: the pair correlation function g_2(r) for a quasiperiodic tiling
has peaks at ALL possible inter-particle distances r_k, which form a dense
(but discrete) set.

The B_N coefficient (Lambda_bar = 2*phi*B_N in 1D) can be computed via:
B_N = lim_{beta->0+} [phi/(2*beta) - (1/2) * sum_{k=1}^inf Z_k * r_k * exp(-beta*r_k^2)]
(Zachary & Torquato 2009, eq. 73)

For QUASIPERIODIC TILINGS, the pair separations r_k form a DISCRETE set
(only finitely many inequivalent separations up to quasiperiodic equivalence).

For the FIBONACCI chain with tiles S=1, L=tau:
  Possible separations: all values m*1 + n*tau for non-negative integers m,n
  with (m,n) not both zero. These form a dense subset of [0,inf).
  But the THETA SERIES only uses those separations that actually appear
  as inter-particle distances in the infinite Fibonacci tiling.

This approach is equivalent to the DIRECT VARIANCE computation:
  Lambda_bar = lim_{R->inf} (1/R) * integral_0^R sigma^2(R') dR'

The Monte Carlo method computes this directly and gives Lambda_bar = 0.25 for Silver.

===== THE ANALYTICAL METHOD =====

The analytical formula from Zachary & Torquato (2009) uses the THETA SERIES:
For a quasiperiodic chain, define the theta series:
  Theta(q) = 1 + sum_{k>=1} Z_k * q^{r_k^2}
where Z_k is the coordination number for the k-th shell, r_k is the shell radius.
Then B_N is extracted from the small-q behavior of Theta(q).

For PERIODIC systems (Bravais lattices), Theta(q) is a standard mathematical
object (Jacobi theta function). For QUASIPERIODIC tilings, it must be computed
from the pair correlation function of the specific tiling.

The simplest way: compute the pair correlation function numerically from a
large chain, then evaluate the sum.

===== NUMERICAL APPROACH =====

We compute:
1. Generate a large chain (N ~ 10^6 points)
2. Compute all pair separations {r_{ij}} for |i-j| <= K (some cutoff)
3. Evaluate B_N = lim [phi/(2*beta) - (1/2)*sum Z_k*r_k*exp(-beta*r_k^2)]
4. Lambda_bar = 2*phi*B_N

This is the EXACT same formula Z&T used for Fibonacci (Λ̄ = 0.20110).
It's exact in the limit K -> infinity and beta -> 0.

For 1D chains, the pair separations form a COUNTABLE DENSE SET,
but only a FINITE number of distinct separations exist up to some cutoff R_max.
The key insight: for substitution tilings, all pair separations have the form
  r = m + n*lambda1  for non-negative integers m,n with m+n > 0
and the coordination numbers Z_k can be computed analytically.

For FIBONACCI (lambda1=tau=1.618...):
  Separations: 0 + 1*tau = tau (nearest neighbors), 1+0=1 (second neighbors?), etc.
  Wait: adjacent tiles contribute spacing 1 or tau. The PAIR SEPARATIONS
  for ALL distances (not just nearest) involve all positive linear combinations.

For SILVER (lambda1=1+sqrt(2)):
  Adjacent spacings: 1 (S tile) or 1+sqrt(2) (L tile).
  Pair separations: m*1 + n*(1+sqrt(2)) for m,n >= 0, (m,n) != (0,0).

The COORDINATION NUMBERS can be computed analytically using the PAIR
CORRELATION FUNCTION of the substitution tiling (Hof 1995, Baake & Moody 1999).

For our purposes (numerical), we use:
  B_N = lim_{beta->0+} F(beta)
  F(beta) = phi/(2*beta) - (1/2)*sum_{distinct separations r_k with Z_k != 0} Z_k*r_k*exp(-beta*r_k^2)

For a finite chain of length L with N points, the sum over all pairs gives
a good approximation when L >> 1/sqrt(beta).

===== IMPLEMENTATION =====

We use three approaches:
A) Direct from pairs: enumerate all inter-particle distances in a large chain
B) Theta-series regularization
C) Variance Monte Carlo (existing, as cross-check)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from substitution_tilings import CHAINS, generate_substitution_sequence, sequence_to_points
from quasicrystal_variance import compute_number_variance_1d, compute_lambda_bar


def compute_B_N_pairs(chain_name, n_iters=None, N_target=200_000, beta_values=None, verbose=True):
    """
    Compute B_N using the Zachary & Torquato theta-series formula:
      B_N = lim_{beta->0+} [phi/(2*beta) - (1/2) * sum_{i!=j} r_{ij} * exp(-beta*r_{ij}^2) / N]
    where the sum is over ALL pairs (i,j) with i < j, r_{ij} = |x_i - x_j|.

    The 1/N factor comes from the volume-average interpretation (eq. 83 of Z&T):
      B_N = lim_{L->inf} (1/L) * integral_0^L B_N(R) dR
    where B_N(R) = (R/D) * [1 - (R/D)^d + (1/N)*sum_{i!=j} alpha(r_{ij};R)]
    in d=1, with alpha(r;R) = 1 - r/2R for r < 2R.

    Simplified approach: compute
      F(beta) = (phi/(2*beta)) - (1/(2*N)) * sum_{i!=j} r_{ij} * exp(-beta*r_{ij}^2)
    and extrapolate F(beta) -> B_N as beta -> 0.

    Then Lambda_bar = 2 * phi * B_N where phi = rho * D (D = length scale = 1/rho for unit dens).
    => Lambda_bar = 2 * rho * (1/rho) * B_N = 2 * B_N.
    Wait: phi = rho * v(D/2) in d=1, v(D/2) = 2*(D/2) = D.
    With D = mean spacing = 1/rho: phi = rho * (1/rho) = 1.
    So Lambda_bar = 2 * phi * B_N = 2 * B_N.
    """
    chain = CHAINS[chain_name]
    M = chain['matrix']

    if n_iters is None:
        for iters in range(5, 50):
            vec = np.array([0,1], dtype=np.int64)
            Mn = np.linalg.matrix_power(M, iters)
            n_pred = int(np.sum(Mn @ vec))
            if n_pred > N_target:
                break
        n_iters = iters

    seq = generate_substitution_sequence(chain_name, n_iters)
    pts, L = sequence_to_points(seq, chain_name)
    N = len(pts)
    rho = N / L
    phi = 1.0  # = rho * D = rho * (1/rho) = 1 in reduced units

    if verbose:
        print(f"  {chain['name']}: N={N:,}, L={L:.1f}, rho={rho:.8f}")

    if beta_values is None:
        # Use geometrically spaced beta values, extrapolate to 0
        beta_values = np.logspace(-5, -1, 20)

    # Compute F(beta) for each beta
    # For each beta: F(beta) = phi/(2*beta) - (1/2N) * sum_{i<j} Z_ij * r_ij * exp(-beta*r_ij^2)
    # where Z_ij = 2 (for the pair and its mirror)
    # = phi/(2*beta) - (1/N) * sum_{i<j} r_ij * exp(-beta*r_ij^2)

    F_values = np.zeros(len(beta_values))

    # For large N, sum over ALL pairs is O(N^2) -- too slow.
    # Instead: for each beta, only pairs with r < 3/sqrt(beta) contribute significantly.
    # We compute the pair sum using a sliding-window approach.
    pts_sorted = np.sort(pts)

    for ib, beta in enumerate(beta_values):
        r_cutoff = 4.0 / np.sqrt(beta)  # exp(-beta*r^2) < exp(-16) for r > r_cutoff
        if r_cutoff > L/2:
            r_cutoff = L/2

        # Sum over pairs within r_cutoff
        pair_sum = 0.0
        for i in range(N):
            # Find j > i with pts[j] - pts[i] < r_cutoff
            hi = np.searchsorted(pts_sorted, pts_sorted[i] + r_cutoff, side='right')
            js = pts_sorted[i+1:hi]
            if len(js) > 0:
                r_ij = js - pts_sorted[i]
                pair_sum += np.sum(r_ij * np.exp(-beta * r_ij**2))

        F_values[ib] = phi/(2*beta) - pair_sum / N

    # Extrapolate F(beta) -> B_N as beta -> 0
    # For small beta: F(beta) = B_N + c1*beta + c2*beta^2 + ...
    # Fit a polynomial in beta to the small-beta values
    small_beta_mask = beta_values < np.median(beta_values)
    if np.sum(small_beta_mask) >= 2:
        coeffs = np.polyfit(beta_values[small_beta_mask], F_values[small_beta_mask], 2)
        B_N_extrap = coeffs[-1]  # constant term = F(0)
    else:
        B_N_extrap = F_values[0]

    lambda_bar = 2.0 * B_N_extrap  # = 2*phi*B_N with phi=1

    if verbose:
        print(f"  B_N (extrapolated): {B_N_extrap:.8f}")
        print(f"  Lambda_bar = 2*B_N: {lambda_bar:.8f}")

    return lambda_bar, F_values, beta_values


def compute_B_N_exact_pairs(chain_name, n_iters=None, N_target=500_000,
                             beta_values=None, verbose=True):
    """
    More accurate computation using the exact pair-distance histogram.

    The key formula (Zachary & Torquato eq. 73, d=1):
      B_N = lim_{beta->0} [phi/(2*beta) - (1/2)*sum_k Z_k*r_k*exp(-beta*r_k^2)]
    where sum is over DISTINCT pair distances r_k with their multiplicities Z_k.

    For a substitution tiling of length L with N points:
      (1/2)*sum_{k} Z_k*r_k*exp(-beta*r_k^2)
    approximates the infinite-chain sum when L >> 1/sqrt(beta).

    We compute this as (1/N)*sum_{i<j} r_{ij}*exp(-beta*r_{ij}^2) * 2  (the 2 counts both i<j and j>i)
    [Note: Z_k counts both directions, so Z_k = 2*(number in one direction)]
    Actually: (1/2)*sum_k Z_k*r_k*exp(-beta*r_k^2) = (1/2)*sum_{i!=j} r_{ij}*exp(-beta*r_{ij}^2) / N
    = (1/N) * sum_{i<j} r_{ij}*exp(-beta*r_{ij}^2)

    So: F(beta) = phi/(2*beta) - (1/N)*sum_{i<j} r_{ij}*exp(-beta*r_{ij}^2)
    """
    chain = CHAINS[chain_name]
    M = chain['matrix']

    if n_iters is None:
        for iters in range(5, 50):
            vec = np.array([0,1], dtype=np.int64)
            Mn = np.linalg.matrix_power(M, iters)
            n_pred = int(np.sum(Mn @ vec))
            if n_pred > N_target:
                break
        n_iters = iters

    seq = generate_substitution_sequence(chain_name, n_iters)
    pts, L = sequence_to_points(seq, chain_name)
    N = len(pts)
    rho = N / L
    phi = 1.0  # rho * D = rho * (1/rho) = 1

    if verbose:
        print(f"  {chain['name']}: N={N:,}, L={L:.1f}, rho={rho:.8f}")

    if beta_values is None:
        beta_values = np.logspace(-6, -2, 30)

    pts_sorted = np.sort(pts)
    F_values = np.zeros(len(beta_values))

    for ib, beta in enumerate(beta_values):
        r_cutoff = 5.0 / np.sqrt(beta)
        if r_cutoff > L/2:
            r_cutoff = L/2

        pair_sum = 0.0
        for i in range(N):
            hi = np.searchsorted(pts_sorted, pts_sorted[i] + r_cutoff, side='right')
            if hi > i+1:
                r_ij = pts_sorted[i+1:hi] - pts_sorted[i]
                pair_sum += np.sum(r_ij * np.exp(-beta * r_ij**2))

        F_values[ib] = phi/(2*beta) - pair_sum/N

        if verbose and ib % 5 == 0:
            print(f"    beta={beta:.2e}: F(beta)={F_values[ib]:.6f}")

    # Extrapolate F(beta -> 0)
    # For small beta, F(beta) should approach B_N + C*sqrt(beta) + ... (from Euler-Maclaurin)
    # Actually for smooth distributions: F(beta) = B_N + O(beta) (analytic in beta)
    # For the substitution tiling (discrete spectrum):
    # F(beta) = B_N + sum_k Z_k*r_k*(something small) + correction
    # Use linear fit in beta for small-beta values:
    mask = beta_values < 1e-3
    if np.sum(mask) >= 3:
        # Fit: F(beta) = a + b*beta + c*beta^2
        betas_fit = beta_values[mask]
        F_fit = F_values[mask]
        coeffs = np.polyfit(betas_fit, F_fit, 2)
        B_N_extrap = coeffs[-1]
    else:
        B_N_extrap = F_values[0]

    lambda_bar = 2.0 * B_N_extrap

    return lambda_bar, F_values, beta_values, pts, rho


def main():
    print("=" * 65)
    print("  Analytical Lambda_bar via Zachary-Torquato theta-series method")
    print("=" * 65)
    print()
    print("Formula: B_N = lim_{beta->0} [phi/(2*beta) - (1/N)*sum_{i<j} r_ij*exp(-beta*r_ij^2)]")
    print("Lambda_bar = 2 * B_N  (with phi = rho * D = 1 in reduced units)")
    print()

    chains_to_test = ['fibonacci', 'silver', 'bronze']
    results = {}

    for chain_name in chains_to_test:
        print(f"\n{'='*50}")
        print(f"  {CHAINS[chain_name]['name']}")
        print(f"{'='*50}")

        lb, F_vals, betas, pts, rho = compute_B_N_exact_pairs(
            chain_name, N_target=100_000, verbose=True)

        print(f"  Lambda_bar = {lb:.8f}")

        results[chain_name] = {'lb': lb, 'F': F_vals, 'betas': betas}

    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Chain':15s}  {'Lambda_bar':>12s}  {'vs 1/4':>10s}  Literature")
    print("  " + "-" * 55)
    for name in chains_to_test:
        lb = results[name]['lb']
        lit = {'fibonacci': '0.20110', 'silver': '(novel ~0.250)', 'bronze': '(novel ~0.282)'}
        is_quarter = 'YES' if abs(lb - 0.25) < 0.001 else '---'
        print(f"  {CHAINS[name]['name']:15s}  {lb:12.6f}  {is_quarter:>10s}  {lit[name]}")

    print()
    lb_silver = results['silver']['lb']
    print(f"Silver: Lambda_bar = {lb_silver:.8f}")
    print(f"        1/4        = {0.25:.8f}")
    print(f"        diff       = {lb_silver - 0.25:.2e}")
    if abs(lb_silver - 0.25) < 1e-4:
        print("  --> CONSISTENT WITH Lambda_bar = 1/4 EXACTLY")
    else:
        print(f"  --> Lambda_bar differs from 1/4 by {abs(lb_silver-0.25):.6f}")

    # Make plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'fibonacci': '#1f77b4', 'silver': '#ff7f0e', 'bronze': '#2ca02c'}
    lit_vals = {'fibonacci': 0.20110, 'silver': 0.25, 'bronze': 0.282}

    for ax, name in zip(axes, chains_to_test):
        r = results[name]
        F = r['F']
        betas = r['betas']
        lb = r['lb']
        ax.semilogx(betas, F, 'o-', color=colors[name], lw=2)
        ax.axhline(lb/2, color='red', ls='--', lw=1.5,
                   label=rf"$B_N = {lb/2:.5f}$ (extrap)")
        ax.set_xlabel(r'$\beta$', fontsize=12)
        ax.set_ylabel(r'$F(\beta)$', fontsize=12)
        ax.set_title(f"{CHAINS[name]['name']}: $\\bar{{\\Lambda}} = {lb:.5f}$")
        ax.legend(fontsize=9)
        ax.grid(True, ls=':', alpha=0.5)

    plt.suptitle(r'$B_N = \lim_{\beta\to 0} F(\beta) = \lim_{\beta\to 0} [\phi/(2\beta) - (1/N)\sum_{i<j} r_{ij} e^{-\beta r_{ij}^2}]$',
                 fontsize=11)
    plt.tight_layout()

    fig_dir = os.path.join(os.path.dirname(__file__), 'results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'fig_lambda_ZT_theta_series.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {fig_path}")
    plt.close()


if __name__ == '__main__':
    main()
