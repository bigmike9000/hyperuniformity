"""
Deep analytical investigation: Silver Lambda-bar = 1/4?

Method 1: Very large N with many R values using exact event-driven variance
Method 2: Analytical theta-function calculation using Zachary 2009 formulas
Method 3: Pattern search for exact formula across metallic means
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys, os

sys.path.insert(0, '/c/Users/minec/OneDrive/Desktop/Hyperuniformity')
from substitution_tilings import CHAINS, generate_substitution_sequence, sequence_to_points, predict_chain_length
from quasicrystal_variance import compute_number_variance_1d

FIGURES_DIR = '/c/Users/minec/OneDrive/Desktop/Hyperuniformity/results/figures'

# =========================================================
# Method 1: Exact variance via event-driven algorithm
# Much faster than Monte Carlo for large N
# =========================================================

def compute_variance_exact_1d(positions, R, L):
    """
    Exact sigma^2(R) by event-driven sweep: O(N log N) per R.

    As the window center sweeps from 0 to L, the count changes
    when a point enters or exits the window.
    Each point x_j creates:
      - entry event at (x_j - R) mod L  (point enters window)
      - exit event at (x_j + R) mod L   (point exits window)

    Returns: exact sigma^2(R) and exact mean count.
    """
    N = len(positions)

    # Build entry/exit events
    enter_pos = (positions - R) % L
    exit_pos  = (positions + R) % L

    # Create event arrays: +1 for enter, -1 for exit
    events_pos = np.concatenate([enter_pos, exit_pos])
    events_delta = np.concatenate([np.ones(N, dtype=np.int32), -np.ones(N, dtype=np.int32)])

    # Sort by position
    sort_idx = np.argsort(events_pos, kind='stable')
    events_pos = events_pos[sort_idx]
    events_delta = events_delta[sort_idx]

    # Count at position 0: number of points in [-R, R] mod L
    # = points in [L-R, L) union [0, R)
    if R >= L/2:
        # Window covers more than half the domain
        count0 = N
    else:
        # Points in [0, R):
        n_right = np.searchsorted(positions, R, side='right')
        # Points in [L-R, L):
        n_left = N - np.searchsorted(positions, L - R, side='left')
        count0 = n_right + n_left

    # Walk through events, accumulating weighted moments
    # Between consecutive events, count is constant
    # Weight = gap_length / L
    total_mean = 0.0
    total_sq   = 0.0
    count = count0

    prev = 0.0
    for i in range(len(events_pos)):
        pos = events_pos[i]
        gap = pos - prev
        # gap should be >= 0 since events are sorted
        w = gap / L
        total_mean += count * w
        total_sq   += count * count * w
        count += events_delta[i]
        prev = pos

    # Final gap from last event back to L (wrap around)
    gap = L - prev
    w = gap / L
    total_mean += count * w
    total_sq   += count * count * w

    variance = total_sq - total_mean * total_mean
    return variance, total_mean


def compute_variance_exact_batch(positions, L, R_array):
    """Compute exact sigma^2(R) for an array of R values."""
    variances = np.zeros(len(R_array))
    means = np.zeros(len(R_array))
    for j, R in enumerate(R_array):
        variances[j], means[j] = compute_variance_exact_1d(positions, R, L)
    return variances, means


# =========================================================
# Method 2: Period-exact averaging
# For a substitution tiling, sigma^2(R) is periodic in R
# with period = L / N_periods or similar.
# We can integrate over exactly one oscillation period.
# =========================================================

def find_oscillation_period(R_array, variances, rho, metal):
    """
    Estimate the oscillation period of sigma^2(R).

    For metallic mean chains, the variance oscillates with period
    approximately equal to 1/rho (the mean inter-point spacing).
    But the actual period is determined by the tile length ratio.

    For Silver: L = 1+sqrt(2), S = 1, rho = 0.5
    The fundamental period in R space is related to the tile lengths.
    """
    # Mean spacing = 1/rho
    mean_spacing = 1.0 / rho

    # Find peaks in variance (local maxima)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(variances, height=np.mean(variances))
    if len(peaks) > 2:
        peak_positions = R_array[peaks]
        periods = np.diff(peak_positions)
        avg_period = np.median(periods)
        return avg_period, mean_spacing
    return mean_spacing, mean_spacing


# =========================================================
# Method 3: Analytical formula via inter-point distance sums
#
# Zachary & Torquato (2009), Eq. (73) for d=1:
#
#  B_N = lim_{beta->0+} [phi/(2*beta) - (1/2) * Theta'(e^{-beta})]
#
# where Theta'(q) = d/dq [Theta(q)] and Theta is the theta series.
#
# For a quasicrystal, the theta series is:
#  Theta(q) = 1 + sum_{k=1}^inf Z_k * q^{r_k^2}
#
# where r_k are the coordination shell radii and Z_k their multiplicities.
#
# Equivalently, from Eq. (82)-(83):
#  B_N = lim_{L->inf} (1/L) * int_0^L B_N(R) dR
# where:
#  B_N(R) = (R/D) * [1 - 2*phi*(R/D)^d + (1/N)*sum_{i!=j} alpha(r_ij; R)]
#
# For d=1, alpha(r; R) = max(0, 1 - r/(2R)) (scaled intersection volume)
# So:
#  B_N(R) = R * [1 - 2*phi*R + (1/N)*sum_{i!=j} max(0, 1 - r_ij/(2R))]
#
# Lambda_bar = (period-average of sigma^2(R))
# And sigma^2(R) = 2*phi*D * B_N(R) / D (with D=1/rho and phi=rho*D=1)
#                = 2 * B_N(R) / 1  ???
#
# Actually the clean relationship is:
# Lambda_bar = 2*phi*D * B_N = 2 * 1 * B_N  (using phi=1, D=1/rho=1/rho)
# No wait: phi = rho * D_char, with D_char = mean spacing = 1/rho
# So phi = rho * (1/rho) = 1
# And Lambda_bar = 2^d * phi * B_N = 2 * 1 * B_N = 2 * B_N
# => Lambda_bar = 2 * B_N
# Table 1 of Zachary: Fibonacci phi*B_N = 0.10055, Lambda_bar = 0.20110 = 2*0.10055  CHECK!
# =========================================================

def compute_BN_exact(positions, L, N_R=5000, R_min_frac=0.01, R_max_frac=0.25):
    """
    Compute B_N exactly using Eq. (82)-(83) of Zachary 2009.

    B_N(R) = (R/D) * [1 - 2*phi*(R/D) + (1/N)*sum_{i!=j} alpha(r_ij; R)]
    where:
    - alpha(r; R) = max(0, 1 - r/(2R)) for d=1
    - D = mean spacing = 1/rho
    - phi = rho*D = 1 (our normalization)

    B_N = lim_{L_avg->inf} (1/L_avg) * int_0^{L_avg} B_N(R) dR

    The integral is the running average of B_N(R), which should plateau.
    """
    N = len(positions)
    rho = N / L
    D = 1.0 / rho
    phi = rho * D  # = 1.0

    R_min = R_min_frac * L
    R_max = R_max_frac * L
    R_array = np.linspace(R_min, R_max, N_R)

    print(f"  Computing exact B_N(R): R in [{R_min:.1f}, {R_max:.1f}], {N_R} values...")
    t0 = time.perf_counter()

    # Compute sigma^2(R) exactly
    variances, means = compute_variance_exact_batch(positions, L, R_array)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    # B_N(R) = sigma^2(R) / (2*phi) = sigma^2(R) / 2
    # (since sigma^2 = 2*phi*B_N_of_R roughly, but let's verify:
    # Zachary eq (82): B_N(R) = (R/D) * [...]
    # sigma^2(R) = 2*phi*(R/D)^0 * B_N (surface term) + ...
    # Actually sigma^2 = 2^d * phi * B_N_asymptotic
    # But B_N(R) as defined in eq (82) is NOT sigma^2/2.
    # Let me use the direct definition:
    # Lambda_bar = mean(sigma^2(R)) for large R
    # And B_N = Lambda_bar / (2*phi) = Lambda_bar / 2

    # Period-average sigma^2
    start = N_R // 3
    lambda_bar = np.mean(variances[start:])
    lambda_bar_std = np.std(variances[start:])

    # Running average
    running = np.cumsum(variances) / np.arange(1, N_R+1)

    return R_array, variances, lambda_bar, lambda_bar_std, running


# =========================================================
# Deeper convergence test: use very fine R sampling
# over exactly many oscillation periods
# =========================================================

def precision_lambda_bar(chain_name, target_N=None, num_exact_R=3000):
    """
    Ultra-high-precision Lambda-bar using exact event-driven variance.

    For the Silver chain, we know rho = 0.5 exactly (L/S = 1+sqrt(2) means
    each fundamental domain has one S and one L tile, total length 2+sqrt(2),
    containing 2 points, so rho = 2/(2+sqrt(2)) = sqrt(2)/(1+sqrt(2)) = sqrt(2)-1? No...
    Wait: Silver substitution L->LLS, S->L.
    Frequencies: f_L/(f_S) = metallic_mean = 1+sqrt(2)
    So f_L = (1+sqrt(2)) * f_S, and f_L + f_S = 1
    => f_S = 1/(2+sqrt(2)), f_L = (1+sqrt(2))/(2+sqrt(2))
    Mean tile length = f_S * 1 + f_L * (1+sqrt(2))
                     = 1/(2+sqrt(2)) + (1+sqrt(2))^2/(2+sqrt(2))
                     = [1 + (1+sqrt(2))^2] / (2+sqrt(2))
                     = [1 + 1 + 2*sqrt(2) + 2] / (2+sqrt(2))
                     = [4 + 2*sqrt(2)] / (2+sqrt(2))
                     = 2*(2+sqrt(2)) / (2+sqrt(2)) = 2
    So mean tile length = 2, and rho = 1/2 = 0.5. CONFIRMED.
    """
    chain = CHAINS[chain_name]
    metal = chain['metallic_mean']
    print(f"\n{'='*65}")
    print(f"  Precision Lambda-bar: {chain['name']}")
    print(f"  Metallic mean mu = {metal:.10f}")
    print(f"{'='*65}")

    # Find iterations for target N
    if target_N is None:
        target_N = 10_000_000

    for iters in range(5, 50):
        n_pred = predict_chain_length(chain_name, iters)
        if n_pred >= target_N:
            break

    print(f"  Generating {chain_name} chain ({iters} iters -> N~{n_pred:,})...")
    t0 = time.perf_counter()
    seq = generate_substitution_sequence(chain_name, iters)
    points, L = sequence_to_points(seq, chain_name)
    N = len(points)
    rho = N / L
    mean_sp = 1.0 / rho
    print(f"  N = {N:,}, L = {L:.1f}, rho = {rho:.8f}, mean_spacing = {mean_sp:.8f}  [{time.perf_counter()-t0:.1f}s]")

    # Verify rho against theoretical value
    if chain_name == 'silver':
        rho_theory = 1.0 / 2.0  # = 0.5 exactly
        print(f"  rho (theory) = {rho_theory:.8f}, rho (computed) = {rho:.8f}, diff = {abs(rho-rho_theory):.2e}")

    # Use exact event-driven algorithm
    # R range: [mean_sp, R_max] where R_max covers many oscillation periods
    R_max = min(L / 8.0, 50000 * mean_sp)
    R_array = np.linspace(mean_sp * 0.5, R_max, num_exact_R)
    n_oscillations = (R_max - mean_sp * 0.5) / mean_sp
    print(f"  R range: [0.5, {R_max/mean_sp:.0f}] mean spacings ({n_oscillations:.0f} oscillation periods)")
    print(f"  Computing exact variance at {num_exact_R} R values...")

    t0 = time.perf_counter()
    variances, means = compute_variance_exact_batch(points, L, R_array)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Lambda-bar: simple mean of last 2/3
    start = num_exact_R // 3
    lb_mean = np.mean(variances[start:])
    lb_std  = np.std(variances[start:])
    lb_sem  = lb_std / np.sqrt(num_exact_R - start)

    # More robust: Monte Carlo-style subsampling to get error
    # Split into 10 blocks, get block means
    n_blocks = 10
    block_size = (num_exact_R - start) // n_blocks
    block_means = []
    for b in range(n_blocks):
        b0 = start + b * block_size
        b1 = start + (b+1) * block_size
        block_means.append(np.mean(variances[b0:b1]))
    block_means = np.array(block_means)
    lb_block = np.mean(block_means)
    lb_block_err = np.std(block_means) / np.sqrt(n_blocks)

    # Running average (cumulative mean)
    running = np.cumsum(variances) / np.arange(1, num_exact_R+1)
    lb_running_final = running[-1]

    # Trapezoid integral
    try:
        lb_trapz = np.trapezoid(variances[start:], R_array[start:]) / (R_array[-1] - R_array[start])
    except AttributeError:
        lb_trapz = np.trapz(variances[start:], R_array[start:]) / (R_array[-1] - R_array[start])

    print(f"\n  Results for {chain['name']}:")
    print(f"    Lambda-bar (simple mean):  {lb_mean:.8f} +/- {lb_sem:.2e} (SEM)")
    print(f"    Lambda-bar (block avg):    {lb_block:.8f} +/- {lb_block_err:.2e} (10 blocks)")
    print(f"    Lambda-bar (running avg):  {lb_running_final:.8f}")
    print(f"    Lambda-bar (trapz):        {lb_trapz:.8f}")
    print(f"    Variance std:              {lb_std:.6f}")
    print(f"    Variance range:            [{variances[start:].min():.6f}, {variances[start:].max():.6f}]")

    return {
        'N': N, 'rho': rho, 'L': L, 'mean_sp': mean_sp,
        'lb_mean': lb_mean, 'lb_sem': lb_sem,
        'lb_block': lb_block, 'lb_block_err': lb_block_err,
        'lb_running': lb_running_final,
        'lb_trapz': lb_trapz,
        'R_array': R_array, 'variances': variances, 'running': running,
        'block_means': block_means,
    }


# =========================================================
# Analytical derivation attempt
# For Silver chain, compute Lambda-bar analytically
# =========================================================

def analytical_silver():
    """
    Attempt analytical computation of Lambda-bar for Silver chain.

    The number variance for a substitution tiling can be computed
    as a sum over Bragg peaks:
        sigma^2(R) = sum_k |w_k|^2 * |sum_j exp(i*k*x_j)|^2 / N
    where w_k is the window function.

    For Class I, Lambda-bar = mean_{R} sigma^2(R).

    Alternative: use the pair correlation function directly.
    For a Silver chain, the pair distances r take values:
        r = p * S + q * L = p + q*(1+sqrt(2))  for non-negative integers p, q
    The probability of finding a pair at distance r depends on the
    substitution matrix frequencies.

    This is a complex calculation. Instead, let's use the known result
    that Lambda-bar = (1/2) * integral_0^inf |h(r)| * r dr / some_normalization
    or equivalently the structural formula.

    From Torquato-Stillinger 2003 / Zachary 2009, for d=1:
    Lambda_bar = B_N where B_N = -(phi/1) * int_0^inf h(r) r dr ... No.

    The correct formula (d=1, phi=1, D=1/rho):
    B_N = -(phi * kappa(d)) / (D * v(D/2)) * int h(r) r dr  [eq 22, d=1]
    where kappa(1) = c(1)/2 = 1, v(D/2) = D (length in 1D)
    => B_N = -(1 * 1) / (D * D) * int_0^inf h(r) r dr
           = -(1/D^2) * int_0^inf h(r) r dr
    and Lambda_bar = 2 * phi * B_N = 2 * (-1/D^2) * int h(r) r dr
                   = -2/D^2 * int h(r) r dr

    For the integer lattice:
    rho = 1, D = 1, h(r) = sum_{n!=0} delta(r-n) - 1 (so that h(r)->0 for r->inf)
    Actually for lattice: g_2(r) = sum_{n=1}^inf [delta(r-n) + delta(r+n)] / rho
    and h(r) = g_2(r) - 1 = sum_{n!=0} delta(r-n)/rho - 1

    Int_0^inf h(r) r dr = -Int_0^inf r dr - sum_{n=1}^inf n*1/rho
    This diverges, so we use the regularized version with Gaussian:

    B_N = lim_{beta->0+} [phi/(2*beta) - (1/2) * sum_{n=1}^inf Z_n r_n exp(-beta r_n^2) / rho * rho]
        = lim_{beta->0+} [phi/(2*beta) - (1/2) * sum_{pairs, i!=j} r_{ij} exp(-beta r_{ij}^2) / N]

    For integer lattice (rho=1, phi=1, D=1):
    B_N = lim_{beta->0+} [1/(2*beta) - sum_{n=1}^inf n exp(-beta n^2)]

    The sum sum_{n=1}^inf n*exp(-beta*n^2) is related to the Jacobi theta function.
    Specifically:
    sum_{n=1}^inf n*exp(-beta*n^2) = -(1/2) * d/d(-beta) sum_{n=1}^inf exp(-beta*n^2)
                                   = (1/2) * d/dbeta sum_{n=1}^inf exp(-beta*n^2)
    For small beta: sum_{n=1}^inf exp(-beta*n^2) ~ sqrt(pi/(4*beta)) - 1/2 + ...
    So (1/2) * d/dbeta [sqrt(pi/4)*beta^{-1/2}] = (1/2) * sqrt(pi/4) * (-1/2) * beta^{-3/2}
                                                  = -sqrt(pi)/(8*beta^{3/2})
    But 1/(2*beta) grows slower than 1/beta^{3/2}, so the limit involves cancellation...

    Actually the Zachary formula works differently. Let me just trust the simulation.
    """
    print("\n  Analytical attempt for Silver (rho=0.5, mu=1+sqrt(2)):")
    print("  Silver has EXACTLY rho=0.5 (2 points per unit cell of length 2)")
    print("  Tile lengths: S=1, L=1+sqrt(2)=2.4142...")
    print("  Tile frequencies: f_S = 1/(2+sqrt(2)), f_L = (1+sqrt(2))/(2+sqrt(2))")

    sqrt2 = np.sqrt(2)
    f_S = 1/(2+sqrt2)
    f_L = (1+sqrt2)/(2+sqrt2)
    mean_tile = f_S * 1 + f_L * (1+sqrt2)
    rho_check = 1/mean_tile

    print(f"  f_S = {f_S:.8f}, f_L = {f_L:.8f}")
    print(f"  Mean tile length = {mean_tile:.8f} (expected 2)")
    print(f"  rho = 1/mean_tile = {rho_check:.8f} (expected 0.5)")

    print("\n  For the Silver chain, B_N = lim_{beta->0+} F(beta)")
    print("  F(beta) = phi/(2*beta) - (1/2) * (1/N) * sum_{i!=j} r_ij * exp(-beta*r_ij^2)")
    print("  phi = rho * D_char = 0.5 * 2 = 1")
    print("  D_char = mean spacing = 1/rho = 2")
    print("")
    print("  Key question: is B_N = 1/8 exactly? => Lambda_bar = 2*B_N = 1/4?")
    print("")
    print("  For integer lattice: B_N = 1/12, Lambda_bar = 2*1/12 = 1/6 (check!)")
    print("  If Silver Lambda_bar = 1/4, then B_N = 1/8.")
    print("  The question reduces to: does the Silver theta-series sum give B_N=1/8?")

    # The Zachary formula for B_N of an integer lattice uses:
    # B_N = 1/12 from the exact result 2*phi*B_N = 1/6 => B_N = 1/(2*6) = 1/12.
    # Their formula phi*B_N = 0.08333 => with phi=1, B_N = 0.08333 = 1/12. CHECK.
    # For Silver: if phi*B_N = 1/8 * phi = 1/8 * 1 = 0.125
    # Their Table 1 shows phi*B_N = 0.08333 for Z, 0.10055 for Fibonacci.
    # Silver would be 0.12500 = 1/8 if Lambda_bar = 1/4.

    print(f"\n  phi*B_N for integer lattice = 1/12 = {1/12:.8f}")
    print(f"  phi*B_N for Fibonacci (Zachary Table 1) = 0.10055")
    print(f"  phi*B_N for Silver (if 1/4) = 1/8 = {1/8:.8f}")
    print(f"  Pattern: 1/12, 0.10055, 1/8 = 0.08333, 0.10055, 0.12500")
    print(f"  Differences: +0.01722, +0.02445")
    print(f"  Not obviously a simple pattern like 1/12, 1/10, 1/8 (= 1/(12-2k))...")

    # Check: is there a formula like phi*B_N = (mu - 1)/(2*(mu + 1))?
    # Lattice: mu->inf: phi*B_N -> 1/2 (wrong, should be 1/12)
    # Try phi*B_N = 1/(4*(mu+1)/(mu-1)) = (mu-1)/(4*(mu+1)):
    # Lattice: (inf-1)/(4*(inf+1)) -> 1/4 (wrong)
    # Try phi*B_N = 1/(12) independent of chain (not true - Fibonacci differs)

    # Let's check if there's a formula phi*B_N = (mu-1)/(2*(mu+1)^2) or similar
    for mu_name, mu in [('Lattice (mu=inf)', np.inf), ('Fibonacci', (1+np.sqrt(5))/2), ('Silver', 1+np.sqrt(2)), ('Bronze', (3+np.sqrt(13))/2)]:
        known_BN = {'Lattice (mu=inf)': 1/12, 'Fibonacci': 0.10055, 'Silver': 0.125, 'Bronze': 0.141}
        if mu != np.inf:
            # Try various formulas
            f1 = 1/(12)
            f2 = (mu - 1) / (4 * (mu + 1))  # wrong but check
            f3 = 1 / (4 * mu)
            f4 = mu / (4 * (mu**2 + 1))
            f5 = (mu + 1) / (4 * (mu**2 + mu + 1))
            f6 = 1 / (4 * (mu + 1/mu))
            print(f"  {mu_name} (mu={mu:.4f}): known~{known_BN.get(mu_name, '?'):.5f}, "
                  f"1/(4*mu)={f3:.5f}, mu/(4*(mu^2+1))={f4:.5f}, "
                  f"(mu+1)/(4*(mu^2+mu+1))={f5:.5f}, 1/(4*(mu+1/mu))={f6:.5f}")


# =========================================================
# Figure generation
# =========================================================

def make_precision_figure(results_dict):
    """Generate the precision convergence figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    colors = {'fibonacci': '#1f77b4', 'silver': '#2ca02c', 'bronze': '#d62728'}
    labels = {'fibonacci': 'Fibonacci (n=1)', 'silver': 'Silver (n=2)', 'bronze': 'Bronze (n=3)'}

    # Panel 1: Variance curve for Silver, zoomed
    ax = axes[0, 0]
    if 'silver' in results_dict:
        r = results_dict['silver']
        ms = r['mean_sp']
        # Show first 200 mean spacings
        mask = r['R_array'] <= 200 * ms
        ax.plot(r['R_array'][mask] / ms, r['variances'][mask],
                '-', color='#2ca02c', lw=0.5, alpha=0.8)
        ax.axhline(0.25, color='red', ls='--', lw=2, label='1/4 = 0.25000')
        ax.axhline(r['lb_mean'], color='orange', ls='-', lw=1.5,
                   label=f"Mean = {r['lb_mean']:.6f}")
        ax.set_xlabel('R / mean spacing')
        ax.set_ylabel(r'$\sigma^2(R)$')
        ax.set_title(f"Silver Variance (N={r['N']:,})")
        ax.legend(fontsize=9)
        ax.grid(True, ls=':', alpha=0.4)

    # Panel 2: Running average convergence for all chains
    ax = axes[0, 1]
    for name in ['fibonacci', 'silver', 'bronze']:
        if name in results_dict:
            r = results_dict[name]
            ms = r['mean_sp']
            ax.plot(r['R_array'] / ms, r['running'],
                    '-', color=colors[name], lw=1.5, label=f"{labels[name]}: {r['lb_running']:.5f}")
    ax.axhline(1/4, color='black', ls='--', lw=1.5, alpha=0.7, label='1/4')
    ax.axhline(0.20110, color='gray', ls=':', lw=1, alpha=0.7, label='Fib. exact')
    ax.set_xlabel('R / mean spacing')
    ax.set_ylabel(r'Running $\bar{\Lambda}(R)$')
    ax.set_title('Running Average Convergence')
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)
    ax.set_ylim([0.18, 0.30])

    # Panel 3: Silver - deviation from 1/4 (convergence plot)
    ax = axes[0, 2]
    if 'silver' in results_dict:
        r = results_dict['silver']
        ms = r['mean_sp']
        delta = r['running'] - 0.25
        ax.plot(r['R_array'] / ms, delta, '-', color='#2ca02c', lw=1)
        ax.axhline(0, color='red', ls='--', lw=2, label='1/4 exactly')
        ax.fill_between([0, r['R_array'][-1]/ms], [-0.001, -0.001], [0.001, 0.001],
                        alpha=0.2, color='red', label='+/- 0.001')
        ax.set_xlabel('R / mean spacing')
        ax.set_ylabel(r'Running $\bar{\Lambda}(R) - 1/4$')
        ax.set_title('Silver: Deviation from 1/4')
        ax.legend(fontsize=9)
        ax.grid(True, ls=':', alpha=0.4)
        ax.set_xlim([50, None])

    # Panel 4: Block mean convergence for Silver
    ax = axes[1, 0]
    if 'silver' in results_dict:
        r = results_dict['silver']
        blocks = r['block_means']
        x_blocks = np.arange(1, len(blocks)+1)
        ax.bar(x_blocks, blocks, color='#2ca02c', alpha=0.7)
        ax.axhline(0.25, color='red', ls='--', lw=2, label='1/4')
        ax.axhline(r['lb_block'], color='orange', ls='-', lw=2,
                   label=f"Mean = {r['lb_block']:.6f} +/- {r['lb_block_err']:.4f}")
        ax.set_xlabel('Block index')
        ax.set_ylabel(r'Block $\bar{\Lambda}$')
        ax.set_title('Silver: Block Average of sigma^2(R)')
        ax.legend(fontsize=9)
        ax.set_ylim([0.20, 0.30])
        ax.grid(True, ls=':', alpha=0.4)

    # Panel 5: All three chains - late-R variance distribution
    ax = axes[1, 1]
    for name in ['fibonacci', 'silver', 'bronze']:
        if name in results_dict:
            r = results_dict[name]
            start = len(r['variances']) // 3
            var_late = r['variances'][start:]
            ax.hist(var_late, bins=50, alpha=0.5, color=colors[name],
                    label=f"{name}: mean={np.mean(var_late):.4f}", density=True)
    ax.set_xlabel(r'$\sigma^2(R)$')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of sigma^2(R) (last 2/3 of R range)')
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)

    # Panel 6: Summary table
    ax = axes[1, 2]
    ax.axis('off')

    lines = ["PRECISION SUMMARY", "="*42, ""]
    for name in ['fibonacci', 'silver', 'bronze']:
        if name in results_dict:
            r = results_dict[name]
            lines.append(f"{labels[name]}:")
            lines.append(f"  N = {r['N']:,}")
            lines.append(f"  lb_mean  = {r['lb_mean']:.7f} +/- {r['lb_sem']:.1e}")
            lines.append(f"  lb_block = {r['lb_block']:.7f} +/- {r['lb_block_err']:.1e}")
            lines.append(f"  lb_trapz = {r['lb_trapz']:.7f}")
            lines.append("")

    lines.append("Reference values:")
    lines.append(f"  Lattice:   1/6    = {1/6:.7f}")
    lines.append(f"  Fibonacci: 0.2011 (Zachary 2009)")
    lines.append(f"  1/4      : {0.25:.7f}")
    lines.append("")
    lines.append("Silver deviation from 1/4:")
    if 'silver' in results_dict:
        r = results_dict['silver']
        diff = r['lb_mean'] - 0.25
        n_sigma = abs(diff) / r['lb_sem'] if r['lb_sem'] > 0 else 0
        lines.append(f"  diff = {diff:.5f}")
        lines.append(f"  = {n_sigma:.1f} sigma from 1/4")
        lines.append(f"  {'Consistent with 1/4' if n_sigma < 2 else 'NOT 1/4 (>2 sigma)'}")

    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Results', fontsize=12)

    plt.suptitle('High-Precision Lambda-bar: Silver = 1/4?', fontsize=14, y=1.01)
    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, 'fig_silver_precision.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outpath}")
    plt.close()


# =========================================================
# Main
# =========================================================

if __name__ == '__main__':
    print("="*70)
    print("  Deep Precision Lambda-bar Analysis for Metallic Mean Chains")
    print("="*70)

    # Run analytical reasoning
    analytical_silver()

    # ---- High-precision simulation for Silver ----
    # Use exact event-driven variance (much more accurate than Monte Carlo)
    # Target N = 9.4M (18 iterations, already generated)
    silver_r = precision_lambda_bar('silver', target_N=5_000_000, num_exact_R=5000)

    # ---- Fibonacci for cross-check ----
    fib_r = precision_lambda_bar('fibonacci', target_N=5_000_000, num_exact_R=5000)

    # ---- Bronze ----
    bronze_r = precision_lambda_bar('bronze', target_N=5_000_000, num_exact_R=3000)

    results = {
        'fibonacci': fib_r,
        'silver': silver_r,
        'bronze': bronze_r,
    }

    # Final summary
    print("\n" + "="*70)
    print("  FINAL PRECISION RESULTS")
    print("="*70)

    names = ['fibonacci', 'silver', 'bronze']
    for name in names:
        r = results[name]
        chain = CHAINS[name]
        print(f"\n  {chain['name']}:")
        print(f"    N = {r['N']:,}  rho = {r['rho']:.8f}")
        print(f"    Lambda-bar = {r['lb_mean']:.7f} +/- {r['lb_sem']:.2e} (SEM)")
        print(f"    Lambda-bar = {r['lb_block']:.7f} +/- {r['lb_block_err']:.2e} (block avg)")
        print(f"    Lambda-bar (trapz) = {r['lb_trapz']:.7f}")

    print(f"\n  Reference: 1/4 = {0.25:.7f}")
    if 'silver' in results:
        r = results['silver']
        diff = r['lb_mean'] - 0.25
        n_sigma = abs(diff) / max(r['lb_block_err'], 1e-8)
        print(f"\n  Silver deviation from 1/4: {diff:+.5f} ({n_sigma:.1f} sigma)")
        if n_sigma < 2.0:
            print("  => CONSISTENT with Lambda-bar = 1/4 exactly")
        else:
            print(f"  => INCONSISTENT with 1/4 at {n_sigma:.1f} sigma")

    # Check candidate expressions for Bronze
    print(f"\n  Bronze Lambda-bar = {bronze_r['lb_mean']:.6f}")
    print(f"  Candidates:")
    bval = bronze_r['lb_mean']
    candidates = [
        ('9/32',         9/32),
        ('17/60',        17/60),
        ('sqrt(2)/5',   np.sqrt(2)/5),
        ('11/39',       11/39),
        ('(sqrt(13)-1)/8', (np.sqrt(13)-1)/8),
        ('sqrt(13)/12', np.sqrt(13)/12),
        ('3/(4+2*sqrt(13))', 3/(4+2*np.sqrt(13))),
        ('(3+sqrt(13))/24', (3+np.sqrt(13))/24),
    ]
    for expr, val in candidates:
        diff = abs(bval - val)
        flag = " <--- POSSIBLE MATCH" if diff < 3e-4 else ("  (close)" if diff < 1e-3 else "")
        print(f"    {expr:30s} = {val:.7f}  diff = {diff:.2e}{flag}")

    # Make figure
    make_precision_figure(results)

    print("\nDone.")
