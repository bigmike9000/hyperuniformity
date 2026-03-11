"""
1D Perturbed Lattice Point Patterns

Generates point patterns by displacing each site of the integer lattice Z
by an i.i.d. random draw from a chosen distribution f. The hyperuniformity
class depends on the small-k behavior of the characteristic function f_tilde(k).

Supported displacement distributions:
  1. Uniform[-a/2, a/2)  (URL model)
     - f_tilde(k) = sin(ka/2)/(ka/2)
     - S(k) = 1 - |f_tilde(k)|^2 + |f_tilde(k)|^2 * S_lattice(k)
     - alpha = 2 (Class I) -- strongest for i.i.d. perturbations
     - At a=1 (cloaked): no Bragg peaks, Lambda_bar = 1/3
     - At a=0: unperturbed lattice, Lambda_bar = 1/6

  2. Gaussian(0, sigma^2)
     - f_tilde(k) = exp(-sigma^2 k^2 / 2)
     - alpha = 2 (Class I)
     - Bragg peaks always present (exponentially damped but never zero)

  3. Cauchy(0, gamma)
     - f_tilde(k) = exp(-gamma |k|)
     - alpha = 1 (Class II)
     - 1st moment finite, 2nd moment diverges

Theory:
  S(k) = 1 - |f_tilde(k)|^2 + |f_tilde(k)|^2 * S_lattice(k)

  The diffuse (continuous) part is 1 - |f_tilde(k)|^2, always non-negative.
  The Bragg part is |f_tilde(k)|^2 evaluated at reciprocal lattice vectors
  k = 2*pi*n (n integer). For the cloaked URL (a=integer), sinc vanishes
  at all reciprocal lattice points and Bragg peaks are eliminated entirely.

References:
  Klatt, Kim, Torquato (2020) PhysRevE 101, 032118
  Kim, Torquato (2018) PhysRevB 97, 054105
  Gabrielli, Joyce, Torquato (2008) PhysRevE 77, 031125
"""

import numpy as np
import time


# ============================================================
# Point pattern generators
# ============================================================

def generate_perturbed_uniform(N, a, rng=None):
    """
    Generate a 1D perturbed lattice with uniform displacements.

    Each integer site n = 0, 1, ..., N-1 is displaced by an independent
    draw from Uniform[-a/2, a/2). Points are wrapped to [0, N) via modulo.

    Parameters
    ----------
    N : int
        Number of lattice sites.
    a : float
        Width of the uniform displacement distribution. a=0 gives the
        unperturbed lattice; a=1 is the "cloaked" case with no Bragg peaks.
    rng : numpy.random.Generator, optional
        Random number generator. If None, a new default generator is used.

    Returns
    -------
    points : ndarray
        Point positions in [0, N), sorted.
    L : float
        Domain length (equal to N).
    """
    if rng is None:
        rng = np.random.default_rng()

    sites = np.arange(N, dtype=np.float64)
    displacements = rng.uniform(-a / 2, a / 2, size=N)
    points = (sites + displacements) % N
    points.sort()
    return points, float(N)


def generate_perturbed_gaussian(N, sigma, rng=None):
    """
    Generate a 1D perturbed lattice with Gaussian displacements.

    Each integer site n = 0, 1, ..., N-1 is displaced by an independent
    draw from N(0, sigma^2). Points are wrapped to [0, N) via modulo.

    Parameters
    ----------
    N : int
        Number of lattice sites.
    sigma : float
        Standard deviation of the Gaussian displacement distribution.
    rng : numpy.random.Generator, optional
        Random number generator. If None, a new default generator is used.

    Returns
    -------
    points : ndarray
        Point positions in [0, N), sorted.
    L : float
        Domain length (equal to N).
    """
    if rng is None:
        rng = np.random.default_rng()

    sites = np.arange(N, dtype=np.float64)
    displacements = rng.normal(0.0, sigma, size=N)
    points = (sites + displacements) % N
    points.sort()
    return points, float(N)


def generate_perturbed_cauchy(N, gamma, rng=None):
    """
    Generate a 1D perturbed lattice with Cauchy displacements.

    Each integer site n = 0, 1, ..., N-1 is displaced by an independent
    draw from Cauchy(0, gamma). Points are wrapped to [0, N) via modulo.

    The Cauchy distribution has finite 1st moment but divergent 2nd moment,
    yielding alpha = 1 (Class II hyperuniform) instead of alpha = 2.

    Parameters
    ----------
    N : int
        Number of lattice sites.
    gamma : float
        Scale parameter (half-width at half-maximum) of the Cauchy distribution.
    rng : numpy.random.Generator, optional
        Random number generator. If None, a new default generator is used.

    Returns
    -------
    points : ndarray
        Point positions in [0, N), sorted.
    L : float
        Domain length (equal to N).
    """
    if rng is None:
        rng = np.random.default_rng()

    sites = np.arange(N, dtype=np.float64)
    # Cauchy via inverse CDF: x = gamma * tan(pi * (u - 0.5))
    u = rng.uniform(0, 1, size=N)
    displacements = gamma * np.tan(np.pi * (u - 0.5))
    points = (sites + displacements) % N
    points.sort()
    return points, float(N)


def generate_perturbed_stable(N, stability, scale=1.0, rng=None):
    """
    Generate a 1D perturbed lattice with symmetric stable displacements.

    Each integer site n = 0, 1, ..., N-1 is displaced by an independent
    draw from a symmetric stable distribution with stability index s
    and scale parameter c.

    The characteristic function is f_tilde(k) = exp(-c^s * |k|^s), giving:
      S(k) ~ |k|^s  as k -> 0
    so alpha = s.

    Special cases:
      s = 2: Gaussian (Class I, alpha = 2)
      s = 1: Cauchy (Class II, alpha = 1)
      0 < s < 1: Class III (0 < alpha < 1)

    Uses the Chambers-Mallows-Stuck algorithm for stable RV generation.

    Parameters
    ----------
    N : int
        Number of lattice sites.
    stability : float
        Stability index s in (0, 2]. Controls the tail heaviness.
        s < 1 gives Class III hyperuniformity with alpha = s.
    scale : float
        Scale parameter c > 0.
    rng : numpy.random.Generator, optional
        Random number generator. If None, a new default generator is used.

    Returns
    -------
    points : ndarray
        Point positions in [0, N), sorted.
    L : float
        Domain length (equal to N).
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (0 < stability <= 2):
        raise ValueError(f"stability must be in (0, 2], got {stability}")

    sites = np.arange(N, dtype=np.float64)

    # Chambers-Mallows-Stuck algorithm for symmetric stable RVs (beta=0)
    s = stability
    V = rng.uniform(-np.pi / 2, np.pi / 2, size=N)
    W = rng.exponential(1.0, size=N)

    if s == 1.0:
        # Cauchy case
        displacements = scale * np.tan(V)
    else:
        # General symmetric stable
        displacements = scale * (
            np.sin(s * V) / np.cos(V) ** (1.0 / s)
            * (np.cos((1.0 - s) * V) / W) ** ((1.0 - s) / s)
        )

    points = (sites + displacements) % N
    points.sort()
    return points, float(N)


# ============================================================
# Exact Lambda_bar for the URL model (Klatt et al. 2020)
# ============================================================

def lambda_bar_url_exact(a):
    """
    Exact Lambda_bar for the uniformly randomized lattice (URL).

    Formula from Klatt, Kim, Torquato (2020), Eq. (B7):
      Lambda_bar = a/3 + frac(a)^2 * (1 - frac(a))^2 / (6 * a^2)

    where frac(a) = a - floor(a) is the fractional part of a.

    Special cases:
      a = 0: Lambda_bar = 1/6 (unperturbed lattice)
      a = integer (cloaked): Lambda_bar = a/3  (since frac(a)=0)
      a = 1: Lambda_bar = 1/3

    Parameters
    ----------
    a : float
        Width of the uniform displacement distribution (a >= 0).

    Returns
    -------
    lambda_bar : float
        The exact asymptotic surface-area coefficient.
    """
    if a == 0:
        return 1.0 / 6.0

    frac_a = a - np.floor(a)
    lambda_bar = a / 3.0 + frac_a ** 2 * (1 - frac_a) ** 2 / (6.0 * a ** 2)
    return lambda_bar


# ============================================================
# Analytical structure factors
# ============================================================

def structure_factor_url_analytical(k_array, a):
    """
    Analytical S(k) for the cloaked URL (a = integer).

    When a is an integer, all Bragg peaks vanish and:
      S(k) = 1 - sinc^2(k*a / (2*pi))

    where sinc(x) = sin(pi*x)/(pi*x) is the normalized sinc function.

    For a=1 specifically: S(k) = 1 - sinc^2(k/(2*pi)).

    This is exact for the continuous part; when a is not an integer, Bragg
    peaks are present and this function returns only the diffuse envelope.

    Parameters
    ----------
    k_array : ndarray
        Wavevector values (positive).
    a : int or float
        Width of the uniform displacement. Function is exact when a is integer.

    Returns
    -------
    S_k : ndarray
        Analytical structure factor.
    """
    # f_tilde(k) = sin(k*a/2) / (k*a/2)  for a > 0
    # |f_tilde(k)|^2 = sinc^2(k*a/(2*pi)) using numpy sinc(x) = sin(pi*x)/(pi*x)
    if a == 0:
        # Unperturbed lattice: S(k) is a sum of Bragg peaks
        # Return zero for the diffuse part
        return np.zeros_like(k_array)

    ft_sq = np.sinc(k_array * a / (2 * np.pi)) ** 2
    S_k = 1.0 - ft_sq
    return S_k


def structure_factor_analytical(k_array, dist_type, param):
    """
    Analytical diffuse part of S(k) for a perturbed lattice.

    S_diffuse(k) = 1 - |f_tilde(k)|^2

    This is the continuous (non-Bragg) contribution to S(k). For the full
    S(k), Bragg peaks at k = 2*pi*n (weighted by |f_tilde(2*pi*n)|^2) must
    be added separately.

    Parameters
    ----------
    k_array : ndarray
        Wavevector values (positive).
    dist_type : str
        One of 'uniform', 'gaussian', 'cauchy', 'stable'.
    param : float or tuple
        Distribution parameter: a for uniform, sigma for Gaussian,
        gamma for Cauchy, (stability, scale) tuple for stable.

    Returns
    -------
    S_diffuse : ndarray
        Diffuse part of the structure factor.
    """
    if dist_type == 'uniform':
        a = param
        if a == 0:
            return np.zeros_like(k_array)
        # f_tilde(k) = sin(ka/2)/(ka/2)
        ft_sq = np.sinc(k_array * a / (2 * np.pi)) ** 2
    elif dist_type == 'gaussian':
        sigma = param
        # f_tilde(k) = exp(-sigma^2 k^2 / 2)
        ft_sq = np.exp(-sigma ** 2 * k_array ** 2)
    elif dist_type == 'cauchy':
        gamma = param
        # f_tilde(k) = exp(-gamma |k|)
        ft_sq = np.exp(-2 * gamma * np.abs(k_array))
    elif dist_type == 'stable':
        # param is (stability_index, scale)
        s, c = param
        # f_tilde(k) = exp(-c^s |k|^s)
        ft_sq = np.exp(-2 * c ** s * np.abs(k_array) ** s)
    else:
        raise ValueError(f"Unknown dist_type: {dist_type!r}. "
                         f"Must be 'uniform', 'gaussian', 'cauchy', or 'stable'.")

    S_diffuse = 1.0 - ft_sq
    return S_diffuse


# ============================================================
# Numerical S(k) via FFT (reuses two_phase_media approach)
# ============================================================

def compute_structure_factor(points, L, M=None):
    """
    Compute S(k) via histogram binning + real FFT.

    Parameters
    ----------
    points : ndarray
        1D point positions in [0, L).
    L : float
        Domain length.
    M : int, optional
        Number of histogram bins (must be power of 2).
        Defaults to next power of 2 >= 2*L.

    Returns
    -------
    k_array : ndarray
        Wavevectors k_n = 2*pi*n/L for n = 1, ..., M//2.
    S_k : ndarray
        Structure factor S(k_n) = |F[n]|^2 / N.
    """
    N = len(points)
    if M is None:
        M = int(2 ** np.ceil(np.log2(2 * L)))

    h = np.histogram(points, bins=M, range=(0, L))[0]
    F = np.fft.rfft(h.astype(np.float64))

    n_max = M // 2
    S_k = np.abs(F[1:n_max + 1]) ** 2 / N
    k_array = 2 * np.pi * np.arange(1, n_max + 1) / L

    return k_array, S_k


# ============================================================
# Number variance and Lambda_bar (local copy for self-contained use)
# ============================================================

def compute_number_variance_1d(points, L, R_array, num_windows=20000, rng=None):
    """
    Compute sigma^2(R) using sorted-array binary search with periodic BCs.

    Parameters
    ----------
    points : ndarray
        1D point positions (will be sorted internally).
    L : float
        Domain length.
    R_array : ndarray
        Array of window half-widths.
    num_windows : int
        Number of random window centers per R value.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    variances : ndarray
        Variance of the point count in a window of half-width R.
    """
    if rng is None:
        rng = np.random.default_rng()

    sorted_pts = np.sort(points)
    N = len(sorted_pts)
    variances = np.empty(len(R_array))

    for j, R in enumerate(R_array):
        centers = rng.uniform(0, L, num_windows)
        lo = centers - R
        hi = centers + R

        idx_hi = np.searchsorted(sorted_pts, hi, side='right')
        idx_lo = np.searchsorted(sorted_pts, lo, side='left')
        counts = (idx_hi - idx_lo).astype(np.float64)

        # Periodic wrap-left
        wrap_left = lo < 0
        if np.any(wrap_left):
            wrapped_lo = L + lo[wrap_left]
            counts[wrap_left] = (
                np.searchsorted(sorted_pts, hi[wrap_left], side='right')
                + N - np.searchsorted(sorted_pts, wrapped_lo, side='left')
            ).astype(np.float64)

        # Periodic wrap-right
        wrap_right = hi > L
        if np.any(wrap_right):
            wrapped_hi = hi[wrap_right] - L
            counts[wrap_right] = (
                N - np.searchsorted(sorted_pts, lo[wrap_right], side='left')
                + np.searchsorted(sorted_pts, wrapped_hi, side='right')
            ).astype(np.float64)

        variances[j] = np.var(counts)

    return variances


def compute_lambda_bar(R_array, variances, R_min_frac=0.2):
    """
    Extract Lambda_bar as the large-R average of sigma^2(R).

    For Class I hyperuniform point patterns, sigma^2(R) oscillates around
    a bounded constant Lambda_bar as R -> infinity.

    Parameters
    ----------
    R_array : ndarray
        Window half-widths.
    variances : ndarray
        Corresponding number variances.
    R_min_frac : float
        Fraction of R_max below which data are discarded (to avoid
        finite-size transients). Default 0.2.

    Returns
    -------
    lambda_bar : float
        Mean variance in the plateau region.
    """
    R_max = R_array[-1]
    mask = R_array > R_min_frac * R_max
    return float(np.mean(variances[mask]))


# ============================================================
# Main: demonstration and verification
# ============================================================

if __name__ == '__main__':
    print('=' * 70)
    print('  Perturbed Lattice Point Patterns -- Demonstration & Verification')
    print('=' * 70)

    rng = np.random.default_rng(42)
    N = 500_000            # points for numerical checks
    num_windows = 30000    # for variance computation

    # ----------------------------------------------------------
    # 1. URL model: sweep over a values, compare Lambda_bar
    # ----------------------------------------------------------
    print('\n--- URL Model (Uniform Perturbation) ---\n')
    header = '  {:>6s}  {:>18s}  {:>16s}  {:>10s}'.format(
        'a', 'Lambda_bar (exact)', 'Lambda_bar (num)', 'Rel Error')
    print(header)
    print('  {}  {}  {}  {}'.format('-'*6, '-'*18, '-'*16, '-'*10))

    a_values = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    url_results = {}

    for a in a_values:
        exact = lambda_bar_url_exact(a)

        if a == 0.0:
            # Unperturbed lattice: just use integer points
            points = np.arange(N, dtype=np.float64)
            L = float(N)
        else:
            points, L = generate_perturbed_uniform(N, a, rng=rng)

        rho = N / L
        R_max = min(200 / rho, L / 4)
        R_array = np.linspace(1.0 / rho, R_max, 500)

        variances = compute_number_variance_1d(points, L, R_array,
                                               num_windows=num_windows, rng=rng)
        numerical = compute_lambda_bar(R_array, variances)

        rel_err = abs(numerical - exact) / exact if exact > 0 else 0.0
        url_results[a] = {'exact': exact, 'numerical': numerical, 'rel_err': rel_err}

        print('  {:6.2f}  {:18.6f}  {:16.6f}  {:9.1%}'.format(
            a, exact, numerical, rel_err))

    # Key check: a=0 should give 1/6, a=1 should give 1/3
    print('\n  Verification: a=0 exact = {:.6f} (expected 1/6 = {:.6f})'.format(
        lambda_bar_url_exact(0), 1/6))
    print('  Verification: a=1 exact = {:.6f} (expected 1/3 = {:.6f})'.format(
        lambda_bar_url_exact(1), 1/3))

    # ----------------------------------------------------------
    # 2. Gaussian perturbation: verify alpha = 2
    # ----------------------------------------------------------
    print('\n\n--- Gaussian Perturbation ---\n')
    sigma_values = [0.1, 0.2, 0.5]

    for sigma in sigma_values:
        points, L = generate_perturbed_gaussian(N, sigma, rng=rng)
        rho = N / L

        # Check S(k) near k=0 analytically: S(k) ~ sigma^2 * k^2
        k_test = np.linspace(0.01, 0.5, 100)
        S_analytical = structure_factor_analytical(k_test, 'gaussian', sigma)
        # Near k=0: 1 - exp(-sigma^2 k^2) ~ sigma^2 k^2
        S_approx = sigma ** 2 * k_test ** 2
        max_err = np.max(np.abs(S_analytical - S_approx) / np.maximum(S_analytical, 1e-30))

        # Numerical Lambda_bar
        R_max = min(200 / rho, L / 4)
        R_array = np.linspace(1.0 / rho, R_max, 500)
        variances = compute_number_variance_1d(points, L, R_array,
                                               num_windows=num_windows, rng=rng)
        numerical = compute_lambda_bar(R_array, variances)

        bragg_arg = sigma**2 * (2*np.pi)**2
        bragg_val = np.exp(-bragg_arg)
        print('  sigma = {:.1f}: Lambda_bar = {:.6f}, '
              'S(k)~sigma^2*k^2 approx error = {:.1%} (small k)'.format(
                  sigma, numerical, max_err))
        print('    Bragg damping: |f_tilde(2*pi)|^2 = exp(-{:.2f}) '
              '= {:.2e}'.format(bragg_arg, bragg_val))

    # ----------------------------------------------------------
    # 3. Cauchy perturbation: verify alpha = 1
    # ----------------------------------------------------------
    print('\n\n--- Cauchy Perturbation ---\n')
    gamma_values = [0.05, 0.1, 0.2]

    for gamma in gamma_values:
        points, L = generate_perturbed_cauchy(N, gamma, rng=rng)
        rho = N / L

        # Check S(k) near k=0 analytically: S(k) ~ 2*gamma*|k|
        k_test = np.linspace(0.01, 0.3, 100)
        S_analytical = structure_factor_analytical(k_test, 'cauchy', gamma)
        # Near k=0: 1 - exp(-2*gamma*k) ~ 2*gamma*k
        S_approx = 2 * gamma * k_test
        max_err = np.max(np.abs(S_analytical - S_approx) / np.maximum(S_analytical, 1e-30))

        # Numerical Lambda_bar -- Cauchy is Class II, variance grows as ln(R)
        R_max = min(200 / rho, L / 4)
        R_array = np.linspace(1.0 / rho, R_max, 500)
        variances = compute_number_variance_1d(points, L, R_array,
                                               num_windows=num_windows, rng=rng)
        var_growth = variances[-1] / variances[len(variances) // 4]

        bragg_arg = 2*gamma*2*np.pi
        bragg_val = np.exp(-bragg_arg)
        print('  gamma = {:.2f}: S(k)~2*gamma*|k| approx error = {:.1%} '
              '(small k)'.format(gamma, max_err))
        print('    Variance ratio (last/quarter): {:.2f} '
              '(>1 expected for Class II, sigma^2 ~ ln R)'.format(var_growth))
        print('    Bragg damping: |f_tilde(2*pi)|^2 = exp(-{:.3f}) '
              '= {:.4f}'.format(bragg_arg, bragg_val))

    # ----------------------------------------------------------
    # 4. S(k) verification: numerical vs analytical for cloaked URL
    # ----------------------------------------------------------
    print('\n\n--- S(k) Verification: Cloaked URL (a=1) ---\n')
    N_sk = 1_000_000
    points_cloaked, L_cloaked = generate_perturbed_uniform(N_sk, a=1.0, rng=rng)
    k_num, S_num = compute_structure_factor(points_cloaked, L_cloaked)

    # Compare in the range k in [0.1, 6] (away from k=0 where S->0)
    mask = (k_num > 0.1) & (k_num < 6.0)
    S_exact_cloaked = structure_factor_url_analytical(k_num[mask], a=1)
    S_num_masked = S_num[mask]

    # Bin-averaged comparison (reduce noise)
    n_bins = 50
    bin_edges = np.linspace(0.1, 6.0, n_bins + 1)
    mean_err_bins = []
    for i in range(n_bins):
        bm = (k_num[mask] >= bin_edges[i]) & (k_num[mask] < bin_edges[i + 1])
        if np.sum(bm) > 0:
            mean_err_bins.append(np.mean(np.abs(S_num_masked[bm] - S_exact_cloaked[bm])))
    mean_abs_err = np.mean(mean_err_bins)
    print('  N = {:,}, k range [0.1, 6.0]'.format(N_sk))
    print('  Mean absolute error (bin-averaged): {:.6f}'.format(mean_abs_err))
    print('  S(k) at k ~ pi: analytical = {:.6f}'.format(1 - np.sinc(0.5)**2))

    # ----------------------------------------------------------
    # 5. Summary table
    # ----------------------------------------------------------
    print('\n\n' + '=' * 70)
    print('  Summary')
    print('=' * 70)
    print('\n  {:20s}  {:>10s}  {:>6s}  {:>8s}  {:>12s}'.format(
        'Distribution', 'Parameter', 'alpha', 'Class', 'Bragg peaks?'))
    print('  {}  {}  {}  {}  {}'.format('-'*20, '-'*10, '-'*6, '-'*8, '-'*12))
    print('  {:20s}  {:>10s}  {:>6s}  {:>8s}  {:>12s}'.format(
        'Uniform (URL)', 'a=1', '2', 'I', 'No (cloaked)'))
    print('  {:20s}  {:>10s}  {:>6s}  {:>8s}  {:>12s}'.format(
        'Uniform (URL)', 'a=0.5', '2', 'I', 'Yes'))
    print('  {:20s}  {:>10s}  {:>6s}  {:>8s}  {:>12s}'.format(
        'Gaussian', 'sigma=0.2', '2', 'I', 'Yes (damped)'))
    print('  {:20s}  {:>10s}  {:>6s}  {:>8s}  {:>12s}'.format(
        'Cauchy', 'gamma=0.1', '1', 'II', 'Yes (damped)'))

    print('\n  URL Lambda_bar exact values:')
    print('    a=0 (lattice):  {:.6f}  (= 1/6)'.format(lambda_bar_url_exact(0)))
    print('    a=1 (cloaked):  {:.6f}  (= 1/3)'.format(lambda_bar_url_exact(1)))
    print('    a=0.5:          {:.6f}'.format(lambda_bar_url_exact(0.5)))
    print('    a=2 (cloaked):  {:.6f}  (= 2/3)'.format(lambda_bar_url_exact(2)))

    print('\n  Done.')
