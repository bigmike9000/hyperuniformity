"""
Phase 4: Two-Phase Media and Diffusion Spreadability

Constructs two-phase media from 1D point patterns by placing non-overlapping
solid rods on each point. Computes the spectral density chi_V(k) and the
diffusion spreadability S(t), then extracts the hyperuniformity exponent alpha
via logarithmic derivative.

Theory (1D):
  - Rod half-length a = phi2 / (2*rho), packing fraction phi2 = 0.35
  - Structure factor S(k) via histogram binning + FFT
  - Spectral density: chi_V(k) = rho * |m_tilde(k)|^2 * S(k)
    where m_tilde(k) = 2*sin(k*a)/k is the rod form factor
  - Excess spreadability: E(t) = (dk/(pi*phi2)) * sum chi_V(k) * exp(-k^2*D*t)
  - Alpha extraction: alpha(t) = -2 * d(ln E)/d(ln t) - 1 -> 3 for quasicrystals

References:
  Torquato (2021) PhysRevE 104
  Hitin-Bialus et al. (2024) PhysRevE 109
"""

import numpy as np


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
        Defaults to next power of 2 >= 2*L (gives bin width <= 0.5).

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

    # S(k) = |F[n]|^2 / N for n >= 1 (skip DC at n=0)
    n_max = M // 2
    S_k = np.abs(F[1:n_max + 1]) ** 2 / N
    k_array = 2 * np.pi * np.arange(1, n_max + 1) / L

    return k_array, S_k


def compute_spectral_density(k_array, S_k, rho, a):
    """
    Compute spectral density chi_V(k) for a two-phase medium of rods.

    chi_V(k) = rho * |m_tilde(k)|^2 * S(k)
    where m_tilde(k) = 2*sin(k*a)/k is the 1D rod form factor.

    Parameters
    ----------
    k_array : ndarray
        Wavevectors (positive).
    S_k : ndarray
        Structure factor at those wavevectors.
    rho : float
        Number density of points.
    a : float
        Rod half-length.

    Returns
    -------
    chi_V : ndarray
        Spectral density at each wavevector.
    """
    form_factor_sq = (2 * np.sin(k_array * a) / k_array) ** 2
    chi_V = rho * form_factor_sq * S_k
    return chi_V


def compute_excess_spreadability(k_array, chi_V, phi2, t_array, D=1.0):
    """
    Compute excess spreadability E(t) = S(inf) - S(t).

    E(t) = (dk / (pi * phi2)) * sum_n chi_V(k_n) * exp(-k_n^2 * D * t)

    Parameters
    ----------
    k_array : ndarray
        Wavevectors (uniformly spaced, dk = k[1] - k[0]).
    chi_V : ndarray
        Spectral density at those wavevectors.
    phi2 : float
        Packing fraction of phase 2.
    t_array : ndarray
        Time values at which to evaluate.
    D : float
        Diffusion coefficient (default 1.0).

    Returns
    -------
    E_t : ndarray
        Excess spreadability at each time value.
    """
    dk = k_array[1] - k_array[0]
    k2 = k_array ** 2
    prefactor = dk / (np.pi * phi2)

    E_t = np.empty(len(t_array))
    for i, t in enumerate(t_array):
        E_t[i] = prefactor * np.sum(chi_V * np.exp(-k2 * D * t))

    return E_t


def compute_lattice_spreadability(phi2, t_array, D=1.0, n_bragg=50):
    """
    Compute E(t) for the integer lattice analytically from Bragg peak positions.

    The integer lattice (spacing d=1, rho=1) has Bragg peaks at k = 2*pi*n.
    Using the exact formula avoids histogram binning artifacts that would
    create a spurious noise floor in S(k).

    E(t) = (2*rho/phi2) * sum_{n=1}^{n_bragg} |m_tilde(2*pi*n)|^2 * exp(-(2*pi*n)^2*D*t)

    Parameters
    ----------
    phi2 : float
        Packing fraction.
    t_array : ndarray
        Time values.
    D : float
        Diffusion coefficient (default 1.0).
    n_bragg : int
        Number of Bragg peaks to include (default 50).

    Returns
    -------
    E_t : ndarray
        Excess spreadability at each time value.
    """
    rho = 1.0
    a = phi2 / (2 * rho)
    prefactor = 2 * rho / phi2

    E_t = np.zeros(len(t_array))
    for n in range(1, n_bragg + 1):
        k = 2 * np.pi * n
        m_sq = (2 * np.sin(k * a) / k) ** 2
        E_t += prefactor * m_sq * np.exp(-k**2 * D * t_array)

    return E_t


def extract_alpha(t_array, excess_spread, window=10):
    """
    Extract effective hyperuniformity exponent alpha(t) via smoothed log derivative.

    n(t) = -d ln[E(t)] / d ln(t)
    alpha(t) = 2*n(t) - 1

    Uses a sliding window linear regression for robust slope estimation,
    avoiding point-by-point noise amplification from np.gradient on
    discrete Bragg peak spectra.

    Parameters
    ----------
    t_array : ndarray
        Time values (must be positive).
    excess_spread : ndarray
        E(t) values (must be positive for log transform).
    window : int
        Half-width of sliding window for slope estimation (default 10).

    Returns
    -------
    alpha_t : ndarray
        Effective exponent at each time value.
    """
    n = len(t_array)
    mask = excess_spread > 0
    log_t = np.log(t_array)
    log_E = np.log(np.maximum(excess_spread, 1e-300))

    alpha_t = np.full(n, np.nan)

    for i in range(n):
        if not mask[i]:
            continue
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        w_mask = mask[lo:hi]
        if np.sum(w_mask) < 3:
            continue
        lt = log_t[lo:hi][w_mask]
        le = log_E[lo:hi][w_mask]
        slope = np.polyfit(lt, le, 1)[0]
        alpha_t[i] = -2 * slope - 1

    return alpha_t


def extract_alpha_period_aware(t_array, excess_spread, period, n_periods=2):
    """
    Extract alpha(t) using a window matched to the oscillation period.

    For quasicrystalline spreadability, E(t) oscillates in log(t) space
    with period 2*ln(mu), where mu is the metallic mean. Matching the
    sliding window to an integer number of periods cancels the oscillatory
    contribution, yielding a much smoother alpha(t) curve.

    Parameters
    ----------
    t_array : ndarray
        Time values (must be positive, log-spaced).
    excess_spread : ndarray
        E(t) values.
    period : float
        Oscillation period in ln(t) space (= 2*ln(mu)).
    n_periods : int
        Number of full periods to span (default 2).

    Returns
    -------
    alpha_t : ndarray
        Effective exponent at each time value.
    """
    n = len(t_array)
    mask = excess_spread > 0
    log_t = np.log(t_array)
    log_E = np.log(np.maximum(excess_spread, 1e-300))

    half_width = n_periods * period / 2.0  # half-width in log(t) space

    alpha_t = np.full(n, np.nan)

    for i in range(n):
        if not mask[i]:
            continue
        center = log_t[i]
        in_window = (log_t >= center - half_width) & (log_t <= center + half_width) & mask
        if np.sum(in_window) < 5:
            continue
        lt = log_t[in_window]
        le = log_E[in_window]
        slope = np.polyfit(lt, le, 1)[0]
        alpha_t[i] = -2 * slope - 1

    return alpha_t


def extract_alpha_fit(t_array, excess_spread, t_min=1e2, t_max=1e5):
    """
    Extract a single alpha value via linear fit of log(E) vs log(t).

    More robust than computing a pointwise log derivative and averaging,
    because the fit uses all data points simultaneously and is less
    sensitive to oscillations.

    E(t) ~ t^{-(1+alpha)/2}  =>  ln(E) = -(1+alpha)/2 * ln(t) + const

    Parameters
    ----------
    t_array : ndarray
        Time values.
    excess_spread : ndarray
        E(t) values.
    t_min, t_max : float
        Time window for the fit.

    Returns
    -------
    alpha : float
        Best-fit hyperuniformity exponent.
    r_squared : float
        Coefficient of determination (R^2) of the fit.
    """
    mask = (t_array >= t_min) & (t_array <= t_max) & (excess_spread > 0)
    if np.sum(mask) < 5:
        return np.nan, 0.0

    lt = np.log(t_array[mask])
    le = np.log(excess_spread[mask])

    coeffs = np.polyfit(lt, le, 1)
    slope = coeffs[0]
    alpha = -2 * slope - 1

    # R^2
    le_pred = np.polyval(coeffs, lt)
    ss_res = np.sum((le - le_pred) ** 2)
    ss_tot = np.sum((le - np.mean(le)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return alpha, r_squared
