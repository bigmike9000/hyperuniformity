# Hyperuniformity in Quasiperiodic Point Patterns — Computational Reference

> **Project:** Junior Paper under Prof. Salvatore Torquato, Princeton University
> **Scope:** Compute hyperuniformity scaling exponent α for 1D and 2D metallic-mean quasicrystals via number variance and diffusion spreadability
> **Key references:** Oğuz et al. (2019), Hitin-Bialus et al. (2024), Torquato (2021), Wang & Torquato (2022)

---

## TABLE OF CONTENTS

1. [Core Definitions & Classification](#1-core-definitions--classification)
2. [Number Variance σ²(R) — Theory & Computation](#2-number-variance-σr--theory--computation)
3. [1D Substitution Tiling Generation](#3-1d-substitution-tiling-generation)
4. [Cut-and-Project (Projection) Method](#4-cut-and-project-projection-method)
5. [Surface-Area Coefficient Λ̄](#5-surface-area-coefficient-λ̄)
6. [Two-Phase Media Mapping](#6-two-phase-media-mapping)
7. [Spectral Density χ̃_V(k)](#7-spectral-density-χ̃_vk)
8. [Diffusion Spreadability S(t)](#8-diffusion-spreadability-st)
9. [2D Extensions — Generalized Dual Method](#9-2d-extensions--generalized-dual-method)
10. [Validation Benchmarks & Expected Results](#10-validation-benchmarks--expected-results)
11. [Implementation Notes & Numerical Pitfalls](#11-implementation-notes--numerical-pitfalls)

---

## 1. Core Definitions & Classification

### 1.1 Hyperuniformity

A point pattern in d dimensions is **hyperuniform** if the number variance σ²_N(R) within a spherical observation window of radius R grows slower than the window volume R^d as R → ∞:

```
lim_{R→∞} σ²_N(R) / v₁(R) = 0
```

where v₁(R) ~ R^d is the volume of a d-dimensional sphere of radius R.

Equivalently, the structure factor S(k) vanishes as wavenumber k → 0:

```
lim_{|k|→0} S(k) = 0
```

### 1.2 Scaling Exponent α

For systems where S(k) varies as a power law near the origin:

```
S(k) ~ B |k|^α    as k → 0,   with B, α > 0
```

The exponent α determines the large-R behavior of σ²_N(R).

### 1.3 Hyperuniformity Classes (in d dimensions)

| Class | α range | Variance scaling (large R) | Examples |
|-------|---------|---------------------------|----------|
| **I** (strong) | α > 1 | σ² ~ R^{d-1} (bounded in 1D) | Crystals, Fibonacci chain, Penrose tiling |
| **II** (logarithmic) | α = 1 | σ² ~ R^{d-1} ln(R) | Period-doubling chain, certain non-substitution quasicrystals |
| **III** (weak) | 0 < α < 1 | σ² ~ R^{d-α} | 0222 limit-periodic chain |
| **Non-hyperuniform** | α = 0 | σ² ~ R^d | Poisson process, typical liquids |
| **Anti-hyperuniform** | α < 0 | σ² grows faster than R^d | Critical systems |

### 1.4 Integrated Fourier Intensity Z(k)

For 1D systems with dense, discontinuous Bragg peaks (like quasicrystals), α is better defined through the integrated intensity:

```
Z(k) = 2 ∫₀ᵏ S(q) dq
```

Then α is defined by:

```
Z(k) ~ k^{1+α}    as k → 0
```

This is the proper definition for quasiperiodic systems where S(k) is not a smooth function.

---

## 2. Number Variance σ²(R) — Theory & Computation

### 2.1 Definition

For a 1D point pattern with number density ρ, place an observation window (interval) of length 2R at a random position x. Count the number of points N(x, R) inside. The number variance is:

```
σ²_N(R) = ⟨N(x,R)²⟩ - ⟨N(x,R)⟩²
```

where ⟨·⟩ denotes averaging over the position x.

### 2.2 Relation to Structure Factor (1D)

```
σ²_N(R) = (ρ / π) ∫₀^∞ S(k) |ω̃(k; R)|² dk
```

where the window function Fourier transform is:

```
ω̃(k; R) = 2 sin(kR) / k
```

Equivalently, using the integrated intensity Z(k):

```
σ²_N(R) = (2ρR / π) ∫₀^∞ Z(k) [sin(2kR) / (2kR)]' dk
```

(See Oğuz et al. 2017, Eq. 2.6 for the full expression.)

### 2.3 Computational Algorithm — Sliding Window Method

This is the primary method for Phase 3 of the project.

**Input:** Array of point positions `x[0], x[1], ..., x[N-1]` sorted in ascending order, total system length L.

**Algorithm:**
```python
def compute_variance(positions, R_values, L):
    """
    Compute σ²(R) via sliding window with periodic boundary conditions.

    Parameters:
        positions: sorted array of N point positions in [0, L)
        R_values: array of window radii to evaluate
        L: system length (for periodic boundaries)

    Returns:
        variance: array of σ²(R) values
    """
    N = len(positions)
    rho = N / L
    variance = np.zeros(len(R_values))

    for i, R in enumerate(R_values):
        window_length = 2 * R
        if window_length >= L:
            # Window covers entire system — variance is 0
            variance[i] = 0.0
            continue

        # For each point, count how many points fall in [x - R, x + R]
        # Use continuous sliding: analytically track count changes
        # as window center moves from 0 to L

        # Method: For each gap between consecutive points,
        # the count in the window is constant.
        # Track count changes as window edges cross points.

        # EFFICIENT APPROACH:
        # Sort entry/exit events for the window
        counts = []
        # Sample M uniformly spaced window positions
        M = max(10000, 10 * N)  # oversample
        centers = np.linspace(0, L, M, endpoint=False)
        for c in centers:
            left = (c - R) % L
            right = (c + R) % L
            if left < right:
                count = np.searchsorted(positions, right) - np.searchsorted(positions, left)
            else:  # wraps around
                count = (N - np.searchsorted(positions, left)) + np.searchsorted(positions, right)
            counts.append(count)

        counts = np.array(counts, dtype=float)
        variance[i] = np.var(counts)

    return variance
```

**IMPORTANT — Exact method (preferred for large N):**

For substitution tilings with N ~ 10^7 points, the sampling approach above is too slow. Instead, use the **exact event-driven method**:

```python
def compute_variance_exact(positions, R, L):
    """
    Exact σ²(R) by tracking all count-change events.

    As the window center sweeps from 0 to L, the count changes
    only when the left or right edge of the window crosses a point.
    Each point x_j creates two events:
      - A point enters the window when center = (x_j - R) mod L
      - A point exits the window when center = (x_j + R) mod L

    Returns: exact σ²(R)
    """
    N = len(positions)

    # Create sorted list of events: (position, +1 for enter, -1 for exit)
    events = []
    for xj in positions:
        enter_pos = (xj - R) % L   # point enters window
        exit_pos = (xj + R) % L    # point exits window
        events.append((enter_pos, +1))
        events.append((exit_pos, -1))

    events.sort(key=lambda e: e[0])

    # Walk through events, tracking count and accumulating
    # weighted first and second moments
    # Between consecutive events, count is constant
    # Weight = gap length / L

    total_mean = 0.0
    total_sq = 0.0
    count = 0

    # Initialize count at position 0
    # Count = number of points in window centered at 0
    # i.e., points in [L - R, R] (mod L)
    for xj in positions:
        if (xj % L) < R or (xj % L) > (L - R):
            count += 1

    prev_pos = 0.0
    for pos, delta in events:
        gap = pos - prev_pos
        if gap < 0:
            gap += L
        weight = gap / L
        total_mean += count * weight
        total_sq += count**2 * weight
        count += delta
        prev_pos = pos

    # Final gap from last event back to 0 (or L)
    gap = L - prev_pos
    weight = gap / L
    total_mean += count * weight
    total_sq += count**2 * weight

    variance = total_sq - total_mean**2
    return variance
```

**Complexity:** O(N log N) per R value (dominated by sorting).

### 2.4 Poisson Benchmark (Phase 1 Validation)

For a 1D Poisson process with density ρ and periodic boundary conditions:

```
σ²_Poisson(R) = 2ρR    for R ≪ L/2
```

This is the **exact** analytical result. Use this to validate your variance computation code.

**Generation:** To create a Poisson point pattern with N expected points in [0, L):
```python
import numpy as np
N_actual = np.random.poisson(N)
positions = np.sort(np.random.uniform(0, L, N_actual))
```

**Validation criteria:** The computed σ²(R) should match 2ρR to within statistical fluctuations (which decrease as N increases). For N = 10^6, expect relative error < 0.1% for R values well below L/2.

---

## 3. 1D Substitution Tiling Generation

### 3.1 Substitution Matrix Framework

A 1D substitution tiling with two tile types (Short S, Long L) is defined by:

1. **Substitution matrix M:** A 2×2 matrix with non-negative integer entries
2. **Substitution rule:** How each tile type maps to a sequence of tiles
3. **Tile length ratio ξ = L/S:** Must be preserved by the substitution

The substitution matrix acts on the column vector (n_S, n_L)^T giving tile counts after one iteration.

### 3.2 Eigenvalue Formula for α

**THIS IS THE KEY THEORETICAL RESULT** (Oğuz et al. 2019, Eq. 28):

For a substitution matrix M with eigenvalues λ₁ (largest) and λ₂ (second largest):

```
α = 1 - 2 ln|λ₂| / ln|λ₁|
```

**Conditions:**
- System is hyperuniform (α > 0) when |λ₂| < |λ₁|^{1/2}
- System is anti-hyperuniform (α < 0) when |λ₂| > |λ₁|^{1/2}
- Result holds for both PV (Pisot-Vijayaraghavan) and non-PV cases
- For PV cases (|λ₂| < 1), the tiling is quasiperiodic with pure Bragg spectrum

### 3.3 Metallic Mean Chains

The metallic mean chains use substitution matrices of the form:

```
M = [[0, 1], [1, n]]    for n = 1, 2, 3, ...
```

with substitution rules S → L, L → L^n S (i.e., n copies of L followed by one S).

| Chain | n | Matrix | λ₁ | λ₂ | det(M) | α (exact) |
|-------|---|--------|----|----|--------|-----------|
| **Fibonacci** (golden) | 1 | [[0,1],[1,1]] | τ = (1+√5)/2 ≈ 1.618 | -1/τ ≈ -0.618 | -1 | 3 |
| **Silver** (0112) | 2 | [[0,1],[1,2]] | 1+√2 ≈ 2.414 | 1-√2 ≈ -0.414 | -1 | 3 |
| **Bronze** | 3 | [[0,1],[1,3]] | (3+√13)/2 ≈ 3.303 | (3-√13)/2 ≈ -0.303 | -1 | 3 |

**Critical observation:** For ALL metallic mean chains, det(M) = -1, so |λ₂| = 1/λ₁. Therefore:

```
α = 1 - 2 ln(1/λ₁) / ln(λ₁) = 1 - 2(-1) = 3
```

**All metallic mean chains have α = 3 (Class I, strongly hyperuniform).**

This is the main theoretical prediction to verify computationally.

### 3.4 Tile Length Ratio

The substitution must preserve the ratio ξ = L/S. This requires (from Oğuz et al. Eq. 3.7):

```
ξ = (n + √(n² + 4)) / 2    (the metallic mean itself)
```

| Chain | ξ (exact) | ξ (decimal) |
|-------|-----------|-------------|
| Fibonacci | (1+√5)/2 = τ | 1.6180339... |
| Silver | 1+√2 | 2.4142135... |
| Bronze | (3+√13)/2 | 3.3027756... |

### 3.5 Substitution Generation Algorithm

```python
def generate_substitution_chain(n_metallic, num_iterations, seed='L'):
    """
    Generate a 1D metallic-mean substitution chain.

    Parameters:
        n_metallic: 1 for Fibonacci, 2 for silver, 3 for bronze
        num_iterations: number of substitution iterations
        seed: starting tile ('S' or 'L')

    Returns:
        tiles: string of 'S' and 'L' characters
        positions: array of point positions (left endpoints of each tile)
    """
    # Substitution rules: S → L, L → L^n S
    def substitute(tile_string):
        result = []
        for tile in tile_string:
            if tile == 'S':
                result.append('L')
            elif tile == 'L':
                result.extend(['L'] * n_metallic + ['S'])
        return result

    # Iterate
    tiles = list(seed)
    for _ in range(num_iterations):
        tiles = substitute(tiles)

    # Compute positions
    xi = (n_metallic + np.sqrt(n_metallic**2 + 4)) / 2  # metallic mean
    S_length = 1.0  # set S = 1
    L_length = xi    # L = ξ·S

    positions = [0.0]
    for tile in tiles[:-1]:  # N tiles → N+1 endpoints, but we want N points
        if tile == 'S':
            positions.append(positions[-1] + S_length)
        else:
            positions.append(positions[-1] + L_length)

    total_length = positions[-1] + (S_length if tiles[-1] == 'S' else L_length)

    return ''.join(tiles), np.array(positions), total_length
```

**Decoration convention:** Place one point at the left endpoint of each tile. This gives N points for N tiles.

**Scaling:** After m iterations starting from a single tile:
- Number of tiles: N_m ~ λ₁^m
- Total length: L_m ~ λ₁^m · (const)
- Number density: ρ = N_m / L_m → constant as m → ∞

For the Fibonacci chain specifically:
```
ρ = τ / (1 + τ) = τ / τ² = 1/τ ≈ 0.618    (with S = 1)
```

More generally, ρ = (component of left eigenvector) · (decoration vector) / (component of left eigenvector) · (length vector). See Oğuz et al. Eq. 3.14.

### 3.6 Growth Rate & Practical Sizes

| Iterations | Fibonacci N | Silver N | Bronze N |
|-----------|------------|---------|---------|
| 10 | 144 | 2,378 | 53,798 |
| 15 | 1,597 | 195,025 | ~2.7×10⁶ |
| 20 | 17,711 | 15,994,428 | ~1.7×10⁸ |
| 25 | 196,418 | ~1.3×10⁹ | — |
| 30 | 2,178,309 | — | — |
| 35 | 24,157,817 | — | — |

**Target: N ~ 10^7.** For Fibonacci use ~35 iterations; for silver ~18; for bronze ~12.

---

## 4. Cut-and-Project (Projection) Method

### 4.1 Overview

Alternative to substitution for generating quasiperiodic chains. Provides an independent validation of Class I behavior.

### 4.2 Construction (Fibonacci Example)

1. Start with a 2D square lattice with basis vectors e₁ = (1,0) and e₂ = (0,1)
2. Define the "physical space" line E∥ with slope 1/τ (for Fibonacci)
3. Define the "perpendicular space" line E⊥ orthogonal to E∥
4. Define a strip of width ω centered on E∥
5. Project all lattice points within the strip onto E∥

**The projected points form the Fibonacci chain.**

### 4.3 Algorithm

```python
def fibonacci_cut_and_project(N_target, tau=None):
    """
    Generate Fibonacci chain via cut-and-project method.

    Parameters:
        N_target: approximate number of desired points
        tau: golden ratio (default: (1+√5)/2)

    Returns:
        positions: sorted array of projected point positions
    """
    if tau is None:
        tau = (1 + np.sqrt(5)) / 2

    # Physical space direction (unit vector)
    e_par = np.array([tau, 1]) / np.sqrt(tau**2 + 1)
    # Perpendicular space direction
    e_perp = np.array([-1, tau]) / np.sqrt(tau**2 + 1)

    # Strip width (CRITICAL: must be "ideal" width)
    # For Fibonacci, ideal width = tau (projected window in perp space)
    # The perpendicular-space projection of the unit cell is
    # max(|e₁·e_perp|, |e₂·e_perp|) ...
    # Actually: omega = (1 + tau) / sqrt(1 + tau^2) ... but more precisely,
    # the ideal strip width in perpendicular space is:
    omega = (1 + tau) / np.sqrt(1 + tau**2)  # = tau / sqrt(1 + tau^2) * (1 + tau)/tau ...
    # Simpler: omega = tau * (1/sqrt(1+tau^2)) + 1 * (tau/sqrt(1+tau^2))
    # = (tau + tau) / sqrt(...) ...

    # CORRECT FORMULA: The acceptance window in perpendicular space
    # has width W = |e₁·e_perp| + |e₂·e_perp|
    #             = 1/sqrt(1+tau^2) + tau/sqrt(1+tau^2)
    #             = (1+tau)/sqrt(1+tau^2) = tau^2/sqrt(1+tau^2)
    W = (1 + tau) / np.sqrt(1 + tau**2)

    # Generate lattice points in a large enough region
    # Need ~N_target points, lattice has density ~1/tau per unit length along E∥
    L_est = N_target * tau  # rough estimate of physical-space extent
    grid_range = int(np.ceil(L_est / np.sqrt(1 + tau**2) * (1 + tau))) + 10

    positions = []
    for i in range(-grid_range, grid_range + 1):
        for j in range(-grid_range, grid_range + 1):
            point = np.array([i, j], dtype=float)
            # Project onto perpendicular space
            x_perp = np.dot(point, e_perp)
            # Check if within strip
            if abs(x_perp) < W / 2:
                # Project onto physical space
                x_par = np.dot(point, e_par)
                positions.append(x_par)

    positions = np.sort(positions)

    # Center and trim to desired count
    positions -= positions[len(positions)//2]

    return positions
```

### 4.4 Critical: Ideal vs Non-Ideal Strip Width

**The strip width must be EXACTLY the ideal width.**

- **Ideal width** → Class I hyperuniformity (α > 1, variance bounded)
- **Non-ideal width** → Degrades to Class II (α = 1, variance ~ ln R)

This is a known subtlety. For validation, compare substitution and projection results.

### 4.5 General Metallic Mean Projection

For silver (n=2) and bronze (n=3) metallic means, the projection method generalizes but uses different lattice embeddings. The substitution method is more straightforward for these cases.

---

## 5. Surface-Area Coefficient Λ̄

### 5.1 Definition

For Class I hyperuniform systems in 1D, the variance σ²_N(R) is bounded and oscillates. The **surface-area coefficient** is:

```
Λ̄ = lim_{R→∞} (1/R) ∫₀ᴿ σ²_N(R') dR'
```

i.e., the time-averaged value of the variance over large R. This serves as a **scalar order metric** that ranks hyperuniform systems by their degree of order.

### 5.2 Known Values

| System | Λ̄ | Notes |
|--------|---|-------|
| 1D integer lattice | 1/6 ≈ 0.1667 | Global minimum among all 1D hyperuniform patterns |
| 1D Fibonacci chain | ~0.2 (to be computed) | Expected to be close to but larger than 1/6 |
| 1D Silver chain | ? (to be computed) | One of the project goals |
| 1D Bronze chain | ? (to be computed) | One of the project goals |
| 1D Poisson | Not defined (σ² ~ R) | Non-hyperuniform |

### 5.3 Computational Method

```python
def compute_lambda_bar(positions, R_max, num_R_points, L):
    """
    Compute Λ̄ by averaging σ²(R) over R ∈ [R_min, R_max].

    For Class I systems, σ²(R) is bounded and oscillating,
    so Λ̄ = mean of σ²(R) over a large R range.
    """
    R_values = np.linspace(1.0, R_max, num_R_points)
    variances = np.array([compute_variance_exact(positions, R, L) for R in R_values])

    # Λ̄ = (1/R_max) ∫₀^{R_max} σ²(R') dR' ≈ mean(σ²)
    lambda_bar = np.trapz(variances, R_values) / (R_values[-1] - R_values[0])

    return lambda_bar, R_values, variances
```

### 5.4 Verification Strategy

- For the integer lattice (ρ=1, positions at 0,1,2,...,N-1): computed Λ̄ should converge to 1/6
- For Fibonacci: σ²(R) should be bounded (piecewise quadratic segments) with no secular growth
- All metallic mean chains should give Class I behavior (bounded σ²)

---

## 6. Two-Phase Media Mapping

### 6.1 Motivation

Quasicrystals have dense, discontinuous Bragg peaks in S(k), making direct extraction of α impossible. The workaround: convert the point pattern to a two-phase medium and use the **spectral density** χ̃_V(k) instead, which is a smooth function.

### 6.2 Procedure

1. Place identical non-overlapping solid inclusions (rods in 1D, disks in 2D) centered at each point
2. Phase 2 = solid (interior of rods/disks), volume fraction φ₂
3. Phase 1 = void (everything else), volume fraction φ₁ = 1 - φ₂

### 6.3 Packing Fractions

From Hitin-Bialus et al. (2024):
- **1D:** φ₂ = 0.35 (rods of half-length a centered at each point, so 2a·ρ = 0.35)
- **2D:** φ₂ = 0.25 (disks of radius a centered at each vertex)

The rod half-length (1D) is:
```
a = φ₂ / (2ρ)
```

### 6.4 Indicator Function

Define the indicator function of phase 2:

```
I₂(x) = 1  if x is inside any rod/disk
I₂(x) = 0  otherwise
```

The two-point correlation function is:
```
S₂(r) = ⟨I₂(x) I₂(x + r)⟩
```

The autocovariance function is:
```
χ_V(r) = S₂(r) - φ₂²
```

The spectral density is the Fourier transform:
```
χ̃_V(k) = FT[χ_V(r)]
```

---

## 7. Spectral Density χ̃_V(k)

### 7.1 Relation to Structure Factor

For a packing of identical inclusions centered at point positions {x_j}, the spectral density is:

```
χ̃_V(k) = ρ |m̃(k)|² S(k)
```

where:
- ρ is the number density
- m̃(k) is the Fourier transform of the single-inclusion indicator function
- S(k) is the structure factor of the point pattern

### 7.2 Single-Inclusion Form Factor

**1D (rod of half-length a):**
```
m̃(k) = 2 sin(ka) / k = 2a sinc(ka/π)    [using sinc(x) = sin(πx)/(πx)]
```

Actually, more precisely:
```
m̃(k) = 2a · sin(ka) / (ka)    (this equals 2a when k=0)
```

**2D (disk of radius a):**
```
m̃(k) = 2πa J₁(ka) / k
```
where J₁ is the Bessel function of the first kind.

### 7.3 Computational Algorithm for 1D Spectral Density

```python
def compute_spectral_density_1d(positions, a, k_values, L):
    """
    Compute χ̃_V(k) for a 1D packing of rods of half-length a.

    Parameters:
        positions: array of N point positions in [0, L)
        a: rod half-length (φ₂ = 2aρ)
        k_values: array of wavenumbers to evaluate
        L: system length (periodic)

    Returns:
        chi_V: spectral density at each k value
    """
    N = len(positions)
    rho = N / L

    chi_V = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        if abs(k) < 1e-15:
            # k = 0: χ̃_V(0) = 0 for hyperuniform systems
            chi_V[i] = 0.0
            continue

        # Structure factor via direct computation
        # S(k) = (1/N) |Σ_j exp(-i k x_j)|²
        phase_sum = np.sum(np.exp(-1j * k * positions))
        S_k = np.abs(phase_sum)**2 / N

        # Form factor of rod
        m_tilde_k = 2 * np.sin(k * a) / k

        # Spectral density
        chi_V[i] = rho * np.abs(m_tilde_k)**2 * S_k

    return chi_V
```

**WARNING:** For quasicrystals, S(k) consists of dense Bragg peaks. The direct computation above samples S(k) at specific k values and will miss most peaks. This is fine for computing the spectral density at specific k values, but you cannot naively "scan" k to see the full peak structure.

### 7.4 Efficient S(k) via FFT

For large N, use FFT-based approach:

```python
def compute_Sk_fft(positions, L, num_k):
    """
    Compute S(k) using histogram + FFT approach.

    Bin point positions onto a fine grid, FFT, then |FFT|²/N.
    """
    # Create density histogram
    n_bins = max(num_k * 4, len(positions) * 4)  # oversample
    hist, edges = np.histogram(positions, bins=n_bins, range=(0, L))

    # FFT (real-to-complex)
    rho_k = np.fft.rfft(hist.astype(float))

    # S(k) = |ρ(k)|² / N
    N = len(positions)
    S_k = np.abs(rho_k)**2 / N

    # Corresponding k values
    dk = 2 * np.pi / L
    k_values = np.arange(len(S_k)) * dk

    return k_values, S_k
```

### 7.5 Small-k Behavior

For the two-phase medium derived from metallic mean chains (all with α = 3):

```
χ̃_V(k) ~ |k|^α = |k|³    as k → 0
```

This is what the spreadability method will extract.

---

## 8. Diffusion Spreadability S(t)

### 8.1 Physical Setup

Consider a two-phase medium. Initially, solute is distributed uniformly in phase 2 (the rods/disks) and absent from phase 1 (the void). Both phases have the same diffusion coefficient D. The spreadability S(t) is the fraction of total solute present in phase 1 at time t.

### 8.2 Exact Formula (Fourier Space)

**In d dimensions** (Torquato 2021, Phys. Rev. E 104, 054102):

```
S(∞) - S(t) = (d · ωd) / ((2π)^d · φ₂) × ∫₀^∞ k^{d-1} χ̃_V(k) exp(-k² D t) dk
```

where ωd = π^{d/2} / Γ(d/2 + 1) is the volume of a d-dimensional unit sphere.

**In 1D (d=1):**

```
S(∞) - S(t) = (1) / (π · φ₂) × ∫₀^∞ χ̃_V(k) exp(-k² D t) dk
```

with S(∞) = φ₁ = 1 - φ₂.

**In 2D (d=2):**

```
S(∞) - S(t) = (1) / (π · φ₂) × ∫₀^∞ k · χ̃_V(k) exp(-k² D t) dk
```

### 8.3 Long-Time Asymptotic Scaling

**Key result:** When χ̃_V(k) ~ B |k|^α as k → 0, the long-time excess spreadability scales as:

```
S(∞) - S(t) ~ C · t^{-(d+α)/2}    as t → ∞
```

where C is a constant depending on B, d, φ₂, and D.

### 8.4 Extracting α via Logarithmic Derivative

Define the **effective exponent**:

```
n(t) = -d ln[S(∞) - S(t)] / d ln(t)
```

At long times, n(t) → (d + α)/2. So:

```
α = 2·n(∞) - d
```

**Wang-Torquato fitting algorithm** (Phys. Rev. Appl. 17, 034022, 2022):

1. Compute S(t) numerically at logarithmically spaced time points
2. Compute the excess spreadability S(∞) - S(t)
3. Take the logarithmic derivative: plot ln[S(∞) - S(t)] vs ln(t)
4. Fit the slope at long times → this gives -(d+α)/2
5. Extract α

### 8.5 Computational Algorithm

```python
def compute_spreadability_1d(chi_V_k, k_values, D, t_values, phi2):
    """
    Compute excess spreadability S(∞) - S(t) for a 1D two-phase medium.

    Parameters:
        chi_V_k: spectral density evaluated at k_values
        k_values: array of wavenumbers (uniformly spaced, k ≥ 0)
        D: diffusion coefficient (set to 1 for simplicity)
        t_values: array of time values (use log spacing)
        phi2: volume fraction of phase 2

    Returns:
        excess_S: array of S(∞) - S(t) values
    """
    dk = k_values[1] - k_values[0] if len(k_values) > 1 else 1.0

    excess_S = np.zeros(len(t_values))
    for i, t in enumerate(t_values):
        # Numerical integration: (1/(π φ₂)) ∫ χ̃_V(k) exp(-k²Dt) dk
        integrand = chi_V_k * np.exp(-k_values**2 * D * t)
        excess_S[i] = np.trapz(integrand, k_values) / (np.pi * phi2)

    return excess_S


def extract_alpha_1d(t_values, excess_S, t_min_fit=None, t_max_fit=None):
    """
    Extract α from long-time spreadability decay.

    In 1D: S(∞) - S(t) ~ t^{-(1+α)/2}
    So log(excess_S) vs log(t) has slope -(1+α)/2
    Therefore: α = -2·slope - 1
    """
    log_t = np.log(t_values)
    log_excess = np.log(excess_S)

    # Select fitting range (long times only)
    if t_min_fit is None:
        t_min_fit = t_values[len(t_values)//2]
    if t_max_fit is None:
        t_max_fit = t_values[-1]

    mask = (t_values >= t_min_fit) & (t_values <= t_max_fit) & (excess_S > 0)

    # Linear regression on log-log plot
    slope, intercept = np.polyfit(log_t[mask], log_excess[mask], 1)

    alpha = -2 * slope - 1  # For 1D: slope = -(1+α)/2

    return alpha, slope
```

### 8.6 Practical Considerations

- **Set D = 1** without loss of generality (just rescales time)
- **Time range:** Use logarithmically spaced points from t ~ 10^{-3} to t ~ 10^6
- **k resolution:** Need enough k-points to resolve the spectral density. For N points in a box of length L, use k_max = π·N/L and dk = 2π/L.
- **Truncation:** Hitin-Bialus et al. show that truncating the small-k region (removing the first few k values) still gives accurate α. This is useful because the smallest k values have the poorest statistics.

### 8.7 Expected Results

For all metallic mean chains in 1D with φ₂ = 0.35:

```
S(∞) - S(t) ~ t^{-(1+3)/2} = t^{-2}
```

So the log-log slope should be -2, giving α = 3.

For the 2D Penrose tiling with φ₂ = 0.25:

```
S(∞) - S(t) ~ t^{-(2+6)/2} = t^{-4}
```

giving α ≈ 6 (Hitin-Bialus et al. found α = 5.97 ± 0.06).

---

## 9. 2D Extensions — Generalized Dual Method

### 9.1 Overview

2D quasiperiodic tilings with n-fold symmetry can be generated using the **Generalized Dual Method (GDM)**, also known as the de Bruijn multigrid method.

### 9.2 GDM Construction

1. Choose n sets of parallel lines (grids), each with a different orientation θ_j = πj/n for j = 0, 1, ..., n-1
2. Each grid j has lines at positions: x · n_j = m + γ_j, where m is an integer and γ_j is a phase shift
3. The line intersections define a tiling of the plane
4. Vertices of the tiling form the quasiperiodic point pattern

### 9.3 Targets

| Tiling | Rotational symmetry | Metallic mean | α (predicted/known) |
|--------|-------------------|---------------|---------------------|
| Penrose | 5-fold | Golden (τ) | ~6 (measured: 5.97 ± 0.06) |
| Ammann-Beenker | 8-fold | Silver (1+√2) | To be determined |
| Bronze equivalent | To be determined | Bronze ((3+√13)/2) | To be determined |

### 9.4 Implementation Notes

The GDM is well-documented in:
- Socolar, Steinhardt, Levine (1985) Phys. Rev. B 32, 5547
- Gähler & Rhyner (1986) J. Phys. A 19, 267

For computational implementation, the key steps are:
1. Generate the multigrid (n families of parallel lines with specified phase shifts)
2. Find all pairwise intersections between lines from different families
3. Map each intersection to a vertex in the tiling
4. Remove duplicates and compile vertex list

This is computationally more involved than 1D substitution and is marked as Phase 5 (future).

---

## 10. Validation Benchmarks & Expected Results

### 10.1 Phase 1: Poisson Benchmark

| Quantity | Expected | Tolerance |
|----------|----------|-----------|
| σ²(R) for Poisson | 2ρR | < 1% relative for N ≥ 10^5, R < L/4 |

### 10.2 Phase 2-3: 1D Metallic Mean Chains

| Quantity | Fibonacci | Silver | Bronze |
|----------|-----------|--------|--------|
| α (from eigenvalue formula) | 3 (exact) | 3 (exact) | 3 (exact) |
| σ²(R) behavior | Bounded, oscillating | Bounded, oscillating | Bounded, oscillating |
| Hyperuniformity class | I | I | I |
| Λ̄ | > 1/6 (compute) | > 1/6 (compute) | > 1/6 (compute) |
| Λ̄ ranking | Compute | Compute | Compute |

**Key check:** σ²(R) should NOT grow with R. If it does, there's a bug. Plot σ²(R) over several decades of R.

**Λ̄ for integer lattice:** Must equal 1/6 exactly. This serves as a validation of the Λ̄ computation.

### 10.3 Phase 4: Spreadability Validation

| Quantity | Expected for metallic means (1D) |
|----------|--------------------------------|
| Long-time slope of ln[S(∞)-S(t)] vs ln(t) | -2.0 |
| Extracted α | 3.0 ± 0.02 |

**Fibonacci benchmark from Hitin-Bialus et al.:** They achieved α within 0.02% of the exact value of 3.

### 10.4 Consistency Checks

1. **Substitution vs Projection:** For Fibonacci, both methods should give identical σ²(R) (up to finite-size effects)
2. **Density check:** Computed ρ should match theoretical ρ = (left eigenvector component)
3. **Tile count ratio:** n_L / n_S → λ₁ component ratio as iterations → ∞

---

## 11. Implementation Notes & Numerical Pitfalls

### 11.1 Periodic Boundary Conditions

- Substitution chains are NOT inherently periodic. For variance computation with PBC, you need to either:
  - Use a very large chain and only compute σ²(R) for R ≪ L/2
  - OR explicitly wrap the chain into a periodic box (but this introduces a discontinuity at the boundary — avoid unless carefully handled)

**Recommendation:** Generate a chain much larger than needed and restrict analysis to R < L/4.

### 11.2 Finite-Size Effects

- σ²(R) must be computed for R ≪ L/2 to avoid finite-size artifacts
- The Λ̄ computation should use R_max ~ L/10 to be safe
- For spreadability, the minimum meaningful k is k_min = 2π/L

### 11.3 Numerical Precision

- Point positions should be computed in double precision (float64)
- For very large chains (N > 10^7), memory may be an issue — store only positions, not the full tile string
- The spectral density integral in the spreadability formula should use adaptive quadrature or very fine k-grid

### 11.4 Parallelization Opportunities

- σ²(R) at different R values are independent → trivially parallelizable
- S(k) computation: the sum Σ exp(-ikx_j) can be parallelized or done via FFT
- Spreadability at different t values are independent

### 11.5 File Organization Suggestion

```
project/
├── KNOWLEDGE.md              ← this file
├── src/
│   ├── poisson_benchmark.py  ← Phase 1
│   ├── substitution.py       ← Generate metallic-mean chains
│   ├── projection.py         ← Cut-and-project (Fibonacci)
│   ├── variance.py           ← σ²(R) computation
│   ├── lambda_bar.py         ← Λ̄ computation
│   ├── spectral_density.py   ← χ̃_V(k) computation
│   ├── spreadability.py      ← S(t) and α extraction
│   └── utils.py              ← Shared utilities
├── tests/
│   ├── test_poisson.py       ← Validates σ² = 2ρR
│   ├── test_lattice.py       ← Validates Λ̄ = 1/6
│   └── test_fibonacci.py     ← Cross-validates substitution vs projection
├── results/
│   └── ...
└── figures/
    └── ...
```

### 11.6 Key Python Dependencies

```
numpy          # Core numerics
scipy          # Integration, special functions (Bessel for 2D)
matplotlib     # Plotting
numba          # JIT compilation for hot loops (variance computation)
```

### 11.7 Performance Targets

| Computation | N = 10^5 | N = 10^6 | N = 10^7 |
|------------|---------|---------|---------|
| Generate substitution chain | < 1s | < 10s | < 2min |
| σ²(R) at single R (exact method) | < 1s | < 5s | < 1min |
| σ²(R) at 1000 R values | < 10min | < 1hr | ~10hr |
| Spectral density (FFT, all k) | < 1s | < 5s | < 1min |
| Spreadability (100 t values) | < 1s | < 10s | < 5min |

For N = 10^7 variance computations, consider using Numba JIT:

```python
from numba import njit

@njit
def variance_at_R_numba(positions, R, L):
    # ... same logic as compute_variance_exact but JIT-compiled
    pass
```

---

## REFERENCES

1. Torquato & Stillinger (2003). "Local density fluctuations, hyperuniformity, and order metrics." Phys. Rev. E 68, 041113.
2. Torquato (2018). "Hyperuniform states of matter." Physics Reports 745, 1-95.
3. Oğuz, Socolar, Steinhardt & Torquato (2019). "Hyperuniformity and anti-hyperuniformity in one-dimensional substitution tilings." Acta Cryst. A 75, 3-13.
4. Torquato (2021). "Diffusion spreadability as a probe of the microstructure of complex media across length scales." Phys. Rev. E 104, 054102.
5. Wang & Torquato (2022). "Dynamic measure of hyperuniformity and nonhyperuniformity in heterogeneous media via the diffusion spreadability." Phys. Rev. Appl. 17, 034022.
6. Hitin-Bialus, Maher, Steinhardt & Torquato (2024). "Hyperuniformity classes of quasiperiodic tilings via diffusion spreadability." Phys. Rev. E 109, 064108.
7. Oğuz, Socolar, Steinhardt & Torquato (2017). "Hyperuniformity of quasicrystals." Phys. Rev. B 95, 054119.
8. Socolar, Steinhardt & Levine (1985). "Quasicrystals with arbitrary orientational symmetry." Phys. Rev. B 32, 5547.
9. Maher, Wang, Shi, Jiao & Torquato (2025). "Precise determination of the long-time asymptotics of the diffusion spreadability." arXiv:2602.17873.
