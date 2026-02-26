# Comprehensive Ranking of Hyperuniform 1D Point Patterns — Project Context

## Principal Investigator: Professor Salvatore Torquato, Princeton University

## 1. Motivation and Scope

Hyperuniformity is a unifying framework that encompasses a remarkably broad range of
ordered and disordered systems with anomalously suppressed density fluctuations [1].
In one dimension, the known landscape of hyperuniform point patterns — particularly
for Class I ($\alpha > 1$) — remains surprisingly sparse. The table in the Physics
Reports review [1] lists only a handful of 1D examples, leaving open the question
of how diverse hyperuniform structures compare in their degree of order.

**This project aims to provide a much more comprehensive ranking of 1D hyperuniform
point configurations**, spanning two broad families:

1. **Quasicrystals** — metallic-mean substitution tilings (Fibonacci, silver, bronze)
   and other aperiodic ordered chains.
2. **Exotic disordered systems** — hyperuniform point patterns that lack long-range
   order yet still suppress large-scale density fluctuations (e.g., stealthy
   hyperuniform, maximally random jammed, cloaked systems, perturbed lattices).

The ranking is built from two complementary metrics: the **hyperuniformity exponent
$\alpha$**, which classifies how strongly fluctuations are suppressed, and the
**surface-area coefficient $\bar{\Lambda}$**, which quantifies the residual
fluctuation amplitude within a given class. Together, these provide a fine-grained
"fingerprint" for each pattern's degree of order.

A key motivation is that hyperuniform systems often exhibit unusual physical
properties — anomalous transport, photonic band gaps, mechanical rigidity — that
correlate with their hyperuniformity class and $\bar{\Lambda}$ value [1, 2]. By
establishing a comprehensive $(\alpha, \bar{\Lambda})$ ranking across diverse 1D
structures, this project lays groundwork for connecting structural metrics to
physical behavior.

## 2. Core Definitions

- **Hyperuniformity:** A point pattern where the local number variance $\sigma^2(R)$
  within a window of radius $R$ grows slower than the window volume [1].

- **Scaling Exponent ($\alpha$):** In reciprocal space, the structure factor $S(k)$
  (or spectral density $\tilde{\chi}_V(k)$) vanishes as $k \to 0$, scaling as
  $S(k) \sim |k|^\alpha$ [2, 3].

- **Hyperuniformity Classes:**
  - **Class I** ($\alpha > 1$): Variance bounded, scales as $R^{d-1}$.
    Includes crystals and quasicrystals [1, 3, 4].
  - **Class II** ($\alpha = 1$): Variance scales as $R^{d-1}\ln(R)$.
    Includes period-doubling limit-periodic chains [3, 5].
  - **Class III** ($0 < \alpha < 1$): Variance scales as $R^{d-\alpha}$.
    Includes the 0222 chain [3, 5].
  - **Anti-hyperuniform** ($\alpha < 0$): Variance grows faster than random [6].

- **Surface-Area Coefficient ($\bar{\Lambda}$):** The amplitude coefficient of the
  leading variance growth term. For Class I systems in 1D this is the time-averaged
  value of the bounded, oscillating variance function. Reference values:
  - Integer lattice: $\bar{\Lambda} = 1/6$
  - Poisson (random baseline, not hyperuniform): $\sigma^2(R) = 2\rho R$ [7]

## 3. 1D Substitution Tilings (Quasicrystals)

Quasicrystals can be generated using a substitution matrix $\mathbf{M}$ applied
iteratively to a seed tile (Short $S$ or Long $L$) [8, 9].

- **Fibonacci Chain (Golden Ratio):**
  $M = \begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}$,
  substitution $S \to L$, $L \to LS$ [8, 10].

- **Silver Ratio (0112) Chain:**
  $M = \begin{pmatrix} 0 & 1 \\ 1 & 2 \end{pmatrix}$,
  substitution $S \to L$, $L \to LLS$; metallic mean $\mu_2 = 1 + \sqrt{2} \approx 2.414$ [11].

- **Bronze Ratio Chain:**
  $M = \begin{pmatrix} 0 & 1 \\ 1 & 3 \end{pmatrix}$,
  substitution $S \to L$, $L \to LLLS$; metallic mean $\mu_3 = (3 + \sqrt{13})/2 \approx 3.303$ [12].

- **Eigenvalue Formula for $\alpha$:** If $\lambda_1$ and $\lambda_2$ are the largest
  and second-largest eigenvalues of $\mathbf{M}$:
  $$\alpha = 1 - \frac{2 \ln|\lambda_2|}{\ln|\lambda_1|}$$
  For all metallic mean chains, $\det(\mathbf{M}) = -1$, so
  $|\lambda_2| = 1/\lambda_1$, which gives $\alpha = 3$ universally [5, 13].

- **Alternative Generation — Projection (Cut-and-Project) Method:**
  Construct a 2D square lattice $\mathbb{Z}^2$ and project points within a strip
  onto a 1D "physical space" line with slope $1/\mu_n$. The strip width $\omega$
  in the perpendicular (internal) direction controls the hyperuniformity class:
  - Ideal width $\omega = \mu_n$: produces a Class I hyperuniform quasicrystal
    with exactly 2 distinct spacings. For Fibonacci, spacing ratio = $\tau$
    (matches substitution). For silver/bronze, spacing ratio = $(\mu+1)/\mu$.
  - Non-ideal width $\omega \neq \mu_n$: degrades the pattern to Class II
    hyperuniformity with logarithmically growing variance envelope.

- **$\bar{\Lambda}$ Rescaling Invariance:** The long-range average of the bounded
  variance $\bar{\Lambda}$ is independent of point density. Under rescaling
  $x \to ax$, $\sigma'^2(R) = \sigma^2(R/a)$, whose long-range average is
  unchanged. This is verified numerically: the projection method (density
  $\rho \approx 0.85$) and substitution method (density $\rho \approx 0.72$)
  give the same $\bar{\Lambda}$ for Fibonacci within 0.1%.

## 4. The Measurement Problem & Two-Phase Media

Many hyperuniform systems (especially quasicrystals with dense, discontinuous Bragg
peaks) make direct evaluation of $\alpha$ from $S(k)$ impossible [14, 15]. The
workaround:

- **Mapping:** Convert the point pattern to a 1D two-phase medium by placing
  non-overlapping solid rods on each point [16, 17].
- **Packing Fraction:** $\phi_2 = 0.35$ [17, 18].

## 5. Diffusion Spreadability $\mathcal{S}(t)$

The dynamic fraction of solute diffusing from particles (phase 2) into void space
(phase 1) over time [19, 20].

- **Formula:**
  $$\mathcal{S}(\infty) - \mathcal{S}(t) = \frac{d\,\omega_d}{(2\pi)^d \phi_2}
  \int_0^\infty k^{d-1}\, \tilde{\chi}_V(k)\, e^{-k^2 D t}\, dk$$
  [21, 22]

- **Extracting $\alpha$:** At long times, the excess spreadability decays as
  $t^{-(d+\alpha)/2}$. The logarithmic derivative of the spreadability curve
  yields $\alpha$ [23–26].

## 6. Current Results

| Pattern | N (tiles) | $\bar{\Lambda}$ | vs Lattice |
|---|---|---|---|
| Integer Lattice | 100,000 | ~1/6 | reference |
| Fibonacci (substitution) | 14,930,352 | 0.200 | 1.21x |
| Silver (substitution) | 22,619,537 | 0.250 | 1.51x |
| Bronze (substitution) | 21,932,293 | 0.282 | 1.71x |
| Fibonacci (projection) | 220,161 | 0.200 | 1.21x |

Projection vs substitution agreement (Fibonacci): 0.1% difference, confirming
$\bar{\Lambda}$ rescaling invariance.

## 7. Project Roadmap (1D Focus)

### Phase 1: Code Benchmarking & Validation — COMPLETE

- **Task:** Generate a 1D Poisson point pattern with periodic boundary conditions.
- **Goal:** Compute $\sigma^2(R)$ via sliding window and confirm it matches the
  exact result $\sigma^2(R) = 2\rho R$.
- **Status:** COMPLETE. Implemented in `Poisson variance benchmark.py`.
  Mean relative error 1.1\% (100 realizations).

### Phase 2: Generating Quasiperiodic Patterns — COMPLETE

- **Substitution method:** Implemented in `substitution_tilings.py`. Vectorized
  generation supports N ~ $10^7$ chains for all three metallic means.
- **Projection method:** Implemented in `projection_method.py`. Cut-and-project
  from $\mathbb{Z}^2$ with parameterized strip width. Demonstrates Class I
  (ideal $\omega = \mu$) vs Class II (non-ideal $\omega \neq \mu$) transition.
- **Status:** COMPLETE for both methods.

### Phase 3: Computing Variance and $\bar{\Lambda}$ — COMPLETE

- Computed $\sigma^2(R)$ for all chains at N ~ $10^7$ scale.
- Verified Class I behavior: bounded, oscillating variance for all chains.
- Verified piecewise quadratic structure via second derivative analysis.
- Exact analytic formula $\sigma^2(R) = 2\{R\}(1-2\{R\})$ validated for lattice.
- $\bar{\Lambda}$ values stable across N = 500k → 10M (convergence confirmed).
- **Status:** COMPLETE. All results in `results/` directory.

### Phase 4: Two-Phase Media and Diffusion Spreadability — COMPLETE

- Decorated point patterns with non-overlapping solid rods ($\phi_2 = 0.35$).
- Computed structure factor $S(k)$ via histogram binning + FFT.
- Computed spectral density $\tilde{\chi}_V(k) = \rho |m̃(k)|^2 S(k)$.
- Evaluated excess spreadability $E(t) = S(\infty) - S(t)$ over $t \in [10^{-2}, 10^8]$.
- Extracted $\alpha$ via logarithmic derivative of $E(t)$.
- Implemented in `two_phase_media.py` (library) and `run_all.py` (Figs 6–8).

**Alpha extraction results (linear fit of $\ln E$ vs $\ln t$ over $[10^2, 10^5]$):**

| Pattern | $\alpha$ | Expected | Error | Status |
|---|---|---|---|---|
| Poisson | 0.001 | 0 | — | OK |
| Integer Lattice | exp. decay | $\infty$ | — | OK |
| Fibonacci (Golden Ratio) | 3.049 | 3 | 1.6% | OK |
| Silver Ratio | 2.992 | 3 | 0.3% | OK |
| Bronze Ratio | 2.987 | 3 | 0.4% | OK |

- Two extraction methods: (1) period-aware log derivative for $\alpha(t)$ curves
  (sliding window matched to oscillation period $2\ln\mu$ in $\ln t$ space), and
  (2) linear fit of $\ln E$ vs $\ln t$ for reported values (more robust, averages
  over all oscillations in the plateau window).
- Integer lattice computed analytically from Bragg peak positions (avoids FFT
  binning artifacts). Its $E(t)$ decays exponentially, reaching zero before the
  plateau window — confirming $\alpha \to \infty$ as expected for crystals.
- **Status:** COMPLETE. All three metallic-mean chains yield $\alpha \approx 3$,
  confirming Class I hyperuniformity via the two-phase media approach.

### Phase 5: Expanding the 1D Catalog (In Progress)

- Extend the analysis pipeline to a broader set of 1D hyperuniform point patterns,
  including disordered hyperuniform systems (e.g., stealthy hyperuniform, perturbed
  lattices, maximally random jammed packings, cloaked systems).
- Populate a comprehensive $(\alpha, \bar{\Lambda})$ ranking table across all three
  hyperuniformity classes.
- Investigate correlations between the structural metrics ($\alpha$, $\bar{\Lambda}$)
  and physical properties (transport, band structure, mechanical response).
