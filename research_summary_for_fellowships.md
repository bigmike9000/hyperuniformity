# Research Summary: Michael Fang

## Project Title
Classification and Measurement of One-Dimensional Hyperuniform Point Patterns

## Advisor
Professor Salvatore Torquato, Princeton University
Second reader: Professor Paul Steinhardt

## What is Hyperuniformity?
Hyperuniform systems are point patterns (or materials) whose large-scale density fluctuations are anomalously suppressed compared to typical disordered systems. Formally, the structure factor S(k) vanishes as the wavenumber k approaches zero. Hyperuniformity appears across physics, materials science, and mathematics — in quasicrystals, biological tissues, the distribution of prime numbers, and engineered "stealthy" materials with designer photonic properties.

## What Michael's Project Does
This project provides the first systematic numerical catalog of hyperuniform point patterns in one dimension, characterizing 27 systems across all three universality classes using three complementary probes: number variance, structure factor, and diffusion spreadability.

The key contributions are:

1. **Metallic-mean quasicrystal family (Fibonacci through Nickel, n=1..5 and beyond):** All share the same hyperuniformity exponent alpha=3 due to the unimodular determinant condition on their substitution matrices. Michael computed the surface-area coefficient (Lambda-bar) for each — four of the five values are novel. He further showed that Lambda-bar converges to exactly 1/3 as the metallic-mean index n goes to infinity, establishing a universal limit for this family.

2. **Silver mean Lambda-bar = 1/4 exactly:** High-precision numerics (N=3.88 million points) confirmed this exact value to within 0.3 standard deviations.

3. **Inaccessibility of the 2 < alpha < 3 gap:** An exhaustive algebraic enumeration of substitution matrices proved that no unimodular matrix of any rank can produce a hyperuniformity exponent in the interval (2, 3). This is a theorem, not just a numerical observation. Michael then showed the gap *can* be accessed using non-unimodular matrices, finding the first known example (alpha=2.071) and 138 total matrices that reach into this interval.

4. **New substitution tilings filling spectral gaps:** The Bombieri-Taylor cubic substitution was identified as the first known 1D quasicrystal with 1 < alpha < 2 (alpha=1.545), and a new 3-letter substitution achieves alpha=2 exactly — a deterministic analogue of the stochastic uncorrelated random lattice.

5. **Generalized scaling coefficients for weaker hyperuniform classes:** New metrics Lambda_II and Lambda_III were introduced to rank Class II and III patterns (where the standard Lambda-bar diverges), using curve-fitting methods that correct systematic biases in plateau-based estimators.

6. **Stealthy hyperuniform configurations:** Analysis of ~4,316 computationally generated stealthy ground states per stealthiness parameter, revealing distinct structural phases that satisfy the same constraint but differ in short-range order.

## Methodology
All computations are performed with custom Python code. Substitution tilings are generated from matrix rules (2x2 through 4x4), and hyperuniformity metrics are extracted from number variance scaling, Fourier-space structure factors, and time-dependent spreadability (following the Yuan-Torquato improved fitting method). Results are validated against exact analytical benchmarks (integer lattice Lambda-bar = 1/6, Zachary-Torquato numerical value for Fibonacci, Klatt et al. exact formula for uncorrelated random lattices).

## Current Status
- Junior Paper: Sections 1-3 (Introduction, Theory, Methods) complete at 9 pages; results sections in preparation.
- Four checkpoint meetings with Prof. Torquato completed.
- Full computational pipeline and figure generation are operational.

## Fields / Keywords
Statistical mechanics, condensed matter theory, quasicrystals, hyperuniformity, substitution tilings, number theory, computational physics
