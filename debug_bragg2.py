"""
Carefully derive the Bragg-sum Lambda_bar formula using small chains
where we can do exact S(k) computation. Focus on getting the normalization right.

Key insight from the integer lattice calibration:
   Lambda_bar = (4*rho) * sum_{G>0} I_G / G^2
where I_G = lim_{N->inf} S(G)/N (structure factor peak per particle).

For the INTEGER LATTICE:
- S(G)/N -> 1 for all Bragg peaks at G=2*pi*m
- 4*1 * sum_{m>=1} 1/(2*pi*m)^2 = (4/4*pi^2) * pi^2/6 = 1/6 ✓

For a quasiperiodic chain, S(G)/N for a Bragg peak is NOT 1; it equals
|F_G|^2 which is the SQUARED MODULUS of the normalized structure factor amplitude.

For the cut-and-project construction:
   F_G = (1/N) * sum_j exp(i G x_j)  -->  |F_G|^2 = S(G)/N^2 * N = S(G)/N

But wait, for the integer lattice: (1/N) * sum_j exp(i G x_j) = 1 for G=2*pi*m.
So F_G = 1 and |F_G|^2 = 1 = S(G)/N. ✓

For quasiperiodic tiling: F_G -> 0 as N -> inf for MOST k values,
but at Bragg peaks it converges to a finite nonzero value.

The QUESTION is: does S(G)/N converge to a FINITE value at Bragg peaks
for a quasiperiodic tiling?

For a quasiperiodic tiling, the structure factor scales as:
   S(G) ~ N * |F_G|^2  (where |F_G|^2 ~ sinc^2(pi*q_perp) from CaP construction)
   ONLY for N -> infinity at a TRUE BRAGG PEAK.

For finite N, the peak heights are affected by nearby peaks and finite-size effects.
The correct way is to use large N and look at the SPECIFIC INTENSITY per particle.

Let me use the exact SEQUENCE method: for a substitution tiling of N=10^5-10^6 tiles,
compute the sum sum_j exp(i G x_j) EXACTLY (using the substitution structure)
and get the exact |F_G|^2 in the infinite-N limit.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from substitution_tilings import CHAINS, generate_substitution_sequence, sequence_to_points

def compute_F_exact(pts, L, G_vals, batch=100):
    """
    Compute F_G = (1/N) * |sum_j exp(i G x_j)|^2 = S(G)/N
    for a given point set.
    """
    N = len(pts)
    F2 = np.zeros(len(G_vals))
    for i in range(0, len(G_vals), batch):
        Gb = G_vals[i:i+batch]
        phase = np.outer(Gb, pts)  # (batch, N)
        rho_hat = np.sum(np.exp(1j * phase), axis=1)  # (batch,)
        F2[i:i+batch] = np.abs(rho_hat)**2 / N
    return F2

print("=" * 65)
print("  Step 1: Integer lattice calibration")
print("=" * 65)

# Integer lattice: N=1000, spacing=1
N_latt = 1000
pts_latt = np.arange(N_latt, dtype=float)
L_latt = float(N_latt)
rho_latt = 1.0
G_latt = np.array([2*np.pi*m for m in range(1, 21)])  # first 20 peaks
F2_latt = compute_F_exact(pts_latt, L_latt, G_latt)
print("Integer lattice Bragg peaks:")
print("G_m / S(G)/N:")
for m, (G, F2) in enumerate(zip(G_latt[:5], F2_latt[:5])):
    print(f"  m={m+1}: G={G:.4f}, S(G)/N={F2:.6f}  (expected 1.0 = N/N = {N_latt}/{N_latt})")

lb_latt = 4 * rho_latt * np.sum(F2_latt / G_latt**2)
print(f"\nLambda_bar (integer lattice, 20 peaks): {lb_latt:.6f}")
print(f"  (expected 1/6 = {1/6:.6f}; sum 1/(2*pi*m)^2 = pi^2/6 so 4/(4*pi^2)*pi^2/6 = 1/6)")

print()
print("=" * 65)
print("  Step 2: Fibonacci chain -- convergence with N")
print("=" * 65)

tau = (1+np.sqrt(5))/2
lam1_fib = tau

for n_iters in [15, 20, 23, 25]:
    seq = generate_substitution_sequence('fibonacci', n_iters)
    pts, L = sequence_to_points(seq, 'fibonacci')
    N = len(pts)
    rho = N/L
    ell_bar = 1.0/rho

    # Key Bragg peak: G_{0,1} = 2*pi*tau/ell_bar (the 'main' peak)
    # and G_{1,0} = 2*pi/ell_bar
    G_01 = 2*np.pi*tau/ell_bar
    G_10 = 2*np.pi/ell_bar
    G_11 = 2*np.pi*(1+tau)/ell_bar
    G_m10 = 2*np.pi*(tau-1)/ell_bar  # = 2*pi*(tau-1)/ell_bar > 0 since tau>1

    G_test = np.array([G_10, G_m10, G_01, G_11])
    F2_test = compute_F_exact(pts, L, G_test)

    # Expected intensities from cut-and-project: I_{p,q} = sinc^2(pi*(q - p/tau))
    def sinc2_perp(p, q, lam1):
        s = q - p/lam1
        if abs(s) < 1e-12:
            return 1.0
        return (np.sin(np.pi*s)/(np.pi*s))**2

    I_10 = sinc2_perp(1, 0, tau)   # should be sinc^2(-pi/tau)
    I_m10 = sinc2_perp(-1, 0, tau)  # G=-2*pi/ell_bar, negative, skip
    I_01 = sinc2_perp(0, 1, tau)   # sinc^2(pi) = 0!!!
    I_11 = sinc2_perp(1, 1, tau)   # sinc^2(pi*(1-1/tau)) = sinc^2(pi/tau^2)

    print(f"N={N:>10,} iters={n_iters}:")
    print(f"  G_{{1,0}}={G_10:.4f}: S/N={F2_test[0]:.6f}, theory sinc^2={I_10:.6f}")
    print(f"  G_{{0,1}}={G_01:.4f}: S/N={F2_test[2]:.6f}, theory sinc^2={I_01:.6f}")
    print(f"  G_{{1,1}}={G_11:.4f}: S/N={F2_test[3]:.6f}, theory sinc^2={I_11:.6f}")

    # Full Lambda_bar from sum
    # Generate all Bragg peaks
    G_all = []
    I_CaP = []
    for pp in range(-50, 51):
        for qq in range(-50, 51):
            if pp==0 and qq==0: continue
            t = pp + qq*lam1_fib
            if t <= 0: continue
            G = 2*np.pi*t/ell_bar
            if G > 200: continue
            s = qq - pp/lam1_fib
            if abs(s) < 1e-12:
                I = 1.0
            else:
                I = (np.sin(np.pi*s)/(np.pi*s))**2
            G_all.append(G)
            I_CaP.append(I)
    G_all = np.array(G_all)
    I_CaP = np.array(I_CaP)
    lb_CaP = 4*rho * np.sum(I_CaP/G_all**2)
    print(f"  Lambda_bar (CaP formula): {lb_CaP:.6f}")

print()
print("Key observation: sinc^2(pi*(q-p/tau)) at (p,q)=(0,1):")
print(f"  q-p/tau = 1-0 = 1 -> sinc^2(pi*1) = sin^2(pi)/(pi)^2 = 0")
print("So the main peak G_{0,1} has ZERO intensity from the CaP formula?")
print("But numerically S(G_{0,1})/N is large! The formula must be WRONG.")
print()
print("Let me check: G_{0,1} = 2*pi*tau/ell_bar. What is tau?")
print(f"  tau = {tau:.8f}")
print(f"  ell_bar = {1/((1+tau)/(1+tau**2)):.8f} wait, ell_bar = 1/rho")
rho_fib_th = (1+tau**2)/(1+tau)**2 * (1+tau)  # = (1+tau)/( ... )
# Actually: ell_bar = f_S*1 + f_L*tau = 1/(1+tau) + tau^2/(1+tau) = (1+tau^2)/(1+tau) = tau (since tau^2=tau+1)
ell_bar_fib_th = (1+tau**2)/(1+tau)
print(f"  ell_bar (theoretical) = (1+tau^2)/(1+tau) = {ell_bar_fib_th:.8f}")
print(f"  But tau^2 = tau+1, so (1+tau+1)/(1+tau) = (2+tau)/(1+tau) = tau+... hmm")
print(f"  (1+tau^2)/(1+tau) = (1+tau+1)/(1+tau) = tau + 2/(1+tau) - 1 = ... = tau/(1+tau)*(1+tau) = tau? No.")
print(f"  Actually: (1+tau^2)/(1+tau) = (1+tau+1)/(1+tau) = 1 + 1/(1+tau)")
print(f"  1/(1+tau) = 1/(tau+1) = tau-1 (since 1/tau = tau-1)")
print(f"  So ell_bar = 1 + tau-1 = tau = {tau:.8f}")
print()

# So for Fibonacci: ell_bar = tau = (1+sqrt(5))/2
# G_{0,1} = 2*pi*tau / tau = 2*pi !!!
# And G_{1,0} = 2*pi / tau = 2*pi*(tau-1) = 2*pi*(0.6180...) ≈ 3.884

# The (p,q)=(0,1) peak: G = 2*pi*tau/ell_bar = 2*pi*tau/tau = 2*pi
# perp component: q - p/lam1 = 1 - 0/tau = 1 -> sinc^2(pi*1) = 0
# But G = 2*pi is the LATTICE PEAK (G = 2*pi/d for d=1), and it should have I=0
# for hyperuniform system! No, S(G=2*pi) for Fibonacci is finite and nonzero.
print("BUT WAIT: G=2*pi corresponds to k=2*pi, which is the reciprocal of the")
print("MEAN spacing ell_bar=tau. For a substitution tiling, this is NOT a Bragg peak")
print("in the usual sense -- wait, it IS because p+q*tau = 0+1*tau = tau and G=2*pi*tau/ell_bar=2*pi.")
print()
print("sinc^2(pi*1) = 0 means the CaP window function gives zero amplitude.")
print("This is WRONG because numerically S(2*pi)/N is large for Fibonacci!")
print()
print("THE ERROR in the CaP formula: the window function width W is NOT 1.")
print("I used W=ell_S=1 but the correct W depends on the normalization of the")
print("perpendicular space. Let me compute the CORRECT formula.")

print()
print("=" * 65)
print("  Step 3: Correct the cut-and-project intensity formula")
print("=" * 65)
print()
# For the Fibonacci chain (n=1 metallic mean):
# The substitution matrix M = [[0,1],[1,1]], eigenvalues lambda1=tau, lambda2=1-tau=-1/tau
# Right eigenvector for lambda1: v = (1, tau) -> tile lengths S=1, L=tau
# In cut-and-project language:
# Physical direction: e_par = (1, tau)/||(1,tau)||  (NOT normalized)
# Perpendicular:     e_perp = (-tau, 1)/||(tau,1)||
#
# A 2D lattice point (m,n) in Z^2 projects to:
# x_par  = (m + n*tau) / norm  (physical position, up to scale)
# x_perp = (-m*tau + n) / norm  (perpendicular component)
#
# The ACCEPTANCE WINDOW in perp space has width W determined by the tile types.
# For Fibonacci: S tiles correspond to large perp projection, L to small.
# Tile S has physical length 1, perpendicular projection length tau.
# Tile L has physical length tau, perpendicular projection length 1.
# The window W = ell_S^perp + ell_L^perp = tau + 1 = tau^2 = tau+1
# Wait: the acceptance window for a 1D quasicrystal via CaP from Z^2:
# The window width = 1/(norm) or something like that.
#
# Let me use the FORMULA from Baake & Grimm "Aperiodic Order" (Cambridge, 2013):
# For the Fibonacci chain with lambda1=tau:
# The window interval has length 1/(tau-1) = tau (since 1/(tau-1) = tau^2/(tau*(tau-1))=...)
# Actually 1/(tau-1) = 1/(tau-1)*(tau+1)/(tau+1) = (tau+1)/(tau^2-1) = (tau+1)/tau = 1/tau + 1...
# Let me just look at what window width W gives the correct Fourier coefficient for the first peak.

# For cut-and-project with window of width W:
# I_{p,q} = (1/ell_bar)^2 * |integral_{-W/2}^{W/2} exp(i*q_perp * x) dx|^2
# = (1/ell_bar)^2 * W^2 * sinc^2(q_perp * W / 2)
# where q_perp = 2*pi * (-p*tau + q) / ell_bar... hmm, scaling issues.
#
# Let me use the NUMERICAL approach to find what W gives the correct intensities.

# For Fibonacci: first peak at G_{1,0} = 2*pi/ell_bar = 2*pi/tau
# perp component (unnormalized): q_perp_index = q - p/tau = 0 - 1/tau = -1/tau
# If I_{1,0} = W^2 * sinc^2(q_perp_index * pi * W)?
# From large-N computation: I_{1,0} = S(G_{1,0})/N -> some value
# Let me compute it for large N:

iters = 23
seq_fib = generate_substitution_sequence('fibonacci', iters)
pts_fib, L_fib = sequence_to_points(seq_fib, 'fibonacci')
N_fib = len(pts_fib)
rho_fib = N_fib/L_fib
ell_bar_fib = 1/rho_fib

print(f"Fibonacci N={N_fib:,}, ell_bar={ell_bar_fib:.8f}=tau={tau:.8f}")

# Compute S(G)/N for a few key peaks
G_10 = 2*np.pi*1/ell_bar_fib   # (1,0) peak
G_01 = 2*np.pi*tau/ell_bar_fib  # (0,1) peak
G_11 = 2*np.pi*(1+tau)/ell_bar_fib  # (1,1)
G_m11 = 2*np.pi*(tau-1)/ell_bar_fib  # (-1,1) = G_01 - G_10 -> = 2*pi*(tau-1)/ell_bar
G_21 = 2*np.pi*(2+tau)/ell_bar_fib  # (2,1)
G_02 = 2*np.pi*2*tau/ell_bar_fib  # (0,2)
G_m12 = 2*np.pi*(2*tau-1)/ell_bar_fib  # (-1,2)

Gs = np.array([G_10, G_01, G_11, G_m11, G_21, G_02, G_m12])
labels = ['(1,0)', '(0,1)', '(1,1)', '(-1,1)', '(2,1)', '(0,2)', '(-1,2)']
F2s = compute_F_exact(pts_fib, L_fib, Gs)

print(f"\nPeak intensities S(G)/N for Fibonacci (N={N_fib:,}):")
for lbl, G, F2 in zip(labels, Gs, F2s):
    # Perp component: q - p/tau
    parts = lbl.strip('()').split(',')
    p_val, q_val = int(parts[0]), int(parts[1])
    s = q_val - p_val/tau
    sinc2_val = 1.0 if abs(s)<1e-12 else (np.sin(np.pi*s)/(np.pi*s))**2
    print(f"  {lbl}: G={G:.4f}, S/N={F2:.6f}, q_perp={s:.4f}, sinc^2(pi*q_perp)={sinc2_val:.6f}")

print()
print("Key insight from (0,1): q_perp = 1, sinc^2(pi) = 0, but S/N is large!")
print("This means the sinc formula uses q_perp in different units!")
print()
print("Let's try: q_perp_CORRECT = 1/(tau+1) for (0,1) -> sinc^2(pi/(tau+1))")
q_perp_01_correct = 1/(tau+1)
sinc2_01 = (np.sin(np.pi*q_perp_01_correct)/(np.pi*q_perp_01_correct))**2
print(f"  sinc^2(pi/(tau+1)) = {sinc2_01:.6f},  S(G_01)/N = {F2s[1]:.6f}")

# Actually, let me think about this more carefully.
# The CaP construction for the Fibonacci chain:
# The physical space E_par is the line y = x/tau in R^2.
# The perpendicular space E_perp is the line y = -tau*x in R^2.
# The acceptance window W is an interval of length 1 in E_perp.
# The PERPENDICULAR COMPONENT of a reciprocal lattice vector (p,q) in Z^2:
# q_perp = (p, q) . (1, -tau) / ||(1,-tau)|| = (p - q*tau) / sqrt(1+tau^2)
# The structure factor amplitude:
# F_{p,q} = (1/|W|) * integral_W exp(i*q_perp*x) dx = sinc(q_perp * W/2)
# where q_perp = 2*pi * (p*(-tau) + q*1) / (ell_bar * ???)
#
# The proper formula uses the STAR MAP which maps a reciprocal vector (p,q)
# to its CONJUGATE in perpendicular space:
# (p,q)^* = (p - q*tau, q - p*tau) ... hmm no.
#
# THE CORRECT FORMULA for Fibonacci CaP:
# The star map for Fibonacci: sigma_* maps tau -> 1/tau (Galois conjugate).
# So (p+q*tau)^* = (p + q*(1/tau)) = (p*tau - q*tau^2 + q) / tau ... no.
# Actually: if xi = p + q*tau, then xi^* = p + q*(1-tau) = p + q - q*tau [Galois conj]
# = p + q*lambda2  where lambda2 = 1-tau = 1-tau ≈ -0.618
#
# q_perp(p,q) = p + q*(1-tau) = p + q*lambda2  [NOT q - p/tau]

print()
print("CORRECTED star map for Fibonacci: q_perp = p + q*(1-tau) = p + q*lambda2")
lam2_fib = 1 - tau  # = -0.6180...
for lbl, G, F2 in zip(labels, Gs, F2s):
    parts = lbl.strip('()').split(',')
    p_val, q_val = int(parts[0]), int(parts[1])
    q_perp_new = p_val + q_val*lam2_fib  # = p + q*(1-tau)
    if abs(q_perp_new) < 1e-12:
        sinc2_new = 1.0
    else:
        sinc2_new = (np.sin(np.pi*q_perp_new)/(np.pi*q_perp_new))**2
    print(f"  {lbl}: S/N={F2:.6f}, q_perp_new={q_perp_new:.4f}, sinc^2={sinc2_new:.6f}")

print()
print("The sinc^2 formula with q_perp = p + q*lambda2:")
print("  (1,0): q_perp = 1+0 = 1 -> sinc^2(pi) = 0")
print("  (0,1): q_perp = 0+lambda2 = 1-tau = -0.618 -> sinc^2(-0.618*pi)")
s_01 = lam2_fib  # = -0.618...
print(f"  sinc^2(pi*{s_01:.4f}) = {(np.sin(np.pi*s_01)/(np.pi*s_01))**2:.6f}")
print(f"  vs S(G_01)/N = {F2s[1]:.6f}")
print()

# Hmm, the sinc^2 for (1,0) is still 0. Let me look at the ACTUAL FORMULA more carefully.
# For the Fibonacci chain specifically, the literature gives:
# The intensity of the Bragg peak at G_{p,q} = 2*pi*(p+q*tau)/tau is:
# I_{p,q} = (f_S + f_L * exp(i*G_{p,q}))^2 * |W(G_perp)|^2
# where W is the window function and G_perp = 2*pi*(p*(-1/tau) + q*tau) / ell_bar ... hmm.
#
# SIMPLEST approach: compute I_{p,q} directly from the finite tiling, extrapolate to N->inf.
# S(G_{p,q})/N is approximately |F_{p,q}|^2 for large N at a TRUE Bragg peak.

print("=" * 65)
print("  Step 4: Direct verification -- compute Lambda_bar from S(k)/N for large N")
print("=" * 65)
print()
# For Fibonacci (N=196418), compute S(G)/N for ALL candidate peaks and sum
# Lambda_bar = 4*rho * sum I_G / G^2

# Use a sufficiently large N to ensure convergence
seq_fib25 = generate_substitution_sequence('fibonacci', 25)
pts_fib25, L_fib25 = sequence_to_points(seq_fib25, 'fibonacci')
N_fib25 = len(pts_fib25)
rho_fib25 = N_fib25/L_fib25
ell_bar_25 = 1/rho_fib25

print(f"Using N={N_fib25:,} for Fibonacci (25 iters)")

# All Bragg peaks with G in [0, 100]:
G_all_fib = []
for p in range(-150, 151):
    for q in range(-150, 151):
        if p==0 and q==0: continue
        G = 2*np.pi*(p + q*tau)/ell_bar_25
        if 0 < G < 100:
            G_all_fib.append(G)
G_all_fib = np.array(sorted(set(round(g, 8) for g in G_all_fib)))
print(f"Candidate peaks in [0,100]: {len(G_all_fib)}")

# Compute in batches
print("Computing S(G)/N...")
batch = 50
F2_all_fib = np.zeros(len(G_all_fib))
for i in range(0, len(G_all_fib), batch):
    Gb = G_all_fib[i:i+batch]
    phase = np.outer(Gb, pts_fib25)  # (batch, N)
    rho_hat = np.sum(np.exp(1j * phase), axis=1)
    F2_all_fib[i:i+batch] = np.abs(rho_hat)**2 / N_fib25
    if i % 500 == 0:
        print(f"  Progress: {i}/{len(G_all_fib)}")

# Lambda_bar from sum
lb_fib_direct = 4*rho_fib25 * np.sum(F2_all_fib/G_all_fib**2)
print(f"\nFibonacci Lambda_bar (direct sum, k<100): {lb_fib_direct:.8f}")
print(f"Expected: 0.20110")
print(f"Ratio: {lb_fib_direct/0.20110:.6f}")

# Show the largest-contributing peaks
contrib = 4*rho_fib25 * F2_all_fib/G_all_fib**2
sorted_idx = np.argsort(contrib)[::-1][:20]
print(f"\nTop 20 contributing peaks:")
for idx in sorted_idx:
    print(f"  G={G_all_fib[idx]:.6f}, S/N={F2_all_fib[idx]:.8f}, contrib={contrib[idx]:.8f}")
