"""
Debug the normalization of the Bragg peak formula for Lambda_bar.
We use the integer lattice as a known calibration: Lambda_bar = 1/6 exactly.
"""
import numpy as np

# =========================================================
# INTEGER LATTICE NORMALIZATION CHECK
# =========================================================
# For integer lattice: spacing d=1, rho=1.
# sigma^2(R) = 2f(1-2f) for f=frac(R) in [0,0.5]  (known exact formula)
# Lambda_bar = period-average = 1/6

def sigma2_lattice_exact(R):
    f = R % 1.0
    if f > 0.5:
        f = 1.0 - f
    return 2.0 * f * (1.0 - 2.0 * f)

# Lambda_bar check
R_vals = np.linspace(0.001, 1000, 500000)
sig2_exact = np.array([sigma2_lattice_exact(R) for R in R_vals])
lb_exact = np.mean(sig2_exact)
print(f"Lambda_bar (integer lattice, exact formula): {lb_exact:.8f}  (expected 1/6={1/6:.8f})")

# =========================================================
# TRY DIFFERENT BRAGG SUM FORMULAS
# =========================================================
# For integer lattice: Bragg peaks at G_m = 2*pi*m (m=1,2,3,...)
# The question is: what is the correct intensity I_m?

# FORMULA 1: Lambda_bar = (2*rho/pi) * sum_{G>0} I_G / G^2
# With I_G = 1 (structure factor at peak / N):
def lb_formula1(M=10000):
    total = 0.0
    rho = 1.0
    for m in range(1, M+1):
        G = 2*np.pi*m
        I = 1.0  # S(G)/N = 1 for integer lattice
        total += I / G**2
    return (2*rho/np.pi) * total

# FORMULA 2: Lambda_bar = (4*rho/pi) * sum_{G>0} I_G / G^2  (extra factor 2)
def lb_formula2(M=10000):
    return 2 * lb_formula1(M)

# FORMULA 3: sigma^2(R) = (rho/pi) * sum_G I_G * (2sin(GR)/G)^2
# Lambda_bar = long-R average = (rho/pi) * sum_G I_G * 2/G^2  [since <sin^2>=1/2]
# = (2*rho/pi) * sum_G I_G / G^2   [same as formula 1... if I_G=1]
# Hmm.

# Let's compute sigma^2(R) from the Bragg sum and compare to exact:
def sigma2_bragg1(R, M=2000):
    """sigma^2 = (rho/pi) * sum_G I_G * (2sin(GR)/G)^2, I_G=1"""
    total = 0.0
    for m in range(1, M+1):
        G = 2*np.pi*m
        total += (2*np.sin(G*R)/G)**2
    return (1.0/np.pi) * total

def sigma2_bragg2(R, M=2000):
    """sigma^2 = (2*rho/pi) * sum_G I_G * (2sin(GR)/G)^2, I_G=1 (extra factor 2)"""
    return 2 * sigma2_bragg1(R, M)

R_test = 0.3
exact_val = sigma2_lattice_exact(R_test)
bragg1 = sigma2_bragg1(R_test)
bragg2 = sigma2_bragg2(R_test)
print(f"\nAt R=0.3:")
print(f"  Exact sigma^2           = {exact_val:.8f}")
print(f"  Bragg formula 1         = {bragg1:.8f}  (ratio: {exact_val/bragg1:.4f})")
print(f"  Bragg formula 2         = {bragg2:.8f}  (ratio: {exact_val/bragg2:.4f})")

# Check ratio for several R values
print("\nRatio exact/bragg1 at various R:")
for R in [0.1, 0.2, 0.3, 0.4, 0.47, 0.15, 0.33]:
    exact = sigma2_lattice_exact(R)
    b1 = sigma2_bragg1(R)
    print(f"  R={R:.2f}: exact={exact:.6f}, bragg1={b1:.6f}, ratio={exact/b1:.4f}")

# =========================================================
# The ratio should be constant if the formula is correct up to normalization
# =========================================================

# If ratio ~ pi, then sigma^2_exact = pi * sigma^2_bragg1, meaning:
# sigma^2_exact = pi * (1/pi) * sum = sum ... which would give Lambda_bar = sum/G^2 = pi/6 ???
# Let me check:
print(f"\n  pi = {np.pi:.8f}")

# Lambda_bar from bragg1 (using long-R average):
R_range = np.linspace(0.001, 500, 100000)
sig2_b1 = np.array([sigma2_bragg1(R, M=500) for R in R_range])
lb_b1 = np.mean(sig2_b1)
print(f"\nLambda_bar from Bragg formula 1 (M=500): {lb_b1:.8f}")
print(f"Expected 1/6 = {1/6:.8f}")
print(f"Ratio: {(1/6)/lb_b1:.6f}  (is this pi?  pi={np.pi:.6f})")

# =========================================================
# DIRECT TEST: what normalization gives Lambda_bar = 1/6?
# =========================================================
# From the long-R average of sigma^2_bragg1:
# Lambda_bar_b1 = (1/pi) * sum (2/G^2) * (1/2)  [since <sin^2>=1/2]
# = (1/pi) * sum 1/G^2
# = (1/pi) * sum 1/(4*pi^2*m^2)
# = (1/(4*pi^3)) * pi^2/6
# = 1/(24*pi) = 0.01326... NOPE.
# With (2*rho/pi): (2/pi)*(1/(4*pi^2)) * pi^2/6 = (2/pi)*(1/24) = 1/(12*pi) = 0.0265

# Clearly something is fundamentally off. Let me look at this from a completely
# different angle: use the REAL SPACE formula and convert to Bragg sums.

# Real space formula (T&S 2003):
# sigma^2(R) = rho*2R * [1 + rho * integral_{0}^{2R} h(r) (1 - r/2R) dr]
# = rho*2R + rho^2 * integral_0^{2R} h(r) (2R - r) dr
# where h(r) = g2(r) - 1.
# For integer lattice (rho=1, d=1):
# g2(r) = sum_{m>=1} 2*delta(r-m) / (rho*v_d(r)) ... wait
# In 1D, g2(r) is defined so that rho*g2(r)*dr = probability of finding particle in [r, r+dr]
# given one particle at origin. For integer lattice: particles at m=1,2,3,...
# rho*g2(r) = sum_{m=1}^inf delta(r-m) + sum_{m=1}^inf delta(r+m)
# But we only consider r > 0 typically:
# rho*g2(r) = 2*sum_{m=1}^inf delta(r-m) for r>0 (factor 2 for +m and -m)
# Wait, for 1D lattice with spacing 1 and rho=1:
# The number of particles per unit length at distance r from origin = delta(r-m) for each m.
# But this is a SUM over all particles, not normalized by rho^2.
# Correct: rho^2 * g2(r) = rho_2(r) = rho * sum_{m!=0} delta(r-m)
# So g2(r) = (1/rho) * sum_{m!=0} delta(r-m) = sum_{m!=0} delta(r-m)
# h(r) = g2(r) - 1 = sum_{m!=0} delta(r-m) - 1

# sigma^2(R) = rho*2R + rho^2 * integral_0^{2R} [sum_{m!=0} delta(r-m) - 1] * (2R-r) dr
# = 2R + integral_0^{2R} [sum_{m=1}^inf [delta(r-m) + delta(r+m)] - 1] * (2R-r) dr
# = 2R + sum_{m=1}^{floor(2R)} 2*(2R-m) - integral_0^{2R} (2R-r) dr
# = 2R + 2*sum_{m=1}^{M} (2R-m) - [2R*r - r^2/2]_0^{2R}
# = 2R + 2*sum_{m=1}^{M} (2R-m) - (4R^2 - 2R^2)
# = 2R + 2*[2R*M - M*(M+1)/2] - 2R^2
# where M = floor(2R)

def sigma2_real_space(R):
    M = int(2*R)
    result = 2*R + 2*(2*R*M - M*(M+1)/2) - 2*R**2
    return result

R_test = 0.3
print(f"\n\nReal-space formula check at R=0.3:")
print(f"  Exact formula: {sigma2_lattice_exact(0.3):.8f}")
print(f"  Real-space:    {sigma2_real_space(0.3):.8f}")
# Note: for R=0.3, M=floor(0.6)=0, so sigma2 = 2*0.3 + 0 - 2*0.09 = 0.6 - 0.18 = 0.42
# But exact is 2*0.3*(1-0.6) = 2*0.3*0.4 = 0.24. Hmm, wrong.
# Let me recheck. For R=0.3, the window has half-width 0.3, so full width 0.6.
# Only 0 or 1 lattice points can be inside (lattice spacing 1).
# Points at j=0,1,2,... Inside [c-0.3, c+0.3] there is at most 1 point.
# Var = E[N^2] - E[N]^2 = E[N] - E[N]^2 * ? No:
# For 0-1 events: Var = p*(1-p) where p = P(one point in window).
# But E[N] = rho*2R = 0.6, and since it's Bernoulli-like: Var = p*(1-p) where p=0.6?
# But that gives 0.24 = 0.6*0.4. YES that matches! So sigma^2 = 0.24.
print(f"\nAt R=0.3: expected sigma^2 = {0.3*2*(1-0.3*2):.6f}")
# = 0.6*(1-0.6) = 0.6*0.4 = 0.24 using Bernoulli with p = 0.6? But formula gives 2*0.3*(1-2*0.3) = 0.24 ✓

# So the variance formula is sigma^2(R) = 2*f*(1-2*f) where f=frac(R).
# The Fourier series of this:
# This is a function of f=R mod 1. Let's compute its Fourier coefficients.
# sigma^2(R) = a_0 + sum_{m=1}^inf a_m * cos(2*pi*m*R)  (even function of R)
# a_0 = integral_0^1 sigma^2(R) dR = 1/6
# a_m = 2*integral_0^1 sigma^2(R) cos(2*pi*m*R) dR
#     = 2*[integral_0^{0.5} 2f(1-2f)cos(2*pi*m*f) df + integral_{0.5}^1 2(1-f)(2f-1)cos(2*pi*m*f) df]
# = 4*integral_0^{0.5} 2f(1-2f)cos(2*pi*m*f) df * 2  [by symmetry f->1-f]
# Wait: let me compute numerically.
R_fine = np.linspace(0, 1, 10000, endpoint=False)
sig2_fine = np.array([sigma2_lattice_exact(R) for R in R_fine])
# Fourier coefficients
a_0 = np.mean(sig2_fine)
print(f"\na_0 (DC) of sigma^2 = {a_0:.8f}  (= Lambda_bar = {1/6:.8f})")

for m in range(1, 6):
    a_m = 2*np.mean(sig2_fine * np.cos(2*np.pi*m*R_fine))
    b_m = 2*np.mean(sig2_fine * np.sin(2*np.pi*m*R_fine))
    print(f"  m={m}: a_m = {a_m:.8f},  b_m = {b_m:.8f}")
    # These should match -1/(2*pi^2*m^2) * something
    print(f"         -1/(pi^2*m^2) = {-1/(np.pi**2*m**2):.8f}")

# So sigma^2(R) = 1/6 + sum_{m=1}^inf a_m * cos(2*pi*m*R)
# Now: compare with Bragg sum formula.
# sigma^2(R) = (const) * sum_{m>=1} I_m * (2sin(G_m*R)/G_m)^2
# = (const) * sum_{m>=1} I_m * (1 - cos(2*G_m*R)) / G_m^2 * 2  [since sin^2 = (1-cos(2x))/2]
# Wait: (2sin(GR)/G)^2 = 4sin^2(GR)/G^2 = 2*(1-cos(2GR))/G^2

# So: sigma^2(R) = (rho/pi) * sum_{m>=1} I_m * 2*(1-cos(2*G_m*R))/G_m^2
# = (2*rho/pi) * sum_m I_m/G_m^2 - (2*rho/pi) * sum_m I_m*cos(2*G_m*R)/G_m^2
#
# Comparing with Fourier series:
# sigma^2(R) = Lambda_bar + oscillating part
#
# The DC (constant) part = (2*rho/pi) * sum_m I_m/G_m^2 = Lambda_bar  ✓
# The oscillating part: -(2*rho/pi) * sum_m I_m * cos(2*G_m*R) / G_m^2
# = sum_m a_m * cos(2*pi*m*R)  [Fourier expansion]
#
# For the oscillating cosines to match, G_m = pi*m (so that 2*G_m = 2*pi*m).
# BUT we said G_m = 2*pi*m for the integer lattice! So 2*G_m = 4*pi*m, not 2*pi*m.
#
# AH WAIT. The issue is the definition of the window.
# sigma^2(R) counts points in the interval [-R, R] (half-width R).
# The Bragg peaks of the INTEGER LATTICE are at G = 2*pi*m (spacing 1 in real space).
# The Fourier series in R has oscillation frequencies 2*G_m = 4*pi*m, but we observe
# oscillation frequency 2*pi*m.
#
# RESOLUTION: the period-1 oscillation of sigma^2(R) for the integer lattice
# has frequency 1 in R (not frequency G/pi). So the Bragg sum formula gives
# oscillations at frequency G/(pi) = 2*pi*m/pi = 2*m per unit R,
# but the actual oscillations are at frequency m per unit R.
#
# This means the formula sigma^2(R) = (rho/pi)*sum I_G * (2sin(GR)/G)^2
# gives oscillations at frequency G*R/pi, while the actual oscillations
# in the INTEGER LATTICE are at frequency R (period 1).
#
# For the integer lattice: Bragg peaks at G=2*pi*m.
# Oscillation period in sigma^2: pi/(G) = 1/2m per unit R.
# But the actual period is 1. This means the sum formula doesn't reproduce
# the correct oscillation PERIOD for the integer lattice!
#
# This is because the formula uses (2sin(GR)/G)^2 which oscillates with period pi/G,
# but the variance oscillates with period 1/rho = 1 (mean spacing).
#
# For the integer lattice specifically:
# pi/(2*pi*m) = 1/(2m) -- the oscillation period from the Bragg formula is 1/(2m),
# but the actual period in sigma^2 is 1.
#
# So the formula sigma^2(R) = (rho/pi)*sum I_G*(2sin(GR)/G)^2 is WRONG for discrete point processes?
#
# NO -- let's check numerically what the sum ACTUALLY gives:
print("\n\nNumerical check: sum formula sigma^2 vs exact, over R in [0,2]:")
R_check = np.linspace(0.01, 2, 200)
for R in R_check[:10]:
    exact = sigma2_lattice_exact(R)
    bragg = sum((2*np.sin(2*np.pi*m*R)/(2*np.pi*m))**2 for m in range(1,200)) / np.pi
    print(f"  R={R:.3f}: exact={exact:.6f}, bragg/pi={bragg:.6f}, ratio={exact/(bragg+1e-15):.4f}")

# =========================================================
# Let me look at this completely fresh using the known T&S formula
# =========================================================
print()
print("=" * 60)
print("FRESH DERIVATION USING KNOWN T&S RESULT")
print("=" * 60)
print()
# From T&S 2003, the asymptotic formula for B_N in d=1 (eq. 73 of Zachary & Torquato):
# B_N = lim_{beta->0+} [phi/(2*beta) - (1/2) * sum_k Z_k * r_k * exp(-beta*r_k^2)]
# where Z_k = coordination number of k-th shell, r_k = radius of k-th shell
# phi = rho * v(D/2) where D = some length scale (mean spacing), v(D/2) = 2*(D/2) = D
# For the integer lattice with D=1 (mean spacing): phi = rho*D = 1*1 = 1.
# Z_k = 2 for each shell (two directions), r_k = k*D = k.
# B_N = lim [1/(2*beta) - (1/2) * sum_{k=1}^inf 2*k*exp(-beta*k^2)]
# = lim [1/(2*beta) - sum_{k=1}^inf k*exp(-beta*k^2)]
#
# Using Euler-Maclaurin: sum_{k=1}^inf k*e^{-beta*k^2} ~ integral_0^inf k*e^{-beta*k^2} dk - (1/2)*lim_{k->0} k*e^{-beta*k^2} + ...
# integral_0^inf k*e^{-beta*k^2} dk = 1/(2*beta)
# correction terms from Euler-Maclaurin = -0 (since f(0)=0) + (1/12)*f'(0) + ...
# f(k) = k*e^{-beta*k^2}, f'(k) = e^{-beta*k^2} - 2*beta*k^2*e^{-beta*k^2}
# f'(0) = 1
# So sum ~ 1/(2*beta) - (1/12) + O(beta)
# B_N = lim [1/(2*beta) - (1/(2*beta) - 1/12)] = 1/12 ✓
# And Lambda_bar = 2*phi*B_N = 2*1*(1/12) = 1/6 ✓

# Now, the BRAGG PEAK FORMULA for B_N in 1D:
# From the Fourier space formula for B_N (using the Epstein zeta function):
# B_N = -(phi*kappa(1)) / (D*v(D/2)) * integral h(r)*r*dr  ... for disordered
# For ORDERED systems, use the theta-series (eq 73 of Z&T).
#
# THE KEY FORMULA for HYPERUNIFORM systems:
# Lambda_bar = 2*phi*B_N (with phi = rho*D, and D = 1/rho for unit density)
# = 2*rho*(1/rho) * B_N = 2*B_N
# B_N = 1/12 -> Lambda_bar = 1/6 ✓
#
# For quasiperiodic chains, we can compute B_N from the pair correlation function
# using the regularized theta-series formula. The key connection to Bragg peaks:
#
# For a purely point-supported pair correlation function (all peaks, no continuous part):
# g_2(r) = sum_k Z_k * delta(r - r_k) / (rho * s_1(r_k))   [eq. 70 of Z&T]
# where s_1(r) = 2 (surface area of 1-sphere = 2 points at ±r in 1D)
# So g_2(r) = sum_{k: r_k>0} Z_k * delta(r - r_k) / (2*rho)
# For integer lattice: Z_k = 2, r_k = k (k=1,2,...), rho=1:
# g_2(r) = sum_{k=1}^inf delta(r-k)  ✓
#
# The B_N formula (T&S 2003, eq. 22, kappa(d=1) = 1):
# B_N = -phi/(D*v(D/2)) * integral h(r)*r*dr
# For integer lattice: h(r) = sum_{k!=0} delta(r-k) - 1
# This integral DIVERGES -- which is why Z&T use the regularization (eq. 71-73).
#
# ALTERNATIVE: The connection between B_N and the structure factor (Fourier space):
# From eq. (22) of Z&T (using Parseval's theorem):
# B_N = -(phi*kappa(d))/(D*v(D/2)) * (1/(2*pi)^d) * integral h_hat(k) / k dk  [d=1]
# But h_hat(k) = (S(k)-1)/rho, so:
# B_N = -(phi*kappa(1))/(D*v(D/2)) * (1/(2*pi)) * integral (S(k)-1)/(rho*k) dk
# In 1D with phi=rho*D, v(D/2)=D, kappa(1)=1:
# B_N = -1/(2*pi) * integral (S(k)-1)/k dk  [rho factors cancel]
# For hyperuniform S(k)->0 as k->0:
# B_N = -(1/(2*pi)) * integral_0^inf 2*(S(k)-1)/k dk  [symmetrize]
# For S(k) with Bragg peaks: S(k) = sum_G I_G * delta(k-G)
# B_N = -(1/(2*pi)) * integral 2*(sum_G I_G delta - 1)/k dk
#
# THIS DOES NOT CONVERGE (delta functions divided by k).
#
# The Bragg sum formula lambda_bar = (2*rho/pi) sum I_G/G^2 does NOT come directly
# from eq. 22 of Z&T. Let me re-derive it from the variance formula.
#
# The CORRECT derivation of Lambda_bar = (2*rho/pi) * sum I_G/G^2:
#
# Starting from: sigma^2(R) = rho*2R + rho^2 * integral h(r) * alpha(r;R) dr
# where alpha(r;R) = 1 - r/(2R) for 0<r<2R.
#
# For Class I hyperuniform system: sigma^2(R) = Lambda_bar + oscillating parts
# (bounded variance, no growth in R). The Lambda_bar is the time-average.
#
# KEY: Lambda_bar = lim_{R->inf} (1/R) * integral_0^R sigma^2(R') dR'
#
# From Fourier representation:
# sigma^2(R) = (rho/pi) * integral_{-inf}^{inf} S_c(k) * sin^2(kR) / k^2 dk * 2
# where S_c(k) = S(k) - rho*2*pi*delta(k) is the CONNECTED structure factor.
# For purely Bragg: S_c(k) = sum_{G!=0} I_G * delta(k-G)
# sigma^2(R) = (rho/pi) * sum_{G!=0} I_G * sin^2(GR) / G^2 * 2  [symmetric, G and -G]
# = (2*rho/pi) * sum_{G>0} I_G * (sin^2(GR) + sin^2(-GR)) / G^2
# = (4*rho/pi) * sum_{G>0} I_G * sin^2(GR) / G^2
#
# Lambda_bar = lim (1/R) * integral_0^R sigma^2 dR
# = (4*rho/pi) * sum_{G>0} I_G / G^2 * lim (1/R) * integral_0^R sin^2(GR') dR'
# = (4*rho/pi) * sum_{G>0} I_G / G^2 * (1/2)
# = (2*rho/pi) * sum_{G>0} I_G / G^2   ✓ (formula is correct)
#
# BUT: what is I_G?
# In the formula sigma^2(R) = (rho/pi) * integral S_c(k) * (2*sin(kR)/k)^2 dk:
# This uses the 1-SIDED convention (integral from 0 to inf):
# sigma^2(R) = (rho/pi) * integral_0^inf S_c(k) * (2*sin(kR)/k)^2 dk * 2/(2pi)... hmm.
#
# Let me be very explicit. In 1D:
# sigma^2(R) = rho*v(R) + rho^2 * integral_{-inf}^{inf} (v(R) - v_{int}(r;R)) h(r) dr
# where v(R)=2R and v_{int}(r;R) = max(0, 2R-|r|) (intersection of two intervals of size 2R)
# = rho*2R + rho^2 * integral_{-2R}^{2R} (2R - |r|) h(r) dr
# = rho*2R + 2*rho^2 * integral_0^{2R} (2R - r) h(r) dr
#
# Taking the Fourier transform of (2R-r)*theta(2R-r) (as function of r):
# integral_{-inf}^{inf} (2R - |r|)*theta(2R-|r|) * e^{-ikr} dr = 4R * sin^2(kR) / k^2
# (this is the standard result for triangular function)
#
# So: sigma^2(R) = rho*2R + 2*rho^2 * (1/(2*pi)) * integral h_hat(k) * 4R*sin^2(kR)/k^2 dk
# [using Parseval: integral f(r)g(r)dr = (1/2pi)*integral f_hat(k)*g_hat(-k)dk]
# = rho*2R + (4*R*rho^2)/(2*pi) * integral (S(k)/rho - 1/rho) * sin^2(kR)/k^2 dk
# Wait, h_hat(k) = (S(k) - 1)/rho for rho=1? Let me be careful.
# h(r) = g_2(r) - 1, and S(k) = 1 + rho*h_hat(k), so h_hat(k) = (S(k)-1)/rho.
#
# sigma^2(R) = rho*2R + (4*rho^2/(2*pi)) * integral (S(k)-1)/rho * R*sin^2(kR)/k^2 dk
# = rho*2R + (2*rho*R/pi) * integral_0^inf 2*(S(k)-1)/k^2 * sin^2(kR) dk  [symmetrize]
#
# For HYPERUNIFORM system (A_N=0, meaning integral (S(k)-1)dk = 0):
# The rho*2R term must be compensated. Let me separate:
# sigma^2(R) = (2*rho*R/pi) * [pi + integral_0^inf 2*(S(k)-1)*sin^2(kR)/k^2 dk]
# Hmm, that doesn't simplify nicely either.
#
# I think the issue is that the CORRECT formula I should use is:
# sigma^2(R) = -(rho/pi) * integral_0^inf S'(k) * (2sin(kR)/k) * (2cos(kR)) dk  [integration by parts]
# But this is getting circular.
#
# BOTTOM LINE: Let me just verify the formula NUMERICALLY for the integer lattice.
# I'll compute sigma^2(R) from the Bragg sum formula and see what prefactor
# is needed to match the exact formula.

print("\n\n===========================")
print("NUMERICAL CALIBRATION")
print("===========================")
print()
print("Integer lattice, rho=1, G_m = 2*pi*m, I_m = ???")
print("Testing formula: sigma^2(R) = C * sum_{m>=1} sin^2(G_m*R) / G_m^2")
print()
# If sigma^2(R) = 2*f*(1-2*f) for f=frac(R):
# The Fourier series of 2*f*(1-2*f) over f in [0,1]:
# a_0 = 1/6
# a_m = -1/(pi^2*m^2) for m>=1 (need to verify)
# Check: 1/6 - sum_{m>=1} cos(2*pi*m*f)/(pi^2*m^2) at f=0:
val_at_0 = 1/6 - sum(1/(np.pi**2*m**2) for m in range(1,10001))
print(f"sigma^2(0) from Fourier series: {val_at_0:.8f} (should be 0)")
# sin^2(G_m*R) = (1-cos(2*G_m*R))/2 = (1 - cos(4*pi*m*R))/2
# This oscillates at frequency 2*G_m = 4*pi*m per unit R. But sigma^2 oscillates at 2*pi*m.
# So the formula sin^2(G_m*R)/G_m^2 does NOT reproduce the Fourier series of sigma^2.
#
# THE RESOLUTION: For the integer lattice, the 'Bragg peaks' in g_2 are at r_k = k
# (integer separations), not at G = 2*pi*m.  The formula sigma^2 = sum_k f(r_k)
# involves the REAL-SPACE peaks, not the Fourier-space peaks.
#
# The formula Lambda_bar = (2*rho/pi) * sum_G I_G/G^2 uses FOURIER SPACE peaks.
# For the integer lattice: G_m = 2*pi*m, I_m = 1.
# Lambda_bar = (2/pi) * sum_{m>=1} 1/(2*pi*m)^2 = (2/pi) * (1/4*pi^2) * pi^2/6 = 1/(12*pi)
# But Lambda_bar should be 1/6!

# The ratio is: (1/6) / (1/(12*pi)) = 2*pi = 6.283...
# So the formula needs an extra factor of 2*pi??

# Let me check: if I_m = 2*pi (= unit cell volume in 1D = spacing = 1? no...)
# lambda_bar = (2/pi) * sum 2*pi/(2*pi*m)^2 = (2/pi) * (2*pi) * sum 1/(4*pi^2*m^2)
# = (4) * (1/(4*pi^2)) * pi^2/6 = 1/6 ✓ !!!

# So the correct formula is Lambda_bar = (2*rho/pi) * sum_{G>0} I_G / G^2
# BUT with I_G = 2*pi * (intensity per peak in S(k)/N).
#
# In other words: I_G should be the WEIGHT of the Bragg delta function,
# i.e., I_G = integral_{G-eps}^{G+eps} S(k) dk / (2*pi)  ... hmm
# Or: I_G = 1/(2*pi) * lim_{N->inf} S(G)/N * 2*pi  ... still 1.
#
# WAIT. Let me think about this more carefully.
#
# The issue is the CONVENTION for the structure factor S(k).
#
# CONVENTION A (physics): S(k) = 1 + rho * h_hat(k) where h_hat is the ordinary FT.
# For integer lattice: S(k) = sum_m e^{-ikm} = 2*pi * sum_m delta(k - 2*pi*m)
# At each G=2*pi*m: S(G) = 2*pi (not 1!).
# So I_G = 2*pi.
#
# CONVENTION B (normalized): S_N(k) = (1/N) * |sum_j e^{ikx_j}|^2
# At G=2*pi*m: S_N(G) = N. So I_G = N (grows with N).
#
# CONVENTION C (per-particle normalized):
# S_pp(k) = (1/N) * |sum_j e^{ikx_j}|^2 at k-grid points 2*pi*p/L
# At G = 2*pi*m: S_pp = N (for the integer lattice, all N particles contribute coherently)
# After normalizing by N: I_G = 1.
#
# For the variance formula (from T&S 2003 eq. 2.13 in 1D):
# sigma^2(R) = rho*2R * [1 + rho * integral h(r) * alpha(r;R) dr]
# = rho*2R + rho^2 * integral h(r) * (2R - |r|) * theta(2R - |r|) dr
# The Fourier approach (using h_hat in CONVENTION A):
# h_hat(k) = (S(k) - 1) / rho   [in convention A, S(0)=rho*(2*pi)*sum delta -> diverges]
# This is the issue: in Convention A, S(k) is a DISTRIBUTION (sum of delta functions)
# and needs to be used with the corresponding Parseval formula:
#
# integral f(r)*g(r) dr = (1/(2*pi)) * integral f_hat(k) * g_hat(-k) dk
#
# integral h(r) * (2R-|r|)*theta(2R-|r|) dr
# = (1/(2*pi)) * integral h_hat(k) * FT{(2R-|r|)*theta}(k) dk
# = (1/(2*pi)) * integral (S(k)/rho - 1/rho) * 4R*sin^2(kR)/k^2 dk
# = (1/(2*pi*rho)) * integral (S(k) - 1) * 4R*sin^2(kR)/k^2 dk
#
# [NB: 1/rho here because h_hat = (S-1)/rho, so rho*h_hat = S-1]
#
# sigma^2(R) = rho*2R + rho^2 * (1/(2*pi*rho)) * integral (S(k)-1) * 4R*sin^2(kR)/k^2 dk
# = rho*2R + (rho/(2*pi)) * integral (S(k)-1) * 4R*sin^2(kR)/k^2 dk
# = rho*2R + (2*rho*R/pi) * integral_0^inf (S(k)-1) * 4*sin^2(kR)/k^2 dk  [even, 0 to inf factor 2]
#
# Hmm, let me re-do:
# integral_{-inf}^{inf} (S(k)-1) * 4R*sin^2(kR)/k^2 dk
# [For S(k) = 2*pi*sum_m delta(k-2*pi*m):
# = 4R * sum_{m!=0} sin^2(2*pi*m*R) / (2*pi*m)^2  [excluding m=0, since S(0)-1 = 2*pi*rho*delta(0)-1]
# Wait this is still confused. Let me just use the FINITE-SUM version.
#
# For a finite system of N points with spacing d=1 in box [0,L] (L=N):
# S(k) = (1/N) * |sum_{j=0}^{N-1} e^{ikj}|^2
# At k=G_m = 2*pi*m/L * L_period ... hmm. Let me use k=2*pi*m (continuum Bragg peaks).
# For large N: S(G_m) -> N for G_m = 2*pi*m (integer m!=0).
# This grows with N -- it's a Bragg peak of strength N.
#
# The CORRECT normalization for the continuum formula is:
# In the thermodynamic limit, S(k) = (L/N) * sum_m I_m * L * delta(k-G_m)... no.
#
# Actually, in the thermodynamic limit for a LATTICE:
# S(k) = rho * FT[g_2(r)] + 1 [= 1 + rho*h_hat(k)]
# For integer lattice (rho=1, d=1):
# g_2(r) = sum_{m>=1} [delta(r-m) + delta(r+m)]  [for r!=0]
# FT[g_2](k) = integral g_2(r) e^{-ikr} dr = 2*sum_{m>=1} cos(km)
# = sum_{m=-inf}^{inf} e^{-ikm} - 1  [subtract m=0]
# = 2*pi * sum_m delta(k-2*pi*m) - 1  [Poisson sum formula]
# h_hat(k) = FT[g_2](k) - FT[1](k) = FT[g_2](k) [since FT[1] = 2*pi*delta(k)]
# Wait: h(r) = g_2(r) - 1, so h_hat(k) = FT[g_2](k) - FT[1](k) = FT[g_2](k) - 2*pi*delta(k)
# S(k) = 1 + rho*h_hat(k) = 1 + 1*(2*pi*sum_m delta(k-2*pi*m) - 1 - 2*pi*delta(k))
# = 1 + 2*pi*sum_{m!=0} delta(k-2*pi*m) - 1 + ???
# This is getting messy. Let me just accept the answer:
#
# For the integer lattice (rho=1, d=1):
# S(k) = 2*pi * sum_{m=-inf}^{inf} delta(k - 2*pi*m)
# [This is the standard result: the structure factor of a 1D lattice is an array of delta functions
#  with weight 2*pi each (the 2*pi comes from the convention int dk/(2*pi))]
#
# So I_G = 2*pi (weight of each Bragg delta function).
#
# Lambda_bar = (2*rho/pi) * sum_{G>0} I_G / G^2
# = (2*1/pi) * sum_{m>=1} 2*pi / (2*pi*m)^2
# = (2/pi) * 2*pi * sum_{m>=1} 1/(4*pi^2*m^2)
# = (4) * (1/(4*pi^2)) * sum 1/m^2
# = (1/pi^2) * pi^2/6 = 1/6  ✓ !!!!!!

print()
print("EUREKA!")
print("For integer lattice: I_G = 2*pi (the weight of each Bragg delta function)")
print("Lambda_bar = (2*rho/pi) * sum_{m>=1} 2*pi/(2*pi*m)^2")
val = (2*1/np.pi) * sum(2*np.pi/(2*np.pi*m)**2 for m in range(1, 100001))
print(f"  = {val:.8f}  (expected 1/6 = {1/6:.8f})")
print()
print("So the CORRECT FORMULA for Bragg peak intensities uses I_G = 2*pi * |F_G|^2")
print("where |F_G|^2 is the normalized structure factor per particle.")
print()
print("For quasiperiodic tilings: I_G = 2*pi * lim_{N->inf} S(G)/N")
print("where S(G) is the structure factor peak height (in the S(k)=|rho_hat|^2/N sense).")
print()
print("The factor of 2*pi arises because S(k) as a distribution has peaks of weight 2*pi,")
print("while in finite-N calculations S(G)/N -> 1 for the integer lattice.")
print()
print("EQUIVALENTLY: Lambda_bar = (4*rho) * sum_{G>0} |F_G|^2 / G^2")
print("where |F_G|^2 = lim S(G)/N (peak of normalized structure factor per particle).")
#
# Now for quasiperiodic tilings: |F_G|^2 = sinc^2(pi*(q - p/lambda1))
# (this is the cut-and-project window function result)
# So: Lambda_bar = (4*rho) * sum_{G>0} sinc^2(pi*(q-p/lam1)) / G^2
#
# Let's verify with the integer lattice first:
# For integer lattice: lambda1=1, G_m = 2*pi*m (only m=p, q=0 gives G=2*pi*p)
# sinc^2(pi*(0 - p/1)) = sinc^2(-pi*p) = sinc^2(pi*p)
# For p=1,2,3,...: sinc(pi*p) = sin(pi*p)/(pi*p) = 0  (p is nonzero integer)!
# That gives 0 intensity for all peaks -- WRONG.
#
# Hmm, the cut-and-project formula doesn't work for the integer lattice (1D Bravais lattice).
# The cut-and-project formula is for QUASIPERIODIC tilings, not for 1D lattices.
# For the quasiperiodic Silver chain (irrational lambda1=1+sqrt(2)):
# sinc^2(pi*(q - p/lambda1)) is generally nonzero since q-p/lambda1 is irrational (not integer).
#
# For the Integer LATTICE, we need a different approach.
# The calibration (2*pi factor) comes from the THERMODYNAMIC LIMIT of S(k):
# S(k) ~ N * delta_{periodic}(k-G_m) (in a box of size L=N, k-space spacing 2*pi/N)
# The weight of the Bragg peak integrated over one k-space period 2*pi/N:
# integral S dk ~ S(G_m) * (2*pi/N) = N * 2*pi/N = 2*pi
# So I_G = 2*pi for the integer lattice.
# For quasiperiodic tiling (finite system, L points in box of size L):
# At a Bragg peak G: S(G) = |sum_j e^{iGx_j}|^2 / N  (normalized per particle)
# The weight = S(G) * (something). For the quasiperiodic tiling, two Bragg peaks at
# nearby k values from the same (p,q) pair and a nearby (p',q') pair... complex.
#
# For PRACTICAL PURPOSES, let me just use the FFT approach:
# 1. Generate chain of N ~ 10^6 points
# 2. Compute S(k) numerically (normalized as S(k) = |rho_hat(k)|^2 / N)
# 3. Find Bragg peak heights S(G_i)
# 4. Lambda_bar = (4*rho) * sum_{G>0} S(G_i)/N / G_i^2
#    [using I_G = 2*pi * S(G)/N, and (2*rho/pi) * 2*pi = 4*rho]
#
# This gives Lambda_bar DIRECTLY from the numerical S(k).

print("=" * 60)
print("CORRECTED LAMBDA_BAR FORMULA:")
print()
print("   Lambda_bar = (4*rho) * sum_{G>0} [S(G)/N] / G^2")
print()
print("where S(G) is the peak height of S(k) = |rho_hat(k)|^2 / N")
print("and the sum is over all positive Bragg wavevectors G.")
print()
print("Equivalently, using the cut-and-project formula I_G = sinc^2(...):")
print("   Lambda_bar = (4*rho) * sum_{G>0} sinc^2(pi*(q - p/lambda1)) / G^2")
print()

# Verify: for integer lattice with I_G = 1 (S(G)/N = 1 at each G=2*pi*m):
val2 = 4*1 * sum(1/(2*np.pi*m)**2 for m in range(1, 100001))
print(f"Integer lattice: Lambda_bar = {val2:.8f}  (expected 1/6={1/6:.8f})")

# For Fibonacci chain with I_G = sinc^2(pi*(q-p/lam1)) and G = 2*pi*(p+q*lam1)/ell_bar:
lam1_fib = (1+np.sqrt(5))/2
ell_bar_fib = (1 + lam1_fib**2)/(1+lam1_fib)
rho_fib = 1/ell_bar_fib

total_fib = 0.0
P = 500
for p in range(-P, P+1):
    for q in range(-P, P+1):
        if p==0 and q==0:
            continue
        t = p + q*lam1_fib
        if t <= 0:
            continue
        s = q - p/lam1_fib
        if abs(s) < 1e-12:
            I = 1.0
        else:
            I = (np.sin(np.pi*s)/(np.pi*s))**2
        G = 2*np.pi*t/ell_bar_fib
        total_fib += I/G**2

lb_fib = 4*rho_fib * total_fib
print(f"Fibonacci: Lambda_bar = {lb_fib:.8f}  (expected 0.20110)")

# Silver
lam1_sil = 1 + np.sqrt(2)
ell_bar_sil = 2.0  # = (1+lam1^2)/(1+lam1) = (1+3+2*sqrt2)/(2+sqrt2) = (4+2sqrt2)/(2+sqrt2) = 2
rho_sil = 0.5

total_sil = 0.0
for p in range(-P, P+1):
    for q in range(-P, P+1):
        if p==0 and q==0:
            continue
        t = p + q*lam1_sil
        if t <= 0:
            continue
        s = q - p/lam1_sil
        if abs(s) < 1e-12:
            I = 1.0
        else:
            I = (np.sin(np.pi*s)/(np.pi*s))**2
        G = 2*np.pi*t/ell_bar_sil
        total_sil += I/G**2

lb_sil = 4*rho_sil * total_sil
print(f"Silver:    Lambda_bar = {lb_sil:.8f}  (expected ~0.2500)")
print(f"           diff from 1/4: {lb_sil - 0.25:.2e}")
