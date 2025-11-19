import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import warnings
from pysindy.utils.axes import AxesWarning
warnings.filterwarnings("ignore", category=AxesWarning)

# -----------------------------------------
# True material (3D J2 eq: Voce + Perzyna + viscoelastic dashpot)
# Interpret eps as ε_eq, sig as σ_eq (von Mises)
# -----------------------------------------
E      = 2000.0          # "elastic" modulus (eq)
eta_ve = 5.0             # viscoelastic dashpot (Kelvin-Voigt-style)
sigy0  = 5.0             # initial yield stress
Q, b   = 12.0, 40.0      # Voce: sig_y = sigy0 + Q(1 - exp(-b*ep))
eta_vp = 20.0            # Perzyna viscosity

def simulate(eps, dt):
    """
    Generate 'true' response for a 3D isotropic J2 eq model:
      σ_eq = E (ε_eq - ε_eq^p) + η_ve * ε̇_eq
      σ_y(ε^p) = sigy0 + Q(1 - exp(-b*ep))
      ε̇^p = <σ_eq - σ_y(ε^p)> / η_vp

    Returns:
      sig   : equivalent stress history
      ep    : equivalent plastic strain history
      sig_y : current yield stress history
      f     : overstress history
    """
    ep    = np.zeros_like(eps)
    sig   = np.zeros_like(eps)
    sig_y = np.zeros_like(eps)
    f     = np.zeros_like(eps)

    epsdot = np.gradient(eps, dt)

    for k in range(1, len(eps)):
        # viscoelastic + elastic trial stress
        sig[k-1]   = E * (eps[k-1] - ep[k-1]) + eta_ve * epsdot[k-1]
        sig_y[k-1] = sigy0 + Q * (1.0 - np.exp(-b * ep[k-1]))  # Voce hardening
        f[k-1]     = sig[k-1] - sig_y[k-1]                     # overstress
        ep_dot     = max(0.0, f[k-1]) / eta_vp                 # Perzyna
        ep[k]      = ep[k-1] + dt * ep_dot                     # explicit Euler

    # last step
    sig[-1]   = E * (eps[-1] - ep[-1]) + eta_ve * epsdot[-1]
    sig_y[-1] = sigy0 + Q * (1.0 - np.exp(-b * ep[-1]))
    f[-1]     = sig[-1] - sig_y[-1]
    return sig, ep, sig_y, f

# -----------------------------
# Strain paths (train / validate)
# (in 3D: interpret as equivalent strain histories)
# -----------------------------
dt = 1e-3
t  = np.arange(0, 20, dt)
eps_tr = 0.006*np.sin(2*np.pi*0.5*t)*np.exp(-0.1*t) + 0.0005*t
eps_va = 0.007*np.sin(2*np.pi*0.35*t)*np.exp(-0.05*t) + 0.0005*t

sig_tr, ep_tr, sigy_tr, f_tr = simulate(eps_tr, dt)
sig_va, ep_va, sigy_va, f_va = simulate(eps_va, dt)

epsdot_tr = np.gradient(eps_tr, dt)
epdot_tr  = np.gradient(ep_tr, dt)
epsdot_va = np.gradient(eps_va, dt)
epdot_va  = np.gradient(ep_va, dt)

# ============================================================
# 1) VISCOELASTIC PART: SINDy discovery of σ^ve = η_ve ε̇
#    (on purely elastic/visco data before yield)
# ============================================================

# Use only data BEFORE plastic starts: ep == 0 region
yield_idx = np.argmax(ep_tr > 0)   # first plastic point
if yield_idx == 0:
    raise RuntimeError("No plasticity in training path, cannot split elastic/plastic.")
mask_ve = np.arange(len(t)) < yield_idx
print(f"Elastic/visco fraction (train): {mask_ve.mean():.3f}")

sig_el     = sig_tr[mask_ve]
eps_el     = eps_tr[mask_ve]
epsdot_el  = epsdot_tr[mask_ve]

# Remove elastic part with known E, leaving dashpot contribution:
#   sig_dash = sig - E*eps = η_ve * epsdot
sig_dash_el = sig_el - E * eps_el

# Build design matrix with SINDy (here it's overkill but consistent with rest)
# We'll regress: sig_dash ≈ a * eps_dot  (polynomial degree 1)
X_ve_raw = epsdot_el.reshape(-1, 1)
names_ve = ["eps_dot"]
poly_ve  = ps.PolynomialLibrary(degree=1, include_bias=False, include_interaction=False)
Theta_ve_raw   = poly_ve.fit_transform(X_ve_raw)
phi_ve_names   = poly_ve.get_feature_names(input_features=names_ve)

# Z-score Θ_ve and y_ve
Xmu_ve  = Theta_ve_raw.mean(0)
Xstd_ve = Theta_ve_raw.std(0) + 1e-12
ymu_ve  = sig_dash_el.mean()
ystd_ve = sig_dash_el.std() + 1e-12

Theta_ve = (Theta_ve_raw - Xmu_ve) / Xstd_ve
y_ve     = (sig_dash_el   - ymu_ve) / ystd_ve

lib_ve = ps.IdentityLibrary()
opt_ve = ps.STLSQ(threshold=0.0, alpha=0.0)  # just linear LS, no sparsity

model_ve = ps.SINDy(feature_library=lib_ve, optimizer=opt_ve)
model_ve.feature_names = phi_ve_names
model_ve.state_names   = ["sig_dash"]
model_ve.fit(Theta_ve, t=dt, x_dot=y_ve)

# Coeffs: y ≈ b_ve + w_ve * Θ_raw
coefs_s_ve    = np.ravel(model_ve.coefficients()[0])
w_unscaled_ve = (ystd_ve / Xstd_ve) * coefs_s_ve
b_unscaled_ve = ymu_ve - np.dot(w_unscaled_ve, Xmu_ve)

eta_ve_hat = w_unscaled_ve[0]   # coefficient multiplying eps_dot
print(f"\n=== Viscoelastic SINDy fit ===")
print(f"True  eta_ve = {eta_ve:.4f}")
print(f"Fit   eta_ve = {eta_ve_hat:.4f}")
print(f"Intercept (dashpot) ~ {b_unscaled_ve:.3e} (should be near 0)")

def sig_dash_fast(epsdot):
    """Dashpot stress from SINDy viscoelastic model: sig_dash ≈ b + w*epsdot."""
    Theta_row = poly_ve.transform(np.array([[epsdot]]))[0]
    return float(b_unscaled_ve + np.dot(w_unscaled_ve, Theta_row))

# ============================================================
# 2) PLASTIC PART: SINDy discovery of ep_dot = g(over, ep, eps_dot)
# ============================================================

# Overstress features: X = [over, ep, eps_dot]
# Estimate initial yield stress from first plastic point
sigy0_hat = sig_tr[np.argmax(ep_tr > 0)]
over_tr   = sig_tr - sigy0_hat
over_va   = sig_va - sigy0_hat

# Plastic-only mask (training)
tol = 1e-10
mask_pl = epdot_tr > tol
print(f"Plastic fraction (train): {mask_pl.mean():.3f}")

X_raw = np.column_stack([over_tr, ep_tr, epsdot_tr])[mask_pl]
y_raw = epdot_tr[mask_pl]  # target ep_dot (plastic only)

# Θ via PolynomialLibrary on [over, ep, eps_dot]
names = ["over","ep","eps_dot"]
poly   = ps.PolynomialLibrary(degree=2, include_bias=False, include_interaction=False)
Phi_raw    = poly.fit_transform(X_raw)
phi_names  = poly.get_feature_names(input_features=names)

# Z-score Θ and y (fit on TRAIN-plastic)
Xmu  = Phi_raw.mean(0)
Xstd = Phi_raw.std(0) + 1e-12
ymu  = y_raw.mean()
ystd = y_raw.std() + 1e-12

Phi = (Phi_raw - Xmu)/Xstd
y   = (y_raw  - ymu)/ystd

# Fit plastic law with SINDy
lib = ps.IdentityLibrary()
opt = ps.SR3(reg_weight_lam=0.1, relax_coeff_nu=1.0, max_iter=10000, tol=1e-12)

model = ps.SINDy(feature_library=lib, optimizer=opt)
model.feature_names = phi_names
model.state_names   = ["ep"]
model.fit(Phi, t=dt, x_dot=y)

# Coefficients (scaled & unscaled) + print
coefs_s    = np.ravel(model.coefficients()[0])       # scaled Θ coefficients
w_unscaled = (ystd / Xstd) * coefs_s                # unscaled Θ coefficients
b_unscaled = ymu - np.dot(w_unscaled, Xmu)          # intercept

w_len = max(12, max(len(n) for n in phi_names))
print("\n=== Plastic law coefficients (scaled) ===")
for c, n in zip(coefs_s, phi_names):
    print(f"{c:+12.6e}  *  {n:<{w_len}}")

print("\nIntercept (unscaled):", f"{b_unscaled:+.6e}")
print("Top terms (|coef|, unscaled):")
order = np.argsort(-np.abs(w_unscaled))
for j in order[:8]:
    if abs(w_unscaled[j]) > 0:
        print(f"{w_unscaled[j]:+.6e}  *  {phi_names[j]}")

def epdot_fast(over, ep, epsdot):
    """Plastic strain rate from SINDy: ep_dot ≈ b + w·Φ_raw(over, ep, eps_dot)."""
    Phi_row = poly.transform(np.array([[over, ep, epsdot]]) )[0]
    g = float(b_unscaled + np.dot(w_unscaled, Phi_row))
    return max(0.0, g)  # enforce non-negative plastic flow

# ============================================================
# 3) Combined rollout on validation path:
#    σ = E(ε - ep) + σ_dash(epsdot)    with both parts from SINDy
# ============================================================

ep_hat  = np.zeros_like(ep_va)
sig_hat = np.zeros_like(sig_va)

# initial conditions: match the true initial state
ep_hat[0]  = ep_va[0]
sig_hat[0] = sig_va[0]

for k in range(1, len(t)):
    # viscoelastic dashpot from SINDy (pure function of epsdot)
    sig_dash_km1 = sig_dash_fast(epsdot_va[k-1])

    # elastic predictor minus plastic part
    sig_trial = E*(eps_va[k-1] - ep_hat[k-1]) + sig_dash_km1

    # plastic evolution law from SINDy (uses overstress = trial - sigy0_hat)
    over_k = sig_trial - sigy0_hat
    gk     = epdot_fast(over_k, ep_hat[k-1], epsdot_va[k-1])
    ep_hat[k] = ep_hat[k-1] + dt * gk

    # final stress at t_k
    sig_dash_k = sig_dash_fast(epsdot_va[k])
    sig_hat[k] = E*(eps_va[k] - ep_hat[k]) + sig_dash_k

# error metrics
rmse_sig = np.sqrt(np.mean((sig_hat - sig_va)**2))
rmse_ep  = np.sqrt(np.mean((ep_hat  - ep_va )**2))
print(f"\nValidation RMSE  σ: {rmse_sig:.4e}   ep: {rmse_ep:.4e}")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(eps_va, sig_va,  lw=2, label="True")
plt.plot(eps_va, sig_hat, "--", lw=2, label="SINDy rollout")
plt.xlabel("ε_eq"); plt.ylabel("σ_eq")
plt.title("Validation: von Mises equivalent stress–strain")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3))
plt.plot(t, ep_va,  lw=2, label="True ep_eq")
plt.plot(t, ep_hat, "--", lw=2, label="SINDy ep_eq")
plt.xlabel("t"); plt.ylabel("ep_eq")
plt.title("Validation: equivalent plastic strain")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3))
plt.plot(ep_va, sig_va,  lw=2, label="True hardening (σ_eq vs ep_eq)")
plt.plot(ep_hat, sig_hat, "--", lw=2, label="SINDy hardening")
plt.xlabel("ep_eq"); plt.ylabel("σ_eq")
plt.title("Validation: equivalent hardening curve")
plt.legend(); plt.tight_layout(); plt.show()
