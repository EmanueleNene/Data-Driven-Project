import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import warnings
from pysindy.utils.axes import AxesWarning
warnings.filterwarnings("ignore", category=AxesWarning)

# -----------------------------
# True material (Voce + Perzyna)
# -----------------------------
E = 2000.0
sigy0 = 5.0
Q, b  = 12.0, 40.0             # Voce: sig_y = sigy0 + Q(1 - exp(-b*ep))
eta_vp = 20.0                  # Perzyna viscosity

def simulate(eps, dt):
    ep = np.zeros_like(eps); sig = np.zeros_like(eps);
    sig_y = np.zeros_like(eps); f = np.zeros_like(eps)
    for k in range(1, len(eps)):
        sig[k-1] = E*(eps[k-1] - ep[k-1]) # elastic predictor / trial stress
        sig_y[k-1] = sigy0 # no hardeneing
        sig_y[k-1] = sigy0 + Q*(1.0 - np.exp(-b*ep[k-1])) # Voce hardening
        f[k-1] = sig[k-1] - sig_y[k-1] # yield function (overstress)
        ep_dot = max(0.0, f[k-1]) # normal flow check
        ep_dot = max(0.0, f[k-1])/eta_vp # Perzyna viscosity
        ep[k] = ep[k-1] + dt * ep_dot # explicit Euler
    # last step
    sig[-1] = E*(eps[-1] - ep[-1]) # last stress value outside loop
    sig_y[-1] = sigy0 + Q*(1.0 - np.exp(-b*ep[-1]))
    f[-1]    = sig[-1] - sig_y[-1]
    return sig, ep, sig_y, f

# -----------------------------
# Strain paths (train / validate)
# -----------------------------
dt = 1e-3
t  = np.arange(0, 20, dt)
eps_tr = 0.006*np.sin(2*np.pi*0.5*t)*np.exp(-0.1*t) + 0.0005*t
eps_va = 0.007*np.sin(2*np.pi*0.35*t)*np.exp(-0.05*t) + 0.0005*t

# note that sigy, f are not used for training, just for plotting
# since they can't be measured in experiments, it's cheating to use them in SINDy
sig_tr, ep_tr, sigy_tr, f_tr = simulate(eps_tr, dt)
sig_va, ep_va, sigy_va, f_va = simulate(eps_va, dt)

epsdot_tr = np.gradient(eps_tr, dt); epdot_tr = np.gradient(ep_tr, dt)
epsdot_va = np.gradient(eps_va, dt); epdot_va = np.gradient(ep_va, dt)

# -----------------------------
# Overstress features: X = [over, ep, eps_dot]
# -----------------------------
# Estimate yield stress from first plastic point
sigy0_hat = sig_tr[np.argmax(ep_tr > 0)]
over_tr   = sig_tr - sigy0_hat
over_va   = sig_va - sigy0_hat

# Plastic-only mask (training)
tol = 1e-10
mask_pl = epdot_tr > tol
print(f"Plastic fraction (train): {mask_pl.mean():.3f}")

X_raw = np.column_stack([over_tr, ep_tr, epsdot_tr])[mask_pl]
y_raw = epdot_tr[mask_pl]  # target ep_dot (plastic only)

# -----------------------------
# Θ via PolynomialLibrary on [over, ep, eps_dot]
# -----------------------------
names = ["over","ep","eps_dot"]
poly = ps.PolynomialLibrary(degree=2, include_bias=False, include_interaction=False)
Phi_raw = poly.fit_transform(X_raw)
phi_names  = poly.get_feature_names(input_features=names)

# -----------------------------
# Z-score Θ and y (fit on TRAIN-plastic)
# -----------------------------
Xmu  = Phi_raw.mean(0);  Xstd = Phi_raw.std(0) + 1e-12 # avoid div-by-0
ymu  = y_raw.mean();     ystd = y_raw.std()   + 1e-12

Phi = (Phi_raw - Xmu)/Xstd
y   = (y_raw  - ymu)/ystd

# -----------------------------
# Fit (no sweep)
# -----------------------------
lib = ps.IdentityLibrary()
opt = ps.STLSQ(threshold=0.05, alpha=0.0)  # <- tune threshold here (0.02..0.15)
opt = ps.SR3(reg_weight_lam=0.3, relax_coeff_nu=1.5,max_iter=10000, tol=1e-12)
opt = ps.SR3(reg_weight_lam=0.1, relax_coeff_nu=1)

model = ps.SINDy(feature_library=lib, optimizer=opt)
model.feature_names = phi_names
model.state_names   = ["ep"]
model.fit(Phi, t=dt, x_dot=y)

# -----------------------------
# Coefficients (scaled & unscaled) + print
# -----------------------------
coefs_s = np.ravel(model.coefficients()[0])         # scaled Θ coefficients
w_unscaled = (ystd / Xstd) * coefs_s                # unscaled Θ coefficients
b_unscaled = ymu - np.dot(w_unscaled, Xmu)          # intercept

w_len = max(12, max(len(n) for n in phi_names))
print("\n=== Coefficients (scaled) ===")
for c, n in zip(coefs_s, phi_names):
    print(f"{c:+12.6e}  *  {n:<{w_len}}")
print("\nIntercept (unscaled):", f"{b_unscaled:+.6e}")
print("Top terms (|coef|, unscaled):")
order = np.argsort(-np.abs(w_unscaled))
for j in order[:8]:
    if abs(w_unscaled[j]) > 0:
        print(f"{w_unscaled[j]:+.6e}  *  {phi_names[j]}")

# -----------------------------
# FAST RHS: ep_dot from (over, ep, eps_dot), no model.predict
# -----------------------------
def epdot_fast(over, ep, epsdot):
    Phi_row = poly.transform(np.array([[over, ep, epsdot]]) )[0]
    g = float(b_unscaled + np.dot(w_unscaled, Phi_row))
#    return g   # real output    
    return max(0.0, g)  # mod so epdot is never negative (physical constraint)

# -----------------------------
# Rollout on VALIDATION
# -----------------------------
ep_hat = np.zeros_like(ep_va)
for k in range(1, len(t)):
    sig_k  = E*(eps_va[k-1] - ep_hat[k-1])
    over_k = sig_k - sigy0_hat
    gk     = epdot_fast(over_k, ep_hat[k-1], epsdot_va[k-1])
    ep_hat[k] = ep_hat[k-1] + dt*gk
sig_hat = E*(eps_va - ep_hat)

rmse_sig = np.sqrt(np.mean((sig_hat - sig_va)**2))
rmse_ep  = np.sqrt(np.mean((ep_hat  - ep_va )**2))
print(f"\nValidation RMSE  σ: {rmse_sig:.4e}   ep: {rmse_ep:.4e}")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(eps_va, sig_va,  lw=2, label="True")
plt.plot(eps_va, sig_hat, "--", lw=2, label="SINDy rollout")
plt.xlabel("ε"); plt.ylabel("σ"); plt.title("Validation: stress–strain")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3))
plt.plot(t, ep_va,  lw=2, label="True ep")
plt.plot(t, ep_hat, "--", lw=2, label="SINDy ep")
plt.xlabel("t"); plt.ylabel("ep"); plt.title("Validation: plastic strain")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3))
plt.plot(ep_va, sig_va,  lw=2, label="True hardening (σ vs ep)")
plt.plot(ep_hat, sig_hat, "--", lw=2, label="SINDy hardening")
plt.xlabel("ep"); plt.ylabel("σ"); plt.title("Validation: hardening curve")
plt.legend(); plt.tight_layout(); plt.show()
