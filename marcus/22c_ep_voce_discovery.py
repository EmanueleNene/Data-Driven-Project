# src/22c_ep_voce_discovery.py
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from pysindy.feature_library import CustomLibrary

# -------- True parameters (Voce + Perzyna) --------
E = 2000.0
sigy0, Q, b = 5.0, 12.0, 40.0      # Voce: sig_y = sigy0 + Q(1 - exp(-b*ep))
eta = 20.0                         # Perzyna viscosity

def simulate(eps, dt):
    ep = np.zeros_like(eps); sig = np.zeros_like(eps)
    for k in range(1, len(eps)):
        sig[k-1] = E*(eps[k-1] - ep[k-1])
        sig_y = sigy0 + Q*(1.0 - np.exp(-b*ep[k-1]))
        f = sig[k-1] - sig_y
        ep[k] = ep[k-1] + dt * max(0.0, f)/eta
    sig[-1] = E*(eps[-1] - ep[-1])
    return sig, ep

# -------- Strain paths (train / validate) --------
dt = 1e-2; t = np.arange(0, 20, dt)
eps_tr = 0.006*np.sin(2*np.pi*0.5*t)*np.exp(-0.1*t) + 0.0005*t
eps_va = 0.007*np.sin(2*np.pi*0.35*t)*np.exp(-0.05*t) + 0.0005*t

sig_tr, ep_tr = simulate(eps_tr, dt)
sig_va, ep_va = simulate(eps_va, dt)

epsdot_tr = np.gradient(eps_tr, dt); epdot_tr = np.gradient(ep_tr, dt)
epsdot_va = np.gradient(eps_va, dt); epdot_va = np.gradient(ep_va, dt)

# -------- Library: linear states + gated (ReLU) combos --------
# inside your discovery script



sigy0 = 5.0  # your yield offset

def relu_sig_minus_y0(X):
    X = np.asarray(X); X = X if X.ndim==2 else X[:,None]
    sig = X[:, 0]                  # x0 = stress
    return np.maximum(0.0, sig - sigy0)     # r

def r_times_ep(X):
    X = np.asarray(X); X = X if X.ndim==2 else X[:,None]
    sig, ep = X[:,0], X[:,1]       # x1 = ep
    r = np.maximum(0.0, sig - sigy0)
    return r * ep

def r_times_ep2(X):
    X = np.asarray(X); X = X if X.ndim==2 else X[:,None]
    sig, ep = X[:,0], X[:,1]
    r = np.maximum(0.0, sig - sigy0)
    return r * ep * ep

custom_lib = CustomLibrary(
    library_functions=[relu_sig_minus_y0, r_times_ep, r_times_ep2],
    function_names=[
        lambda names: "relu(sig-sigy0)",
        lambda names: "relu(sig-sigy0)*ep",
        lambda names: "relu(sig-sigy0)*ep^2",
    ],
)

lib = ps.IdentityLibrary() + custom_lib   # identity gives linear [sig, ep, eps, eps_dot]


# -------- Arrange data: learn ep_dot = g(sig, ep, eps, eps_dot) --------
X_tr = np.column_stack([sig_tr, ep_tr, eps_tr, epsdot_tr])
X_va = np.column_stack([sig_va, ep_va, eps_va, epsdot_va])

# (Optional) scale features (helps sparsity)
Xmu, Xstd = X_tr.mean(0), X_tr.std(0) + 1e-12
Ymu, Ystd = epdot_tr.mean(), epdot_tr.std() + 1e-12
X_tr_s = (X_tr - Xmu)/Xstd; y_tr_s = (epdot_tr - Ymu)/Ystd
X_va_s = (X_va - Xmu)/Xstd

opt = ps.STLSQ(threshold=3e-3)  # tune 1e-3..1e-2 to get ~2–5 terms
model = ps.SINDy(feature_library=lib, optimizer=opt)
model.feature_names = ["sig","ep","eps","eps_dot"]
model.state_names   = ["ep"]
model.fit(X_tr_s, t=dt, x_dot=y_tr_s)

# ---- Robust print (bypass pretty-printer differences) ----
import numpy as np
feat = np.array(model.get_feature_names())
coef = np.ravel(model.coefficients()[0])
nz   = np.abs(coef) > 1e-10
print("\n=== Discovered evolution law (scaled) ===")
for c, n in sorted(zip(coef[nz], feat[nz]), key=lambda z: -abs(z[0])):
    print(f"{c:+.4g} * {n}")

# 1) See what features SINDy *actually* had access to:
feat = model.get_feature_names()
print("Features in library:", feat)
print("Num features:", len(feat))

# 2) How many ReLU columns survived the fit?
coef = np.ravel(model.coefficients()[0])
for c, n in zip(coef, feat):
    if "relu" in n:
        print(f"{n:35s}  coeff={c:+.3e}")

# 3) Was the gate ever on? (using your train data)
r = np.maximum(0.0, sig_tr - sigy0)  # or whatever gate you intended
print("Gate active fraction:", (r > 0).mean())


# -------- Rollout on validation with learned model --------
def predict_epdot_scaled(sig, ep, eps, epsdot):
    x = np.array([sig, ep, eps, epsdot])
    xs = (x - Xmu)/Xstd
    ys = model.predict(xs[None, :]).item()
    return ys*Ystd + Ymu

ep_hat = np.zeros_like(ep_va)
for k in range(1, len(t)):
    sig_k = E*(eps_va[k-1] - ep_hat[k-1])
    ep_hat[k] = ep_hat[k-1] + dt * predict_epdot_scaled(sig_k, ep_hat[k-1], eps_va[k-1], epsdot_va[k-1])
sig_hat = E*(eps_va - ep_hat)

rmse_sig = np.sqrt(np.mean((sig_hat - sig_va)**2))
rmse_ep  = np.sqrt(np.mean((ep_hat  - ep_va )**2))
print(f"\nValidation RMSE  σ: {rmse_sig:.4e}   ep: {rmse_ep:.4e}")

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
