import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from pysindy.feature_library import CustomLibrary

# -----------------------------
# Ground truth: elastic–perfectly-plastic + Perzyna
# -----------------------------
E = 2000.0        # elastic modulus
sig_y = 5.0       # yield stress (tension only)
eta_vp = 20.0     # viscosity -> ep_dot = <sig - sig_y>/eta_vp

def simulate_perzyna(eps, dt):
    ep  = np.zeros_like(eps)
    sig = np.zeros_like(eps)
    for k in range(1, len(eps)):
        sig[k-1] = E*(eps[k-1] - ep[k-1])
        f = sig[k-1] - sig_y
        ep_dot = max(0.0, f)/eta_vp
        ep[k] = ep[k-1] + dt*ep_dot
    sig[-1] = E*(eps[-1] - ep[-1])
    return sig, ep

# -----------------------------
# Strain paths (train vs val)
# -----------------------------
dt = 1e-3
t  = np.arange(0, 20, dt)
# TRAIN: damped sine + ramp
eps_tr = 0.006*np.sin(2*np.pi*0.5*t)*np.exp(-0.1*t) + 0.0005*t
# VAL: different frequency/amplitude & no ramp
eps_va = 0.007*np.sin(2*np.pi*0.35*t)*np.exp(-0.05*t)

sig_tr, ep_tr = simulate_perzyna(eps_tr, dt)
sig_va, ep_va = simulate_perzyna(eps_va, dt)

# Targets (time derivatives)
epdot_tr = np.gradient(ep_tr, dt)
epdot_va = np.gradient(ep_va, dt)
epsdot_tr = np.gradient(eps_tr, dt)
epsdot_va = np.gradient(eps_va, dt)

# -----------------------------
# Library: linear terms + ReLU yield & ReLU^2 (NO “cheating”: SINDy must choose)
#   X columns passed to library = [sig, ep, eps, eps_dot]
# -----------------------------
def relu_yield(X):
    X = np.asarray(X);  X = X if X.ndim==2 else X[:,None]
    sig_col = X[:,0]
    return np.maximum(0.0, sig_col - sig_y)

def relu_yield_sq(X):
    r = relu_yield(X)
    return r*r

lin_lib   = ps.IdentityLibrary()  # linear in [sig, ep, eps, eps_dot]
custom    = CustomLibrary(
    library_functions=[relu_yield, relu_yield_sq],
    function_names=["relu(sig-sigy)", "relu(sig-sigy)^2"]
)
lib = lin_lib + custom

# -----------------------------
# Arrange data for SINDy: learn ep_dot = g(sig, ep, eps, eps_dot)
# -----------------------------
X_tr = np.column_stack([sig_tr, ep_tr, eps_tr, epsdot_tr])
X_va = np.column_stack([sig_va, ep_va, eps_va, epsdot_va])

opt = ps.STLSQ(threshold=2e-3)    # tune up/down if too dense/sparse
model = ps.SINDy(feature_library=lib, optimizer=opt)
model.feature_names = ["sig","ep","eps","eps_dot"]  # names for linear part
model.state_names   = ["ep"]

model.fit(X_tr, t=dt, x_dot=epdot_tr)


# --- robust print of the learned RHS, bypassing pretty-printer ---
import numpy as np

try:
    feat = np.array(model.get_feature_names())
except Exception:
    # absolute fallback if feature names fail:
    feat = np.array([f"f{i}" for i in range(len(model.coefficients()[0].ravel()))])

coef = np.ravel(model.coefficients()[0])  # single state (ep)
rhs = " + ".join(f"{c:+.6g}*{n}" for c, n in zip(coef, feat) if abs(c) > 1e-12)
print("\n=== Discovered evolution law ===")
print(f"(ep)' = {rhs or '0'}")


model.equations

print(model)  # uses __str__ under the hood; avoids the shadowed attribute


print("\n=== Discovered evolution law (TRAIN) ===")
#for eq in model.equations():
#    print(eq)

# -----------------------------
# Validate: simulate with learned ep_dot on the VAL path
# -----------------------------
def predict_epdot_from_state(sig, ep, eps, eps_dot):
    Xrow = np.array([[sig, ep, eps, eps_dot]])
    return float(model.predict(Xrow)[0, 0]) # ep_dot

# forward Euler rollout using learned model on validation path
ep_hat = np.zeros_like(ep_va)
for k in range(1, len(t)):
    sig_k = E*(eps_va[k-1] - ep_hat[k-1])           # stress from elasticity
    ep_hat[k] = ep_hat[k-1] + dt * predict_epdot_from_state(
        sig_k, ep_hat[k-1], eps_va[k-1], epsdot_va[k-1]
    )
sig_hat = E*(eps_va - ep_hat)

# -----------------------------
# Simple metrics
# -----------------------------
rmse_sig = np.sqrt(np.mean((sig_hat - sig_va)**2))
rmse_ep  = np.sqrt(np.mean((ep_hat  - ep_va )**2))
print(f"\nValidation RMSE  σ: {rmse_sig:.4e}   ep: {rmse_ep:.4e}")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(eps_va, sig_va,  lw=2, label="True")
plt.plot(eps_va, sig_hat, "--", lw=2, label="SINDy rollout")
plt.xlabel("strain ε"); plt.ylabel("stress σ")
plt.title("Validation path: stress–strain")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3))
plt.plot(t, ep_va,  lw=2, label="True ep")
plt.plot(t, ep_hat, "--", lw=2, label="SINDy ep")
plt.xlabel("t"); plt.ylabel("ep")
plt.title("Validation: plastic strain vs time")
plt.legend(); plt.tight_layout(); plt.show()
