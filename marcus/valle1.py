import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary, FourierLibrary, CustomLibrary

# --------------------------------
# 1. Ground-truth model generator
# Choose one: Neo-Hookean, Perzyna, Maxwell, Anand, etc.
# --------------------------------
def simulate_material(eps, dt, model="perzyna"):
    E = 2000.0
    sig_y = 5.0
    eta = 20.0
    H   = 50.0   # hardening modulus for plasticity
    ep = np.zeros_like(eps)
    sig = np.zeros_like(eps)

    for k in range(1, len(eps)):
        sig[k-1] = E*(eps[k-1] - ep[k-1])

        if model == "elastic":
            ep[k] = 0.0

        elif model == "perzyna":
            f = sig[k-1] - sig_y
            ep_dot = max(0, f)/eta
            ep[k] = ep[k-1] + dt*ep_dot

        elif model == "perzyna_hard":
            f = sig[k-1] - (sig_y + H*ep[k-1])
            ep_dot = max(0, f)/eta
            ep[k] = ep[k-1] + dt*ep_dot

        # add more cases: "anand", "maxwell", "kv", etc.

    sig[-1] = E*(eps[-1] - ep[-1])
    return sig, ep


# --------------------------------
# 2. Generate train/validation data
# --------------------------------
dt = 1e-2
t  = np.arange(0, 20, dt)
eps_tr = 0.006*np.sin(2*np.pi*0.5*t)*np.exp(-0.1*t) + 0.0005*t
eps_va = 0.007*np.sin(2*np.pi*0.35*t)*np.exp(-0.05*t)

sig_tr, ep_tr = simulate_material(eps_tr, dt, model="perzyna_hard")
sig_va, ep_va = simulate_material(eps_va, dt, model="perzyna_hard")

# Derivatives
epdot_tr  = np.gradient(ep_tr, dt)
epdot_va  = np.gradient(ep_va, dt)
epsdot_tr = np.gradient(eps_tr, dt)
epsdot_va = np.gradient(eps_va, dt)


# --------------------------------
# 3. Build general feature library
# --------------------------------
def relu_yield(X):
    X = np.atleast_2d(X)
    # fill with zeros if missing columns
    ncols = X.shape[1]
    sig_col = X[:,0] if ncols > 0 else np.zeros(X.shape[0])
    ep_col  = X[:,1] if ncols > 1 else np.zeros(X.shape[0])

    sig_y, H = 5.0, 50.0
    return np.maximum(0.0, sig_col - (sig_y + H*ep_col))

def sinh_sig(X):
    X = np.atleast_2d(X)
    return np.sinh(X[:,0])

def exp_sig(X):
    X = np.atleast_2d(X)
    return np.exp(X[:,0])

poly   = PolynomialLibrary(degree=3)
four   = FourierLibrary(n_frequencies=2)
custom = CustomLibrary(
    library_functions=[relu_yield, sinh_sig, exp_sig],
    function_names=["relu(sig - sigy - H*ep)", "sinh(sig)", "exp(sig)"]
)

lib = poly + four + custom


# --------------------------------
# 4. Fit SINDy model
# --------------------------------
X_tr = np.column_stack([sig_tr, ep_tr, eps_tr, epsdot_tr])
X_va = np.column_stack([sig_va, ep_va, eps_va, epsdot_va])

opt = ps.STLSQ(threshold=1e-3)
model = ps.SINDy(feature_library=lib, optimizer=opt)
model.feature_names = ["sig","ep","eps","eps_dot"]
model.state_names   = ["ep"]

model.fit(X_tr, t=dt, x_dot=epdot_tr)

print("\n=== Discovered evolution law ===")
#for eq in model.equations():
#    print(eq)


# --------------------------------
# 5. Validation: rollout
# --------------------------------
def predict_epdot(sig, ep, eps, epsdot):
    row = np.array([[sig, ep, eps, epsdot]])
    return float(model.predict(row)[0,0])

ep_hat = np.zeros_like(ep_va)
for k in range(1, len(t)):
    sig_k = 2000*(eps_va[k-1] - ep_hat[k-1])
    ep_hat[k] = ep_hat[k-1] + dt * predict_epdot(sig_k, ep_hat[k-1], eps_va[k-1], epsdot_va[k-1])
sig_hat = 2000*(eps_va - ep_hat)
print("\n=== Discovered evolution law ===")
# --------------------------------
# 6. Plots
# --------------------------------
plt.figure()
plt.plot(eps_va, sig_va, label="True")
plt.plot(eps_va, sig_hat, "--", label="SINDy")
plt.xlabel("strain ε"); plt.ylabel("stress σ")
plt.legend(); plt.title("Stress-strain (Validation)")

plt.figure()
plt.plot(t, ep_va, label="True ep")
plt.plot(t, ep_hat, "--", label="SINDy ep")
plt.xlabel("t"); plt.ylabel("plastic strain ep")
plt.legend(); plt.title("Plastic strain evolution")
plt.show()