import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from pysindy.feature_library import CustomLibrary

# ----- true parameters -----
E = 2000.0          # elastic modulus
sig_y = 5.0         # yield stress (tension only here)
eta_vp = 20.0       # viscoplastic viscosity (Perzyna)
dt = 1e-3
t = np.arange(0, 20, dt)

# strain history with loading/unloading/cycles
eps = 0.006*np.sin(2*np.pi*0.5*t) + 0.0005*t

# integrate constitutive response (explicit Euler for clarity)
ep = np.zeros_like(eps)      # plastic strain
sig = np.zeros_like(eps)
for k in range(1, len(t)):
    sig[k-1] = E*(eps[k-1] - ep[k-1])
    f = sig[k-1] - sig_y               # yield function (perfect plasticity)
    ep_dot = max(0.0, f) / eta_vp      # Perzyna
    ep[k] = ep[k-1] + dt * ep_dot
# last stress value
sig[-1] = E*(eps[-1] - ep[-1])

# ----- build SINDy dataset for the evolution law ep_dot = g(sig) -----
# Target derivative (from the simulated ep)
ep_dot_data = np.gradient(ep, dt)

# Feature: ReLU(sig - sig_y)
def relu_yield(X):
    # X columns: [sig]
    X = np.asarray(X)
    if X.ndim == 1: X = X[:, None]
    return np.maximum(0.0, X[:, 0] - sig_y)

feature_lib = CustomLibrary(
    library_functions=[relu_yield],
    function_names=["relu(sig - sig_y)"]
)

# we give SINDy just one feature column (the ReLU); linear should suffice
X = sig.reshape(-1, 1)  # states passed to library (only sigma)
y = ep_dot_data         # target is ep_dot

lib = feature_lib
opt = ps.STLSQ(threshold=1e-8)  # keep it linear & precise
model = ps.SINDy(feature_library=lib, optimizer=opt)
model.feature_names = ["sig"]    # input name for printing
model.state_names   = ["ep"]     # state whose derivative we model

# Fit continuous-time ODE: ep_dot = c1 * relu(sig - sig_y)
model.fit(X, t=dt, x_dot=y)

#print("\n=== Discovered evolution law ===")
#model.print()

# Extract coefficient (should be ~ 1/eta_vp)
c = float(model.coefficients()[0][0])
print(f"\nTrue 1/eta_vp = {1/eta_vp:.6f}")
print(f"Fit  1/eta_vp = {c:.6f}")

# ----- quick validation: rebuild ep using discovered c -----
ep_hat = np.zeros_like(ep)
for k in range(1, len(t)):
    sig_k = E*(eps[k-1] - ep_hat[k-1])
    ep_hat[k] = ep_hat[k-1] + dt * c * max(0.0, sig_k - sig_y)
sig_hat = E*(eps - ep_hat)

# ----- plots -----
plt.figure(figsize=(6,4))
plt.plot(ep, sig, lw=2, label="True")
plt.plot(ep, sig_hat, "--", lw=2, label="Rebuilt (SINDy c)")
plt.xlabel("strain ε"); plt.ylabel("stress σ"); plt.title("Elastic–perfectly-plastic (Perzyna)")
plt.legend(); plt.tight_layout(); plt.show()
