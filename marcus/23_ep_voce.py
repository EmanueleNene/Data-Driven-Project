import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt


# -------- True parameters (inital yield + Voce + Perzyna) --------

E = 2000.0
sigy0 = 5.0            # initial yield stress
Q, b = 12.0, 40.0      # Voce: sig_y = sigy0 + Q(1 - exp(-b*ep))
eta_vp = 20.0             # Perzyna viscosity

def simulate(eps, dt):
    ep = np.zeros_like(eps); sig = np.zeros_like(eps)
    for k in range(1, len(eps)):
        sig[k-1] = E*(eps[k-1] - ep[k-1]) # elastic predictor / trial stress
        sig_y = sigy0 # no hardeneing
        sig_y = sigy0 + Q*(1.0 - np.exp(-b*ep[k-1])) # Voce hardening
        f = sig[k-1] - sig_y # yield function (overstress)
        ep_dot = max(0.0, f) # normal flow check
        ep_dot = max(0.0, f)/eta_vp # Perzyna viscosity
        ep[k] = ep[k-1] + dt * ep_dot # explicit Euler
    sig[-1] = E*(eps[-1] - ep[-1]) # last stress value outside loop
    return sig, ep

# -------- Strain paths (train / validate) --------
dt = 1e-2; t = np.arange(0, 20, dt)
eps_tr = 0.006*np.sin(2*np.pi*0.5*t)*np.exp(-0.1*t) + 0.0005*t
eps_va = 0.007*np.sin(2*np.pi*0.35*t)*np.exp(-0.05*t) + 0.0005*t

sig_tr, ep_tr = simulate(eps_tr, dt)
sig_va, ep_va = simulate(eps_va, dt)

epsdot_tr = np.gradient(eps_tr, dt); epdot_tr = np.gradient(ep_tr, dt)
epsdot_va = np.gradient(eps_va, dt); epdot_va = np.gradient(ep_va, dt)

# -------- Arrange data: learn ep_dot = g(sig, ep, eps, eps_dot) --------
#X_tr = np.column_stack([sig_tr, ep_tr, eps_tr, epsdot_tr])
#X_va = np.column_stack([sig_va, ep_va, eps_va, epsdot_va])

# (Optional) scale features (helps sparsity)
#Xmu, Xstd = X_tr.mean(0), X_tr.std(0) + 1e-12
#Ymu, Ystd = epdot_tr.mean(), epdot_tr.std() + 1e-12
#X_tr_s = (X_tr - Xmu)/Xstd; y_tr_s = (epdot_tr - Ymu)/Ystd
#X_va_s = (X_va - Xmu)/Xstd


# -----------------------------
# 0) Pack TRAIN / VAL into arrays
# -----------------------------
# (Assumes these already exist from your simulator.)
# TRAIN
X_tr_raw = np.column_stack([sig_tr, ep_tr, eps_tr, epsdot_tr])  # cols: [sig, ep, eps, eps_dot]
y_tr_raw = epdot_tr                                             # target: ep_dot
names = ["sig","ep","eps","eps_dot"]

# VALIDATION
X_va_raw = np.column_stack([sig_va, ep_va, eps_va, epsdot_va])

# -----------------------------
# 1) Build manual design matrix Φ with targeted ReLUs
# -----------------------------

# Estimate sigy0 from data (first index where plastic strain becomes > 0)   
sigy0 = sig_tr[np.argmax(ep_tr > 0)]; print("Estimated sigy0:", sigy0)


def build_phi(X_raw, sigy0):
    sig = X_raw[:, 0]
    ep  = X_raw[:, 1]
    eps = X_raw[:, 2]
    eps_dot = X_raw[:, 3]
    r   = np.maximum(0.0, sig - sigy0)          # relu(sig - sigy0)     * np.maximum(0.0, np.sign(eps_dot)*1.)
    Phi = np.column_stack([
#        X_raw,                 # identity: sig, ep, eps, eps_dot
        r,                     # relu(sig - sigy0)
        r * ep,                # relu(sig - sigy0) * ep
        r * ep * ep,            # relu(sig - sigy0) * ep^2
        r * eps_dot,             # relu(sig - sigy0) * eps_dot
        r * ep * eps_dot,        # relu(sig - sigy0) * ep * eps_dot
        r * eps_dot * eps_dot,    # relu(sig - sigy0) * eps_dot^2
    ])
#    phi_names =  names + ["relu(sig-sigy0)",
    phi_names =           ["relu(sig-sigy0)",
                         "relu(sig-sigy0)*ep",
                         "relu(sig-sigy0)*ep^2",
                         "relu(sig-sigy0)*eps_dot",
                         "relu(sig-sigy0)*ep*eps_dot",
                         "relu(sig-sigy0)*eps_dot^2"]
    return Phi, phi_names

Phi_tr_raw, phi_names = build_phi(X_tr_raw, sigy0)
Phi_va_raw, _         = build_phi(X_va_raw, sigy0)

# Gate activation (TRAIN, raw)
gate_frac = (np.maximum(0.0, X_tr_raw[:,0] - sigy0) > 0).mean()
print(f"Gate active fraction (train): {gate_frac:.3f}")

# -----------------------------
# 2) Z-score scaling (fit on TRAIN, apply to both)
# -----------------------------
Xmu   = Phi_tr_raw.mean(axis=0)
Xstd  = Phi_tr_raw.std(axis=0) + 1e-12
ymu   = y_tr_raw.mean()
ystd  = y_tr_raw.std() + 1e-12

Phi_tr = (Phi_tr_raw - Xmu) / Xstd
y_tr   = (y_tr_raw  - ymu) / ystd
Phi_va = (Phi_va_raw - Xmu) / Xstd

# -----------------------------
# 3) Fit SINDy on scaled Θ (Identity library)
# -----------------------------
lib = ps.IdentityLibrary()
#opt = ps.STLSQ(threshold=1e-1) 
opt = ps.SR3(reg_weight_lam=0.1, relax_coeff_nu=1)
model = ps.SINDy(feature_library=lib, optimizer=opt)
model.feature_names = phi_names
model.state_names   = ["ep"]
model.fit(Phi_tr, t=dt, x_dot=y_tr)

# -----------------------------
# 4) Inspect ALL features & coefficients
# -----------------------------
coefs_s = np.ravel(model.coefficients()[0])  # scaled-space coefficients (Ξ row)
print("\n=== ALL features & coefficients (scaled) ===")
for c, n in zip(coefs_s, phi_names):
    print(f"{c:+.4e}  *  {n}")

# Unscale coefficients if you want the law in original units:
coefs_unscaled = (ystd / Xstd) * coefs_s     # elementwise
intercept      = ymu - np.dot(coefs_unscaled, Xmu)
print("\nIntercept (unscaled):", f"{intercept:+.4e}")
print("Top terms (|coef|, unscaled):")
order = np.argsort(-np.abs(coefs_unscaled))
for j in order[:6]:
    print(f"{coefs_unscaled[j]:+.4e}  *  {phi_names[j]}")



## ---- Robust print (bypass pretty-printer differences) ----
#import numpy as np
##feat = np.array(model.get_feature_names())
#feat = np.array(lib.get_feature_names(input_features=["relu(sig-sigy0)", "relu(sig-sigy0)*ep", "relu(sig-sigy0)*ep^2", "relu(sig-sigy0)*eps_dot", "relu(sig-sigy0)*ep*eps_dot", "relu(sig-sigy0)*eps_dot^2"]))
#coef = np.ravel(model.coefficients()[0])
#nz   = np.abs(coef) > 1e-10
#print(" ");print(" ");print(" ");
#print("\n=== Discovered evolution law (scaled) ===")
#for c, n in sorted(zip(coef[nz], feat[nz]), key=lambda z: -abs(z[0])):
#    print(f"{c:+.4g} * {n}")
#
## 1) See what features SINDy *actually* had access to:
##feat = model.get_feature_names()
#print(" ");print(" ");print(" ");
#print("Features in library:", feat)
#print("Num features:", len(feat))
#
## 2) How many ReLU columns survived the fit?
#coef = np.ravel(model.coefficients()[0])
#for c, n in zip(coef, feat):
##    if "relu" in n:
#        print(f"{n:35s}  coeff={c:+.3e}")
#
## 3) Was the gate ever on? (using your train data)
#r = np.maximum(0.0, sig_tr - sigy0)  # or whatever gate you intended
#print("Gate active fraction:", (r > 0).mean())
#
## Build names from the library (works even across versions)
#feat_names = np.array(lib.get_feature_names(input_features=["relu(sig-sigy0)", "relu(sig-sigy0)*ep", "relu(sig-sigy0)*ep^2", "relu(sig-sigy0)*eps_dot", "relu(sig-sigy0)*ep*eps_dot", "relu(sig-sigy0)*eps_dot^2"]))
#coefs = np.ravel(model.coefficients()[0])   # Ξ row for ep'
#
#print("\n=== ALL features and coefficients (scaled fit) ===")
#for c, n in zip(coefs, feat_names):
#    print(f"{c:+.4e}  *  {n}")







# -----------------------------
# 5) Optional: view Θ and |Ξ|
# -----------------------------
print("\nTheta (scaled) shape:", Phi_tr.shape, "  Xi shape:", model.coefficients().shape)

plt.figure(figsize=(9,3))
plt.bar(range(len(coefs_s)), np.abs(coefs_s)[order])
plt.xticks(range(len(coefs_s)), [phi_names[j] for j in order], rotation=75, ha='right', fontsize=8)
plt.ylabel("|coef| (scaled)")
plt.title("SINDy coefficients (sorted)")
plt.tight_layout(); plt.show()

# -----------------------------
# 6) Fast rollout on VALIDATION using unscaled coefficients
# -----------------------------
# Build fast RHS in *unscaled* space: ep_dot ≈ b + sum_j w_j * Φ_raw_j
w = coefs_unscaled.copy()
b = intercept

def epdot_fast_raw(sig, ep, eps, eps_dot):
    r = max(0.0, sig - sigy0)
#    phi = np.array([sig, ep, eps, eps_dot, r, r*ep, r*ep*ep,r * eps_dot,r * ep * eps_dot,r * eps_dot * eps_dot])
    phi = np.array([r, r*ep, r*ep*ep,r * eps_dot,r * ep * eps_dot,r * eps_dot * eps_dot])
    g = float(b + np.dot(w, phi))
#    return g   # real output    
    return max(0.0, g)   # mod so it's never negative (physical constraint)

E = 2000.0  # your elastic modulus (use your variable)
ep_hat = np.zeros_like(ep_va)
for k in range(1, len(t)):
    sig_k = E*(eps_va[k-1] - ep_hat[k-1])
    ep_hat[k] = ep_hat[k-1] + dt * epdot_fast_raw(sig_k, ep_hat[k-1], eps_va[k-1], epsdot_va[k-1])
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

plt.figure(figsize=(6,3))
plt.plot(t, ep_va,  lw=2, label="True ep")
plt.plot(t, ep_hat, "--", lw=2, label="SINDy ep")
plt.plot(t, eps_va, "--", lw=2, label="eps_va")
plt.plot(t, epsdot_va, "--", lw=2, label="epsdot_va")
plt.xlabel("t"); plt.ylabel("ep"); plt.title("Validation: plastic strain")
plt.legend(); plt.tight_layout(); plt.show()

