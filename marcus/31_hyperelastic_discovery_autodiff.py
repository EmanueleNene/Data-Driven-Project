import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLarsIC

# ---------- USER SWITCHES ----------
DISCOVERY_MODE = "B_potential"     # "A_stress" or "B_potential"
TRAIN_PATHS    = {"uniaxial", "shear"}   # held-out: the others are validation
# -----------------------------------

# --- JAX autodiff ---
import jax.numpy as jnp
from jax import grad

# ---------- KINEMATICS & INVARIANTS ----------
def invariants_from_F(F):
    B = F @ F.T
    J = np.linalg.det(F)
    I1 = np.trace(B)
    I2 = 0.5 * (I1**2 - np.trace(B @ B))
    return I1, I2, J, B

def F_uniaxial(lam):
    return np.diag([lam, 1/np.sqrt(lam), 1/np.sqrt(lam)])

def F_equibiaxial(lam):
    return np.diag([lam, lam, 1/(lam**2)])

def F_simple_shear(gamma):
    F = np.eye(3); F[0,1] = gamma; return F

# ---------- TRUE MODEL (compressible Neo-Hookean, for data gen) ----------
mu_true, kappa_true = 1.7, 50.0
def W_true(I1, I2, J):
    return 0.5*mu_true*(I1-3) - mu_true*jnp.log(J) + 0.5*kappa_true*(jnp.log(J))**2

def cauchy_from_W_autodiff(F, W_func):
    I1, I2, J, B = invariants_from_F(F)
    Wvec = lambda v: W_func(v[0], v[1], v[2])
    gW = grad(Wvec)
    dI1, dI2, dJ = map(float, gW(jnp.array([I1, I2, J])))
    I = np.eye(3)
    return (2.0/J)*(dI1*B + dI2*(np.trace(B)*I - B)) + dJ*I

# ---------- LOAD PATHS & DATA ----------
def make_paths():
    return [
        ("uniaxial",   "λ", np.linspace(0.7, 1.6, 120),
         lambda s: F_uniaxial(s),  (0,0)),  # plot σ11
        ("equibiaxial","λ", np.linspace(0.7, 1.4, 100),
         lambda s: F_equibiaxial(s),(0,0)),
        ("shear",      "γ", np.linspace(0.0, 1.0, 100),
         lambda s: F_simple_shear(s),(0,1)), # plot σ12
    ]

def generate_dataset():
    rows = []
    for name, xlabel, grid, Fbuilder, comp in make_paths():
        for s in grid:
            F = Fbuilder(float(s))
            I1, I2, J, B = invariants_from_F(F)
            sigma = cauchy_from_W_autodiff(F, W_true)
            rows.append(dict(path=name, s=float(s), F=F, I1=I1, I2=I2, J=J, B=B, sigma=sigma, comp=comp, xlabel=xlabel))
    return rows












# ---------- FEATURE LIBRARIES ----------
# Keep it modest but general (no "cheating"): center near reference and include volumetric terms
def scalar_features(I1, I2, J):
    i1 = I1 - 3.0
    i2 = I2 - 3.0
    lnJ = np.log(J)
    return np.array([
        i1, i1**2,         # isochoric-ish I1 terms
        i2,                # I2
        lnJ, (lnJ**2),     # volumetric
        (J-1.0), (J-1.0)**2
    ], dtype=float)

feat_names_A = ["(I1-3)","(I1-3)^2","(I2-3)","lnJ","(lnJ)^2","(J-1)","(J-1)^2"]

# For potential discovery, define basis φ_k(I1,I2,J)
def phi_list(I1, I2, J):
    i1 = I1 - 3.0
    i2 = I2 - 3.0
    lnJ = np.log(J)
    return np.array([
        i1, i1**2, i2, lnJ, (lnJ**2), (J-1.0), (J-1.0)**2
    ], dtype=float)

phi_names_B = ["(I1-3)","(I1-3)^2","(I2-3)","lnJ","(lnJ)^2","(J-1)","(J-1)^2"]

# ---------- DISCOVERY A) DIRECT STRESS ----------
def discover_stress(rows):
    # Train separate sparse models for each component we plot: (σ11 for stretch paths, σ12 for shear)
    # Build datasets per 'path' target component:
    models = {}
    for target_path in {"uniaxial","equibiaxial","shear"}:
        X, y = [], []
        for r in rows:
            use = (r["path"] in TRAIN_PATHS)
            if not use: continue
            i,j = r["comp"]
            # Train σ11 with stretch paths; σ12 with shear
            if target_path in {"uniaxial","equibiaxial"} and (i,j)!=(0,0): continue
            if target_path=="shear" and (i,j)!=(0,1): continue
            X.append(scalar_features(r["I1"], r["I2"], r["J"]))
            y.append(r["sigma"][i,j])
        if not X: continue
        X, y = np.asarray(X), np.asarray(y)
        reg = LassoLarsIC(criterion="bic").fit(X, y)
        models[target_path] = (reg, X.shape[1])
    return models

def predict_stress(r, models):
    i,j = r["comp"]
    if (i,j)==(0,0):
        key="uniaxial" if r["path"]!="equibiaxial" else "equibiaxial"
    else:
        key="shear"
    if key not in models: return None
    reg,_ = models[key]
    return float(reg.predict(scalar_features(r["I1"], r["I2"], r["J"]).reshape(1,-1))[0])

# ---------- DISCOVERY B) POTENTIAL ----------
# W = Σ θ_k φ_k(I1,I2,J). Use autodiff on each φ_k to build a linear map to stress.
def discover_potential(rows):
    # Precompute grads of each φ_k wrt (I1,I2,J) via JAX
    def phi_k(k, v):
        I1,I2,J = v
        i1 = I1-3.0; i2 = I2-3.0; lnJ = jnp.log(J)
        if   k==0: return i1
        elif k==1: return i1**2
        elif k==2: return i2
        elif k==3: return lnJ
        elif k==4: return (lnJ**2)
        elif k==5: return (J-1.0)
        elif k==6: return (J-1.0)**2
        else: return 0.0
    grads = [grad(lambda v, kk=kk: phi_k(kk, v)) for kk in range(len(phi_names_B))]

    # Build linear system A θ ≈ b for the components used in training
    A_rows, b = [], []
    for r in rows:
        if r["path"] not in TRAIN_PATHS: continue
        I1,I2,J,B = r["I1"], r["I2"], r["J"], r["B"]
        I = np.eye(3)
        g = np.array([np.array(gv(np.array([I1,I2,J]))) for gv in grads])  # shape (K,3) -> dφ/d[I1,I2,J]
        # Each θ_k contributes: (2/J)*(dφ_k/dI1 * B + dφ_k/dI2 * (I1 I - B)) + (dφ_k/dJ) * I
        # For a chosen component (i,j), collect its contribution -> column in A
        i,j = r["comp"]
        for k in range(g.shape[0]):
            dI1,dI2,dJ = g[k]
            term = (2.0/J)*(dI1*B + dI2*(np.trace(B)*I - B)) + dJ*I
            A_rows.append(term[i,j])
        b.append(r["sigma"][i,j])
    if not A_rows:
        raise RuntimeError("No training rows — check TRAIN_PATHS")
    # Reshape to (n_samples, K)
    K = len(phi_names_B)
    A = np.asarray(A_rows, float).reshape(-1, K)
    b = np.asarray(b, float)
    reg = LassoLarsIC(criterion="bic").fit(A, b)
    theta = reg.coef_
    return theta

def predict_stress_from_potential(r, theta):
    # Build the same linear combo per-sample
    I1,I2,J,B = r["I1"], r["I2"], r["J"], r["B"]
    I = np.eye(3); i,j = r["comp"]
    # grads of each φ_k
    def phi_k(k, v):
        I1,I2,J = v
        i1 = I1-3.0; i2 = I2-3.0; lnJ = jnp.log(J)
        if   k==0: return i1
        elif k==1: return i1**2
        elif k==2: return i2
        elif k==3: return lnJ
        elif k==4: return (lnJ**2)
        elif k==5: return (J-1.0)
        elif k==6: return (J-1.0)**2
        else: return 0.0
    grads = [grad(lambda v, kk=kk: phi_k(kk, v)) for kk in range(len(phi_names_B))]
    g = np.array([np.array(gv(np.array([I1,I2,J]))) for gv in grads])  # (K,3)
    comp_cols = []
    for k,(dI1,dI2,dJ) in enumerate(g):
        term = (2.0/J)*(dI1*B + dI2*(np.trace(B)*I - B)) + dJ*I
        comp_cols.append(term[i,j])
    comp_cols = np.asarray(comp_cols)
    return float(comp_cols @ theta)

# ---------- RUN ----------
if __name__ == "__main__":
    rows = generate_dataset()

    # Train
    if DISCOVERY_MODE == "A_stress":
        models = discover_stress(rows)
    else:
        theta = discover_potential(rows)

    # Report (compact)
    print("\n=== Discovery mode:", DISCOVERY_MODE, "===")
    if DISCOVERY_MODE == "A_stress":
        for key,(reg,p) in models.items():
            nz = np.abs(reg.coef_) > 1e-9
            print(f"[{key}] kept {nz.sum()}/{p} terms:")
            for n,c in zip(feat_names_A, reg.coef_):
                if abs(c)>1e-9: print(f"  {n:>10s}: {c:+.4f}")
    else:
        nz = np.abs(theta) > 1e-9
        print(f"[W-basis] kept {nz.sum()}/{len(phi_names_B)} terms:")
        for n,c in zip(phi_names_B, theta):
            if abs(c)>1e-9: print(f"  {n:>10s}: {c:+.4f}")

    # Plots: true vs discovered per path (train vs val)
    paths = make_paths()
    fig, ax = plt.subplots(1, len(paths), figsize=(14,4), sharey=True)
    for k,(name, xlabel, grid, Fbuilder, comp) in enumerate(paths):
        xs, y_true, y_disc = [], [], []
        for s in grid:
            F = Fbuilder(float(s))
            I1,I2,J,B = invariants_from_F(F)
            sig = cauchy_from_W_autodiff(F, W_true)
            xs.append(s); y_true.append(sig[comp])

            r = dict(path=name, s=float(s), F=F, I1=I1, I2=I2, J=J, B=B, comp=comp, xlabel=xlabel)
            if DISCOVERY_MODE == "A_stress":
                yhat = predict_stress(r, models)
            else:
                yhat = predict_stress_from_potential(r, theta)
            y_disc.append(yhat if yhat is not None else np.nan)

        xs, y_true, y_disc = np.array(xs), np.array(y_true), np.array(y_disc)
        ax[k].plot(xs, y_true, lw=2, label="true")
        ax[k].plot(xs, y_disc, "--", lw=2, label="discovered")
        ax[k].set_title(f"{name}  " + ("[train]" if name in TRAIN_PATHS else "[val]"))
        ax[k].set_xlabel(xlabel); ax[k].grid(True)
        if k==0: ax[k].set_ylabel("Cauchy stress component")
    ax[-1].legend()
    plt.tight_layout(); plt.show()
