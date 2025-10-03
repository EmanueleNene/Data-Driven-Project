import numpy as np

# -----------------------------
# Helpers: kinematics & invariants
# -----------------------------
def invariants_from_F(F):
    # Right Cauchy–Green: C = F^T F ; Left: B = F F^T
    B = F @ F.T
    J = np.linalg.det(F)
    I1 = np.trace(B)
    I2 = 0.5*(I1**2 - np.trace(B @ B))
    return I1, I2, J, B

# -----------------------------
# Compressible Neo-Hookean potential (Simo/Ortiz form)
# W(I1,J) = (mu/2)(I1 - 3) - mu ln J + (kappa/2)(ln J)^2
# Cauchy stress:  sigma = (mu/J)(B - I) + (kappa ln J / J) I
# -----------------------------
def W_neo_hookean(I1, J, mu, kappa):
    return 0.5*mu*(I1 - 3.0) - mu*np.log(J) + 0.5*kappa*(np.log(J))**2

def cauchy_from_W_neo_hookean(F, mu, kappa):
    I = np.eye(3)
    I1, I2, J, B = invariants_from_F(F)
    sigma = (mu/J)*(B - I) + (kappa*np.log(J)/J)*I
    return sigma  # 3x3

# (optional) First Piola, if you need it later in a UMAT:
# P = J * sigma * F^{-T}
def PK1_from_sigma(F, sigma):
    return np.linalg.det(F) * sigma @ np.linalg.inv(F).T

# -----------------------------
# Load paths: explicit F(·)
# -----------------------------
def F_uniaxial(lam):
    # stretch along e1 with traction-free laterals for incompressible *response* (we still keep full F here)
    return np.diag([lam, 1/np.sqrt(lam), 1/np.sqrt(lam)])

def F_equibiaxial(lam):
    return np.diag([lam, lam, 1/(lam**2)])

def F_simple_shear(gamma):
    F = np.eye(3)
    F[0,1] = gamma
    return F

# -----------------------------
# Make a dataset over multiple paths (inputs: F, outputs: sigma)
# -----------------------------
def make_dataset(mu=1.7, kappa=50.0):
    # grids
    lam_u = np.linspace(0.7, 1.6, 101)     # uniaxial
    lam_b = np.linspace(0.7, 1.3, 81)      # equibiaxial
    gam_s = np.linspace(0.0, 1.0, 81)      # shear

    rows = []
    # Uniaxial
    for lam in lam_u:
        F = F_uniaxial(lam)
        I1, I2, J, B = invariants_from_F(F)
        sigma = cauchy_from_W_neo_hookean(F, mu, kappa)
        rows.append(dict(path="uniaxial",  x=lam,   F=F, I1=I1, I2=I2, J=J, sigma=sigma))
    # Equibiaxial
    for lam in lam_b:
        F = F_equibiaxial(lam)
        I1, I2, J, B = invariants_from_F(F)
        sigma = cauchy_from_W_neo_hookean(F, mu, kappa)
        rows.append(dict(path="equibiaxial", x=lam, F=F, I1=I1, I2=I2, J=J, sigma=sigma))
    # Simple shear
    for gam in gam_s:
        F = F_simple_shear(gam)
        I1, I2, J, B = invariants_from_F(F)
        sigma = cauchy_from_W_neo_hookean(F, mu, kappa)
        rows.append(dict(path="shear",     x=gam,  F=F, I1=I1, I2=I2, J=J, sigma=sigma))

    return rows

if __name__ == "__main__":
    data = make_dataset(mu=1.7, kappa=50.0)
    # quick sanity: print one sample per path
    for p in ("uniaxial","equibiaxial","shear"):
        d = next(r for r in data if r["path"]==p)
        print(f"\n[{p}] x={d['x']:.3f} | I1={d['I1']:.4f}  J={d['J']:.4f}")
        print("sigma =\n", np.array_str(d["sigma"], precision=4, suppress_small=True))
