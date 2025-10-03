import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLarsIC

# ---------- synthetic Neo-Hookean (incompressible), uniaxial ----------
# First Piola stress: P( x) =   ( x -  x^{-2})
mu_true = 1.7
lam = np.linspace(0.7, 1.6, 200)
P_clean = mu_true * (lam - lam**(-2))

# add mild noise
rng = np.random.default_rng(1)
noise_level = 0.01
P = P_clean + noise_level*np.std(P_clean)*rng.standard_normal(P_clean.shape)

# ---------- sparse feature build ----------
# basis: [1,  x,  x^2,  x^3,  x^{-1},  x^{-2}]
Phi = np.column_stack([
    np.ones_like(lam),
    lam,
    lam**2,
    lam**3,
    1/lam,
    1/lam**2
])

# sparse fit P   Phi @ theta
reg = LassoLarsIC(criterion="bic").fit(Phi, P)
theta = np.r_[reg.intercept_, reg.coef_]

print("Basis = [1,  x,  x^2,  x^3,  x^{-1},  x^{-2}]")
print(" x (intercept first):", np.round(theta, 6))

# estimate   from combination ( x -  x^{-2})
mu_est = 0.5 * (theta[1+0] + theta[1+5]) if theta[1+0]*theta[1+5] < 0 else np.nan
print(f"Estimated     {mu_est:.4f}   |   True   = {mu_true:.4f}")

# prediction
P_pred = Phi @ theta[1:] + theta[0]

# plots
plt.figure()
plt.plot(lam, P, ".", label="noisy data", alpha=0.6)
plt.plot(lam, P_clean, lw=2, label="true: mu*(lam - lam**(-2))")
plt.plot(lam, P_pred, "--", label="sparse fit")
plt.xlabel("stretch lam"); plt.ylabel("First Piola stress P")
plt.title("Neo-Hookean uniaxial: sparse rediscovery")
plt.legend(); plt.tight_layout(); plt.show()
