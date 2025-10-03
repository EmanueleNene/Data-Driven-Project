import numpy as np
from sklearn.linear_model import LassoLarsIC

# synthetic uniaxial, incompressible: P(λ) = μ (λ - λ**-2)
mu_true = 1.7
lam = np.linspace(0.7, 1.6, 200)
P = mu_true*(lam - lam**-2)

# library with no bias; physics-guided terms
Phi = np.column_stack([lam, lam**-2])   # features = [λ, λ^{-2}]

# sparse fit with automatic model size (BIC)
reg = LassoLarsIC(criterion="bic").fit(Phi, P)
coef = reg.coef_                        # should be [μ, -μ]
mu_est = 0.5*(coef[0] - coef[1])        # combine to estimate μ

print("True  μ:", mu_true)
print("Coef (λ, λ^{-2}):", coef.round(6))
print("Est.  μ:", round(mu_est, 6))
