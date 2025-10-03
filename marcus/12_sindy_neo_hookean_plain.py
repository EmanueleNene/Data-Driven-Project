import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

# ---------------------------
# Parameters
# ---------------------------
mu = 2.0   # shear modulus

# Deformation
lam = np.linspace(0.5, 2, 500).reshape(-1, 1)   # stretch ratio, column vector

# Stress (Neo-Hookean, Cauchy σ11 in uniaxial tension)
stress = mu * (lam**2 - 1/lam)

# ---------------------------
# Plot true relation
# ---------------------------
plt.figure()
plt.plot(lam, stress, "b", linewidth=2)
plt.xlabel("λ"); plt.ylabel("σ")
plt.title("Neo-Hookean: stress vs stretch")
plt.grid(True)
plt.show()

# ---------------------------
# 1. Feature library
from pysindy.feature_library import CustomLibrary

# Polynomial terms up to degree 5 (no bias, no cross terms)
poly_lib = ps.PolynomialLibrary(degree=3, include_interaction=False, include_bias=False)

# Add 1/lam via a CustomLibrary
library_functions = [
    lambda x: np.exp(x),
    lambda x: 1.0 / x,
]
library_function_names = [
    lambda x: "exp(" + x + ")",
    lambda x: "1/" + x,
]
custom_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
)
# Final library: polynomials + reciprocal
lib = poly_lib + custom_library


# ---------------------------
# 2. Sparse regression (STLSQ)
# ---------------------------
optimizer = ps.STLSQ(threshold=0.1)  # like lambda_sparsify
model = ps.SINDy(feature_library=lib, optimizer=optimizer, discrete_time=True)

# Fit: treat λ as "state", stress as "derivative" (but here it's algebraic)
# Trick: use fit with discrete_time=True so it's just regression
t_dummy = np.arange(len(lam))
model.fit(lam, t=t_dummy, x_dot=stress, feature_names=["lam"])

# ---------------------------
# 3. Display discovered model
# ---------------------------
print("\n=== Discovered model terms ===")
model.print()

# ---------------------------
# 4. Compare true vs predicted
# ---------------------------
stress_pred = model.predict(lam)

plt.figure()
plt.plot(lam, stress, "k", linewidth=2, label="Neo-Hookean (true)")
plt.plot(lam, stress_pred, "r--", linewidth=2, label="SINDy (discovered)")
plt.legend()
plt.title("Comparison: Neo-Hookean vs SINDy")
plt.xlabel("λ"); plt.ylabel("σ")
plt.grid(True)
plt.show()
