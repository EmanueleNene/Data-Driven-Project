import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp

# --- true system: m x'' + c x' + k x = 0
m, c, k = 2.0, 0.30, 4.00

def f_true(t, z):
    x, v = z
    return [v, -(k/m)*x - (c/m)*v]

# simulate
t = np.linspace(0, 20, 2000)
dt = t[1]-t[0]
z0 = [1.0, 0.0]
sol = solve_ivp(f_true, (t[0], t[-1]), z0, t_eval=t)
X = np.vstack([sol.y[0], sol.y[1]]).T

# fit SINDy
lib = ps.IdentityLibrary()
opt = ps.STLSQ(threshold=1e-5)
try:
    model = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["x","v"])
except TypeError:
    model = ps.SINDy(feature_library=lib, optimizer=opt)
    model.feature_names = ["x","v"]
model.fit(X, t=dt)

# print equations
print("=== True system ===")
print("(x)' = v")
print(f"(v)' = -({k/m}) x -({c/m}) v")

print("\n=== Discovered by SINDy ===")
model.print()
print("\n=== Coefficient matrix ===")
print(model.coefficients()) # rows: x', v'; cols: x, v  # should be [[0,1],[-k/m,-c/m]]
print("=== Feature names ===")
print(model.get_feature_names())    # should be ['x', 'v']  # matches coeffs cols   
