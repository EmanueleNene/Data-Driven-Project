import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import solve_ivp

# Duffing oscillator: x'' + d x' + a x + � x^3 = ? cos(? t)
delta, alpha, beta, gamma, omega = 0.2, -1.0, 1.0, 0.3, 1.2

def f(t, z):
    x, v = z
    return [v, -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)]

# simulate
t_span = (0, 50)
t = np.linspace(*t_span, 5000)
z0 = [1.0, 0.0]
sol = solve_ivp(f, t_span, z0, t_eval=t)
X = np.vstack([sol.y[0], sol.y[1]]).T  # columns: x, v
dt = t[1] - t[0]

# SINDy with polynomial features up to cubic terms
optimizer = ps.STLSQ(threshold=0.05)
lib = ps.PolynomialLibrary(degree=3, include_interaction=True, include_bias=False)

# Try the new API first (v2.0.0+)
try:
    model = ps.SINDy(optimizer=optimizer, feature_library=lib, feature_names=["x", "v"])
except TypeError:
    # Fall back for newer versions where feature_names must be set separately
    model = ps.SINDy(optimizer=optimizer, feature_library=lib)
    model.feature_names = ["x", "v"]

model.fit(X, t=dt)
model.print()


# quick visual: phase portrait
plt.figure()
plt.plot(X[:,0], X[:,1], linewidth=1)
plt.xlabel("x"); plt.ylabel("v")
plt.title("Duffing trajectory")
plt.tight_layout()
plt.show()
