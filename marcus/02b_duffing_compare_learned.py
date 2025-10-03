import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import solve_ivp
import pandas as pd

# ---------- TRUE SYSTEM (forced Duffing) ----------
# x'' + d x' + a x + ß x^3 = ? cos(? t)
delta, alpha, beta, gamma, omega = 0.2, -1.0, 1.0, 0.3, 1.2

def f_true(t, z):
    x, v = z
    return [v, -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)]

t_span = (0, 50)
t = np.linspace(*t_span, 5000)
dt = t[1] - t[0]
z0 = [1.0, 0.0]

sol_true = solve_ivp(f_true, t_span, z0, t_eval=t)
X = np.vstack([sol_true.y[0], sol_true.y[1]]).T  # [x, v]

# ---------- FIT SINDy ----------
optimizer = ps.STLSQ(threshold=0.05)
lib = ps.PolynomialLibrary(degree=3, include_interaction=True, include_bias=False)

# version-safe feature_names handling
try:
    model = ps.SINDy(optimizer=optimizer, feature_library=lib, feature_names=["x","v"])
except TypeError:
    model = ps.SINDy(optimizer=optimizer, feature_library=lib)
    model.feature_names = ["x","v"]

model.fit(X, t=dt)

print("\n=== Discovered SINDy equations ===")
for eq in model.equations():
    print(eq)

# Pretty coefficient tables
fnames = np.array(model.get_feature_names())
coefs = model.coefficients()  # list/array per state
rows = []
for state_i, state_name in enumerate(["x'", "v'"]):
    c = np.ravel(coefs[state_i])
    nz = np.abs(c) > 1e-10
    df = pd.DataFrame({"feature": fnames[nz], "coef": c[nz]})
    df = df.sort_values("feature").reset_index(drop=True)
    print(f"\n--- Nonzero coefficients for {state_name} ---")
    print(df.to_string(index=False))
    rows.append(df.assign(state=state_name))
coef_table = pd.concat(rows, ignore_index=True)

# ---------- SIMULATE THE LEARNED MODEL ----------
def f_learned(t, z):
    # model.predict(x) expects row vector(s)
    xdot = model.predict(np.array([z]))[0]  # returns [x', v']
    return xdot

sol_learn = solve_ivp(f_learned, t_span, z0, t_eval=t)

# ---------- TRAJECTORY COMPARISON ----------
x_true, v_true = sol_true.y
x_learn, v_learn = sol_learn.y

# RMSE in phase space (rough sanity)
rmse = np.sqrt(np.mean((x_true - x_learn)**2 + (v_true - v_learn)**2))
print(f"\nRMSE in (x,v) over trajectory: {rmse:.4e}")

plt.figure()
plt.plot(x_true, v_true, label="True system", lw=2)
plt.plot(x_learn, v_learn, "--", label="SINDy-learned", lw=2)
plt.xlabel("x"); plt.ylabel("v"); plt.title("Duffing: trajectory comparison")
plt.legend(); plt.tight_layout(); plt.show()

# Optional: time-series overlay of x(t)
plt.figure()
plt.plot(t, x_true, label="True x(t)", lw=1.5)
plt.plot(t, x_learn, "--", label="Learned x(t)", lw=1.5)
plt.xlabel("t"); plt.ylabel("x"); plt.title("Duffing: x(t) comparison")
plt.legend(); plt.tight_layout(); plt.show()
