import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import solve_ivp
import pandas as pd

# ---------- TRUE DUFFING WITH UNKNOWN (TO SINDY) TIME FORCING ----------
# x'' + d x' + a x + � x^3 = ? cos(? t)
delta, alpha, beta, gamma, omega = 0.2, -1.0, 1.0, 0.3, 1.2

def f_true(t, z):
    x, v = z
    return [v, -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)]

# simulate data
t_span = (0, 50)
t = np.linspace(*t_span, 5000)
dt = t[1] - t[0]
z0 = [1.0, 0.0]
sol = solve_ivp(f_true, t_span, z0, t_eval=t)
x = sol.y[0]; v = sol.y[1]

# ---------- AUGMENT STATE WITH TIME ----------
X_aug = np.column_stack([x, v, t])  # columns: [x, v, t]

# ---------- LIBRARY: POLY(x,v,t)  +  FOURIER(t only) ----------
poly_lib = ps.PolynomialLibrary(degree=3, include_interaction=True, include_bias=False)

# Apply Fourier ONLY to the 3rd column (t), and just pass-thru for x,v:
time_fourier = ps.FourierLibrary(n_frequencies=3)
lib_time_only = ps.GeneralizedLibrary([
    ps.IdentityLibrary(),     # x as-is (no extra functions here)
    ps.IdentityLibrary(),     # v as-is
    time_fourier              # sin(k t), cos(k t) for k=1..3
])

# Final library = all polynomials in (x,v,t)  UNION  {sin(k t), cos(k t)}
full_lib = poly_lib + lib_time_only

# ---------- SINDy (version-safe names) ----------
try:
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.06),
                     feature_library=full_lib,
                     feature_names=["x", "v", "t"])
except TypeError:
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.06),
                     feature_library=full_lib)
    model.feature_names = ["x", "v", "t"]

model.fit(X_aug, t=dt)

print("\\n=== Discovered equations with time-augmented state ===")
for eq in model.equations():
    print(eq)

# ---------- Inspect which time-Fourier features show up in v' ----------
features = np.array(model.get_feature_names())
coefs    = model.coefficients()  # list/array per state row
c_v      = np.ravel(coefs[1])    # coefficients for (v)'

mask_time_fourier = np.array([("sin(" in f or "cos(" in f) and "t" in f for f in features])
sel = (np.abs(c_v) > 1e-8) & mask_time_fourier

print("\\n--- Nonzero time-Fourier terms in (v)' ---")
if np.any(sel):
    df = pd.DataFrame({"feature": features[sel], "coef": c_v[sel]})
    df = df.sort_values("coef", key=np.abs, ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))
else:
    print("(none)")

# ---------- Compare trajectories: true vs learned ----------
def f_learned(ti, z_aug):
    # z_aug = [x, v, t]
    return model.predict(np.array([z_aug]))[0]  # returns [x', v', t']

z0_aug = [z0[0], z0[1], t[0]]
sol_learn = solve_ivp(f_learned, t_span, z0_aug, t_eval=t)

x_learn, v_learn = sol_learn.y[0], sol_learn.y[1]

rmse = np.sqrt(np.mean((x - x_learn)**2 + (v - v_learn)**2))
print(f"\nRMSE in (x,v): {rmse:.4e}")

# plots
plt.figure()
plt.plot(x, v, lw=2, label="True")
plt.plot(x_learn, v_learn, "--", lw=2, label="SINDy (time-augmented)")
plt.xlabel("x"); plt.ylabel("v")
plt.title("Trajectory: true vs learned (time-augmented library)")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(t, x, lw=1.5, label="True x(t)")
plt.plot(t, x_learn, "--", lw=1.5, label="Learned x(t)")
plt.xlabel("t"); plt.ylabel("x")
plt.title("Time series: true vs learned (time-augmented)")
plt.legend(); plt.tight_layout(); plt.show()
