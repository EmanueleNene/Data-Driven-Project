"""
25_voce_perzyna_residual.py
Reviewer comment 3: residual of the SINDy-identified polynomial against the EXACT
exponential Voce-Perzyna hardening law, over the plastic-strain range actually
reached during training. Reproduces the 24d fit (no hard-coded coefficients) and
compares SINDy vs 2nd-order Taylor vs exact.

Produces: voce_perzyna_hardening_residual.png  and prints LaTeX-ready numbers.
"""
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import warnings
import os
try:
    from pysindy.utils.axes import AxesWarning
    warnings.filterwarnings("ignore", category=AxesWarning)
except Exception:
    pass

dir_path = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(dir_path, "figures", "plastic")
os.makedirs(fig_dir, exist_ok=True)

# ---------- true material (identical to 24d_vocePerzyna1D_finalPres.py) ----------
E, sigy0, Q, b, eta_vp = 2000.0, 5.0, 12.0, 40.0, 20.0

def simulate(eps, dt):
    ep = np.zeros_like(eps); sig = np.zeros_like(eps)
    for k in range(1, len(eps)):
        sig[k-1] = E*(eps[k-1] - ep[k-1])
        sig_y = sigy0 + Q*(1.0 - np.exp(-b*ep[k-1]))
        ep[k] = ep[k-1] + dt*max(0.0, sig[k-1]-sig_y)/eta_vp
    sig[-1] = E*(eps[-1] - ep[-1])
    return sig, ep

# ---------- training path (identical to 24d) ----------
dt = 1e-3; t = np.arange(0, 20, dt)
eps_tr = 0.006*np.sin(2*np.pi*0.5*t)*np.exp(-0.1*t) + 0.0005*t
sig_tr, ep_tr = simulate(eps_tr, dt)
epsdot_tr = np.gradient(eps_tr, dt); epdot_tr = np.gradient(ep_tr, dt)

# ---------- reproduce the SINDy fit (plastic-only, z-scored, SR3) ----------
sigy0_hat = sig_tr[np.argmax(ep_tr > 0)]
over_tr = sig_tr - sigy0_hat
mask = epdot_tr > 1e-10
X = np.column_stack([over_tr, ep_tr, epsdot_tr])[mask]
y = epdot_tr[mask]

poly = ps.PolynomialLibrary(degree=2, include_bias=False, include_interaction=False)
Phi_raw = poly.fit_transform(X)
phi_names = poly.get_feature_names(input_features=["over", "ep", "eps_dot"])
Xmu, Xstd = Phi_raw.mean(0), Phi_raw.std(0)+1e-12
ymu, ystd = y.mean(), y.std()+1e-12
model = ps.SINDy(feature_library=ps.IdentityLibrary(),
                 optimizer=ps.SR3(reg_weight_lam=0.1, relax_coeff_nu=1))
model.feature_names = phi_names; model.state_names = ["ep"]
model.fit((Phi_raw-Xmu)/Xstd, t=dt, x_dot=(y-ymu)/ystd)
w = (ystd/Xstd)*np.ravel(model.coefficients()[0])
coef = dict(zip([n.replace(" ", "") for n in phi_names], w))
c_over, w1, w2 = coef["over"], coef["ep"], coef["ep^2"]
print(f"Identified (reproduced): 1/eta={c_over:.4f}, ep={w1:.4f}, ep^2={w2:.4f}")

# ---------- residual over the ACTUAL training range ----------
ep_max = ep_tr.max()
epg = np.linspace(0.0, ep_max, 4000)

# rate hardening contribution  d(eps_p)/dt |_hardening  (units 1/s)
rate_exact  = -(Q/eta_vp)*(1.0 - np.exp(-b*epg))
rate_sindy  =  w1*epg + w2*epg**2
rate_taylor = -(Q*b/eta_vp)*epg + (Q*b**2/(2*eta_vp))*epg**2   # -24 ep + 480 ep^2

rmse_poly   = np.sqrt(np.mean((rate_sindy  - rate_exact)**2))
rmse_taylor = np.sqrt(np.mean((rate_taylor - rate_exact)**2))
mean_rate   = np.mean(np.abs(rate_exact))

print("\n=== Reviewer comment 3: residual over training range ===")
print(f"eps_p_max (training) = {ep_max:.5f}")
print(f"mean |exact hardening rate| = {mean_rate:.5f} 1/s")
print(f"RMSE_poly   = {rmse_poly:.4e} 1/s  = {100*rmse_poly/mean_rate:.2f}% of mean")
print(f"RMSE_taylor = {rmse_taylor:.4e} 1/s  = {100*rmse_taylor/mean_rate:.2f}% of mean")

print("\n--- LaTeX fill (rate units, 1/s) ---")
print(f"EPMAX        = {ep_max:.4f}")
print(f"RMSE_poly    = {rmse_poly:.2e} s^-1")
print(f"XX           = {100*rmse_poly/mean_rate:.1f}")
print(f"RMSE_taylor  = {rmse_taylor:.2e} s^-1")

# ---------- figure ----------
plt.figure(figsize=(7,5))
plt.plot(epg, -rate_exact,  'k-',  lw=2.5, label='Exact Voce (exponential)')
plt.plot(epg, -rate_taylor, 'g--', lw=2,   label='2nd-order Taylor')
plt.plot(epg, -rate_sindy,  'r:',  lw=3,   label='SINDy polynomial')
plt.axvline(ep_max, color='gray', ls=':', alpha=0.6)
plt.text(ep_max, 0.002, r' $\varepsilon_p^{\max}$', color='gray', va='bottom')
plt.xlabel(r'plastic strain $\varepsilon_p$')
plt.ylabel(r'hardening contribution to $\dot\varepsilon_p$  (1/s, sign-flipped)')
plt.title('Hardening rate: exact vs Taylor vs SINDy (training range)')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
fig_path = os.path.join(fig_dir, "04_hardening_residual.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved: {fig_path}")
