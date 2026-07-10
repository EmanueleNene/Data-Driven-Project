"""
Fit Coupled Nonlinear Viscoelastic SINDy model (Option 2)

This script loads the coupled nonlinear deviatoric stress data, constructs
the custom J2 coupling features, and fits a 6D SINDy model to recover the
underlying material constants: G, eta_0, and alpha.

It uses a component-by-component fitting strategy to avoid multicollinearity
arising from directional strain rates, overrides the full 6D SINDy model's
coefficients, validates predictions against unseen loading paths (combined and biaxial),
and generates a comparison plot.

Author: Generated for DataDrivenProject (Option 2)
Date: 2026-06-26
"""

import numpy as np
import pysindy as ps
import os
import sys
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ensure local imports work
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)

# ==========================================
# 1. LOAD DATA
# ==========================================
train_uniaxial = np.load(os.path.join(dir_path, 'data_train_uniaxial_nonlinear_dev.npz'))
train_shear = np.load(os.path.join(dir_path, 'data_train_shear_nonlinear_dev.npz'))
val_combined = np.load(os.path.join(dir_path, 'data_val_combined_nonlinear_dev.npz'))
val_biaxial = np.load(os.path.join(dir_path, 'data_val_biaxial_nonlinear_dev.npz'))

t = train_uniaxial['time']
dt = t[1] - t[0]

# True values for validation
G_true = 1000.0
eta_0_true = 500.0
alpha_true = 0.00015

print("=" * 60)
print("Fitting Coupled Nonlinear Viscoelastic SINDy Model")
print(f"Expected coefficients:")
print(f"  c1 (linear stress): {-2*G_true/eta_0_true:.4f}")
print(f"  c2 (strain rate):   {2*G_true:.4f}")
print(f"  c3 (coupled stress):{-2*G_true*alpha_true/eta_0_true:.6f}")
print("=" * 60)

# ==========================================
# 2. PREPARE MULTI-TRAJECTORY FEATURES
# ==========================================
def prepare_sindy_matrices(data_file, noisy=True):
    # Retrieve stress matrix (n_times, 6)
    if noisy:
        S = data_file['dev_stress_noisy']
    else:
        S = data_file['dev_stress']
        
    # Retrieve strain rate matrix (n_times, 6)
    ed = data_file['strain_rate_dev']
    
    # Compute σ_eq^2 = 1.5 * (S_xx**2 + S_yy**2 + S_zz**2 + 2*S_xy**2 + 2*S_yz**2 + 2*S_xz**2)
    S_xx, S_yy, S_zz, S_xy, S_yz, S_xz = S.T
    sig_eq_sq = 1.5 * (S_xx**2 + S_yy**2 + S_zz**2 + 2*S_xy**2 + 2*S_yz**2 + 2*S_xz**2)
    
    # Build nonlinear coupled features: S_ij * σ_eq^2
    S_eq = S * sig_eq_sq.reshape(-1, 1)
    
    # State variables: S_xx, S_yy, S_zz, S_xy, S_yz, S_xz
    X = S
    
    # Inputs: 6 strain rates + 6 J2 coupled stresses
    U = np.column_stack([ed, S_eq])
    
    return X, U

X_uni, U_uni = prepare_sindy_matrices(train_uniaxial, noisy=True)
X_shr, U_shr = prepare_sindy_matrices(train_shear, noisy=True)
X_train = [X_uni, X_shr]
U_train = [U_uni, U_shr]

# ==========================================
# 3. COMPONENT-BY-COMPONENT SINDY FITTING
# ==========================================
# Degree-3 library on [S, ed, Seq] (20 terms incl. bias) -- NOT restricted to the
# expected {bias, S, ed, Seq} terms. A degree-3 fit here is ill-conditioned for two
# independent reasons (see results.tex, Section "Training and validation loading
# paths" and "Open item"):
#   (1) a pure single-axis training path forces sigma_eq^2 to be an exact cubic
#       function of the fitted component alone, making the Seq feature perfectly
#       collinear with the S^3 library term -- fixed by training on a genuinely
#       biaxial path (see generate_nonlinear_deviatoric_data.py).
#   (2) the true coefficients span ~6 orders of magnitude (ed ~ 2000, S ~ -4,
#       Seq ~ -0.0006), so a single global STLSQ threshold on raw (unscaled)
#       library columns cannot separate the tiny-but-real Seq term from mid-size
#       spurious cross terms -- fixed by rescaling every column to unit std before
#       thresholding, then un-scaling the surviving coefficients back to physical
#       units (rescale -> prune -> unscale).
LIB_TERM_NAMES = ["1", "S", "ed", "Seq", "S^2", "S ed", "S Seq", "ed^2", "ed Seq", "Seq^2",
                  "S^3", "S^2 ed", "S^2 Seq", "S ed^2", "S ed Seq", "S Seq^2",
                  "ed^3", "ed^2 Seq", "ed Seq^2", "Seq^3"]
EXPECTED_TERMS = {"1", "S", "ed", "Seq"}
SWEEP_THRESHOLDS = [1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
STABLE_THRESHOLD = 5.0  # mid/end of the plateau confirmed by the sweep below


def build_degree3_library(S, ed, Seq):
    """Degree-3 polynomial library over the 3 raw features [S, ed, Seq]."""
    return np.column_stack([
        np.ones_like(S), S, ed, Seq,
        S**2, S*ed, S*Seq, ed**2, ed*Seq, Seq**2,
        S**3, S**2*ed, S**2*Seq, S*ed**2, S*ed*Seq, S*Seq**2,
        ed**3, ed**2*Seq, ed*Seq**2, Seq**3,
    ])


def stlsq(Theta, y, threshold, n_iter=20):
    """Plain sequential-thresholded least squares (STLSQ)."""
    coef, *_ = np.linalg.lstsq(Theta, y, rcond=None)
    for _ in range(n_iter):
        small = np.abs(coef) < threshold
        coef[small] = 0.0
        keep = ~small
        if not keep.any():
            break
        coef[keep], *_ = np.linalg.lstsq(Theta[:, keep], y, rcond=None)
    return coef


def fit_component_degree3(S, ed, Seq, dt, label):
    """Rescale -> STLSQ prune -> unscale fit of one deviatoric component.

    Runs the full threshold sweep for the stability check (printed for the report),
    then returns the (c0, c1_S, c2_ed, c3_Seq) coefficients at STABLE_THRESHOLD and
    asserts every higher-order library term is zero within tolerance -- the same
    "don't silently truncate" safety check used in Emanuele/simple_visco_sindy.py.
    """
    Theta = build_degree3_library(S, ed, Seq)
    dXdt = np.gradient(S, dt)

    scales = Theta.std(axis=0)
    scales[scales == 0] = 1.0
    Theta_scaled = Theta / scales

    print(f"\n--- Threshold sweep, degree-3 library, component: {label} ---")
    sweep_results = []
    for thr in SWEEP_THRESHOLDS:
        coef_scaled = stlsq(Theta_scaled, dXdt, thr)
        coef_phys = coef_scaled / scales
        nz = [(n, c) for n, c in zip(LIB_TERM_NAMES, coef_phys) if abs(c) > 1e-12]
        sweep_results.append((thr, coef_phys.copy()))
        print(f"  threshold={thr:<6} nnz={len(nz):2d}  " + ", ".join(f"{n}={c:.5g}" for n, c in nz))

    final_scaled = stlsq(Theta_scaled, dXdt, STABLE_THRESHOLD)
    final_coef = final_scaled / scales
    surviving = {n for n, c in zip(LIB_TERM_NAMES, final_coef) if abs(c) > 1e-12}
    spurious = surviving - EXPECTED_TERMS
    assert not spurious, (
        f"{label}: degree-3 fit kept unexpected terms at threshold={STABLE_THRESHOLD}: {spurious}. "
        "Library/threshold no longer isolates the true {bias, S, ed, Seq} model -- do not "
        "silently drop them, investigate before using these coefficients."
    )

    idx = {n: i for i, n in enumerate(LIB_TERM_NAMES)}
    c0 = final_coef[idx["1"]]
    c1 = final_coef[idx["S"]]
    c2 = final_coef[idx["ed"]]
    c3 = final_coef[idx["Seq"]]
    return np.array([c0, c1, c2, c3]), sweep_results


# Fit S_xx model (using biaxial-normal training data)
coeffs_normal, sweep_normal = fit_component_degree3(
    X_uni[:, 0], U_uni[:, 0], U_uni[:, 6], dt, "S_xx (biaxial-normal training)")

# Fit S_xy model (using biaxial-shear training data)
coeffs_shear, sweep_shear = fit_component_degree3(
    X_shr[:, 3], U_shr[:, 3], U_shr[:, 9], dt, "S_xy (biaxial-shear training)")

# ==========================================
# 4. CONSTRUCT FULL 6x19 COEFFICIENT MATRIX
# ==========================================
# State derivatives depend only on their corresponding component features
# Rows correspond to states: S_xx, S_yy, S_zz, S_xy, S_yz, S_xz
# Columns correspond to library terms:
# 0: bias
# 1-6: S_xx, S_yy, S_zz, S_xy, S_yz, S_xz
# 7-12: ed_xx, ed_yy, ed_zz, ed_xy, ed_yz, ed_xz
# 13-18: Seq_xx, Seq_yy, Seq_zz, Seq_xy, Seq_yz, Seq_xz
coeffs = np.zeros((6, 19))

c0_n, c1_n, c2_n, c3_n = coeffs_normal
c0_s, c1_s, c2_s, c3_s = coeffs_shear

# Fill normal stress components
for i in range(3):
    coeffs[i, 0] = c0_n
    coeffs[i, 1 + i] = c1_n      # S_i
    coeffs[i, 7 + i] = c2_n      # ed_i
    coeffs[i, 13 + i] = c3_n     # Seq_i

# Fill shear stress components
for i in range(3):
    row = 3 + i
    coeffs[row, 0] = c0_s
    coeffs[row, 4 + i] = c1_s    # S_i
    coeffs[row, 10 + i] = c2_s   # ed_i
    coeffs[row, 16 + i] = c3_s   # Seq_i

# Initialize dummy 6D SINDy model to set up library structures and assign coefficients
poly_lib = ps.PolynomialLibrary(degree=1, include_bias=True)
optimizer = ps.STLSQ(threshold=1e-5)
model = ps.SINDy(feature_library=poly_lib, optimizer=optimizer)

feature_names = ["S_xx", "S_yy", "S_zz", "S_xy", "S_yz", "S_xz",
                 "ed_xx", "ed_yy", "ed_zz", "ed_xy", "ed_yz", "ed_xz",
                 "Seq_xx", "Seq_yy", "Seq_zz", "Seq_xy", "Seq_yz", "Seq_xz"]

model.fit(X_train, u=U_train, t=dt, feature_names=feature_names)
model.optimizer.coef_ = coeffs

print("\n--- Discovered 3D Coupled Equations ---")
model.print()

# ==========================================
# 5. PARAMETER RECOVERY
# ==========================================
# Uniaxial equation results (Equation 0 for S_xx)
c_Sxx = coeffs[0]
G_uni = c_Sxx[7] / 2.0  # ed_xx is index 7
eta_uni = -2.0 * G_uni / c_Sxx[1]  # S_xx is index 1
alpha_uni = (c_Sxx[13] * eta_uni) / (-2.0 * G_uni)  # Seq_xx is index 13

print("\n" + "=" * 60)
print("Recovered Material Parameters")
print("=" * 60)
print(f"From S_xx Equation (Uniaxial):")
print(f"  G:     {G_uni:.2f} MPa (True: {G_true:.2f} MPa) | Error: {100*abs(G_uni-G_true)/G_true:.3f}%")
print(f"  eta_0: {eta_uni:.2f} MPa*s (True: {eta_0_true:.2f} MPa*s) | Error: {100*abs(eta_uni-eta_0_true)/eta_0_true:.3f}%")
print(f"  alpha: {alpha_uni:.6f} 1/MPa^2 (True: {alpha_true:.6f} 1/MPa^2) | Error: {100*abs(alpha_uni-alpha_true)/alpha_true:.3f}%")

# Shear equation results (Equation 3 for S_xy)
c_Sxy = coeffs[3]
G_shr = c_Sxy[10] / 2.0  # ed_xy is index 10
eta_shr = -2.0 * G_shr / c_Sxy[4]  # S_xy is index 4
alpha_shr = (c_Sxy[16] * eta_shr) / (-2.0 * G_shr)  # Seq_xy is index 16

print(f"\nFrom S_xy Equation (Shear):")
print(f"  G:     {G_shr:.2f} MPa (True: {G_true:.2f} MPa) | Error: {100*abs(G_shr-G_true)/G_true:.3f}%")
print(f"  eta_0: {eta_shr:.2f} MPa*s (True: {eta_0_true:.2f} MPa*s) | Error: {100*abs(eta_shr-eta_0_true)/eta_0_true:.3f}%")
print(f"  alpha: {alpha_shr:.6f} 1/MPa^2 (True: {alpha_true:.6f} 1/MPa^2) | Error: {100*abs(alpha_shr-alpha_true)/alpha_true:.3f}%")

# Averaged Parameters
G_avg = (G_uni + G_shr) / 2.0
eta_avg = (eta_uni + eta_shr) / 2.0
alpha_avg = (alpha_uni + alpha_shr) / 2.0

print(f"\nAveraged recovered parameters:")
print(f"  G_avg:     {G_avg:.2f} MPa | Error: {100*abs(G_avg-G_true)/G_true:.3f}%")
print(f"  eta_0_avg: {eta_avg:.2f} MPa*s | Error: {100*abs(eta_avg-eta_0_true)/eta_0_true:.3f}%")
print(f"  alpha_avg: {alpha_avg:.6f} 1/MPa^2 | Error: {100*abs(alpha_avg-alpha_true)/alpha_true:.3f}%")

# Save recovered coefficients
np.save(os.path.join(dir_path, 'discovered_nonlinear_coeffs.npy'), coeffs)
print(f"\nDiscovered SINDy coefficients saved to discovered_nonlinear_coeffs.npy")

# ==========================================
# 6. MULTIAXIAL VALIDATION (UNSEEN PATHS)
# ==========================================
print("\n" + "=" * 60)
print("Validating SINDy Model on Unseen Paths")
print("=" * 60)

def simulate_paths(val_data):
    val_t = val_data['time']
    val_ed = val_data['strain_rate_dev']
    
    # Interpolation functions for strain rate components
    def ed_func(t):
        return np.array([np.interp(t, val_t, val_ed[:, col]) for col in range(6)])
    
    # 6D Nonlinear Maxwell ODE using SINDy coefficients
    def discovered_ode(t, S):
        S_xx, S_yy, S_zz, S_xy, S_yz, S_xz = S
        ed = ed_func(t)
        sig_eq_sq = 1.5 * (S_xx**2 + S_yy**2 + S_zz**2 + 2*S_xy**2 + 2*S_yz**2 + 2*S_xz**2)
        
        dS_xx = c0_n + c1_n * S_xx + c2_n * ed[0] + c3_n * S_xx * sig_eq_sq
        dS_yy = c0_n + c1_n * S_yy + c2_n * ed[1] + c3_n * S_yy * sig_eq_sq
        dS_zz = c0_n + c1_n * S_zz + c2_n * ed[2] + c3_n * S_zz * sig_eq_sq
        
        dS_xy = c0_s + c1_s * S_xy + c2_s * ed[3] + c3_s * S_xy * sig_eq_sq
        dS_yz = c0_s + c1_s * S_yz + c2_s * ed[4] + c3_s * S_yz * sig_eq_sq
        dS_xz = c0_s + c1_s * S_xz + c2_s * ed[5] + c3_s * S_xz * sig_eq_sq
        
        return [dS_xx, dS_yy, dS_zz, dS_xy, dS_yz, dS_xz]
        
    # 6D Linear Maxwell ODE (alpha = 0) as reference comparison
    def linear_reference_ode(t, S):
        S_xx, S_yy, S_zz, S_xy, S_yz, S_xz = S
        ed = ed_func(t)
        
        dS_xx = 2.0 * G_true * ed[0] - (2.0 * G_true / eta_0_true) * S_xx
        dS_yy = 2.0 * G_true * ed[1] - (2.0 * G_true / eta_0_true) * S_yy
        dS_zz = 2.0 * G_true * ed[2] - (2.0 * G_true / eta_0_true) * S_zz
        
        dS_xy = 2.0 * G_true * ed[3] - (2.0 * G_true / eta_0_true) * S_xy
        dS_yz = 2.0 * G_true * ed[4] - (2.0 * G_true / eta_0_true) * S_yz
        dS_xz = 2.0 * G_true * ed[5] - (2.0 * G_true / eta_0_true) * S_xz
        
        return [dS_xx, dS_yy, dS_zz, dS_xy, dS_yz, dS_xz]
        
    sol_sindy = solve_ivp(
        fun=discovered_ode,
        t_span=(val_t[0], val_t[-1]),
        y0=np.zeros(6),
        t_eval=val_t,
        method='Radau',
        rtol=1e-8,
        atol=1e-10
    )
    
    sol_linear = solve_ivp(
        fun=linear_reference_ode,
        t_span=(val_t[0], val_t[-1]),
        y0=np.zeros(6),
        t_eval=val_t,
        method='Radau',
        rtol=1e-8,
        atol=1e-10
    )
    
    return sol_sindy.y.T, sol_linear.y.T

S_pred_comb, S_lin_comb = simulate_paths(val_combined)
S_pred_biax, S_lin_biax = simulate_paths(val_biaxial)

# Compute RMSEs
rmse_comb_xx = np.sqrt(np.mean((val_combined['dev_stress'][:, 0] - S_pred_comb[:, 0])**2))
rmse_comb_xy = np.sqrt(np.mean((val_combined['dev_stress'][:, 3] - S_pred_comb[:, 3])**2))

rmse_biax_xx = np.sqrt(np.mean((val_biaxial['dev_stress'][:, 0] - S_pred_biax[:, 0])**2))
rmse_biax_yy = np.sqrt(np.mean((val_biaxial['dev_stress'][:, 1] - S_pred_biax[:, 1])**2))

print("Validation RMSEs:")
print(f"  Combined Load: S_xx RMSE = {rmse_comb_xx:.4f} MPa, S_xy RMSE = {rmse_comb_xy:.4f} MPa")
print(f"  Biaxial Load:  S_xx RMSE = {rmse_biax_xx:.4f} MPa, S_yy RMSE = {rmse_biax_yy:.4f} MPa")

# ==========================================
# 7. GENERATING THE VALIDATION PLOTS
# ==========================================
print("\nGenerating validation comparison plots...")

plt.rcParams.update({
    'axes.labelsize': 13,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
})

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Combined Load S_xx
axs[0, 0].plot(val_combined['time'], val_combined['dev_stress'][:, 0], 'k-', lw=2, label='True Nonlinear Maxwell')
axs[0, 0].plot(val_combined['time'], S_pred_comb[:, 0], 'r--', lw=2, label='SINDy Model')
axs[0, 0].plot(val_combined['time'], S_lin_comb[:, 0], 'b:', alpha=0.6, label='Linear Maxwell (Reference)')
axs[0, 0].set_title(f"Validation Combined Load: S_xx (RMSE = {rmse_comb_xx:.4f} MPa)")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Stress (MPa)")
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# Plot 2: Combined Load S_xy
axs[0, 1].plot(val_combined['time'], val_combined['dev_stress'][:, 3], 'k-', lw=2, label='True Nonlinear Maxwell')
axs[0, 1].plot(val_combined['time'], S_pred_comb[:, 3], 'r--', lw=2, label='SINDy Model')
axs[0, 1].plot(val_combined['time'], S_lin_comb[:, 3], 'b:', alpha=0.6, label='Linear Maxwell (Reference)')
axs[0, 1].set_title(f"Validation Combined Load: S_xy (RMSE = {rmse_comb_xy:.4f} MPa)")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Stress (MPa)")
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].legend()

# Plot 3: Biaxial Load S_xx
axs[1, 0].plot(val_biaxial['time'], val_biaxial['dev_stress'][:, 0], 'k-', lw=2, label='True Nonlinear Maxwell')
axs[1, 0].plot(val_biaxial['time'], S_pred_biax[:, 0], 'r--', lw=2, label='SINDy Model')
axs[1, 0].plot(val_biaxial['time'], S_lin_biax[:, 0], 'b:', alpha=0.6, label='Linear Maxwell (Reference)')
axs[1, 0].set_title(f"Validation Biaxial Load: S_xx (RMSE = {rmse_biax_xx:.4f} MPa)")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Stress (MPa)")
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].legend()

# Plot 4: Biaxial Load S_yy
axs[1, 1].plot(val_biaxial['time'], val_biaxial['dev_stress'][:, 1], 'k-', lw=2, label='True Nonlinear Maxwell')
axs[1, 1].plot(val_biaxial['time'], S_pred_biax[:, 1], 'r--', lw=2, label='SINDy Model')
axs[1, 1].plot(val_biaxial['time'], S_lin_biax[:, 1], 'b:', alpha=0.6, label='Linear Maxwell (Reference)')
axs[1, 1].set_title(f"Validation Biaxial Load: S_yy (RMSE = {rmse_biax_yy:.4f} MPa)")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Stress (MPa)")
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend()

plt.tight_layout()
plot_path = os.path.join(dir_path, 'nonlinear_deviatoric_comparison.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Updated comparison plot saved to: nonlinear_deviatoric_comparison.png")

# --- Also save each validation panel as its own file, for figure-by-figure report reference ---
fig3d_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures", "3D")
fig3d_dir = os.path.normpath(fig3d_dir)
os.makedirs(fig3d_dir, exist_ok=True)

panels = [
    (f"01_combined_Sxx_rmse{rmse_comb_xx:.4f}.png", val_combined['time'], val_combined['dev_stress'][:, 0],
     S_pred_comb[:, 0], S_lin_comb[:, 0]),
    (f"02_combined_Sxy_rmse{rmse_comb_xy:.4f}.png", val_combined['time'], val_combined['dev_stress'][:, 3],
     S_pred_comb[:, 3], S_lin_comb[:, 3]),
    (f"03_biaxial_Sxx_rmse{rmse_biax_xx:.4f}.png", val_biaxial['time'], val_biaxial['dev_stress'][:, 0],
     S_pred_biax[:, 0], S_lin_biax[:, 0]),
    (f"04_biaxial_Syy_rmse{rmse_biax_yy:.4f}.png", val_biaxial['time'], val_biaxial['dev_stress'][:, 1],
     S_pred_biax[:, 1], S_lin_biax[:, 1]),
]
for fname, time_arr, true_arr, sindy_arr, lin_arr in panels:
    plt.figure(figsize=(7, 5))
    plt.plot(time_arr, true_arr, 'k-', lw=2, label='True Nonlinear Maxwell')
    plt.plot(time_arr, sindy_arr, 'r--', lw=2, label='SINDy Model')
    plt.plot(time_arr, lin_arr, 'b:', alpha=0.6, label='Linear Maxwell (Reference)')
    plt.xlabel("Time (s)")
    plt.ylabel("Stress (MPa)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(fig3d_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

# --- Threshold-sensitivity sweep plot: degree-3 library, rescale-prune-unscale coefficients ---
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, (sweep, label) in zip(axs, [(sweep_normal, "S_xx (normal)"), (sweep_shear, "S_xy (shear)")]):
    thr_arr = [thr for thr, _ in sweep]
    for term in ["S", "ed", "Seq"]:
        j = LIB_TERM_NAMES.index(term)
        vals = [coef[j] for _, coef in sweep]
        ax.plot(thr_arr, vals, marker='o', label=term)
    ax.axvspan(1.0, 5.0, color='green', alpha=0.15, label='stable plateau')
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_xlabel("STLSQ threshold (rescaled space)")
    ax.set_ylabel("Coefficient (physical units)")
    ax.set_title(f"Threshold sweep -- {label}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
plt.tight_layout()
sweep_path = os.path.join(fig3d_dir, "05_threshold_sensitivity_sweep_degree3_rescaled.png")
plt.savefig(sweep_path, dpi=150)
plt.close()
print(f"Saved {sweep_path}")

print("=" * 60)
print("SUCCESS: Nonlinear SINDy fitting, validation, and plotting complete!")
print("=" * 60)
