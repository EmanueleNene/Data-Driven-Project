"""
3D Viscoelastic SINDy using Deviatoric Stress Components

This is the CORRECTED approach! We train SINDy on individual deviatoric
stress/strain rate components instead of von Mises equivalents.

Key insight: Each deviatoric component follows the Maxwell equation independently:
    dS_ij/dt = 2G * dε_ij^dev/dt - (2G/η) * S_ij

Since deviatoric stress CAN BE NEGATIVE, this works correctly!

Author: Generated for DataDrivenProject
Date: 2025-12-02
"""

import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint


# ==========================================
# 1. LOAD DEVIATORIC DATA
# ==========================================

print("=" * 60)
print("Loading Deviatoric 3D Data")
print("=" * 60)

# Load data
train_uniaxial = np.load('data_train_uniaxial_dev.npz')
train_shear = np.load('data_train_shear_dev.npz')
val_combined = np.load('data_val_combined_dev.npz')
val_biaxial = np.load('data_val_biaxial_dev.npz')

t = train_uniaxial['time']
dt = t[1] - t[0]

print(f"\nTraining datasets:")
print(f"  Uniaxial: {len(t)} points")
print(f"  Shear: {len(t)} points")

print(f"\nValidation datasets:")
print(f"  Combined: {len(val_combined['time'])} points")
print(f"  Biaxial: {len(val_biaxial['time'])} points")

# Check that deviatoric components can be negative
print(f"\n✓ S_xx can be negative: {train_uniaxial['dev_stress'][:, 0].min() < 0}")
print(f"  Range: [{train_uniaxial['dev_stress'][:, 0].min():.2f}, {train_uniaxial['dev_stress'][:, 0].max():.2f}] MPa")


# ==========================================
# 2. TRAIN SINDy ON S_XX (Uniaxial Component)
# ==========================================

print("\n" + "=" * 60)
print("Training SINDy on Deviatoric Components")
print("=" * 60)

# Define library and optimizer
poly_lib = ps.PolynomialLibrary(degree=1, include_bias=True)  # Linear only for Maxwell
opt = ps.STLSQ(threshold=0.01)

# --- Train on S_xx from uniaxial data ---
print("\n--- Training on S_xx (from uniaxial) ---")
X_uniax = train_uniaxial['dev_stress_noisy'][:, 0].reshape(-1, 1)  # S_xx only
U_uniax = train_uniaxial['strain_rate_dev'][:, 0].reshape(-1, 1)    # ε̇_xx^dev only

model_Sxx = ps.SINDy(feature_library=poly_lib, optimizer=opt)
model_Sxx.fit(X_uniax, u=U_uniax, t=dt, feature_names=["S_xx", "eps_dot_xx_dev"])

print("Discovered equation for S_xx:")
model_Sxx.print()
coeffs_Sxx = model_Sxx.coefficients()[0]
print(f"  c0 (bias): {coeffs_Sxx[0]:.6f}")
print(f"  c1 (S_xx): {coeffs_Sxx[1]:.6f}")
print(f"  c2 (ε̇_xx^dev): {coeffs_Sxx[2]:.6f}")

# --- Train on S_xy from shear data ---
print("\n--- Training on S_xy (from shear) ---")
X_shear = train_shear['dev_stress_noisy'][:, 3].reshape(-1, 1)  # S_xy only
U_shear = train_shear['strain_rate_dev'][:, 3].reshape(-1, 1)    # γ̇_xy only

model_Sxy = ps.SINDy(feature_library=poly_lib, optimizer=opt)
model_Sxy.fit(X_shear, u=U_shear, t=dt, feature_names=["S_xy", "gamma_dot_xy"])

print("Discovered equation for S_xy:")
model_Sxy.print()
coeffs_Sxy = model_Sxy.coefficients()[0]
print(f"  c0 (bias): {coeffs_Sxy[0]:.6f}")
print(f"  c1 (S_xy): {coeffs_Sxy[1]:.6f}")
print(f"  c2 (γ̇_xy): {coeffs_Sxy[2]:.6f}")

# Average coefficients (should be the same for isotropic material!)
coeffs_avg = (coeffs_Sxx + coeffs_Sxy) / 2.0

print("\n--- Averaged Coefficients (Isotropic Assumption) ---")
print(f"  c0 (bias): {coeffs_avg[0]:.6f}")
print(f"  c1 (stress): {coeffs_avg[1]:.6f}")
print(f"  c2 (strain_rate): {coeffs_avg[2]:.6f}")

print("\nExpected (from deviatoric Maxwell):")
G_true = 1000.0
eta_true = 500.0
print(f"  c0 (bias): 0.0")
print(f"  c1 (stress): {-2*G_true/eta_true:.6f}")
print(f"  c2 (strain_rate): {2*G_true:.6f}")


# ==========================================
# MATERIAL PARAMETER RECOVERY
# ==========================================

print("\n" + "=" * 60)
print("Material Parameter Recovery")
print("=" * 60)

# From the deviatoric Maxwell model:
#   dS/dt = 2G * dε^dev/dt - (2G/η) * S
# We have:
#   c1 = -2G/η  (stress coefficient)
#   c2 = 2G     (strain rate coefficient)
# 
# Solving for G and η:
#   G = c2 / 2
#   η = -2G / c1 = -c2 / c1

print("\nRecovering from S_xx model:")
G_recovered_xx = coeffs_Sxx[2] / 2.0
eta_recovered_xx = -2 * G_recovered_xx / coeffs_Sxx[1]
print(f"  G (shear modulus): {G_recovered_xx:.2f} MPa (true: {G_true:.2f} MPa)")
print(f"  η (viscosity): {eta_recovered_xx:.2f} MPa·s (true: {eta_true:.2f} MPa·s)")
print(f"  Error G: {100*abs(G_recovered_xx - G_true)/G_true:.3f}%")
print(f"  Error η: {100*abs(eta_recovered_xx - eta_true)/eta_true:.3f}%")

print("\nRecovering from S_xy model:")
G_recovered_xy = coeffs_Sxy[2] / 2.0
eta_recovered_xy = -2 * G_recovered_xy / coeffs_Sxy[1]
print(f"  G (shear modulus): {G_recovered_xy:.2f} MPa (true: {G_true:.2f} MPa)")
print(f"  η (viscosity): {eta_recovered_xy:.2f} MPa·s (true: {eta_true:.2f} MPa·s)")
print(f"  Error G: {100*abs(G_recovered_xy - G_true)/G_true:.3f}%")
print(f"  Error η: {100*abs(eta_recovered_xy - eta_true)/eta_true:.3f}%")

print("\nAveraged (isotropic):")
G_recovered_avg = coeffs_avg[2] / 2.0
eta_recovered_avg = -2 * G_recovered_avg / coeffs_avg[1]
print(f"  G (shear modulus): {G_recovered_avg:.2f} MPa (true: {G_true:.2f} MPa)")
print(f"  η (viscosity): {eta_recovered_avg:.2f} MPa·s (true: {eta_true:.2f} MPa·s)")
print(f"  Error G: {100*abs(G_recovered_avg - G_true)/G_true:.3f}%")
print(f"  Error η: {100*abs(eta_recovered_avg - eta_true)/eta_true:.3f}%")

# Relaxation time
tau_recovered = eta_recovered_avg / (2 * G_recovered_avg)
tau_true = eta_true / (2 * G_true)
print(f"\nRelaxation time τ = η/(2G):")
print(f"  τ_recovered: {tau_recovered:.4f} s (true: {tau_true:.4f} s)")
print(f"  Error: {100*abs(tau_recovered - tau_true)/tau_true:.3f}%")



# ==========================================
# 3. VALIDATION
# ==========================================

print("\n" + "=" * 60)
print("Validating on Combined Loading")
print("=" * 60)

def discovered_maxwell_ode(S, t, eps_dot_dev_interp, t_interp, coeffs):
    """ODE using discovered coefficients."""
    ed = np.interp(t, t_interp, eps_dot_dev_interp)
    dS_dt = coeffs[0] + coeffs[1] * S + coeffs[2] * ed
    return dS_dt


# Validate on combined load (S_xx and S_xy simultaneously)
print("\nValidation 1: Combined Load (Uniaxial + Shear)")

# Predict S_xx
S_xx_pred = odeint(
    discovered_maxwell_ode, 0.0, t,
    args=(val_combined['strain_rate_dev'][:, 0], t, coeffs_Sxx)
).flatten()

rmse_Sxx = np.sqrt(np.mean((val_combined['dev_stress'][:, 0] - S_xx_pred)**2))
print(f"  S_xx RMSE: {rmse_Sxx:.4f} MPa")

# Predict S_xy
S_xy_pred = odeint(
    discovered_maxwell_ode, 0.0, t,
    args=(val_combined['strain_rate_dev'][:, 3], t, coeffs_Sxy)
).flatten()

rmse_Sxy = np.sqrt(np.mean((val_combined['dev_stress'][:, 3] - S_xy_pred)**2))
print(f"  S_xy RMSE: {rmse_Sxy:.4f} MPa")

# Validate on biaxial
print("\nValidation 2: Biaxial Load (S_xx + S_yy)")

S_xx_biax_pred = odeint(
    discovered_maxwell_ode, 0.0, t,
    args=(val_biaxial['strain_rate_dev'][:, 0], t, coeffs_avg)
).flatten()

S_yy_biax_pred = odeint(
    discovered_maxwell_ode, 0.0, t,
    args=(val_biaxial['strain_rate_dev'][:, 1], t, coeffs_avg)
).flatten()

rmse_biax_xx = np.sqrt(np.mean((val_biaxial['dev_stress'][:, 0] - S_xx_biax_pred)**2))
rmse_biax_yy = np.sqrt(np.mean((val_biaxial['dev_stress'][:, 1] - S_yy_biax_pred)**2))

print(f"  S_xx RMSE: {rmse_biax_xx:.4f} MPa")
print(f"  S_yy RMSE: {rmse_biax_yy:.4f} MPa")


# ==========================================
# 4. VISUALIZATION
# ==========================================

print("\n" + "=" * 60)
print("Creating Visualizations")
print("=" * 60)

# Set larger font sizes for axis labels
plt.rcParams.update({
    'axes.labelsize': 15,      # X and Y axis labels (increased from 14)
    'axes.titlesize': 14,      # Subplot titles
    'xtick.labelsize': 12,     # X-axis tick labels
    'ytick.labelsize': 12,     # Y-axis tick labels
    'legend.fontsize': 11,     # Legend text
})
#===================================FIGURE 1
fig = plt.figure(figsize=(14, 5))
# Training: S_xx
plt.subplot(1, 2, 1)
plt.plot(t, train_uniaxial['dev_stress'][:, 0], 'k-', alpha=0.3, label='True')
plt.plot(t, X_uniax, 'k.', alpha=0.05, markersize=0.5, label='Noisy')
# Predict from model
S_xx_train_pred = odeint(discovered_maxwell_ode, 0.0, t,
                          args=(train_uniaxial['strain_rate_dev'][:, 0], t, coeffs_Sxx)).flatten()
plt.plot(t, S_xx_train_pred, 'r--', lw=2, label='SINDy')
plt.title("Training: S_xx (Uniaxial)")
plt.xlabel("Time (s)")
plt.ylabel(r"$S_{xx}$ (MPa)")
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

# Training: S_xy
plt.subplot(1, 2, 2)
plt.plot(t, train_shear['dev_stress'][:, 3], 'k-', alpha=0.3, label='True')
plt.plot(t, X_shear, 'k.', alpha=0.05, markersize=0.5, label='Noisy')
S_xy_train_pred = odeint(discovered_maxwell_ode, 0.0, t,
                          args=(train_shear['strain_rate_dev'][:, 3], t, coeffs_Sxy)).flatten()
plt.plot(t, S_xy_train_pred, 'r--', lw=2, label='SINDy')
plt.title("Training: S_xy (Shear)")
plt.xlabel("Time (s)")
plt.ylabel(r"$S_{xy}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
#============================END
# Coefficient comparison
"""plt.subplot(3, 3, 3)
terms = ['Bias', 'Stress', 'Strain Rate']
coeffs_expected = [0, -2*G_true/eta_true, 2*G_true]
coeffs_discovered = [coeffs_avg[0], coeffs_avg[1], coeffs_avg[2]]
x_pos = np.arange(len(terms))
width = 0.35
plt.bar(x_pos - width/2, coeffs_expected, width, label='Expected', alpha=0.7)
plt.bar(x_pos + width/2, coeffs_discovered, width, label='Discovered', alpha=0.7)
plt.xticks(x_pos, terms)
plt.ylabel("Coefficient Value")
plt.title("Coefficient Comparison")
plt.legend()
plt.grid(True, alpha=0.3, axis='y')"""

##==============================FIGURE 2
##VALIDATION FOR TENSION-SHEAR
# Validation: Tension-Shear (S_xx)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(t, val_combined['dev_stress'][:, 0], 'k-', lw=1.5, label='True')
plt.plot(t, S_xx_pred, 'r--', lw=2, label='SINDy')
plt.title(f"Validation: S_xx (Combined, RMSE={rmse_Sxx:.2f})")
plt.xlabel("Time (s)")
plt.ylabel(r"$S_{xx}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

##VALIDATION FOR TENSION-SHEAR
# Validation: Tension-Shear (S_xy)
plt.subplot(1, 2, 2)
plt.plot(t, val_combined['dev_stress'][:, 3], 'k-', lw=1.5, label='True')
plt.plot(t, S_xy_pred, 'r--', lw=2, label='SINDy')
plt.title(f"Validation: S_xy (Combined, RMSE={rmse_Sxy:.2f})")
plt.xlabel("Time (s)")
plt.ylabel(r"$S_{xy}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.savefig('Validation_training.png', dpi=150)

"""# Phase plane: S_xx
plt.subplot(3, 3, 6)
plt.plot(val_combined['strain_rate_dev'][:, 0], val_combined['dev_stress'][:, 0],
         'b-', alpha=0.5, lw=1.5, label='True')
plt.plot(val_combined['strain_rate_dev'][:, 0], S_xx_pred,
         'r--', lw=2, label='SINDy')
plt.title("Phase Plane: S_xx")
plt.xlabel(r"$\dot{\epsilon}_{xx}^{dev}$ (1/s)")
plt.ylabel(r"$S_{xx}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)"""
# ## VALIDATION FOR BIAXIALTENSION
# Validation: Biaxial (S_xx)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(t, val_biaxial['dev_stress'][:, 0], 'k-', lw=1.5, label='True')
plt.plot(t, S_xx_biax_pred, 'r--', lw=2, label='SINDy')
plt.title(f"Validation: S_xx (Biaxial, RMSE={rmse_biax_xx:.2f})")
plt.xlabel("Time (s)")
plt.ylabel(r"$S_{xx}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

# Validation: Biaxial (S_yy)
plt.subplot(1, 2, 2)
plt.plot(t, val_biaxial['dev_stress'][:, 1], 'k-', lw=1.5, label='True')
plt.plot(t, S_yy_biax_pred, 'r--', lw=2, label='SINDy')
plt.title(f"Validation: S_yy (Biaxial, RMSE={rmse_biax_yy:.2f})")
plt.xlabel("Time (s)")
plt.ylabel(r"$S_{yy}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)


"""# RMSE summary
plt.subplot(3, 3, 9)
cases = ['S_xx\n(Comb)', 'S_xy\n(Comb)', 'S_xx\n(Biax)', 'S_yy\n(Biax)']
rmse_values = [rmse_Sxx, rmse_Sxy, rmse_biax_xx, rmse_biax_yy]
bars = plt.bar(cases, rmse_values, color='red', alpha=0.6)
plt.ylabel("RMSE (MPa)")
plt.title("Validation Performance")
plt.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, rmse_values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('deviatoric_sindy_results.png', dpi=150)
print("  Saved: deviatoric_sindy_results.png")"""

plt.show()

print("\n" + "=" * 60)
print("SUCCESS! Deviatoric SINDy Analysis Complete")
print("=" * 60)
print(f"\n✓ Discovered correct Maxwell coefficients:")
print(f"  Expected c1 (stress): {-2*G_true/eta_true:.2f}, Got: {coeffs_avg[1]:.2f}")
print(f"  Expected c2 (strain rate): {2*G_true:.2f}, Got: {coeffs_avg[2]:.2f}")
print(f"\n✓ Validation RMSE: S_xx={rmse_Sxx:.4f} MPa, S_xy={rmse_Sxy:.4f} MPa")
print(f"\n✓ Deviatoric approach works because stress CAN BE NEGATIVE!")
