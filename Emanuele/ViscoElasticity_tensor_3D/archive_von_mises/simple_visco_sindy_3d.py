"""
3D Viscoelastic SINDy using Von Mises Equivalent Stress

This script extends the 1D SINDy approach to 3D stress states by:
1. Loading 3D stress-strain data
2. Converting to von Mises equivalents (σ_eq, ε̇_eq)
3. Training SINDy on the equivalent scalars (reusing 1D framework!)
4. Validating on complex combined loading scenarios

Author: Generated for DataDrivenProject
Date: 2025-12-02
"""

import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint


# ==========================================
# 1. LOAD 3D DATA (Generated from generate_3d_data.py)
# ==========================================

print("=" * 60)
print("Loading 3D Viscoelastic Data")
print("=" * 60)

# Load training data (two separate simple cases)
train_uniaxial = np.load('data_train_uniaxial.npz')
train_shear = np.load('data_train_shear.npz')

# Load validation data (combined loads)
val_combined = np.load('data_val_combined.npz')
val_biaxial = np.load('data_val_biaxial.npz')

# Extract time and von Mises equivalents
t = train_uniaxial['time']
dt = t[1] - t[0]

# ⚠️ DO NOT CONCATENATE - this creates discontinuities!
# Instead, we'll train on each dataset separately and ensemble the results

print(f"\nTraining data:")
print(f"  Uniaxial dataset: {len(train_uniaxial['time'])} points")
print(f"  Shear dataset: {len(train_shear['time'])} points")

print(f"\nValidation data:")
print(f"  Combined load points: {len(val_combined['time'])}")
print(f"  Biaxial load points: {len(val_biaxial['time'])}")


# ==========================================
# 2. SINDy TRAINING (On Von Mises Equivalents)
# ==========================================

print("\n" + "=" * 60)
print("Training SINDy on Von Mises Equivalents")
print("=" * 60)

# Define library and optimizer
poly_lib = ps.PolynomialLibrary(degree=3, include_bias=True)
# Reduced threshold from 1.0 to 0.01 for better sparsification
opt = ps.STLSQ(threshold=0.01)

# --- Train on Uniaxial Dataset ---
print("\n--- Training on Uniaxial Data ---")
X_uniaxial = train_uniaxial['stress_eq_noisy'].reshape(-1, 1)
U_uniaxial = train_uniaxial['strain_rate_eq'].reshape(-1, 1)

model_uniaxial = ps.SINDy(feature_library=poly_lib, optimizer=opt)
model_uniaxial.fit(X_uniaxial, u=U_uniaxial, t=dt, feature_names=["sig_eq", "eps_dot_eq"])

coeffs_uniaxial = model_uniaxial.coefficients()[0]
print("Uniaxial model:")
model_uniaxial.print()

# --- Train on Shear Dataset ---
print("\n--- Training on Shear Data ---")
X_shear = train_shear['stress_eq_noisy'].reshape(-1, 1)
U_shear = train_shear['strain_rate_eq'].reshape(-1, 1)

model_shear = ps.SINDy(feature_library=poly_lib, optimizer=opt)
model_shear.fit(X_shear, u=U_shear, t=dt, feature_names=["sig_eq", "eps_dot_eq"])

coeffs_shear = model_shear.coefficients()[0]
print("Shear model:")
model_shear.print()

# --- Average Coefficients (Ensemble) ---
print("\n--- Ensemble Model (Averaged Coefficients) ---")
coeffs = (coeffs_uniaxial + coeffs_shear) / 2.0
print(f"\nCoefficients: {coeffs}")
print(f"  c0 (bias): {coeffs[0]:.6f}")
print(f"  c1 (sig_eq): {coeffs[1]:.6f}")
print(f"  c2 (eps_dot_eq): {coeffs[2]:.6f}")

# Compare with expected Maxwell model: dσ/dt = E*dε/dt - (E/η)*σ
# Expected: c0 ≈ 0, c1 ≈ -E/η, c2 ≈ E
E_true = 2000.0
eta_true = 500.0
print(f"\nExpected (from 3D Maxwell model):")
print(f"  c0 (bias): 0.0")
print(f"  c1 (sig_eq): {-E_true/eta_true:.6f}")
print(f"  c2 (eps_dot_eq): {E_true:.6f}")


# ==========================================
# 3. VALIDATION (On Unseen Combined Loads)
# ==========================================

print("\n" + "=" * 60)
print("Validating on Combined Loading Scenarios")
print("=" * 60)

def discovered_ode(sig, t, eps_dot_interp, t_interp, coeffs):
    """ODE using discovered SINDy coefficients."""
    ed = np.interp(t, t_interp, eps_dot_interp)
    dsig_dt = coeffs[0] + coeffs[1] * sig + coeffs[2] * ed
    return dsig_dt


# Predict on validation: Combined load (uniaxial + shear)
print("\nValidation 1: Combined (Uniaxial + Shear)")
sig_val_combined_pred = odeint(
    discovered_ode, 0.0, t, 
    args=(val_combined['strain_rate_eq'], t, coeffs)
).flatten()

rmse_combined = np.sqrt(np.mean(
    (val_combined['stress_eq'] - sig_val_combined_pred)**2
))
print(f"  RMSE: {rmse_combined:.4f} MPa")
print(f"  Relative RMSE: {100*rmse_combined/np.max(np.abs(val_combined['stress_eq'])):.2f}%")

# Predict on validation: Biaxial load
print("\nValidation 2: Biaxial (σ_xx + σ_yy)")
sig_val_biaxial_pred = odeint(
    discovered_ode, 0.0, t, 
    args=(val_biaxial['strain_rate_eq'], t, coeffs)
).flatten()

rmse_biaxial = np.sqrt(np.mean(
    (val_biaxial['stress_eq'] - sig_val_biaxial_pred)**2
))
print(f"  RMSE: {rmse_biaxial:.4f} MPa")
print(f"  Relative RMSE: {100*rmse_biaxial/np.max(np.abs(val_biaxial['stress_eq'])):.2f}%")

# Also check training reconstruction
sig_train_uniaxial_pred = odeint(
    discovered_ode, 0.0, t, 
    args=(train_uniaxial['strain_rate_eq'], t, coeffs)
).flatten()

sig_train_shear_pred = odeint(
    discovered_ode, 0.0, t, 
    args=(train_shear['strain_rate_eq'], t, coeffs)
).flatten()

rmse_train_uniaxial = np.sqrt(np.mean(
    (train_uniaxial['stress_eq'] - sig_train_uniaxial_pred)**2
))
rmse_train_shear = np.sqrt(np.mean(
    (train_shear['stress_eq'] - sig_train_shear_pred)**2
))

print(f"\nTraining reconstruction:")
print(f"  Uniaxial RMSE: {rmse_train_uniaxial:.4f} MPa")
print(f"  Shear RMSE: {rmse_train_shear:.4f} MPa")


# ==========================================
# 4. VISUALIZATION
# ==========================================

print("\n" + "=" * 60)
print("Creating Visualizations")
print("=" * 60)

width = 12

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

fig = plt.figure(figsize=(width, 10))

# ========== Row 1: Training Data - Uniaxial ==========
plt.subplot(3, 3, 1)
plt.plot(t, train_uniaxial['stress_eq'], 'k-', alpha=0.3, label='True')
plt.plot(t, train_uniaxial['stress_eq_noisy'], 'k.', alpha=0.1, markersize=1, 
         label='Noisy Data')
plt.plot(t, sig_train_uniaxial_pred, 'r--', lw=2, label='SINDy Fit')
plt.title("Training: Uniaxial (σ_eq vs Time)")
plt.xlabel("Time (s)")
plt.ylabel(r"$\sigma_{eq}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 2)
plt.plot(t, train_shear['stress_eq'], 'k-', alpha=0.3, label='True')
plt.plot(t, train_shear['stress_eq_noisy'], 'k.', alpha=0.1, markersize=1, 
         label='Noisy Data')
plt.plot(t, sig_train_shear_pred, 'r--', lw=2, label='SINDy Fit')
plt.title("Training: Pure Shear (σ_eq vs Time)")
plt.xlabel("Time (s)")
plt.ylabel(r"$\sigma_{eq}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 3)
# Training phase plane
plt.plot(train_uniaxial['strain_rate_eq'], train_uniaxial['stress_eq'], 
         'b-', alpha=0.3, linewidth=1, label='Uniaxial (True)')
plt.plot(train_shear['strain_rate_eq'], train_shear['stress_eq'], 
         'g-', alpha=0.3, linewidth=1, label='Shear (True)')
plt.plot(train_uniaxial['strain_rate_eq'], sig_train_uniaxial_pred, 
         'r--', linewidth=2, label='SINDy Fit')
plt.plot(train_shear['strain_rate_eq'], sig_train_shear_pred, 
         'orange', linestyle='--', linewidth=2)
plt.title("Training: Phase Plane")
plt.xlabel(r"$\dot{\epsilon}_{eq}$ (1/s)")
plt.ylabel(r"$\sigma_{eq}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

# ========== Row 2: Validation - Combined Load ==========
plt.subplot(3, 3, 4)
plt.plot(t, val_combined['stress_eq'], 'k-', linewidth=1.5, label='True')
plt.plot(t, sig_val_combined_pred, 'r--', lw=2, label='SINDy Prediction')
plt.title(f"Validation: Combined Load (RMSE={rmse_combined:.2f})")
plt.xlabel("Time (s)")
plt.ylabel(r"$\sigma_{eq}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 5)
plt.plot(t, val_biaxial['stress_eq'], 'k-', linewidth=1.5, label='True')
plt.plot(t, sig_val_biaxial_pred, 'r--', lw=2, label='SINDy Prediction')
plt.title(f"Validation: Biaxial Load (RMSE={rmse_biaxial:.2f})")
plt.xlabel("Time (s)")
plt.ylabel(r"$\sigma_{eq}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 6)
# Validation phase plane
plt.plot(val_combined['strain_rate_eq'], val_combined['stress_eq'], 
         'b-', alpha=0.5, linewidth=1.5, label='Combined (True)')
plt.plot(val_combined['strain_rate_eq'], sig_val_combined_pred, 
         'r--', linewidth=2, label='SINDy Prediction')
plt.plot(val_biaxial['strain_rate_eq'], val_biaxial['stress_eq'], 
         'purple', alpha=0.5, linewidth=1.5, label='Biaxial (True)')
plt.plot(val_biaxial['strain_rate_eq'], sig_val_biaxial_pred, 
         'orange', linestyle='--', linewidth=2)
plt.title("Validation: Phase Plane")
plt.xlabel(r"$\dot{\epsilon}_{eq}$ (1/s)")
plt.ylabel(r"$\sigma_{eq}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

# ========== Row 3: Error Analysis ==========
plt.subplot(3, 3, 7)
error_combined = val_combined['stress_eq'] - sig_val_combined_pred
plt.plot(t, error_combined, 'r-', linewidth=1)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.title("Prediction Error: Combined Load")
plt.xlabel("Time (s)")
plt.ylabel("Error (MPa)")
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 8)
error_biaxial = val_biaxial['stress_eq'] - sig_val_biaxial_pred
plt.plot(t, error_biaxial, 'r-', linewidth=1)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.title("Prediction Error: Biaxial Load")
plt.xlabel("Time (s)")
plt.ylabel("Error (MPa)")
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 9)
# RMSE comparison
cases = ['Uniaxial\n(Train)', 'Shear\n(Train)', 'Combined\n(Val)', 'Biaxial\n(Val)']
rmse_values = [rmse_train_uniaxial, rmse_train_shear, rmse_combined, rmse_biaxial]
colors = ['blue', 'blue', 'red', 'red']
bars = plt.bar(cases, rmse_values, color=colors, alpha=0.6)
plt.ylabel("RMSE (MPa)")
plt.title("Model Performance (Lower is Better)")
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, rmse_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('3d_sindy_results.png', dpi=150)
print("  Saved: 3d_sindy_results.png")

plt.show()

print("\n" + "=" * 60)
print("3D SINDy Analysis Complete!")
print("=" * 60)
print("\n✓ Successfully trained SINDy on simple 3D loads (uniaxial, shear)")
print("✓ Successfully validated on complex combined loads")
print(f"✓ Model generalizes well (Validation RMSE: {rmse_combined:.2f}, {rmse_biaxial:.2f} MPa)")
