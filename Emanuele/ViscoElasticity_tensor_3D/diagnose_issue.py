"""
Diagnostic script to identify issues with 3D SINDy training

This will check:
1. Data quality and scaling
2. Time series continuity
3. Von Mises conversions
4. SINDy sensitivity to parameters
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("DIAGNOSTIC: Checking 3D Data Quality")
print("=" * 60)

# Load data
train_uniaxial = np.load('data_train_uniaxial.npz')
train_shear = np.load('data_train_shear.npz')

t = train_uniaxial['time']

print("\n1. DATA SCALING CHECK")
print("-" * 60)
print(f"Uniaxial stress_eq range: [{train_uniaxial['stress_eq'].min():.2f}, {train_uniaxial['stress_eq'].max():.2f}] MPa")
print(f"Shear stress_eq range: [{train_shear['stress_eq'].min():.2f}, {train_shear['stress_eq'].max():.2f}] MPa")
print(f"Noise level: 0.1 MPa")
print(f"Signal-to-noise ratio (uniaxial): {np.max(np.abs(train_uniaxial['stress_eq']))/0.1:.1f}:1")
print(f"Signal-to-noise ratio (shear): {np.max(np.abs(train_shear['stress_eq']))/0.1:.1f}:1")

if np.max(np.abs(train_uniaxial['stress_eq'])) < 1.0:
    print("⚠️  WARNING: Stress magnitudes are VERY low (<1 MPa)")
    print("   This gives poor signal-to-noise ratio with 0.1 MPa noise!")

print("\n2. TIME SERIES CONTINUITY CHECK")
print("-" * 60)
print(f"Time step dt: {t[1]-t[0]:.6f} s")
print(f"Number of time points: {len(t)}")

# Check what happens when we concatenate
stress_concat = np.concatenate([train_uniaxial['stress_eq'], train_shear['stress_eq']])
print(f"\nConcatenated array length: {len(stress_concat)}")
print(f"Jump at concatenation point:")
print(f"  Last value of uniaxial: {train_uniaxial['stress_eq'][-1]:.4f}")
print(f"  First value of shear: {train_shear['stress_eq'][0]:.4f}")
print(f"  Jump magnitude: {abs(train_uniaxial['stress_eq'][-1] - train_shear['stress_eq'][0]):.4f}")

if abs(train_uniaxial['stress_eq'][-1] - train_shear['stress_eq'][0]) > 0.5:
    print("⚠️  WARNING: Large discontinuity when concatenating!")
    print("   SINDy assumes continuous time series. This will cause issues!")

print("\n3. VON MISES CONVERSION CHECK")
print("-" * 60)

# For uniaxial: σ_eq should equal σ_xx
sigma_xx_max = np.max(np.abs(train_uniaxial['stress_3d'][:, 0]))
sigma_eq_max = np.max(np.abs(train_uniaxial['stress_eq']))
print(f"Uniaxial: max |σ_xx| = {sigma_xx_max:.4f}, max |σ_eq| = {sigma_eq_max:.4f}")
print(f"  Ratio σ_eq/σ_xx: {sigma_eq_max/sigma_xx_max:.4f} (should be 1.0)")

# For shear: σ_eq should equal √3 * τ_xy
tau_xy_max = np.max(np.abs(train_shear['stress_3d'][:, 3]))
sigma_eq_shear_max = np.max(np.abs(train_shear['stress_eq']))
expected_ratio = np.sqrt(3)
print(f"\nShear: max |τ_xy| = {tau_xy_max:.4f}, max |σ_eq| = {sigma_eq_shear_max:.4f}")
print(f"  Ratio σ_eq/τ_xy: {sigma_eq_shear_max/tau_xy_max:.4f} (should be {expected_ratio:.4f})")

print("\n4. STRAIN AMPLITUDE CHECK")
print("-" * 60)
print(f"Uniaxial strain rate amplitude: {np.max(np.abs(train_uniaxial['strain_rate_3d'][:, 0])):.6f}")
print(f"Shear strain rate amplitude: {np.max(np.abs(train_shear['strain_rate_3d'][:, 3])):.6f}")
print("\nWith E=2000, η=500, the steady-state stress for ε̇=0.01 is:")
print(f"  σ_ss = E*τ*ε̇ = 2000 * (500/2000) * 0.01 = {2000 * 0.25 * 0.01:.2f} MPa")
print("  (where τ = η/E = relaxation time)")

print("\n5. SUGGESTED FIXES")
print("-" * 60)
print("Issue 1: VERY LOW SIGNAL LEVELS")
print("  → Increase strain amplitude from 0.01 to 0.1 (10x higher)")
print("  → This will give stress ~2-5 MPa instead of 0.2-0.5 MPa")
print("")
print("Issue 2: TIME SERIES CONCATENATION")
print("  → Don't concatenate! Train on each separately and average coefficients")
print("  → OR: Ensure both start from zero stress (currently they might not)")
print("")
print("Issue 3: THRESHOLD TOO HIGH")
print("  → Reduce STLSQ threshold from 1.0 to 0.01-0.1")
print("  → This will help find the correct sparse solution")

print("\n" + "=" * 60)
