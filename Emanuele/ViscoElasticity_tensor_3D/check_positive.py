"""
Check if von Mises data is always positive
"""
import numpy as np
import matplotlib.pyplot as plt

# Load data
train_uniaxial = np.load('data_train_uniaxial.npz')
train_shear = np.load('data_train_shear.npz')

t = train_uniaxial['time']

print("=" * 60)
print("CHECKING: Are von Mises quantities always positive?")
print("=" * 60)

# Original stress (can be negative)
sigma_xx = train_uniaxial['stress_3d'][:, 0]
tau_xy = train_shear['stress_3d'][:, 3]

# Von Mises (should be non-negative)
sigma_eq_uniaxial = train_uniaxial['stress_eq']
sigma_eq_shear = train_shear['stress_eq']

# Strain rates
eps_dot_xx = train_uniaxial['strain_rate_3d'][:, 0]
eps_dot_eq_uniaxial = train_uniaxial['strain_rate_eq']

print("\n1. ORIGINAL STRESS (1D component)")
print(f"  σ_xx range: [{sigma_xx.min():.2f}, {sigma_xx.max():.2f}] MPa")
print(f"  Can be negative? {sigma_xx.min() < 0}")

print("\n2. VON MISES STRESS (equivalent)")
print(f"  σ_eq range: [{sigma_eq_uniaxial.min():.2f}, {sigma_eq_uniaxial.max():.2f}] MPa")
print(f"  Can be negative? {sigma_eq_uniaxial.min() < 0}")
print(f"  ❌ Always non-negative: {sigma_eq_uniaxial.min() >= 0}")

print("\n3. STRAIN RATE (original vs equivalent)")
print(f"  ε̇_xx range: [{eps_dot_xx.min():.6f}, {eps_dot_xx.max():.6f}]")
print(f"  Can be negative? {eps_dot_xx.min() < 0}")
print(f"  ε̇_eq range: [{eps_dot_eq_uniaxial.min():.6f}, {eps_dot_eq_uniaxial.max():.6f}]")
print(f"  Can be negative? {eps_dot_eq_uniaxial.min() < 0}")
print(f"  ❌ Always non-negative: {eps_dot_eq_uniaxial.min() >= 0}")

print("\n" + "=" * 60)
print("ROOT CAUSE IDENTIFIED!")
print("=" * 60)
print("""
The Maxwell model is: dσ/dt = E*dε/dt - (E/η)*σ

For this to work, σ must be able to be BOTH positive AND negative!

When σ = -40 MPa and ε̇ = 0:
  dσ/dt = 0 - (E/η)*(-40) = +160  ✓ (stress increases toward zero)

But von Mises stress is ALWAYS ≥ 0, so we lose sign information!

When σ_eq = 40 MPa and ε̇_eq ≈ 0:
  dσ_eq/dt = 0 - (E/η)*(40) = -160  ❌ (wrong! should increase)

This is why SINDy can't discover the correct model - the physics
is fundamentally different when stress can't be negative!
""")

print("\n" + "=" * 60)
print("SOLUTION")
print("=" * 60)
print("""
For 3D viscoelasticity, we need a DIFFERENT approach:

Option 1: Use SIGNED von Mises (not standard)
  - Keep track of loading direction
  - Sign σ_eq based on whether material is in tension/compression
  
Option 2: Model DEVIATORIC stress components separately
  - Track all 6 stress components
  - Don't reduce to scalar
  
Option 3: Use INCREMENTAL formulation
  - Model Δσ_eq vs Δε_eq instead of absolute values
  
Option 4: Assume ALWAYS tensile loading
  - Only valid if material never sees compression
  - Modify Maxwell to: dσ_eq/dt = E*dε_eq/dt - (E/η)*(σ_eq - σ_0)
    where σ_0 is a reference stress

Would you like me to implement one of these solutions?
""")
