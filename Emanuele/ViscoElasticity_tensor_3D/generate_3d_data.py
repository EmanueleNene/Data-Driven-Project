"""
Generate Synthetic 3D Viscoelastic Data for SINDy Training

This script generates synthetic 3D stress-strain data from a 3D Maxwell model,
then converts to von Mises equivalents for use in SINDy training.

Training Data: Two separate simple loading cases
  1. Uniaxial tension in x-direction
  2. Pure shear in xy-plane

Validation Data: Combined loading scenarios to test generalization
  1. Uniaxial + shear combined
  2. Biaxial tension

Author: Generated for DataDrivenProject
Date: 2025-12-02
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from von_mises_utils import von_mises_stress, von_mises_strain_rate


# ==========================================
# 3D Maxwell Model Parameters
# ==========================================
E_true = 2000.0      # Elastic modulus (same as 1D case)
eta_true = 500.0     # Viscosity (same as 1D case)
dt = 0.001           # Time step
t = np.arange(0, 10, dt)  # Time array


def maxwell_3d_ode(stress_vec, t, eps_dot_func, t_array):
    """
    3D Maxwell model ODEs for isotropic material.
    
    For each stress component:
    dσ_ij/dt = E * dε_ij/dt - (E/η) * σ_ij
    
    Parameters
    ----------
    stress_vec : array (6,)
        Current stress state [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
    t : float
        Current time
    eps_dot_func : callable
        Function that returns strain rate vector at time t
    t_array : array
        Time array for interpolation
    
    Returns
    -------
    dsigma_dt : array (6,)
        Time derivative of stress components
    """
    # Get strain rate at current time
    eps_dot = eps_dot_func(t, t_array)
    
    # Maxwell equation for each component
    # dσ/dt = E * dε/dt - (E/η) * σ
    dsigma_dt = E_true * eps_dot - (E_true / eta_true) * stress_vec
    
    return dsigma_dt


# ==========================================
# Strain Rate Functions (Rich Signals)
# ==========================================

def uniaxial_strain_rate(t, t_array):
    """Uniaxial tension in x-direction with rich frequency content."""
    # Increased from 0.01 to 0.1 for better signal-to-noise ratio
    eps_xx_dot = 0.1 * (np.sin(2 * np.pi * 0.5 * t) + 
                         0.5 * np.sin(2 * np.pi * 1.5 * t))
    return np.array([eps_xx_dot, 0, 0, 0, 0, 0])


def pure_shear_strain_rate(t, t_array):
    """Pure shear in xy-plane with rich frequency content."""
    # Increased from 0.01 to 0.1 for better signal-to-noise ratio
    gamma_xy_dot = 0.1 * (np.sin(2 * np.pi * 0.5 * t) + 
                           0.5 * np.sin(2 * np.pi * 1.5 * t))
    return np.array([0, 0, 0, gamma_xy_dot, 0, 0])


def combined_uniaxial_shear_strain_rate(t, t_array):
    """Combined uniaxial + shear (validation case)."""
    # Increased amplitudes for better signal levels
    eps_xx_dot = 0.15 * np.sin(2 * np.pi * 0.2 * t)
    gamma_xy_dot = 0.1 * np.sin(2 * np.pi * 0.3 * t)
    return np.array([eps_xx_dot, 0, 0, gamma_xy_dot, 0, 0])


def biaxial_strain_rate(t, t_array):
    """Biaxial tension in x and y (validation case)."""
    # Increased amplitudes for better signal levels
    eps_xx_dot = 0.15 * np.sin(2 * np.pi * 0.25 * t)
    eps_yy_dot = 0.1 * np.sin(2 * np.pi * 0.35 * t)
    return np.array([eps_xx_dot, eps_yy_dot, 0, 0, 0, 0])


# ==========================================
# Data Generation Function
# ==========================================

def generate_3d_data(strain_rate_func, t_array, label=""):
    """
    Generate 3D stress-strain data and convert to von Mises equivalents.
    
    Parameters
    ----------
    strain_rate_func : callable
        Function that returns 6-component strain rate vector
    t_array : array
        Time array
    label : str
        Label for this dataset
    
    Returns
    -------
    dict with keys:
        - 'time': time array
        - 'stress_3d': 3D stress components (n_times, 6)
        - 'strain_rate_3d': 3D strain rate components (n_times, 6)
        - 'stress_eq': von Mises equivalent stress
        - 'strain_rate_eq': von Mises equivalent strain rate
        - 'label': dataset label
    """
    print(f"\nGenerating: {label}")
    
    # Compute strain rates at all times
    strain_rate_3d = np.array([strain_rate_func(ti, t_array) for ti in t_array])
    
    # Integrate to get stress history using 3D Maxwell model
    stress_init = np.zeros(6)
    stress_3d = odeint(maxwell_3d_ode, stress_init, t_array, 
                       args=(strain_rate_func, t_array))
    
    # Convert to von Mises equivalents
    stress_eq = von_mises_stress(stress_3d)
    strain_rate_eq = von_mises_strain_rate(strain_rate_3d)
    
    print(f"  Max σ_eq: {np.max(np.abs(stress_eq)):.2f}")
    print(f"  Max ε̇_eq: {np.max(np.abs(strain_rate_eq)):.6f}")
    
    return {
        'time': t_array,
        'stress_3d': stress_3d,
        'strain_rate_3d': strain_rate_3d,
        'stress_eq': stress_eq,
        'strain_rate_eq': strain_rate_eq,
        'label': label
    }


# ==========================================
# Generate All Datasets
# ==========================================

print("=" * 60)
print("Generating 3D Synthetic Viscoelastic Data")
print(f"E = {E_true} MPa, η = {eta_true} MPa·s")
print("=" * 60)

# TRAINING DATA (separate simple cases)
print("\n--- TRAINING DATA ---")
train_uniaxial = generate_3d_data(uniaxial_strain_rate, t, "Uniaxial Tension (Training)")
train_shear = generate_3d_data(pure_shear_strain_rate, t, "Pure Shear (Training)")

# VALIDATION DATA (combined loads)
print("\n--- VALIDATION DATA ---")
val_combined = generate_3d_data(combined_uniaxial_shear_strain_rate, t, 
                                 "Uniaxial + Shear (Validation)")
val_biaxial = generate_3d_data(biaxial_strain_rate, t, 
                               "Biaxial Tension (Validation)")


# ==========================================
# Add Noise to Training Data
# ==========================================
np.random.seed(42)
train_uniaxial['stress_eq_noisy'] = (train_uniaxial['stress_eq'] + 
                                      np.random.normal(0, 0.1, len(t)))
train_shear['stress_eq_noisy'] = (train_shear['stress_eq'] + 
                                   np.random.normal(0, 0.1, len(t)))


# ==========================================
# Save Data to Files
# ==========================================
print("\n--- SAVING DATA ---")

def save_dataset(data, filename):
    """Save dataset to NPZ file."""
    np.savez(filename,
             time=data['time'],
             stress_3d=data['stress_3d'],
             strain_rate_3d=data['strain_rate_3d'],
             stress_eq=data['stress_eq'],
             strain_rate_eq=data['strain_rate_eq'],
             stress_eq_noisy=data.get('stress_eq_noisy', data['stress_eq']),
             label=data['label'])
    print(f"  Saved: {filename}")

save_dataset(train_uniaxial, 'data_train_uniaxial.npz')
save_dataset(train_shear, 'data_train_shear.npz')
save_dataset(val_combined, 'data_val_combined.npz')
save_dataset(val_biaxial, 'data_val_biaxial.npz')


# ==========================================
# Visualization
# ==========================================
print("\n--- CREATING VISUALIZATIONS ---")

fig = plt.figure(figsize=(14, 10))

# Row 1: Training - Uniaxial
plt.subplot(3, 3, 1)
plt.plot(t, train_uniaxial['stress_3d'][:, 0], 'b-', label=r'$\sigma_{xx}$')
plt.plot(t, train_uniaxial['stress_eq'], 'r--', lw=2, label=r'$\sigma_{eq}$ (von Mises)')
plt.title("Training: Uniaxial Tension")
plt.xlabel("Time (s)")
plt.ylabel("Stress (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 2)
plt.plot(t, train_uniaxial['strain_rate_3d'][:, 0], 'g-', label=r'$\dot{\epsilon}_{xx}$')
plt.plot(t, train_uniaxial['strain_rate_eq'], 'orange', linestyle='--', lw=2, 
         label=r'$\dot{\epsilon}_{eq}$')
plt.title("Strain Rate (Uniaxial)")
plt.xlabel("Time (s)")
plt.ylabel("Strain Rate (1/s)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 3)
plt.plot(train_uniaxial['strain_rate_eq'], train_uniaxial['stress_eq'], 'b-', alpha=0.5)
plt.title("ε̇-σ Space (Uniaxial)")
plt.xlabel(r"$\dot{\epsilon}_{eq}$ (1/s)")
plt.ylabel(r"$\sigma_{eq}$ (MPa)")
plt.grid(True, alpha=0.3)

# Row 2: Training - Shear
plt.subplot(3, 3, 4)
plt.plot(t, train_shear['stress_3d'][:, 3], 'b-', label=r'$\tau_{xy}$')
plt.plot(t, train_shear['stress_eq'], 'r--', lw=2, label=r'$\sigma_{eq}$ (von Mises)')
plt.title("Training: Pure Shear")
plt.xlabel("Time (s)")
plt.ylabel("Stress (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 5)
plt.plot(t, train_shear['strain_rate_3d'][:, 3], 'g-', label=r'$\dot{\gamma}_{xy}$')
plt.plot(t, train_shear['strain_rate_eq'], 'orange', linestyle='--', lw=2, 
         label=r'$\dot{\epsilon}_{eq}$')
plt.title("Strain Rate (Shear)")
plt.xlabel("Time (s)")
plt.ylabel("Strain Rate (1/s)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 6)
plt.plot(train_shear['strain_rate_eq'], train_shear['stress_eq'], 'b-', alpha=0.5)
plt.title("ε̇-σ Space (Shear)")
plt.xlabel(r"$\dot{\epsilon}_{eq}$ (1/s)")
plt.ylabel(r"$\sigma_{eq}$ (MPa)")
plt.grid(True, alpha=0.3)

# Row 3: Validation - Combined loads
plt.subplot(3, 3, 7)
plt.plot(t, val_combined['stress_eq'], 'r-', lw=2, label='Combined (σ_xx + τ_xy)')
plt.plot(t, val_biaxial['stress_eq'], 'purple', lw=2, label='Biaxial (σ_xx + σ_yy)')
plt.title("Validation: Combined Loads")
plt.xlabel("Time (s)")
plt.ylabel(r"$\sigma_{eq}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 8)
plt.plot(t, val_combined['strain_rate_eq'], 'orange', lw=2, label='Combined')
plt.plot(t, val_biaxial['strain_rate_eq'], 'brown', lw=2, label='Biaxial')
plt.title("Validation: Strain Rates")
plt.xlabel("Time (s)")
plt.ylabel(r"$\dot{\epsilon}_{eq}$ (1/s)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 9)
plt.plot(val_combined['strain_rate_eq'], val_combined['stress_eq'], 
         'orange', alpha=0.5, label='Combined')
plt.plot(val_biaxial['strain_rate_eq'], val_biaxial['stress_eq'], 
         'purple', alpha=0.5, label='Biaxial')
plt.title("Validation: ε̇-σ Space")
plt.xlabel(r"$\dot{\epsilon}_{eq}$ (1/s)")
plt.ylabel(r"$\sigma_{eq}$ (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('3d_data_generation.png', dpi=150)
print("  Saved: 3d_data_generation.png")

plt.show()

print("\n" + "=" * 60)
print("Data generation complete!")
print("=" * 60)
