"""
Generate Synthetic 3D Viscoelastic Data using DEVIATORIC Components

This is the CORRECTED approach that uses deviatoric stress/strain components
instead of von Mises equivalents. Deviatoric components CAN BE NEGATIVE,
which is essential for the Maxwell model to work correctly.

For isotropic Maxwell model in 3D:
    dS_ij/dt = 2G * dε_ij^dev/dt - (2G/η) * S_ij
    
where S_ij is deviatoric stress, which looks like the 1D Maxwell equation!

Author: Generated for DataDrivenProject
Date: 2025-12-02
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from deviatoric_utils import deviatoric_stress, deviatoric_strain_rate


# ==========================================
# 3D Maxwell Model Parameters
# ==========================================
# Using G (shear modulus) instead of E for deviatoric formulation
# For simplicity, assume ν ≈ 0, so E ≈ 2G
G_true = 1000.0      # Shear modulus (G = E/2 if ν=0)
eta_true = 500.0     # Viscosity
dt = 0.001           # Time step
t = np.arange(0, 10, dt)  # Time array


def maxwell_3d_deviatoric_ode(dev_stress_vec, t, eps_dot_func, t_array):
    """
    3D Maxwell model for DEVIATORIC stress components.
    
    For each deviatoric component:
    dS_ij/dt = 2G * dε_ij^dev/dt - (2G/η) * S_ij
    
    This is exactly the Maxwell form, applied to each component independently!
    
    Parameters
    ----------
    dev_stress_vec : array (6,)
        Current deviatoric stress [S_xx, S_yy, S_zz, S_xy, S_yz, S_xz]
    t : float
        Current time
    eps_dot_func : callable
        Function that returns strain rate vector at time t
    t_array : array
        Time array
    
    Returns
    -------
    dS_dt : array (6,)
        Time derivative of deviatoric stress
    """
    # Get total strain rate at current time
    eps_dot_total = eps_dot_func(t, t_array)
    
    # Convert to deviatoric strain rate
    eps_dot_dev = deviatoric_strain_rate(eps_dot_total)
    
    # Maxwell equation for deviatoric components
    # dS/dt = 2G * dε^dev/dt - (2G/η) * S
    dS_dt = 2 * G_true * eps_dot_dev - (2 * G_true / eta_true) * dev_stress_vec
    
    return dS_dt


# ==========================================
# Strain Rate Functions (same as before)
# ==========================================

def uniaxial_strain_rate(t, t_array):
    """Uniaxial tension in x-direction."""
    eps_xx_dot = 0.1 * (np.sin(2 * np.pi * 0.5 * t) + 
                         0.5 * np.sin(2 * np.pi * 1.5 * t))
    return np.array([eps_xx_dot, 0, 0, 0, 0, 0])


def pure_shear_strain_rate(t, t_array):
    """Pure shear in xy-plane."""
    gamma_xy_dot = 0.1 * (np.sin(2 * np.pi * 0.5 * t) + 
                           0.5 * np.sin(2 * np.pi * 1.5 * t))
    return np.array([0, 0, 0, gamma_xy_dot, 0, 0])


def combined_uniaxial_shear_strain_rate(t, t_array):
    """Combined uniaxial + shear (validation)."""
    eps_xx_dot = 0.15 * np.sin(2 * np.pi * 0.2 * t)
    gamma_xy_dot = 0.1 * np.sin(2 * np.pi * 0.3 * t)
    return np.array([eps_xx_dot, 0, 0, gamma_xy_dot, 0, 0])


def biaxial_strain_rate(t, t_array):
    """Biaxial tension (validation)."""
    eps_xx_dot = 0.15 * np.sin(2 * np.pi * 0.25 * t)
    eps_yy_dot = 0.1 * np.sin(2 * np.pi * 0.35 * t)
    return np.array([eps_xx_dot, eps_yy_dot, 0, 0, 0, 0])


# ==========================================
# Data Generation Function
# ==========================================

def generate_3d_deviatoric_data(strain_rate_func, t_array, label=""):
    """
    Generate 3D viscoelastic data using deviatoric formulation.
    
    Returns deviatoric stress and strain rate components that can be
    BOTH positive AND negative.
    """
    print(f"\nGenerating: {label}")
    
    # Compute strain rates
    strain_rate_3d = np.array([strain_rate_func(ti, t_array) for ti in t_array])
    
    # Convert to dev iatoric strain rates
    strain_rate_dev = deviatoric_strain_rate(strain_rate_3d)
    
    # Integrate deviatoric Maxwell model
    dev_stress_init = np.zeros(6)
    dev_stress = odeint(maxwell_3d_deviatoric_ode, dev_stress_init, t_array,
                         args=(strain_rate_func, t_array))
    
    # Show range including negative values
    print(f"  Deviatoric stress S_xx range: [{dev_stress[:, 0].min():.2f}, {dev_stress[:, 0].max():.2f}] MPa")
    print(f"  Deviatoric strain rate ε̇_xx^dev range: [{strain_rate_dev[:, 0].min():.6f}, {strain_rate_dev[:, 0].max():.6f}]")
    print(f"  ✓ Can be negative: {dev_stress[:, 0].min() < 0}")
    
    return {
        'time': t_array,
        'dev_stress': dev_stress,
        'strain_rate_dev': strain_rate_dev,
        'label': label
    }


# ==========================================
# Generate All Datasets
# ==========================================

print("=" * 60)
print("Generating 3D Deviatoric Viscoelastic Data")
print(f"G = {G_true} MPa (shear modulus), η = {eta_true} MPa·s")
print(f"Expected coefficients: c1 = -2G/η = {-2*G_true/eta_true:.2f}")
print(f"                       c2 = 2G = {2*G_true:.2f}")
print("=" * 60)

# TRAINING DATA
print("\n--- TRAINING DATA ---")
train_uniaxial = generate_3d_deviatoric_data(uniaxial_strain_rate, t, 
                                               "Uniaxial Tension (Training)")
train_shear = generate_3d_deviatoric_data(pure_shear_strain_rate, t,
                                            "Pure Shear (Training)")

# VALIDATION DATA
print("\n--- VALIDATION DATA ---")
val_combined = generate_3d_deviatoric_data(combined_uniaxial_shear_strain_rate, t,
                                             "Combined Load (Validation)")
val_biaxial = generate_3d_deviatoric_data(biaxial_strain_rate, t,
                                            "Biaxial Tension (Validation)")


# ==========================================
# Add Noise
# ==========================================
np.random.seed(42)
train_uniaxial['dev_stress_noisy'] = (train_uniaxial['dev_stress'] +
                                       np.random.normal(0, 0.1, train_uniaxial['dev_stress'].shape))
train_shear['dev_stress_noisy'] = (train_shear['dev_stress'] +
                                     np.random.normal(0, 0.1, train_shear['dev_stress'].shape))


# ==========================================
# Save Data
# ==========================================
print("\n--- SAVING DATA ---")

def save_dataset(data, filename):
    np.savez(filename,
             time=data['time'],
             dev_stress=data['dev_stress'],
             strain_rate_dev=data['strain_rate_dev'],
             dev_stress_noisy=data.get('dev_stress_noisy', data['dev_stress']),
             label=data['label'])
    print(f"  Saved: {filename}")

save_dataset(train_uniaxial, 'data_train_uniaxial_dev.npz')
save_dataset(train_shear, 'data_train_shear_dev.npz')
save_dataset(val_combined, 'data_val_combined_dev.npz')
save_dataset(val_biaxial, 'data_val_biaxial_dev.npz')


# ==========================================
# Visualization
# ==========================================
print("\n--- CREATING VISUALIZATIONS ---")

fig = plt.figure(figsize=(14, 10))

# Training: Uniaxial
plt.subplot(3, 3, 1)
plt.plot(t, train_uniaxial['dev_stress'][:, 0], 'b-', label=r'$S_{xx}$ (deviatoric)')
plt.title("Training: Uniaxial (Deviatoric Stress)")
plt.xlabel("Time (s)")
plt.ylabel("Stress (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(3, 3, 2)
plt.plot(t, train_uniaxial['strain_rate_dev'][:, 0], 'g-', label=r'$\dot{\epsilon}_{xx}^{dev}$')
plt.title("Strain Rate (Deviatoric)")
plt.xlabel("Time (s)")
plt.ylabel("Strain Rate (1/s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(3, 3, 3)
plt.plot(train_uniaxial['strain_rate_dev'][:, 0], train_uniaxial['dev_stress'][:, 0], 
         'b-', alpha=0.5)
plt.title("Phase Space (Uniaxial)")
plt.xlabel(r"$\dot{\epsilon}_{xx}^{dev}$ (1/s)")
plt.ylabel(r"$S_{xx}$ (MPa)")
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)

# Training: Shear
plt.subplot(3, 3, 4)
plt.plot(t, train_shear['dev_stress'][:, 3], 'b-', label=r'$S_{xy}$ (deviatoric shear)')
plt.title("Training: Pure Shear (Deviatoric)")
plt.xlabel("Time (s)")
plt.ylabel("Stress (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(3, 3, 5)
plt.plot(t, train_shear['strain_rate_dev'][:, 3], 'g-', label=r'$\dot{\gamma}_{xy}$')
plt.title("Shear Strain Rate")
plt.xlabel("Time (s)")
plt.ylabel("Strain Rate (1/s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(3, 3, 6)
plt.plot(train_shear['strain_rate_dev'][:, 3], train_shear['dev_stress'][:, 3],
         'b-', alpha=0.5)
plt.title("Phase Space (Shear)")
plt.xlabel(r"$\dot{\gamma}_{xy}$ (1/s)")
plt.ylabel(r"$S_{xy}$ (MPa)")
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)

# Validation
plt.subplot(3, 3, 7)
plt.plot(t, val_combined['dev_stress'][:, 0], 'r-', label=r'$S_{xx}$')
plt.plot(t, val_combined['dev_stress'][:, 3], 'orange', label=r'$S_{xy}$')
plt.title("Validation: Combined Load")
plt.xlabel("Time (s)")
plt.ylabel("Deviatoric Stress (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(3, 3, 8)
plt.plot(t, val_biaxial['dev_stress'][:, 0], 'purple', label=r'$S_{xx}$')
plt.plot(t, val_biaxial['dev_stress'][:, 1], 'brown', label=r'$S_{yy}$')
plt.title("Validation: Biaxial Tension")
plt.xlabel("Time (s)")
plt.ylabel("Deviatoric Stress (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.subplot(3, 3, 9)
# Show that deviatoric components can be negative!
components = ['S_xx', 'S_yy', 'S_zz', 'S_xy', 'S_yz', 'S_xz']
min_vals = [train_uniaxial['dev_stress'][:, i].min() for i in range(6)]
max_vals = [train_uniaxial['dev_stress'][:, i].max() for i in range(6)]
x_pos = np.arange(len(components))
plt.bar(x_pos, max_vals, 0.4, label='Max', alpha=0.7)
plt.bar(x_pos + 0.4, min_vals, 0.4, label='Min', alpha=0.7)
plt.xticks(x_pos + 0.2, components)
plt.ylabel("Stress (MPa)")
plt.title("Deviatoric Stress Range (Train Uniaxial)")
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(0, color='k', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig('3d_deviatoric_data.png', dpi=150)
print("  Saved: 3d_deviatoric_data.png")

plt.show()

print("\n" + "=" * 60)
print("Deviatoric data generation complete!")
print("=" * 60)
