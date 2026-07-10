"""
Generate Synthetic 3D Nonlinear Viscoelastic Data using DEVIATORIC Components

This script generates data using a coupled nonlinear Maxwell model in deviatoric space:
    dS_ij/dt = 2G * dε_ij^dev/dt - (2G/η(σ_eq)) * S_ij

where the viscosity is stress-dependent (J2-coupled):
    1/η(σ_eq) = (1/η_0) * (1 + α * σ_eq^2)
    σ_eq^2 = 1.5 * S_ij * S_ij = 1.5 * (S_xx^2 + S_yy^2 + S_zz^2 + 2*S_xy^2 + 2*S_yz^2 + 2*S_xz^2)

This creates a cubic coupling between all stress components, providing a true
multiaxial generalization test for SINDy.

Author: Generated for DataDrivenProject (Option 2)
Date: 2026-06-26
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import sys

# Ensure we can import deviatoric_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from deviatoric_utils import deviatoric_strain_rate

# ==========================================
# Material Parameters
# ==========================================
G_true = 1000.0        # Shear modulus (MPa)
eta_0_true = 500.0     # Reference viscosity (MPa*s)
alpha_true = 0.00015   # Nonlinear coupling parameter (1/MPa^2)
dt = 0.001             # Time step
t_span = (0.0, 10.0)
t_eval = np.arange(t_span[0], t_span[1], dt)

# ==========================================
# Nonlinear Coupled ODE
# ==========================================
def maxwell_3d_nonlinear_ode(t, S, eps_dot_func, t_array):
    """
    S: current deviatoric stress [S_xx, S_yy, S_zz, S_xy, S_yz, S_xz]
    """
    # Interpolate input strain rate at time t
    eps_dot_total = eps_dot_func(t, t_array)
    eps_dot_dev = deviatoric_strain_rate(eps_dot_total)
    
    # Compute equivalent stress squared: σ_eq^2 = 3/2 * S : S
    S_xx, S_yy, S_zz, S_xy, S_yz, S_xz = S
    sig_eq_sq = 1.5 * (S_xx**2 + S_yy**2 + S_zz**2 + 2*S_xy**2 + 2*S_yz**2 + 2*S_xz**2)
    
    # Viscosity model: 1/eta(S) = (1/eta_0) * (1 + alpha * sig_eq_sq)
    inv_eta = (1.0 / eta_0_true) * (1.0 + alpha_true * sig_eq_sq)
    
    # Coupled evolution: dS_ij/dt = 2G * dε_ij^dev/dt - (2G/η) * S_ij
    dS_dt = 2.0 * G_true * eps_dot_dev - (2.0 * G_true * inv_eta) * S

    return dS_dt

# ==========================================
# Strain Rate Loadings
# ==========================================
def uniaxial_strain_rate(t, t_array):
    """Biaxial, multi-tone normal loading (Training, S_xx model).

    A pure single-axis path forces S_yy = -0.5*S_xx exactly at every instant
    (isotropic deviatoric projection of a single-component strain input), which makes
    sigma_eq^2 an exact cubic function of S_xx alone -- i.e. the "Seq" coupling feature
    and the S^3 library term become perfectly collinear, regardless of how the signal
    is shaped. Exciting an independent second normal direction (e_yy, different
    frequencies/amplitude than e_xx) breaks that exact kinematic identity, which is a
    prerequisite for fitting a degree>1 library (Section on richer library, results.tex).
    """
    eps_xx_dot = 0.20 * (np.sin(2 * np.pi * 0.5 * t) + 0.7 * np.sin(2 * np.pi * 2.3 * t)
                          + 0.5 * np.sin(2 * np.pi * 5.1 * t) + 0.4 * np.sin(2 * np.pi * 8.7 * t)
                          + 0.3 * np.sin(2 * np.pi * 13.0 * t))
    eps_yy_dot = 0.15 * (np.sin(2 * np.pi * 0.9 * t) + 0.6 * np.sin(2 * np.pi * 3.3 * t)
                          + 0.4 * np.sin(2 * np.pi * 7.2 * t) + 0.3 * np.sin(2 * np.pi * 11.5 * t))
    return np.array([eps_xx_dot, eps_yy_dot, 0, 0, 0, 0])

def pure_shear_strain_rate(t, t_array):
    """Biaxial, multi-tone shear loading (Training, S_xy model).

    Same reasoning as above applies to shear: a single shear channel forces the other
    two shear stresses to stay identically zero, making sigma_eq^2 an exact cubic
    function of S_xy alone. Adding an independent second shear channel (g_xz, different
    frequencies/amplitude than g_xy) breaks that identity.
    """
    gamma_xy_dot = 0.20 * (np.sin(2 * np.pi * 0.6 * t) + 0.7 * np.sin(2 * np.pi * 2.5 * t)
                            + 0.5 * np.sin(2 * np.pi * 5.4 * t) + 0.4 * np.sin(2 * np.pi * 9.1 * t)
                            + 0.3 * np.sin(2 * np.pi * 12.6 * t))
    gamma_xz_dot = 0.15 * (np.sin(2 * np.pi * 1.1 * t) + 0.6 * np.sin(2 * np.pi * 3.6 * t)
                            + 0.4 * np.sin(2 * np.pi * 7.9 * t) + 0.3 * np.sin(2 * np.pi * 11.2 * t))
    return np.array([0, 0, 0, gamma_xy_dot, 0, gamma_xz_dot])

def combined_strain_rate(t, t_array):
    """Combined multiaxial loading: uniaxial x + shear xy (Validation)."""
    eps_xx_dot = 0.18 * np.sin(2 * np.pi * 0.2 * t)
    gamma_xy_dot = 0.12 * np.sin(2 * np.pi * 0.3 * t)
    return np.array([eps_xx_dot, 0, 0, gamma_xy_dot, 0, 0])

def biaxial_strain_rate(t, t_array):
    """Biaxial tension: uniaxial x + uniaxial y (Validation)."""
    eps_xx_dot = 0.18 * np.sin(2 * np.pi * 0.25 * t)
    eps_yy_dot = 0.12 * np.sin(2 * np.pi * 0.35 * t)
    return np.array([eps_xx_dot, eps_yy_dot, 0, 0, 0, 0])

# ==========================================
# Generation function
# ==========================================
def generate_dataset(strain_rate_func, label):
    print(f"Generating: {label} ...")
    
    # Strain rate history
    strain_rate_3d = np.array([strain_rate_func(ti, t_eval) for ti in t_eval])
    strain_rate_dev = deviatoric_strain_rate(strain_rate_3d)
    
    # Solve coupled stiff ODE using Radau
    sol = solve_ivp(
        fun=maxwell_3d_nonlinear_ode,
        t_span=t_span,
        y0=np.zeros(6),
        t_eval=t_eval,
        method='Radau',
        args=(strain_rate_func, t_eval),
        rtol=1e-8,
        atol=1e-10
    )
    
    dev_stress = sol.y.T
    
    print(f"  S_xx range: [{dev_stress[:,0].min():.2f}, {dev_stress[:,0].max():.2f}] MPa")
    print(f"  S_xy range: [{dev_stress[:,3].min():.2f}, {dev_stress[:,3].max():.2f}] MPa")
    
    return {
        'time': t_eval,
        'dev_stress': dev_stress,
        'strain_rate_dev': strain_rate_dev,
        'label': label
    }

# ==========================================
# Run Generator
# ==========================================
print("=" * 60)
print("Generating Coupled Nonlinear Viscoelastic Datasets (Option 2)")
print(f"G = {G_true} MPa, eta_0 = {eta_0_true} MPa*s, alpha = {alpha_true} 1/MPa^2")
print("=" * 60)

train_uniaxial = generate_dataset(uniaxial_strain_rate, "Biaxial Normal, multi-tone (Training)")
train_shear = generate_dataset(pure_shear_strain_rate, "Biaxial Shear, multi-tone (Training)")
val_combined = generate_dataset(combined_strain_rate, "Combined Load (Validation)")
val_biaxial = generate_dataset(biaxial_strain_rate, "Biaxial Tension (Validation)")

# Add noise to training stress for robustness check
np.random.seed(42)
train_uniaxial['dev_stress_noisy'] = train_uniaxial['dev_stress'] + np.random.normal(0, 0.15, train_uniaxial['dev_stress'].shape)
train_shear['dev_stress_noisy'] = train_shear['dev_stress'] + np.random.normal(0, 0.15, train_shear['dev_stress'].shape)

# Save datasets
dir_path = os.path.dirname(os.path.abspath(__file__))
def save_dataset(data, filename):
    filepath = os.path.join(dir_path, filename)
    np.savez(filepath,
             time=data['time'],
             dev_stress=data['dev_stress'],
             strain_rate_dev=data['strain_rate_dev'],
             dev_stress_noisy=data.get('dev_stress_noisy', data['dev_stress']),
             label=data['label'])
    print(f"Saved: {filename}")

print("\nSaving data...")
save_dataset(train_uniaxial, 'data_train_uniaxial_nonlinear_dev.npz')
save_dataset(train_shear, 'data_train_shear_nonlinear_dev.npz')
save_dataset(val_combined, 'data_val_combined_nonlinear_dev.npz')
save_dataset(val_biaxial, 'data_val_biaxial_nonlinear_dev.npz')

# ==========================================
# Visualizing Linear vs Nonlinear difference
# ==========================================
print("\nCreating linear vs nonlinear comparison plot...")

# Solve linear case for comparison (alpha = 0)
def maxwell_3d_linear_ode(t, S, eps_dot_func, t_array):
    eps_dot_total = eps_dot_func(t, t_array)
    eps_dot_dev = deviatoric_strain_rate(eps_dot_total)
    dS_dt = 2.0 * G_true * eps_dot_dev - (2.0 * G_true / eta_0_true) * S
    return dS_dt

sol_linear = solve_ivp(
    fun=maxwell_3d_linear_ode,
    t_span=t_span,
    y0=np.zeros(6),
    t_eval=t_eval,
    method='Radau',
    args=(combined_strain_rate, t_eval)
)
dev_stress_linear = sol_linear.y.T

fig = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_eval, dev_stress_linear[:, 0], 'k--', label='Linear Maxwell')
plt.plot(t_eval, val_combined['dev_stress'][:, 0], 'r-', label='Nonlinear Maxwell (Coupled)')
plt.title("Validation Combined Load: S_xx Response")
plt.xlabel("Time (s)")
plt.ylabel("Stress (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(t_eval, dev_stress_linear[:, 3], 'k--', label='Linear Maxwell')
plt.plot(t_eval, val_combined['dev_stress'][:, 3], 'r-', label='Nonlinear Maxwell (Coupled)')
plt.title("Validation Combined Load: S_xy Response")
plt.xlabel("Time (s)")
plt.ylabel("Stress (MPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(dir_path, 'nonlinear_deviatoric_comparison.png')
plt.savefig(plot_path, dpi=150)
print(f"Comparison plot saved: nonlinear_deviatoric_comparison.png")
plt.close()
print("Data generation complete!")
