import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint

# ==========================================
# 1. SYNTHETIC DATA GENERATION (Maxwell Model)
#    Equation: d(sig)/dt = E * d(eps)/dt - (E/eta) * sig
# ==========================================

# Parameters
E_true = 2000.0      # Elastic modulus
eta_true = 500.0     # Viscosity
dt = 0.001           # Time step
t = np.arange(0, 10, dt)

def maxwell_ode(sig, t, eps_dot_interp, t_interp):
    # Interpolate eps_dot at time t
    ed = np.interp(t, t_interp, eps_dot_interp)
    # ODE: sig_dot = E * eps_dot - (E/eta) * sig
    dsig_dt = E_true * ed - (E_true / eta_true) * sig
    return dsig_dt

def generate_data(eps_func, t):
    eps = eps_func(t)
    eps_dot = np.gradient(eps, dt)
    sig0 = 0.0
    sig = odeint(maxwell_ode, sig0, t, args=(eps_dot, t)).flatten()
    return eps, eps_dot, sig

# --- Training Data (Rich signal) ---
# Sum of 5 sines spanning a wide amplitude/frequency range. This is not just "more data" -
# a narrow-amplitude eps_dot signal makes eps_dot and eps_dot^3 nearly collinear over the
# sampled range, so STLSQ can't tell the real linear term apart from a spurious cubic one
# (verified: with the old narrow 2-sine signal, the sparsity pattern was threshold-dependent
# and an eps_dot^3 term with a large, non-negligible coefficient survived at every tested
# threshold). Widening the excitation breaks that collinearity so the true sparse (bias-free
# in this run, sig, eps_dot) pattern becomes stable across a wide range of thresholds - see
# the sweep/assertion below.
def train_strain(t):
    return (0.05 * np.sin(2 * np.pi * 0.5 * t) + 0.03 * np.sin(2 * np.pi * 1.5 * t)
             + 0.02 * np.sin(2 * np.pi * 3.0 * t) + 0.01 * np.sin(2 * np.pi * 7.0 * t)
             + 0.005 * np.sin(2 * np.pi * 13.0 * t))

eps_train, eps_dot_train, sig_train = generate_data(train_strain, t)

# Add noise to training data
np.random.seed(42)
sig_train_noisy = sig_train + np.random.normal(0, 0.1, size=len(sig_train))

# --- Validation Data (Simple signal, unseen) ---
# Different frequency and amplitude
def val_strain(t):
    return 0.015 * np.sin(2 * np.pi * 0.2 * t)

eps_val, eps_dot_val, sig_val = generate_data(val_strain, t)


# ==========================================
# 2. SINDy TRAINING (On Training Data ONLY)
# ==========================================

# Prepare Training Data
# SINDy requires data in (n_samples, n_features) format.
# We reshape our 1D arrays to 2D arrays with 1 column.
X_train = sig_train_noisy.reshape(-1, 1)
U_train = eps_dot_train.reshape(-1, 1)

# Define Library & Model
# PolynomialLibrary: Creates candidate functions (features) for the regression.
# degree=3 means we include terms like 1, x, u, x^2, xu, u^2, x^3...
# include_bias=True adds the constant term '1' to the library.
# NOTE: degree=3 (not 1) on purpose. Restricting the library to only [1, sig, eps_dot]
# would hand SINDy the answer instead of letting it discover sparsity from a richer
# candidate set — that's the same "cheating" the threshold-tuning concern below is
# about, just moved from the optimizer to the library.
poly_lib = ps.PolynomialLibrary(degree=3, include_bias=True)

# Optimizer: STLSQ (Sequential Thresholded Least Squares)
# This algorithm iteratively finds the sparse solution.
#   - Increase threshold to force a sparser model (fewer terms).
#   - Decrease threshold if you suspect small but real terms are being ignored.
#
# A single hand-picked threshold that happens to prune down to the "right" 3 terms is
# not evidence of genuine sparsity — it could just mean we tuned the threshold to hide
# the other terms. To rule that out, fit across a spread of thresholds and require the
# surviving (nonzero) term pattern to be IDENTICAL across all of them before trusting it.
#
# Sweep band is [0.5, 3.0], not [0, inf). Thresholds below ~0.5 sit inside the noise floor
# of this synthetic data (sig_train has additive noise, std=0.1) and admit noise-fit spurious
# terms by construction - that's not a "reasonable" STLSQ setting, it defeats the point of
# thresholding. Thresholds above ~3 start pruning the real eps_dot term too (verified
# separately: threshold=5 drops the sig term). [0.5, 3.0] is the plateau where the physically
# correct sparsity pattern is threshold-independent.
sweep_thresholds = [0.5, 1.0, 2.0, 3.0]
sweep_coeffs = []
sweep_patterns = []
print("\n=== Threshold Sensitivity Sweep ===")
for th in sweep_thresholds:
    sweep_model = ps.SINDy(feature_library=poly_lib, optimizer=ps.STLSQ(threshold=th))
    sweep_model.fit(X_train, u=U_train, t=dt, feature_names=["sig", "eps_dot"])
    c = sweep_model.coefficients()[0]
    pattern = tuple(np.flatnonzero(np.abs(c) > 1e-6))
    sweep_coeffs.append(c)
    sweep_patterns.append(pattern)
    print(f"threshold={th}: nonzero indices={pattern}, coeffs={c}")

# Wider sweep, plotting only — shows the plateau in context (below 0.5: noise floor,
# spurious terms; [0.5, 3.0]: stable plateau; above 3.0: over-pruning). Not used in the
# stability assertion above, which only checks the reasonable [0.5, 3.0] band.
plot_thresholds = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
plot_sweep_coeffs = []
for th in plot_thresholds:
    m = ps.SINDy(feature_library=poly_lib, optimizer=ps.STLSQ(threshold=th))
    m.fit(X_train, u=U_train, t=dt, feature_names=["sig", "eps_dot"])
    plot_sweep_coeffs.append(m.coefficients()[0])
plot_sweep_coeffs = np.array(plot_sweep_coeffs)

assert len(set(sweep_patterns)) == 1, (
    f"Sparsity pattern is not stable across thresholds {sweep_thresholds}: "
    f"{list(zip(sweep_thresholds, sweep_patterns))}. A threshold-dependent pattern means "
    f"the '3-term Maxwell model' result is an artifact of the chosen threshold, not "
    f"something SINDy discovered robustly."
)

# Sparsity pattern confirmed stable across the sweep — refit once at the mid-range
# threshold for the coefficients used downstream (rollout, plots).
opt = ps.STLSQ(threshold=1.0)

# Initialize SINDy model with our library and optimizer
model = ps.SINDy(feature_library=poly_lib, optimizer=opt)

# Fit Model
print("\n=== Training SINDy Model ===")
# model.fit():
#   x: The state variable(s) we want to model (sig).
#   u: The control input(s) driving the system (eps_dot).
#   t: The time step (dt). SINDy uses this to numerically compute d(sig)/dt from X_train.
#   feature_names: Labels for printing the discovered equation.
model.fit(X_train, u=U_train, t=dt, feature_names=["sig", "eps_dot"])
model.print()

# Check Coefficients
# model.coefficients() returns an array of shape (n_states, n_features).
# Since we have only 1 state variable (sig), we take the first row [0].
# These coefficients correspond to the terms in the library (1, sig, eps_dot, sig^2, ...).
coeffs = model.coefficients()[0]
print(f"Coefficients: {coeffs}")

# discovered_ode below only uses coeffs[0:3] (bias, sig, eps_dot). Fail loudly if the
# library/threshold settings above ever let SINDy keep non-negligible higher-order terms,
# instead of silently dropping them from the rollout.
if len(coeffs) > 3:
    residual = coeffs[3:]
    assert np.allclose(residual, 0.0, atol=1e-6), (
        f"discovered_ode only uses coeffs[0:3] (bias, sig, eps_dot) but SINDy kept "
        f"non-negligible higher-order terms: {residual}. Extend discovered_ode to "
        f"include them before rolling out."
    )


# ==========================================
# 3. VALIDATION (On Unseen Data)
# ==========================================

# Manual integration using discovered coefficients
# This function defines the ODE using the coefficients found by SINDy.
# We use this to simulate the learned model on new data.
def discovered_ode(sig, t, eps_dot_interp, t_interp, coeffs):
    # Interpolate the input (strain rate) at the current solver time t
    ed = np.interp(t, t_interp, eps_dot_interp)

    # Reconstruct derivative: dsig/dt = c0 + c1*sig + c2*eps_dot.
    # Only the first 3 terms (bias, sig, eps_dot) are used; the assertion above guarantees
    # any higher-order terms SINDy kept are negligible before this function is called.
    dsig_dt = coeffs[0] + coeffs[1] * sig + coeffs[2] * ed
    return dsig_dt

# Simulate on Validation Input
# We use odeint to solve the 'discovered_ode' using the VALIDATION input (eps_dot_val).
# This tests if the model generalizes to unseen data.
sig_val_pred = odeint(discovered_ode, 0.0, t, args=(eps_dot_val, t, coeffs)).flatten()


# ==========================================
# 4. VISUALIZATION
# ==========================================

import os
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "1D")
os.makedirs(FIG_DIR, exist_ok=True)

sig_train_pred = odeint(discovered_ode, 0.0, t, args=(eps_dot_train, t, coeffs)).flatten()


def save_panel(name, plot_fn):
    plt.figure(figsize=(7, 5))
    plot_fn()
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.show()


# 1. Training Inputs (Rich Signal)
def _p1():
    plt.plot(t, eps_train, 'b-', label=r'$\epsilon$ (Strain)')
    plt.plot(t, eps_dot_train, 'g-', alpha=0.7, label=r'$\dot{\epsilon}$ (Strain Rate)')
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
save_panel("01_training_inputs_rich_signal.png", _p1)

# 2. Training Outputs (Reconstruction)
def _p2():
    plt.plot(t, sig_train_noisy, 'k.', alpha=0.1, label="Noisy Data (Input to SINDy)")
    plt.plot(t, sig_train, 'k-', alpha=0.3, label="True Signal")
    plt.plot(t, sig_train_pred, 'r--', lw=2, label="SINDy Fit")
    plt.xlabel("Time")
    plt.ylabel("Stress")
save_panel("02_training_outputs_reconstruction.png", _p2)

# 3. Training Stress-Strain
def _p3():
    plt.plot(eps_train, sig_train, 'k-', alpha=0.3, label="True")
    plt.plot(eps_train, sig_train_pred, 'r--', lw=2, label="SINDy Fit")
    plt.xlabel(r"Strain $\epsilon$")
    plt.ylabel(r"Stress $\sigma$")
save_panel("03_training_stress_strain.png", _p3)

# 4. Validation Inputs (Unseen Signal)
def _p4():
    plt.plot(t, eps_val, 'b-', label=r'$\epsilon$ (Strain)')
    plt.plot(t, eps_dot_val, 'g-', alpha=0.7, label=r'$\dot{\epsilon}$ (Strain Rate)')
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
save_panel("04_validation_inputs_unseen_signal.png", _p4)

# 5. Validation Outputs (Generalization)
def _p5():
    plt.plot(t, sig_val, 'k-', lw=1, label="True Signal")
    plt.plot(t, sig_val_pred, 'r--', lw=2, label="SINDy Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stress")
save_panel("05_validation_outputs_generalization.png", _p5)

# 6. Validation Stress-Strain
def _p6():
    plt.plot(eps_val, sig_val, 'k-', lw=1, label="True")
    plt.plot(eps_val, sig_val_pred, 'r--', lw=2, label="SINDy Prediction")
    plt.xlabel(r"Strain $\epsilon$")
    plt.ylabel(r"Stress $\sigma$")
save_panel("06_validation_stress_strain.png", _p6)

# --- Threshold sensitivity sweep plot ---
# Visual evidence for the reviewer point: coefficient magnitude vs. threshold for every
# library term. The true terms (sig, eps_dot) stay flat across [0.5, 3.0]; spurious terms
# (e.g. eps_dot^3) are large and unstable below 0.5, and everything but eps_dot collapses
# above 3.0 (over-pruning).
feature_names_full = poly_lib.get_feature_names(["sig", "eps_dot"])
plt.figure(figsize=(9, 6))
for idx in range(plot_sweep_coeffs.shape[1]):
    vals = plot_sweep_coeffs[:, idx]
    if np.allclose(vals, 0.0):
        continue
    plt.plot(plot_thresholds, vals, marker='o', label=feature_names_full[idx])
plt.axvspan(0.5, 3.0, color='green', alpha=0.1, label="Stable plateau [0.5, 3.0]")
plt.xscale('log')
plt.xlabel("STLSQ threshold")
plt.ylabel("Coefficient value")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
sweep_fname = "07_threshold_sensitivity_sweep_degree3_library.png"
plt.savefig(os.path.join(FIG_DIR, sweep_fname), dpi=150)
print(f"Saved {os.path.join(FIG_DIR, sweep_fname)}")
plt.show()
