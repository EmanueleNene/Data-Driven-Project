import numpy as np
import pysindy as ps
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


# ----- synth data: Kelvin–Voigt -----
E_true, eta_true = 5.0, 0.8
T, dt = 20.0, 0.001
t = np.arange(0.0, T, dt)
eps = 0.05*np.sin(2*np.pi*0.7*t) * np.exp(-0.1*t)         # strain history
eps_dot = np.gradient(eps, dt)                             # true derivative
sig = E_true*eps + eta_true*eps_dot                        # stress

# (optional) add noise and smooth
#rng = np.random.default_rng(0); sig += 0.002*rng.standard_normal(sig.shape)
eps_s = savgol_filter(eps, 101, 3)                         # smooth strain
eps_dot_s = np.gradient(eps_s, dt)
sig_s = savgol_filter(sig, 101, 3)    

#plt.figure()
#plt.plot(eps, sig, "k", linewidth=2, label="clean data")
#plt.plot(eps_s, sig_s, "r--", linewidth=2, label="noisy data")
#plt.legend()
#plt.title("noise")
#plt.xlabel("eps"); plt.ylabel("σ")
#plt.grid(True)
#plt.show()

# features and target
X = np.column_stack([eps_s, eps_dot_s])   # [ε, ε̇]
y = sig_s                               # σ

# SINDy as plain sparse regression
lib = ps.IdentityLibrary()
#lib = lib + ps.PolynomialLibrary(degree=3, include_bias=False)
opt = ps.STLSQ(threshold=0.10)
model = ps.SINDy(feature_library=lib, optimizer=opt, discrete_time=True)
t_dummy = np.arange(len(eps))
model.feature_names = ["eps","eps_dot"]   # names for columns
model.state_names   = ["sig"]             # just for printing
model.fit(X, t=t_dummy, x_dot=y)

# coefficients -> E, η
coefs = model.coefficients()[0]  # [a, b] with σ ≈ a*ε + b*ε̇
E_est, eta_est = float(coefs[0]), float(coefs[1])
print(f"E ≈ {E_est:.4f},  eta ≈ {eta_est:.4f}")
model.print()

# --- Noise analysis & plots ---

# "True" stress from the generator (before any smoothing)
sig_true = E_true*eps + eta_true*eps_dot

# Measured (raw) vs smoothed
sig_raw = sig          # before smoothing (add noise above if you want)
sig_smooth = sig_s     # after Savitzky–Golay

# Simple noise estimate: raw - smoothed
noise_est = sig_raw - sig_smooth

# Metrics
rms_signal = np.sqrt(np.mean((sig_true - np.mean(sig_true))**2))
rms_noise  = np.sqrt(np.mean(noise_est**2))
snr_db = np.inf if rms_noise == 0 else 20*np.log10(rms_signal / rms_noise)
noise_pct = 0.0 if rms_noise == 0 else 100 * rms_noise / rms_signal

print(f"\nNoise RMS = {rms_noise:.3e}")
print(f"Signal RMS = {rms_signal:.3e}")
print(f"SNR = {snr_db:.1f} dB   |   Noise ≈ {noise_pct:.2f}% of signal RMS")

# Plots
#import matplotlib.pyplot as plt

# 1) Signal vs smoothed vs raw
plt.figure(figsize=(9,4))
plt.plot(t, sig_true,  lw=2, label="σ true")
plt.plot(t, sig_raw,   lw=1, alpha=0.6, label="σ raw")
plt.plot(t, sig_smooth,"--", lw=2, label="σ smoothed")
plt.xlabel("t"); plt.ylabel("σ"); plt.title("Stress: true vs raw vs smoothed")
plt.legend(); plt.tight_layout(); plt.show()

# 2) Noise time series
plt.figure(figsize=(9,3))
plt.plot(t, noise_est, lw=1, label="noise estimate = raw - smoothed")
plt.xlabel("t"); plt.ylabel("noise")
plt.title("Noise estimate over time")
plt.tight_layout(); plt.show()

# 3) Noise histogram
plt.figure(figsize=(6,4))
plt.hist(noise_est, bins=50)
plt.xlabel("noise"); plt.ylabel("count"); plt.title("Noise histogram")
plt.tight_layout(); plt.show()
