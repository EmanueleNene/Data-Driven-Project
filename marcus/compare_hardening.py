import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Physics Parameters
# -----------------------------
Q = 12.0
b = 40.0
eta = 20.0

# -----------------------------
# 2. SINDy Learned Coefficients (from your output)
# -----------------------------
# model: ep_dot = w_over*over + w1*ep + w2*ep^2
# H_sindy = -eta * (w1*ep + w2*ep^2)
w1 = -24.76597
w2 = 407.3312

# -----------------------------
# 3. Define Plastic Strain Range
# -----------------------------
# Go up to a reasonable max strain, e.g., 0.05 (5%)
ep = np.linspace(0, 0.05, 500)

# -----------------------------
# 4. Calculate Hardening Curves
# -----------------------------

# A. True Elastic-Exponential Hardening (Voce)
# H = Q * (1 - exp(-b * ep))
H_true = Q * (1.0 - np.exp(-b * ep))

# B. Theoretical Taylor Expansion (2nd Order)
# H ≈ Q * (b*ep - (b*ep)^2 / 2)
# H ≈ (Qb)*ep - (Qb^2/2)*ep^2
term_lin_theor = Q * b
term_quad_theor = - (Q * b**2) / 2.0
H_taylor = term_lin_theor * ep + term_quad_theor * ep**2

# C. SINDy Discovered Hardening
# H_sindy = -eta * (w1*ep + w2*ep^2)
#          = (-eta*w1)*ep + (-eta*w2)*ep^2
term_lin_sindy  = -eta * w1
term_quad_sindy = -eta * w2
H_sindy = term_lin_sindy * ep + term_quad_sindy * ep**2

# -----------------------------
# 5. Print Comparison
# -----------------------------
print(f"{'Term':<20} | {'Theoretical (Taylor)':<20} | {'SINDy Discovered':<20}")
print("-" * 66)
print(f"{'Linear Coef':<20} | {term_lin_theor:<20.4f} | {term_lin_sindy:<20.4f}")
print(f"{'Quadratic Coef':<20} | {term_quad_theor:<20.4f} | {term_quad_sindy:<20.4f}")

# -----------------------------
# 6. Plot
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(ep, H_true, 'k-', lw=2.5, label='True Voce (Exponential)')
plt.plot(ep, H_taylor, 'g--', lw=2, label='Theoretical Taylor (2nd Order)')
plt.plot(ep, H_sindy, 'r:', lw=3, label='SINDy Discovered (Polynomial)')

plt.title("Hardening Law Comparison: True vs Taylor vs SINDy")
plt.xlabel("Plastic Strain $\\epsilon_p$")
plt.ylabel("Hardening Stress $H(\\epsilon_p)$ [MPa]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
