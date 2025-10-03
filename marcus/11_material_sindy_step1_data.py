import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------
# Hyperelastic model: incompressible Neo-Hookean
# Cauchy stress sigma for standard load paths
# mu: shear modulus
# --------------------------------------------

def neo_hookean_uniaxial_sigma11(lam, mu):
    # Uniaxial stretch along 1 with traction-free lateral faces
    # Principal stretches: [lam, lam^{-1/2}, lam^{-1/2}]
    # Cauchy: sigma_11 = mu (lam^2 - lam^{-1}), sigma_22 = sigma_33 = 0
    return mu * (lam**2 - lam**-1)

def neo_hookean_equibiaxial_sigma11(lam, mu):
    # Equibiaxial stretch in 1 and 2, traction-free in 3
    # Principal stretches: [lam, lam, lam^{-2}]
    # Cauchy: sigma_11 = sigma_22 = mu (lam^2 - lam^{-4}), sigma_33 = 0
    return mu * (lam**2 - lam**-4)

def neo_hookean_simple_shear_sigma12(gamma, mu):
    # Simple shear with F = [[1, gamma, 0],[0,1,0],[0,0,1]]
    # For incompressible Neo-Hookean: sigma_12 = mu * gamma
    # Normal stress difference N1 = sigma_11 - sigma_22 = mu * gamma^2 (we won't plot it here)
    return mu * gamma

# --------------------------------------------
# Load paths
# --------------------------------------------

def make_loadpaths():
    # Ranges you can tweak
    lam_u = np.linspace(0.7, 1.6, 120)    # uniaxial lambda
    lam_b = np.linspace(0.7, 1.4, 120)    # equibiaxial lambda (keep moderate to avoid huge stresses)
    gam_s = np.linspace(0.0, 1.0, 120)    # simple shear gamma

    paths = [
        {"name": "uniaxial",   "abscissa_name": "λ",     "x": lam_u, "kind": "stretch"},
        {"name": "equibiaxial","abscissa_name": "λ",     "x": lam_b, "kind": "stretch"},
        {"name": "shear",      "abscissa_name": "γ",     "x": gam_s, "kind": "shear"},
    ]
    return paths

# --------------------------------------------
# Generate dataset from chosen model
# --------------------------------------------

def generate_dataset(mu=1.7):
    paths = make_loadpaths()
    data = []

    for p in paths:
        if p["name"] == "uniaxial":
            x = p["x"]
            y = neo_hookean_uniaxial_sigma11(x, mu)
            y_label = "σ₁₁ (uniaxial)"
        elif p["name"] == "equibiaxial":
            x = p["x"]
            y = neo_hookean_equibiaxial_sigma11(x, mu)
            y_label = "σ₁₁ (equibiaxial)"
        elif p["name"] == "shear":
            x = p["x"]
            y = neo_hookean_simple_shear_sigma12(x, mu)
            y_label = "σ₁₂ (simple shear)"
        else:
            continue

        data.append({
            "path": p["name"],
            "abscissa_name": p["abscissa_name"],
            "x": x,
            "y": y,
            "y_label": y_label
        })
    return data

# --------------------------------------------
# Plot helper
# --------------------------------------------

def plot_paths(data, title="Neo-Hookean (μ true)"):
    n = len(data)
    plt.figure(figsize=(9, 6))
    for d in data:
        plt.plot(d["x"], d["y"], lw=2, label=f'{d["y_label"]}')
    # x-label depends on mixed paths; just show a generic one with symbols
    plt.xlabel("λ (stretch) or γ (shear)")
    plt.ylabel("Cauchy stress component")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------------------------
# Main (Step 1)
# --------------------------------------------

if __name__ == "__main__":
    mu_true = 1.7
    data = generate_dataset(mu=mu_true)
    plot_paths(data, title=f"Neo-Hookean (μ = {mu_true}) — True curves")
