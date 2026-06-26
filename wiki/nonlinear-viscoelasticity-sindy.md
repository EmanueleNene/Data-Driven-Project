---
tags: [sindy, viscoelasticity, j2-plasticity, nonlinear]
---
# Nonlinear Viscoelasticity SINDy Model

**Summary**: Physics-informed SINDy model for J2-coupled 3D nonlinear viscoelasticity in deviatoric space, including derivation, library design, and results.
**Sources**: `generate_nonlinear_deviatoric_data.py`, `fit_nonlinear_deviatoric_sindy.py`
**Last updated**: 2026-06-26

---

## Physical Derivation
For an isotropic viscoelastic material, the shape-changing (deviatoric) stress components $S_{ij}$ drive the viscous flow. The governing Maxwell model in deviatoric space is:
$$\frac{dS_{ij}}{dt} = 2G \dot{\varepsilon}^{dev}_{ij} - \frac{2G}{\eta(\sigma_{eq})} S_{ij}$$

The viscosity $\eta(\sigma_{eq})$ is stress-dependent and modeled using J2 flow theory:
$$\frac{1}{\eta(\sigma_{eq})} = \frac{1}{\eta_0}(1 + \alpha \sigma_{eq}^2)$$
where $\sigma_{eq}^2 = 1.5 (\mathbf{S} : \mathbf{S})$. Expanding this yields:
$$\frac{dS_{ij}}{dt} = 2G \dot{\varepsilon}^{dev}_{ij} - \frac{2G}{\eta_0} S_{ij} - \frac{2G\alpha}{\eta_0} (S_{ij} \sigma_{eq}^2)$$

Thus, the rate of change of each stress component is a combination of three terms:
1. Deviatoric strain rate ($\dot{\varepsilon}^{dev}_{ij}$)
2. Linear stress component ($S_{ij}$)
3. Nonlinear J2 coupled stress component ($S_{ij} \sigma_{eq}^2$)

## Physics-Informed SINDy Library Design
Instead of using a naive cubic polynomial library of 12 variables (which generates **455 candidate terms** and leads to severe overfitting and collinearity issues), we pre-compute the J2 coupled stress term:
$$Seq_{ij} = S_{ij} \sigma_{eq}^2$$
We pass $Seq_{ij}$ as a known control input variable to SINDy, reducing the library size to just **3 candidate terms** (bias, linear stress, strain rate, and J2-coupled stress). This guides the optimizer to the exact physical form.

## Split Training vs. Coupled Validation
- **Split Training**: During uniaxial tension and pure shear tests, stress/strain components are collinear (e.g. $S_{yy} = -0.5 S_{xx}$ in uniaxial tension). Directly fitting a 6D SINDy model splits the coefficients. We fit SINDy component-by-component ($S_{xx}$ on uniaxial data, $S_{xy}$ on shear data) to isolate the independent coefficients.
- **Coupled Validation**: Because $\sigma_{eq}^2$ depends on all stress components, the components are physically coupled. To validate, we integrate the 6 discovered ODEs together as a **coupled 6D system**.

## Results
- **Shear Modulus ($G$)**: $999.91\text{ MPa}$ (True: $1000.0\text{ MPa}$ | Error: $0.009\%$)
- **Reference Viscosity ($\eta_0$)**: $499.88\text{ MPa}\cdot\text{s}$ (True: $500.0\text{ MPa}\cdot\text{s}$ | Error: $0.024\%$)
- **Nonlinear Parameter ($\alpha$)**: $0.00015\text{ 1/MPa}^2$ (True: $0.00015\text{ 1/MPa}^2$ | Error: $0.228\%$)

## Related pages
- [[project-overview]]
- [[reviewer-comments-resolution]]
