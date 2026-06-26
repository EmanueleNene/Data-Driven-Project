# Implementation Plan: Nonlinear Viscoelastic Benchmark (Option 2)

This plan details the steps to introduce a true multiaxial generalization benchmark using a **nonlinear viscoelastic material model** with coupled deviatoric components.

## 1. Objective & Source Issue
* **Source:** Point 4 & 7 in [Review_A.md](file:///D:/Data-driven%20project/Data-Driven-Project/Emanuele/Review_A.md)
* **Goal:** Prove that the SINDy framework generalizes to complex multiaxial responses where the components do not evolve independently (uncoupled). By making viscosity stress-dependent, all deviatoric components are algebraically coupled through the second invariant of the stress deviator (equivalent stress $\sigma_{eq}$).

---

## 2. Mathematical Formulation (Nonlinear Maxwell)
In an isotropic nonlinear Maxwell model, the deviatoric stress rate is:
$$\dot{S}_{ij} = 2G \dot{e}_{ij} - \frac{2G}{\eta(\sigma_{eq})} S_{ij}$$

To make this discoverable by SINDy's polynomial library, we define the relaxation rate ($1/\eta$) as a polynomial function of the equivalent stress squared (which is proportional to the second invariant $J_2 = \frac{1}{2} \boldsymbol{S} : \boldsymbol{S}$):
$$\frac{1}{\eta(\sigma_{eq})} = \frac{1}{\eta_0} \left(1 + \alpha \sigma_{eq}^2\right)$$
where $\sigma_{eq} = \sqrt{\frac{3}{2} \boldsymbol{S} : \boldsymbol{S}}$.

Substituting this back into the evolution law yields:
$$\dot{S}_{ij} = 2G \dot{e}_{ij} - \frac{2G}{\eta_0} S_{ij} - \frac{2G\alpha}{\eta_0} S_{ij} \sigma_{eq}^2$$

This creates a **cubic coupling** between components:
* $\dot{S}_{xx}$ depends on $S_{xx}^3$, $S_{xx} S_{xy}^2$, $S_{xx} S_{yy}^2$, etc.
* SINDy must discover these cross-component cubic terms to recover the correct physical parameters ($G$, $\eta_0$, and $\alpha$).

---

## 3. Concrete Actions & Affected Files

| # | Action | Target File | Difficulty | Effort |
|---|---|---|---|---|
| 1 | Create a coupled ODE data generator that integrates the nonlinear deviatoric Maxwell equations. | `Emanuele/ViscoElasticity_tensor_3D/generate_nonlinear_deviatoric_data.py` | Medium | 4–6 h |
| 2 | Develop a SINDy script that sets up a custom or polynomial library (degree 3) to fit the coupled components. | `Emanuele/ViscoElasticity_tensor_3D/fit_nonlinear_deviatoric_sindy.py` | High | 6–10 h |
| 3 | Validate parameter recovery ($G, \eta_0, \alpha$) and test predictions on unseen loading paths. | `Emanuele/ViscoElasticity_tensor_3D/validate_nonlinear_deviatoric.py` | Medium | 3–5 h |
| 4 | Clean up plots and compare linear vs. nonlinear fits to highlight the improvement. | `Emanuele/ViscoElasticity_tensor_3D/plot_nonlinear_results.py` | Low | 2–3 h |

---

## 4. Feasibility, Difficulty & Estimated Effort
* **Feasibility:** High. PySINDy easily handles custom inputs and polynomial libraries of degree 3.
* **Difficulty:** High. Coupled ODE integration requires numerical care (implicit solvers if equations become stiff), and SINDy fitting requires threshold tuning to prevent overfitting of spurious cubic terms.
* **Total Estimated Effort:** **15–24 hours**.

---

## 5. Dependencies & Key Decisions
1. **Cubic Library Size:** A full 3D polynomial library of degree 3 for 6 stress components + 6 strain rates will contain a huge number of features, leading to high variance and potential fitting failure.
   * *Mitigation:* We should restrict the SINDy feature library to candidate terms that are physically meaningful (e.g. $S_{ij}$, $\dot{e}_{ij}$, and $S_{ij} (\boldsymbol{S} : \boldsymbol{S})$) rather than a brute-force polynomial search.
2. **Stiffness:** At high values of $\alpha$ or stress, the ODEs may become stiff. We will use a robust solver like `scipy.integrate.solve_ivp` with the `'Radau'` method instead of `odeint`.

---

## 6. Completion Criteria
* SINDy recovers the parameters $2G$, $-2G/\eta_0$, and $-2G\alpha/\eta_0$ within $5\%$ tolerance under moderate noise.
* The model trained on uniaxial and shear data correctly predicts combined multiaxial paths at different stress levels where the linear model fails.
