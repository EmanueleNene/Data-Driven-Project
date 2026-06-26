# Operation Log

All changes made to the repository wiki and modeling code are tracked here.

## 2026-06-26 — Initial Vault Scaffold and SINDy Correction
- Scaffolded the Obsidian wiki vault with `index.md`, `log.md`, `project-overview.md`, `nonlinear-viscoelasticity-sindy.md`, and `reviewer-comments-resolution.md`.
- Corrected the J2-coupled nonlinear deviatoric SINDy model in `fit_nonlinear_deviatoric_sindy.py`:
  - Implemented component-wise fitting ($S_{xx}$ and $S_{xy}$) to resolve multicollinearity in simple load cases.
  - Tuned the STLSQ sparsity threshold from `0.005` to `1e-5` to capture the small nonlinear coefficient ($c_3 = -0.0006$).
  - Consolidated 1D parameters into a robust, isotropic 6D SINDy model.
- Validated the 6D SINDy model on unseen paths (combined tension-shear and biaxial tension) with under $0.1\%$ error (RMSE $< 0.02\text{ MPa}$).
- Generated and saved the validation curves to `NonLinear_3D/nonlinear_deviatoric_comparison.png`.
- Compiled a LaTeX PDF report `results.pdf` inside `NonLinear_3D/latex/`.
- Reorganized files: moved all nonlinear assets to `Emanuele/ViscoElasticity_tensor_3D/NonLinear_3D/` and archived linear assets into `Emanuele/ViscoElasticity_tensor_3D/archive/`.
