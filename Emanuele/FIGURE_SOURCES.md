# Figure / Result Source Mapping

Per Review_A point 10: every reported figure should map to one named script or notebook cell.
Paste the relevant row(s) into the report where each figure is introduced.

| Figure file | Source script | Method |
|---|---|---|
| `figures/1D/01_training_inputs_rich_signal.png` | `Emanuele/simple_visco_sindy.py` | 5-sine training excitation (data generation) |
| `figures/1D/02_training_outputs_reconstruction.png` | `Emanuele/simple_visco_sindy.py` | STLSQ, `threshold=1.0`, `PolynomialLibrary(degree=3)`, stable over sweep `[0.5, 3.0]` |
| `figures/1D/03_training_stress_strain.png` | `Emanuele/simple_visco_sindy.py` | same fit as above, stress-strain view |
| `figures/1D/04_validation_inputs_unseen_signal.png` | `Emanuele/simple_visco_sindy.py` | single-frequency validation excitation (data generation) |
| `figures/1D/05_validation_outputs_generalization.png` | `Emanuele/simple_visco_sindy.py` | `discovered_ode` rollout, coefficients at `threshold=1.0` |
| `figures/1D/06_validation_stress_strain.png` | `Emanuele/simple_visco_sindy.py` | same rollout as above, stress-strain view |
| `figures/1D/07_threshold_sensitivity_sweep_degree3_library.png` | `Emanuele/simple_visco_sindy.py` | threshold sweep `[0.01, 2.0]`, degree-3 library, all 10 terms plotted |
| `figures/3D/01_combined_Sxx_rmse0.0022.png` | `Emanuele/ViscoElasticity_tensor_3D/NonLinear_3D/fit_nonlinear_deviatoric_sindy.py` (data from `generate_nonlinear_deviatoric_data.py`) | degree-3 library, rescale→prune→unscale STLSQ, `threshold=5.0` (rescaled space), biaxial training paths |
| `figures/3D/02_combined_Sxy_rmse0.0051.png` | same as above | same |
| `figures/3D/03_biaxial_Sxx_rmse0.0032.png` | same as above | same |
| `figures/3D/04_biaxial_Syy_rmse0.0028.png` | same as above | same |
| `figures/3D/05_threshold_sensitivity_sweep_degree3_rescaled.png` | `Emanuele/ViscoElasticity_tensor_3D/NonLinear_3D/fit_nonlinear_deviatoric_sindy.py` | threshold sweep `[1e-3, 5.0]` (rescaled space), degree-3 library, $S$/$ed$/$Seq$ terms for both normal and shear component fits |
| `figures/plastic/01_validation_stress_strain.png` | `Emanuele/24d_vocePerzyna1D_finalPres.py` | Voce-Perzyna 1D standalone script, validation stress-strain rollout |
| `figures/plastic/02_validation_plastic_strain_accumulation.png` | `Emanuele/24d_vocePerzyna1D_finalPres.py` | same fit, plastic strain vs time |
| `figures/plastic/03_validation_hardening_curve.png` | `Emanuele/24d_vocePerzyna1D_finalPres.py` | same fit, hardening curve (sigma vs ep) |
| `figures/plastic/04_hardening_residual.png` | `Emanuele/25_voce_perzyna_residual.py` | reproduces 24d fit, residual of SINDy poly vs exact Voce law and vs 2nd-order Taylor, over training range of ep |
