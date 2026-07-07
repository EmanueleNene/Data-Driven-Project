# Figure / Result Source Mapping

Per Review_A point 10: every reported figure should map to one named script or notebook cell.
Paste the relevant row(s) into the report where each figure is introduced.

| Figure / result | Source | Method |
|---|---|---|
| 1D Maxwell viscoelastic fit/validation plots | `Emanuele/simple_visco_sindy.py` | STLSQ, `threshold=1.0` (stable across sweep `[0.5, 3.0]`), `PolynomialLibrary(degree=3)` |
| Plastic (Voce-Perzyna) identification | `Emanuele/Data_driven_3D_single_load.ipynb`, SR3 cell (`# --- SINDy sparse regression (SR3) ---`) | SR3, `reg_weight_lam=0.10`, `relax_coeff_nu=1.00` |

3D deviatoric figures (Section 2.5/2.6) are not yet listed here — add once that pipeline is
finalized per action-plan items 1, 5, 6.
