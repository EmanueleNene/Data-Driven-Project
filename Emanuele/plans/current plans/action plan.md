# Review A Action Plan

This plan covers the issues raised in `Review_A.md`. Estimates apply to implementation and verification in `Emanuele/`; report rewriting is excluded unless noted.

| # | Action | Feasibility | Difficulty | Estimated effort |
|---|---|---|---|---|
| 1 | Standardize the 3D viscoelastic workflow on deviatoric components. Remove scalar von Mises viscoelastic code and results from the active pipeline. | High | Medium | 3–5 h |
| 2 | Correct Equations 15 and 28 in the report. Check code comments and plot labels for the same notation errors. | High | Low | 0.5–1 h |
| 3 | Extend the plastic notebook to compare the learned polynomial with the exact Voce–Perzyna law over the plastic-strain range reached during training. Report RMSE, normalized RMSE, and maximum error. | High | Medium | 3–5 h |
| 4 | Rename the current 3D “generalization” results as sanity checks or unseen-loading-path validation. A true generalization benchmark requires nonlinear, coupled, or anisotropic constitutive behavior. | High for relabeling; medium for a new benchmark | Low / High | 1–2 h / 12–24 h |
| 5 | Remove or archive `generate_3d_data.py`, its non-deviatoric datasets, `simple_visco_sindy_3d.py`, and related figures. Make the deviatoric generator authoritative. | High | Medium | 2–4 h |
| 6 | Make `ViscoElasticity_tensor_3D/simple_visco_sindy_deviatoric.py` fully reproducible using deterministic paths, named outputs, documented dependencies, and a clean generation-to-fitting command sequence. | High | Medium | 4–8 h |
| 7 | Remove unsupported orthotropic and coupled-material claims from the report or label them as future work. Supporting them requires a new model, generator, feature library, and validation experiment. | High for claim removal; uncertain for experimental proof | Low / Very high | Report-only / 20–40+ h |
| 8 | Expand the report reference list. Recording library versions and citations in the repository is useful but does not replace the report references. | High | Low | 0.5–1 h |
| 9 | Fix encoding corruption, inconsistent notation, spelling, filenames, plot labels, comments, and dead or duplicated code. Add formatting and basic checks. | High | Medium | 4–8 h |
| 10 | Make manual rollout evaluate the complete fitted feature library, or assert that ignored coefficients are negligible. Document which script or notebook cell produces every report figure and distinguish STLSQ viscoelastic results from SR3 plastic results. | High | Medium | 3–6 h |

## Recommended Order

1. Resolve the conflicting 3D pipelines: points 1 and 5.
2. Make the deviatoric workflow reproducible: point 6.
3. Correct rollout handling and figure provenance: point 10.
4. Add the Voce–Perzyna residual metrics: point 3.
5. Clean terminology, encoding, and validation claims: points 4 and 9.
6. Complete report-only corrections: points 2, 7, and 8.

## Overall Estimate

The minimal defensible code revision is approximately **17–31 hours**. Adding a genuinely coupled or anisotropic benchmark would increase the total to approximately **37–70+ hours**.
