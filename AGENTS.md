# Repository Guidelines

## Project Structure & Module Organization

This repository contains exploratory Sparse Identification of Nonlinear Dynamics (SINDy) work for constitutive material models.

- `Emanuele/` contains Python scripts and notebooks for viscoelasticity experiments. Its `ViscoElasticity_tensor_3D/` subdirectory includes data generators, model-fitting scripts, `.npz` datasets, utilities, and result figures.
- `marcus/`[OLD] contains numbered Python experiments for elastic, viscoelastic, and viscoplastic models. Higher-numbered files generally represent later investigations; see `marcus/readme.txt`.
- `SINDY for matlab/` contains MATLAB examples and supporting `.mat` data.
- Root-level notebooks, CSV/PDF inputs, and HTML exports support broader experiments and presentations.

Keep generated data and figures beside the experiment that produces them. Avoid committing notebook checkpoint directories or temporary interpreter files.

## Build, Test, and Development Commands

There is no packaged build or shared dependency file. Use a virtual environment and install dependencies required by the target script, commonly:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy scipy matplotlib pysindy jupyter
```

Run scripts from their own directory because several use relative data paths:

```powershell
cd Emanuele\ViscoElasticity_tensor_3D
python generate_deviatoric_data.py
python simple_visco_sindy_deviatoric.py
```

Launch notebooks with `jupyter notebook`, or use Google Colab as described in `README.md`. Run MATLAB examples from `SINDY for matlab/`.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for functions and variables, `UPPER_CASE` for constants, and descriptive module names. Preserve the existing numbered naming pattern in `marcus/` when extending a sequence. Keep physical units explicit in names, comments, or plot labels. Prefer small reusable helpers over duplicated tensor or invariant calculations.

## Testing Guidelines

No automated test suite or coverage threshold is configured. Validate changes by running the affected generator and fitting script end-to-end. Check that outputs remain finite, expected coefficients are recovered within documented tolerances, and plots/data files are produced. For reusable utilities, add focused `pytest` tests under a new `tests/` directory using names such as `test_von_mises_utils.py`.

## Commit & Pull Request Guidelines

History uses short imperative summaries, for example `Update README with improved descriptions`. Use a specific subject naming the experiment or model; avoid generic messages such as `Add files via upload`.

Pull requests should describe the model, changed loading path, dependencies, and validation performed. Link related issues and include representative plots when numerical behavior changes. Do not mix unrelated notebook outputs or large regenerated datasets into the same review.
