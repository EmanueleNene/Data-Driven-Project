---
tags: [overview, documentation, structure]
---
# Project Overview

**Summary**: A general overview of the repository structure, modules, and setup instructions.
**Sources**: `AGENTS.md`, `README.md`
**Last updated**: 2026-06-26

---

## Directory Organization
The repository is split into several experimental directories exploring Sparse Identification of Nonlinear Dynamics (SINDy) for material models:

- **Emanuele/**: Main workspace for viscoelasticity tensor 3D modeling.
  - **ViscoElasticity_tensor_3D/**: Active benchmark directory.
    - **NonLinear_3D/**: Contains the latest J2-coupled nonlinear SINDy model, datasets, plots, and LaTeX report.
    - **archive/**: Archived linear deviatoric SINDy scripts, datasets, and plots.
    - **archive_von_mises/**: Older, non-working von Mises 3D exploration files.
- **marcus/**: Contains legacy numbered Python experiments for elastic, viscoelastic, and viscoplastic models.
- **SINDY for matlab/**: MATLAB scripts and supporting `.mat` data.

## Setup and Installation
To run the modeling and fitting scripts, set up a Python virtual environment and install the required dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy scipy matplotlib pysindy jupyter
```

## Related pages
- [[nonlinear-viscoelasticity-sindy]]
- [[reviewer-comments-resolution]]
