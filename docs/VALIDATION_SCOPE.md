# Validation scope

## Included

- The latest remote Transolver experiment entrypoints for Darcy, elasticity,
  Navier–Stokes, and plasticity.
- Physics-embedding utilities used by those experiments.
- A standalone PINN toolkit for manufactured 2D PDE cases.
- Unit checks that do not require benchmark data or a GPU.

## Not included

- Private datasets, checkpoints, logs, figures, or reports.
- Cluster-specific watchers, process-management scripts, and absolute paths.
- Older Phase-1 implementations that were superseded by the official benchmark
  wrappers.
- Third-party source copied into this repository. Transolver is referenced as a
  Git submodule at its official upstream repository.

## Interpretation

The experiment groups are retained to make the research process reproducible.
Their presence does not imply that every constraint improves accuracy.
Strong-form residuals, reduced constraints, and conservation penalties must be
validated against the data-generation discretization and output semantics.
