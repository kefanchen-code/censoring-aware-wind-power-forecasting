# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-09-10

First public software release supporting the manuscript “Probabilistic
forecasting of potential available power under control-induced censoring for a
one-turbine wind-to-hydrogen off-grid microgrid.”

### Included

- Frozen S0--S5 semi-synthetic benchmark configurations and code.
- Censoring-aware PMF, point-label, deletion, reconstruction, quantile,
  censored-quantile, Tobit, climatology, persistence, and oracle models.
- Unit and integration tests for information separation, censoring losses,
  temporal splitting, inference, and recovered preprocessing.
- Compact reference summaries and scripts for all six quantitative manuscript
  figures.
- Auditable Altahullion release-event protocol reconstruction (v1.3), with the
  historical v1.2 protocol identifier and results retained separately.
- Data-source, licensing, citation, and reproducibility documentation.
- A machine-independent experiment fingerprint based only on the frozen
  protocol, input-data hashes, segment assignment, and comparison plan.

### Data scope

Raw Altahullion and Hill of Towie turbine data are not redistributed. They are
obtained from the source Zenodo records described in `DATA.md`. Tracked compact
derived data artifacts are released under CC BY 4.0; source code is released
under the BSD 3-Clause License.
