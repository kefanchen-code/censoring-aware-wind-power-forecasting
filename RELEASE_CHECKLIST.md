# Pre-submission release checklist

## Blocking before public release

- [x] Confirm the software author and copyright holder (Kefan Chen; affiliation: Zhejiang University of Technology).
- [x] Add the BSD 3-Clause License for code and separate CC BY 4.0 notices for data-derived artifacts.
- [x] Align the software title, version, author affiliation, and citation metadata with the manuscript.
- [x] Document the internal model identifiers used in configurations and archived outputs.
- [x] Remove the out-of-scope multi-turbine aggregation-sensitivity experiment from the Paper 1 release.
- [x] Translate the remaining Chinese-only analysis-script docstrings and console labels needed by external users.
- [x] Validate the pinned environment in a clean, isolated environment.
- [x] Run the full benchmark from freshly downloaded data and compare all paper numbers.
- [x] Run a secret and personal-path scan on the final Git tree.
- [x] Replace this private candidate with a tagged `v1.0.0` release.
- [ ] Archive the tagged release in Zenodo and add its version DOI to `CITATION.cff` and the manuscript.

## Already completed in this candidate

- [x] Paper 1 isolated from the wind-to-hydrogen value-study code.
- [x] Main source package, frozen configurations, and tests collected.
- [x] Local absolute path in the main Altahullion preparation script removed.
- [x] Exact result-generation environment recorded.
- [x] Public Altahullion download URL and archive checksum recorded.
- [x] Compact reference outputs collected for figure and table checking.
- [x] Core test suite executed successfully before repository creation.
- [x] Hill of Towie fast-log-to-Parquet conversion restored and checked exactly across two archived turbine-days.
- [x] Missing release-event protocol replaced by an explicitly reconstructed v1.3 protocol without overwriting the legacy v1.2 freeze.
- [x] Historical truth-channel assets rebuilt with exact archived route counts and a recomputed hold-out error.
- [x] Full three-seed v1.3 release-event run reproduced the shared v1.2 accounting and statistical results.
