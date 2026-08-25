# Pre-submission release checklist

## Blocking before public release

- [ ] Confirm the complete author list and software copyright holder.
- [ ] Select and add an open-source `LICENSE` approved by the supervisor or institution.
- [ ] Restore or rewrite the Hill of Towie fast-log-to-Parquet preparation step.
- [ ] Recover or replace the missing frozen protocol source used by the real release-event validation.
- [ ] Decide whether the historical truth-channel analysis is reproducible enough to retain in the public code release.
- [ ] Translate the remaining Chinese-only analysis-script docstrings and console labels needed by external users.
- [ ] Validate the locked environment on a second clean machine.
- [ ] Run the full benchmark from freshly downloaded data and compare all paper numbers.
- [ ] Run a secret and personal-path scan on the final Git tree.
- [ ] Replace this private candidate with a tagged `v1.0.0-paper1` release.
- [ ] Archive the tagged release in Zenodo and add its version DOI to `CITATION.cff` and the manuscript.

## Already completed in this candidate

- [x] Paper 1 isolated from the wind-to-hydrogen value-study code.
- [x] Main source package, frozen configurations, and tests collected.
- [x] Local absolute path in the main Altahullion preparation script removed.
- [x] Exact result-generation environment recorded.
- [x] Public Altahullion download URL and archive checksum recorded.
- [x] Compact reference outputs collected for figure and table checking.
- [x] Core test suite executed successfully before repository creation.
