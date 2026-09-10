# Compact reference results

This directory contains compact, read-only summaries from the frozen manuscript
runs. It is intended for table checking and figure regeneration without
retraining. Large model weights, per-window predictions, and training artifacts
are deliberately excluded.

The included Altahullion semi-synthetic excerpt is derived from the CC BY 4.0
dataset identified in `../DATA.md`. All other files are numerical summaries
created by the scripts in this repository.

Fresh benchmark runs write to `results/`, not to this directory.

The frozen WIS-decomposition tables preserve the original manuscript model set.
The current decomposition script additionally emits rows for `B2_recon` and
`CLQR`; all rows in the historical domain reproduce numerically.

The `release_event_validation/` directory preserves the historical v1.2
summary whose protocol document is no longer available. The executable
replacement is frozen in `protocols/release_event_protocol_v1_3.md`; its full
three-seed summary is stored in `release_event_validation_v1_3/` and matches
the shared v1.2 event accounting, split counts, comparisons, and statistics.
