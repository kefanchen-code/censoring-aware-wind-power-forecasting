# Compact reference results

This directory contains compact, read-only summaries from the frozen manuscript
runs. It is intended for table checking and figure regeneration without
retraining. Large model weights, per-window predictions, and training artifacts
are deliberately excluded.

The included Altahullion semi-synthetic excerpt is derived from the CC BY 4.0
dataset identified in `../DATA.md`. All other files are numerical summaries
created by the scripts in this repository.

Fresh benchmark runs write to `results/`, not to this directory.
